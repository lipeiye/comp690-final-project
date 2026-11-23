"""Post-test study: smarter electrode selection than per-electrode relevance and redundancy.

Review question: the redundancy-aware rule barely changed the chosen electrodes. Are there
better ways to reduce the montage? The frozen rule scores electrodes one at a time; a
motor-imagery decoder works through spatial filters, which combine electrodes, so the
methods compared here judge electrodes jointly or bring in outside knowledge:

  riemann       Barachant and Bonnet (2011): keep the subset whose class-mean spatial
                covariances stay farthest apart (mibci.selectors.riemann_order)
  csp           one-vs-rest common spatial patterns (mibci.selectors.csp_order)
  wrapper       backward elimination by cross-validated accuracy of a fast surrogate decoder
                (tangent space + logistic regression, session-1 runs held out in pairs)
  learned       a concrete (Gumbel-softmax) selector trained end to end in front of EEGNet,
                one per electrode count (mibci.concrete; Strypsteen and Bertrand 2021)
  relevance     the frozen rule's relevance order (counts 4 and 6 are new; the others exist)
  prior         a fixed, physiology-based order around C3, Cz and C4 (no data at all)
  pop-*         one montage for a new user: the mean rank of riemann / relevance / wrapper
                over the other eight participants (leave-one-participant-out), so the
                held-out participant's own data never chooses their electrodes

Each method x electrode count {4, 6, 8, 10, 12, 16} x participant x seed {0, 1, 2} is
trained with the frozen recipe on all of session 1 and scored on session 2 (exploratory:
the main study's test results were known). Rankings use session 1 only.

Stages: rank (session-1 orders per participant), plan, train, evaluate.
Use --shard i/n to split train/evaluate across SLURM array tasks.
Tables: scripts/analyze_selection_study.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from mibci import concrete, data, selectors
from mibci.artifacts import require_files, sha256, write_json
from mibci.experiment import run_pool, train_run
from mibci.layout import FINAL_DIR, frozen_config
from mibci.model import load_bundle
from mibci.study import Study, shard
from mibci.train import ChannelStandardizer, TrainConfig, predict, scores

STUDY = Study(
    "selection_study",
    "scripts/selection_study.py",
    sources=["src/mibci/selectors.py", "src/mibci/concrete.py", "src/mibci/study.py", "src/mibci/regions.py"],
)
COUNTS = (4, 6, 8, 10, 12, 16)
SEEDS = (0, 1, 2)
OWN = ("riemann", "csp", "wrapper")
POPULATION = {"pop-riemann": "riemann", "pop-relevance": "relevance", "pop-wrapper": "wrapper"}
# around the three classic sites, nearest first, alternating hemispheres; then the rest by row
PRIOR = ["C3", "C4", "Cz", "CP3", "CP4", "FC3", "FC4", "C1", "C2", "CPz", "C5", "C6", "FCz", "CP1", "CP2",
         "FC1", "FC2", "P1", "P2", "Pz", "Fz", "POz"]  # fmt: skip
RANKINGS = STUDY.out / "rankings"


def rank_job(subject: int) -> None:
    """Session-1 orders of the three joint methods; the relevance order is the archived one."""
    ses = data.load_session(subject, 1)
    keep = ~ses["artifact"]  # the training trials of the frozen recipe: all artifact-free session-1 trials
    X, y, runs = ses["X"][keep], ses["y"][keep], ses["run"][keep]
    train = np.ones(len(y), bool)
    order, curve = selectors.wrapper_order(X[train], y[train], runs[train])
    archived = json.loads((FINAL_DIR / "rankings" / f"sub-{subject:02d}_seed-0.json").read_text())
    standardized = ChannelStandardizer().fit(X[train]).transform(X[train])
    learned = {str(k): concrete.learn_subset(standardized, y[train].astype(np.int64), k) for k in COUNTS}
    write_json(RANKINGS / f"sub-{subject:02d}.json", {
        "subject": subject, "riemann": selectors.riemann_order(X[train], y[train]),
        "csp": selectors.csp_order(X[train], y[train]), "wrapper": order, "wrapper_curve": curve,
        "relevance": archived["orders"]["relevance"], "learned": learned, "n_train": int(train.sum()),
    })  # fmt: skip


def population_order(rankings: dict[int, dict], method: str, held_out: int) -> list[int]:
    """Mean rank over the other participants; ties broken by montage order."""
    ranks = np.zeros(len(data.EEG_CHANNELS))
    for subject, r in rankings.items():
        if subject != held_out:
            ranks[r[method]] += np.arange(len(data.EEG_CHANNELS))
    return [int(c) for c in np.argsort(ranks, kind="stable")]


def plan_jobs(rankings: dict[int, dict]) -> list[dict]:
    prior = [data.EEG_CHANNELS.index(n) for n in PRIOR]
    jobs = []
    for subject in data.SUBJECTS:
        orders = {m: rankings[subject][m] for m in OWN} | {"prior": prior}
        orders["relevance"] = rankings[subject]["relevance"]
        orders |= {p: population_order(rankings, m, subject) for p, m in POPULATION.items()}
        jobs.extend(
            {"name": f"sub-{subject:02d}_seed-{seed}_learned-{k}", "subject": subject, "seed": seed,
             "method": "learned", "k": k, "channels": rankings[subject]["learned"][str(k)]}
            for k in COUNTS
            for seed in SEEDS
        )  # fmt: skip
        for method, order in orders.items():
            for k in COUNTS:
                subset = sorted(order[:k])
                if method == "relevance" and k > 6:
                    continue  # archived (seeds 0-2) and the electrode-count study already have these
                jobs.extend(
                    {"name": f"sub-{subject:02d}_seed-{seed}_{method}-{k}", "subject": subject, "seed": seed,
                     "method": method, "k": k, "channels": subset}
                    for seed in SEEDS
                )  # fmt: skip
    return jobs


def create_plan() -> None:
    paths = [RANKINGS / f"sub-{s:02d}.json" for s in data.SUBJECTS]
    require_files(paths, "selection-study rankings (run --stage rank first)")
    rankings = {s: json.loads(p.read_text()) for s, p in zip(data.SUBJECTS, paths, strict=True)}
    plan = STUDY.create(
        plan_jobs(rankings),
        purpose="Compare joint (covariance-based, wrapper, learned), physiology-prior and population montages with the "
        "frozen per-electrode rule at 4 to 16 electrodes.",
        claim_boundary="Exploratory, planned after the main study's test results were known. Rankings use "
        "session 1 only; session-2 accuracies are not a prospective test.",
        counts=list(COUNTS),
        seeds=list(SEEDS),
        epochs=frozen_config()["epochs"],
        prior=PRIOR,
        ranking_sha256={p.name: sha256(p) for p in paths},
    )
    print(f"{len(plan['jobs'])} jobs in the plan")


def train_job(job: dict, epochs: int) -> None:
    path = STUDY.model(job)
    record = train_run(job["subject"], "final", job["name"], job["channels"],
                       TrainConfig(epochs=epochs, seed=job["seed"]), path)  # fmt: skip
    record.update(method=job["method"], k=job["k"], model_sha256=sha256(path),
                  protocol_sha256=sha256(STUDY.protocol_path))  # fmt: skip
    write_json(STUDY.record("runs", job), record)


def evaluate_job(job: dict) -> None:
    path = STUDY.model(job)
    model, scaler, bundle = load_bundle(path)
    if bundle["channels"] != job["channels"]:
        raise RuntimeError(f"Bundle channels differ from the plan: {path}")
    ses = data.load_session(job["subject"], 2, final_evaluation=True)
    logits = predict(model, scaler.transform(ses["X"][:, bundle["channels"]]))
    clean = ~ses["artifact"]
    write_json(STUDY.record("test", job), {
        **job, "model_sha256": sha256(path), "protocol_sha256": sha256(STUDY.protocol_path),
        "primary": scores(logits[clean], ses["y"][clean]), "secondary": scores(logits, ses["y"]),
        "pred": logits.argmax(1).tolist(), "y": ses["y"].tolist(), "artifact": ses["artifact"].astype(int).tolist(),
    })  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["rank", "plan", "train", "evaluate"], required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=data.SUBJECTS)
    parser.add_argument("--shard", help="i/n: run only array task i of n (0-based)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "rank":
        todo = [s for s in args.subjects if not (RANKINGS / f"sub-{s:02d}.json").exists()]
        run_pool(rank_job, [(f"sub-{s:02d}", {"subject": s}) for s in todo], args.workers, 1)
        return
    if args.stage == "plan":
        create_plan()
        return
    plan = STUDY.load()
    jobs = [j for j in shard(plan["jobs"], args.shard) if j["subject"] in args.subjects]
    if args.stage == "train":
        STUDY.log_environment(f"train-{(args.shard or 'all').replace('/', 'of')}")
        pending = STUDY.pending(
            jobs,
            "runs",
            lambda j: [STUDY.model(j)],
            lambda j: {"subject": j["subject"], "seed": j["seed"], "epochs": plan["epochs"]},
        )
        run_pool(train_job, [(j["name"], {"job": j, "epochs": plan["epochs"]}) for j in pending], args.workers, 1)
    else:
        require_files([STUDY.model(j) for j in jobs], "selection-study evaluation inputs (run --stage train first)")
        pending = STUDY.pending(jobs, "test", lambda j: [STUDY.model(j)])
        run_pool(evaluate_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
