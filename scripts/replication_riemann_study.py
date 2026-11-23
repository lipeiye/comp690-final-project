"""Prospective test of Riemannian electrode selection on Lee et al. (2019), 54 people.

On BCI IV 2a (nine people; scripts/selection_study.py, exploratory, run after the test results
were known) the Riemannian rule of Barachant and Bonnet (2011), which keeps the electrodes whose
class-average covariance matrices stay farthest apart, beat the frozen one-electrode-at-a-time
relevance rule at 4 to 10 electrodes (+4.4 points at 8) but not at 16 (-0.7). This study turns
that into hypotheses for the 54 Lee2019 participants and fixes them, and the tests, before any
Lee2019 session-2 trial is scored. It was planned on 2026-11-21 while the replication study
(scripts/replication_study.py) was still training; that study's evaluation was held until
this plan existed.

New here: riemann-K, K in 8, 16, 20 (seeds 0, 1, 2; the frozen recipe, as the replication
study). The order comes from mibci.selectors.riemann_order on each participant's session-1
trials; electrodes the replication plan marks as dropouts (zero in some session-1 trial) are
left out of the ordering and placed last, as there. The comparison relevance-K comes from the
replication study's records: same participants, seeds, recipe and scoring.

Stages: plan, train, evaluate. Tests: scripts/analyze_replication_riemann_study.py.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from mibci import lee2019, replication, selectors
from mibci.artifacts import require_files, sha256, write_json
from mibci.experiment import device_for, run_pool
from mibci.layout import frozen_config
from mibci.model import build_eegnet
from mibci.paths import RESULTS_DIR
from mibci.study import Study, shard
from mibci.train import ChannelStandardizer, TrainConfig, fit, predict, scores, set_seed

STUDY = Study(
    "replication_riemann_study",
    "scripts/replication_riemann_study.py",
    sources=["src/mibci/lee2019.py", "src/mibci/replication.py", "src/mibci/selectors.py", "src/mibci/study.py",
             "src/mibci/layout.py"],
)  # fmt: skip
REPLICATION = RESULTS_DIR / "supplementary" / "replication_study" / "protocol.json"
COUNTS = (8, 16, 20)
SEEDS = (0, 1, 2)
HYPOTHESES = {
    "R1": {"primary": True, "claim": "Choosing electrodes jointly beats one at a time at 8 electrodes",
           "contrast": "riemann-8 minus relevance-8, FP32, all session-2 trials",
           "test": "95% participant bootstrap interval lower bound above 0"},
    "R2": {"claim": "The same at 16 electrodes (on BCI IV 2a, 16 was where the advantage had gone)",
           "contrast": "riemann-16 minus relevance-16", "test": "interval lower bound above 0"},
    "R3": {"claim": "The same at 20 electrodes", "contrast": "riemann-20 minus relevance-20",
           "test": "interval lower bound above 0"},
}  # fmt: skip


def ranking(subject: int, dropouts: list[int]) -> dict:
    ses = lee2019.load_session(subject, 1)
    usable = [c for c in range(ses["X"].shape[1]) if c not in dropouts]
    order = selectors.riemann_order(ses["X"][:, usable], ses["y"])
    return {"order": [usable[i] for i in order] + list(dropouts), "dropouts": list(dropouts)}


def create_plan(workers: int) -> None:
    source = json.loads(REPLICATION.read_text())
    dropouts = {s: source["rankings"][str(s)]["dropouts"] for s in lee2019.SUBJECTS}
    with ProcessPoolExecutor(workers) as pool:
        found = pool.map(ranking, lee2019.SUBJECTS, [dropouts[s] for s in lee2019.SUBJECTS])
        orders = dict(zip(lee2019.SUBJECTS, found, strict=True))
    jobs = [
        {"name": f"sub-{s:02d}_seed-{seed}_riemann-{k}", "subject": s, "seed": seed, "condition": f"riemann-{k}",
         "channels": sorted(orders[s]["order"][:k])}
        for s in lee2019.SUBJECTS for k in COUNTS for seed in SEEDS
    ]  # fmt: skip
    plan = STUDY.create(
        jobs,
        purpose="Prospective test, on 54 new people, of the exploratory BCI IV 2a finding that Riemannian "
        "electrode selection beats the frozen relevance rule at small electrode counts.",
        claim_boundary="Prospective for Lee2019: hypotheses and tests were fixed before any Lee2019 session-2 trial "
        "was scored (the replication study's evaluation was held until this plan existed), but the hypothesis "
        "comes from an exploratory BCI IV 2a result seen after its test day. One dataset, two classes, 54 people; "
        "it does not change the frozen 2a configuration.",
        status="prospective_replication",
        lee2019_session2_scored_before_plan=False,
        hypotheses=HYPOTHESES,
        analysis="all session-2 trials; seeds averaged within each participant; 95% participant bootstrap "
        "(10,000 resamples, seed 690); relevance-K from the replication study's records, paired by participant",
        comparison_plan_sha256=sha256(REPLICATION),
        counts=list(COUNTS),
        seeds=list(SEEDS),
        epochs=frozen_config()["epochs"],
        rankings={str(s): r for s, r in orders.items()},
    )
    print(f"{len(plan['jobs'])} jobs")


def train_job(job: dict, epochs: int) -> None:
    """The replication study's recipe (scripts/replication_study.py train_job, unaligned)."""
    ses = lee2019.load_session(job["subject"], 1)
    X, y = ses["X"][:, job["channels"]], ses["y"].astype(np.int64)
    scaler = ChannelStandardizer().fit(X)
    cfg = TrainConfig(epochs=epochs, seed=job["seed"])
    set_seed(cfg.seed)
    model = build_eegnet(len(job["channels"]), n_classes=len(lee2019.CLASSES))
    device = device_for(job["subject"])
    result = fit(model, scaler.transform(X), y, cfg, device=device)
    path = STUDY.model(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"subject": job["subject"], "seed": job["seed"], "condition": job["condition"], "epochs": epochs,
            "device": device}  # fmt: skip
    replication.save(path, model, job["channels"], scaler, **meta)
    write_json(STUDY.record("runs", job), {
        **meta, "name": job["name"], "n_train": len(y), "train_time_s": result["train_time_s"],
        "train_acc": scores(predict(model, scaler.transform(X)), y)["acc"], "model_sha256": sha256(path),
        "protocol_sha256": sha256(STUDY.protocol_path),
    })  # fmt: skip


def evaluate_job(job: dict) -> None:
    path = STUDY.model(job)
    model, scaler, bundle = replication.load(path)
    if bundle["channels"] != job["channels"]:
        raise RuntimeError(f"Bundle channels differ from the plan: {path}")
    ses = lee2019.load_session(job["subject"], 2, final_evaluation=True)
    pred = predict(model, scaler.transform(ses["X"][:, job["channels"]])).argmax(1)
    y, run = ses["y"], ses["run"]
    write_json(STUDY.record("test", job), {
        **job, "acc_all": float((pred == y).mean()), "acc_offline": float((pred[run == 0] == y[run == 0]).mean()),
        "acc_online": float((pred[run == 1] == y[run == 1]).mean()), "pred": pred.tolist(), "y": y.tolist(),
        "run": run.tolist(), "model_sha256": sha256(path), "protocol_sha256": sha256(STUDY.protocol_path),
    })  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "train", "evaluate"], required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=lee2019.SUBJECTS)
    parser.add_argument("--shard", help="i/n: run only array task i of n (0-based)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "plan":
        create_plan(args.workers)
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
        require_files([STUDY.model(j) for j in jobs], "Riemannian replication inputs (run --stage train first)")
        pending = STUDY.pending(jobs, "test", lambda j: [STUDY.model(j)])
        run_pool(evaluate_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
