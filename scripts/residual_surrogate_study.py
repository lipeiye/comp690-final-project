"""Post-test study: how much information do the dropped electrodes add, and how much do they cost as inputs?

scripts/input_kind_study.py showed that an extra input costs what the other inputs cannot predict, and read
the cost of the real dropped electrodes off its curve of copies mixed with noise: about 1.7 points on BCI IV
2a and 4.5 on Lee2019, so about 1.4 and 4.3 points of information. Both are rough: the curve rises steeply
near zero (too small) and the mixes' noise is independent across inputs while the real residuals are
correlated (too large). This study measures both terms directly (mibci.residual_surrogate). Per participant
and training split, a least-squares fit from the 16 kept electrodes (the input-cost plan's) predicts every
dropped electrode, fitted on the training trials only; the extra inputs, after the 16 kept electrodes:

  real    the dropped electrodes themselves (prediction + the trial's own residual), the order-only twin of
          full-22 / full-62
  pred    the prediction alone
  surr-dK the prediction + another trial's residual: within the same set of trials (training trials, held-out
          runs, or the next day) the residuals of all dropped electrodes of a trial move together to another
          trial (derangement K, fixed per participant and split, shared by all seeds); no labels are used

  cost = surr (mean of the draws) - 16 electrodes alone;  information = real - surr;  their sum = real - 16 alone

  BCI IV 2a  6 extra: real, pred, surr-d0/d1/d2; final models seeds 0-9, fold-0/1/2 (4 runs) seeds 0-4
  Lee2019    46 extra: real, pred, surr-d0/d1 final models seeds 0-2; pred, surr-d0 also fold-0/1, seeds 0-2

References (not retrained): 16 electrodes alone is relevance-16 (2a: scripts/overfit_study.py; Lee2019:
scripts/electrode_data_study.py), all electrodes full-22 / full-62 (scripts/overfit_study.py; on Lee2019 also the
held-out stand-in for real), at the same splits and seeds. Lee2019 predictions S1-S4 are fixed in the plan
before any of these models is scored on Lee2019 session 2; BCI IV 2a is exploratory. On BCI IV 2a's next day the
residuals are deranged within the artifact-free trials and, separately, within the others, so that the primary
(artifact-free) score never receives an artifact's residual.

Stages: plan, train, evaluate. Tables: scripts/analyze_residual_surrogate_study.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mibci import data, lee2019, noise_mechanism, overfit, replication
from mibci import residual_surrogate as rs
from mibci.artifacts import require_files, sha256, write_json
from mibci.experiment import device_for, run_pool
from mibci.layout import frozen_config
from mibci.model import build_eegnet
from mibci.paths import RESULTS_DIR
from mibci.study import Study, shard
from mibci.train import ChannelStandardizer, TrainConfig, fit, predict, scores, set_seed

STUDY = Study(
    "residual_surrogate_study",
    "scripts/residual_surrogate_study.py",
    sources=[
        "src/mibci/residual_surrogate.py",
        "src/mibci/noise_mechanism.py",
        "src/mibci/overfit.py",
        "src/mibci/lee2019.py",
        "src/mibci/replication.py",
        "src/mibci/splits.py",
        "src/mibci/study.py",
    ],
)
SUPP = RESULTS_DIR / "supplementary"
COST_PLAN = SUPP / "input_cost_study" / "protocol.json"
KIND_PLAN = SUPP / "input_kind_study" / "protocol.json"
DATASETS = {
    "bnci2a": {
        "added": 6,
        "classes": 4,
        "electrodes": 22,
        "final_conditions": ("real", "pred", "surr-d0", "surr-d1", "surr-d2"),
        "final_seeds": tuple(range(10)),
        "fold_conditions": ("real", "pred", "surr-d0", "surr-d1", "surr-d2"),
        "fold_seeds": tuple(range(5)),
        "splits": ("fold-0", "fold-1", "fold-2"),
    },
    "lee2019": {
        "added": 46,
        "classes": 2,
        "electrodes": 62,
        "final_conditions": ("real", "pred", "surr-d0", "surr-d1"),
        "final_seeds": (0, 1, 2),
        "fold_conditions": ("pred", "surr-d0"),
        "fold_seeds": (0, 1, 2),
        "splits": ("fold-0", "fold-1"),
    },
}
REFERENCES = {
    "none": "relevance-16 of scripts/overfit_study.py (2a) and scripts/electrode_data_study.py (Lee2019)",
    "full": "full-22 / full-62 of scripts/overfit_study.py: the same electrodes as real in another input order; on "
    "Lee2019 also the held-out-runs stand-in for real, which is not trained on the folds here",
}
PREDICTIONS = {
    "S1": {
        "claim": "The dropped electrodes' unpredictable part carries information",
        "contrast": "Lee2019 final models, next day: real-46 minus surr-46 (mean of draws d0, d1)",
        "test": "interval lower bound above 0",
    },
    "S2": {
        "claim": "A residual unrelated to the trial still costs accuracy",
        "contrast": "Lee2019 final models, next day: surr-46 (mean of d0, d1) minus relevance-16 "
        "(scripts/electrode_data_study.py)",
        "test": "interval upper bound below 0",
    },
    "S3": {
        "claim": "An input the kept electrodes predict completely has no visible cost",
        "contrast": "Lee2019 final models, next day: pred-46 minus relevance-16",
        "test": "interval lower bound above -2 points (as K1 of scripts/input_kind_study.py)",
    },
    "S4": {
        "claim": "The less of the dropped electrodes the kept ones predict, the larger a person's cost",
        "contrast": "Spearman correlation over the 54 people between the unexplained share of the dropped "
        "electrodes (scripts/input_kind_study.py predictability: session 1, mean over its folds and the dropped "
        "electrodes) and the person's cost, surr-46 (mean of d0, d1) minus relevance-16, final models, next day",
        "test": "participant-bootstrap interval (10,000 resamples, seed 690) upper bound below 0",
    },
}
READINGS = {
    "S1 and S2 hold": "information and cost both exist and cancel: report the measured numbers",
    "S2 does not hold": "the dropped electrodes neither cost nor add much: 16 suffice because the others are "
    "redundant; rewrite the course report's Section 6.2 and slides 12-13 accordingly",
    "S2 holds, S1 does not": "a cost without information would make 16 clearly better than all electrodes, which "
    "contradicts the earlier results; find out why before writing anything",
}


def day1(dataset: str, subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training-day trials (2a: artifact-free only, as in every training run), labels and run index."""
    if dataset == "bnci2a":
        ses = data.load_session(subject, 1)
        keep = ~ses["artifact"]
        return ses["X"][keep], ses["y"][keep].astype(np.int64), ses["run"][keep]
    ses = lee2019.load_session(subject, 1)
    return ses["X"], ses["y"].astype(np.int64), ses["run"]


def permutation(job: dict, n: int, part: str) -> np.ndarray | None:
    if job["kind"] != "surr":
        return None
    return rs.derangement(n, rs.derangement_key(job["dataset"], job["subject"], job["split"], part, job["draw"]))


def day2_permutation(job: dict, clean: np.ndarray) -> np.ndarray | None:
    """A derangement of the next day's trials within the artifact-free ones and within the others (2a)."""
    if job["kind"] != "surr":
        return None
    perm = np.arange(len(clean))
    for part, mask in (("day2", clean), ("day2-artifact", ~clean)):
        idx = np.flatnonzero(mask)
        if len(idx) >= 2:
            perm[idx] = idx[permutation(job, len(idx), part)]
    return perm


def model_inputs(job: dict) -> list[int]:
    """One bundle entry per network input: kept electrodes, then the dropped electrodes for 'real' and
    -1, -2, ... for the synthetic extra inputs."""
    extra = job["dropped"] if job["kind"] == "real" else [-(i + 1) for i in range(len(job["dropped"]))]
    return [*job["kept"], *extra]


def plan_jobs() -> list[dict]:
    kept = {(j["dataset"], j["subject"], j["split"]): j["channels"] for j in json.loads(COST_PLAN.read_text())["jobs"]}
    jobs = []
    for dataset, spec in DATASETS.items():
        for subject in sorted({s for d, s, _ in kept if d == dataset}):
            for split in ("final", *spec["splits"]):
                channels = kept[(dataset, subject, split)]
                dropped = [c for c in range(spec["electrodes"]) if c not in channels]
                if len(dropped) != spec["added"]:
                    raise ValueError(f"{dataset} {subject} {split}: {len(dropped)} dropped electrodes")
                final = split == "final"
                for condition in spec["final_conditions" if final else "fold_conditions"]:
                    kind, _, draw = condition.partition("-d")
                    for seed in spec["final_seeds" if final else "fold_seeds"]:
                        name = f"{kind}-{spec['added']}" + (f"-d{draw}" if draw else "")
                        jobs.append({
                            "name": f"{dataset}_sub-{subject:02d}_seed-{seed}_{name}_{split}",
                            "dataset": dataset, "subject": subject, "seed": seed, "split": split,
                            "kind": kind, "draw": int(draw) if draw else None, "added": spec["added"],
                            "condition": name, "kept": channels, "dropped": dropped,
                        })  # fmt: skip
    return jobs


def create_plan() -> None:
    kind = json.loads((SUPP / "input_kind_study" / "analysis.json").read_text())["a_equivalent"]
    plan = STUDY.create(
        plan_jobs(),
        purpose="Measure directly how much information the dropped electrodes add and how much they cost as inputs, "
        "with surrogates that keep the part of each dropped electrode the kept ones predict and replace the rest by "
        "another trial's residual.",
        claim_boundary="Exploratory for BCI IV 2a, planned after its test results and after the input-kind study. For "
        "Lee2019 the predictions are fixed before any of these models is scored on Lee2019 session 2, but the "
        "input-kind results (including its interpolated cost and information) and the references' test results were "
        "known; four predictions, no correction for multiplicity. It explains a result; it does not change the "
        "frozen 2a configuration.",
        lee2019_new_models_scored_before_plan=False,
        predictions=PREDICTIONS,
        readings=READINGS,
        references=REFERENCES,
        interpolated_from_input_kind={d: {k: v[k] for k in ("a", "next_day_cost_at_a")} for d, v in kind.items()},
        kinds={
            "real": "the dropped electrodes (prediction + the trial's own residual)",
            "pred": "the least-squares prediction from the kept electrodes alone",
            "surr": "the prediction + another trial's residual, all dropped electrodes of a trial moved together",
        },
        fit="per participant and split: least squares with intercept from the kept electrodes to each dropped "
        "electrode, every time sample of every training trial one observation (final: all of session 1; folds: "
        "their training runs); the same fit as mibci.input_kind.unexplained_fraction",
        derangements="uniform by rejection, seeded by SHA-256 of (dataset, participant, split, set of trials, draw): "
        "the training trials, the held-out runs, session 2 (2a: artifact-free trials and the others separately)",
        datasets={
            name: {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}
            for name, spec in DATASETS.items()
        },
        analysis="seeds (and draws, and folds) averaged within each participant; 95% participant bootstrap (10,000, "
        "seed 690); BCI IV 2a on artifact-free session-2 trials, Lee2019 on all; 2a also with the seed error",
        epochs=frozen_config()["epochs"],
        input_cost_protocol_sha256=sha256(COST_PLAN),
        input_kind_protocol_sha256=sha256(KIND_PLAN),
    )
    counts: dict[str, int] = {}
    for job in plan["jobs"]:
        key = f"{job['dataset']} {'final' if job['split'] == 'final' else 'folds'}"
        counts[key] = counts.get(key, 0) + 1
    print(f"{len(plan['jobs'])} jobs: {counts}")


def train_job(job: dict, epochs: int) -> None:
    X, y, runs = day1(job["dataset"], job["subject"])
    train, held = overfit.split_masks(runs, job["split"], job["dataset"])
    f = rs.fit(X[train], job["kept"], job["dropped"])
    X_train = rs.model_input(X[train], f, job["kind"], permutation(job, int(train.sum()), "train"))
    scaler = ChannelStandardizer().fit(X_train)
    cfg = TrainConfig(epochs=epochs, seed=job["seed"])
    set_seed(cfg.seed)
    model = build_eegnet(X_train.shape[1], n_classes=DATASETS[job["dataset"]]["classes"])
    device = device_for(job["subject"])
    X_fit = scaler.transform(X_train)
    result = fit(model, X_fit, y[train], cfg, device=device)
    path = STUDY.model(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: job[k] for k in ("dataset", "subject", "seed", "split", "condition", "kind", "draw", "added")}
    meta |= {"epochs": epochs, "device": device, "fit_sha256": f.digest()}
    replication.save(path, model, model_inputs(job), scaler, **meta)
    extra = list(range(len(job["kept"]), len(job["kept"]) + len(job["dropped"])))
    left = rs.residual(X[train], f)
    record = {
        **meta,
        "name": job["name"],
        "n_train": int(train.sum()),
        "train_time_s": result["train_time_s"],
        "train_acc": scores(predict(model, X_fit), y[train])["acc"],
        "final_train_loss": result["history"]["train_loss"][-1],
        "train_loss": [round(v, 5) for v in result["history"]["train_loss"]],
        "residual_share_train": float(np.mean(left.var(axis=(0, 2)) / X[train][:, job["dropped"]].var(axis=(0, 2)))),
        "extra_share": noise_mechanism.noise_share(model, X_fit, extra).tolist(),
        "weight_rms_kept": noise_mechanism.weight_rms(model, list(range(len(job["kept"])))),
        "weight_rms_extra": noise_mechanism.weight_rms(model, extra),
        "model_sha256": sha256(path),
        "protocol_sha256": sha256(STUDY.protocol_path),
    }
    if held is not None:
        X_held = rs.model_input(X[held], f, job["kind"], permutation(job, int(held.sum()), "held"))
        logits = predict(model, scaler.transform(X_held))
        record |= {
            "n_held": int(held.sum()),
            "within_day": scores(logits, y[held]),
            "within_day_pred": logits.argmax(1).tolist(),
        }
    write_json(STUDY.record("runs", job), record)


def evaluate_job(job: dict) -> None:
    torch.set_num_threads(1)
    path = STUDY.model(job)
    model, scaler, bundle = replication.load(path)
    if bundle["channels"] != model_inputs(job) or bundle["condition"] != job["condition"]:
        raise RuntimeError(f"Bundle differs from the plan: {path}")
    X1, _, runs = day1(job["dataset"], job["subject"])
    train, _ = overfit.split_masks(runs, job["split"], job["dataset"])
    f = rs.fit(X1[train], job["kept"], job["dropped"])
    if f.digest() != bundle["fit_sha256"]:
        raise RuntimeError(f"{job['name']}: the refitted prediction differs from the one used in training")
    if job["dataset"] == "bnci2a":
        ses = data.load_session(job["subject"], 2, final_evaluation=True)
        clean = ~ses["artifact"]
    else:
        ses = lee2019.load_session(job["subject"], 2, final_evaluation=True)
        clean = np.ones(len(ses["y"]), bool)
    y = ses["y"].astype(np.int64)
    inputs = rs.model_input(ses["X"], f, job["kind"], day2_permutation(job, clean))
    logits = predict(model, scaler.transform(inputs))
    write_json(
        STUDY.record("test", job),
        {
            **{k: v for k, v in job.items() if k not in ("kept", "dropped")},
            "model_sha256": sha256(path),
            "protocol_sha256": sha256(STUDY.protocol_path),
            "primary": scores(logits[clean], y[clean]),
            "secondary": scores(logits, y),
            "pred": logits.argmax(1).tolist(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "train", "evaluate"], required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--subjects", nargs="+", type=int, help="only these participants (local smoke tests)")
    parser.add_argument("--shard", help="i/n: run only array task i of n (0-based)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "plan":
        create_plan()
        return
    plan = STUDY.load()
    jobs = [
        j
        for j in shard(plan["jobs"], args.shard)
        if j["dataset"] in args.datasets and (not args.subjects or j["subject"] in args.subjects)
    ]
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
        require_files([STUDY.model(j) for j in jobs], "residual-surrogate evaluation inputs (run --stage train first)")
        pending = STUDY.pending(jobs, "test", lambda j: [STUDY.model(j)])
        run_pool(evaluate_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
