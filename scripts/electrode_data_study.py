"""Post-test study: does the best electrode count move with the amount of training data?

On BCI IV 2a, accuracy over the relevance-ranked electrode counts rises gently from 22 to 16 and
falls below 14 (ten seeds). If that peak is a bias-variance balance (each extra input costs some
variance, each informative electrode adds signal), it must move: with less training data the best
count should drop, with more it should rise. If it stays at 16 whatever the data, the six dropped
electrodes hurt for another reason (what they contain). This study traces the whole curve at
several amounts of training data on both datasets, with the frozen recipe on Longleaf:

  BCI IV 2a  relevance-K for K in 4, 6, 8, 10, 12, 14, 18, 20 (16 and 22 come from
             scripts/overfit_study.py, same splits, seeds and recipe), trained on
             all six session-1 runs (final, seeds 0-9), on four runs (fold-k) and on two (rfold-k)
             (seeds 0-4, each of the three folds)
  Lee2019    relevance-K for K in 8, 16, 20, 32 (45 and 62 from scripts/overfit_study.py), trained
             on both session-1 runs (final) and on one (fold-0, fold-1); seeds 0-2

Electrodes are ranked on the training partition only (mibci.overfit.relevance_order; the archived
rankings for final models). Fold models are scored on the held-out runs of session 1 and on
session 2; final models on session 2. PREDICTIONS for Lee2019 were fixed before any of these
Lee2019 models was scored; for BCI IV 2a the study is exploratory.

Stages: plan, train, evaluate. Tables: scripts/analyze_electrode_data_study.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mibci import data, lee2019, overfit, replication
from mibci.artifacts import require_files, sha256, write_json
from mibci.experiment import device_for, run_pool
from mibci.layout import FINAL_DIR, frozen_config
from mibci.model import build_eegnet
from mibci.paths import RESULTS_DIR
from mibci.study import Study, shard
from mibci.train import ChannelStandardizer, TrainConfig, fit, predict, scores, set_seed

STUDY = Study(
    "electrode_data_study",
    "scripts/electrode_data_study.py",
    sources=["src/mibci/overfit.py", "src/mibci/lee2019.py", "src/mibci/replication.py", "src/mibci/selection.py",
             "src/mibci/splits.py", "src/mibci/study.py"],
)  # fmt: skip
REPLICATION_PLAN = RESULTS_DIR / "supplementary" / "replication_study" / "protocol.json"
DATASETS = {
    "bnci2a": {"counts": (4, 6, 8, 10, 12, 14, 18, 20), "classes": 4, "final_seeds": tuple(range(10)),
               "fold_seeds": tuple(range(5)),
               "splits": ("fold-0", "fold-1", "fold-2", "rfold-0", "rfold-1", "rfold-2")},
    "lee2019": {"counts": (8, 16, 20, 32), "classes": 2, "final_seeds": (0, 1, 2), "fold_seeds": (0, 1, 2),
                "splits": ("fold-0", "fold-1")},
}  # fmt: skip
PREDICTIONS = {
    "D1": {"claim": "With half the training data, fewer electrodes gain relative to all of them",
           "contrast": "Lee2019: (relevance-16 minus full-62) of fold models (one run) minus the same of final models "
           "(both runs), session 2; full-62 from scripts/overfit_study.py",
           "test": "interval lower bound above 0"},
    "D2": {"claim": "The best electrode count is lower with half the data than with all of it",
           "contrast": "Lee2019: the count with the highest mean accuracy over participants among 8, 16, 20, 32, 45 "
           "and 62, for fold models against final models (session 2), and a participant bootstrap of that argmax",
           "test": "argmax(fold) < argmax(final), in at least 95% of the 10,000 bootstrap resamples"},
}  # fmt: skip
H = ("seeds (and folds) averaged within each participant; 95% participant bootstrap (10,000, seed 690); BCI IV 2a "
     "scored on artifact-free session-2 trials, Lee2019 on all session-2 trials")  # fmt: skip


def day1(dataset: str, subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training-day trials (2a: artifact-free only), labels and run index, as scripts/overfit_study.py."""
    if dataset == "bnci2a":
        ses = data.load_session(subject, 1)
        keep = ~ses["artifact"]
        return ses["X"][keep], ses["y"][keep].astype(np.int64), ses["run"][keep]
    ses = lee2019.load_session(subject, 1)
    return ses["X"], ses["y"].astype(np.int64), ses["run"]


def orders(dataset: str, subject: int) -> dict[str, list[int]]:
    """Relevance order per split: archived for final models, ranked on the training runs for folds."""
    X, y, runs = day1(dataset, subject)
    if dataset == "bnci2a":
        archived = json.loads((FINAL_DIR / "rankings" / f"sub-{subject:02d}_seed-0.json").read_text())
        out = {"final": archived["orders"]["relevance"]}
    else:
        out = {"final": json.loads(REPLICATION_PLAN.read_text())["rankings"][str(subject)]["order"]}
    if overfit.relevance_order(X, y) != out["final"]:
        raise RuntimeError(f"{dataset} participant {subject}: the relevance rule no longer reproduces the archive")
    for split in DATASETS[dataset]["splits"]:
        train, _ = overfit.split_masks(runs, split, dataset)
        out[split] = overfit.relevance_order(X[train], y[train])
    return out


def plan_jobs(ranked: dict[tuple[str, int], dict[str, list[int]]]) -> list[dict]:
    jobs = []
    for (dataset, subject), order in ranked.items():
        spec = DATASETS[dataset]
        for split in ("final", *spec["splits"]):
            seeds = spec["final_seeds"] if split == "final" else spec["fold_seeds"]
            jobs.extend(
                {"name": f"{dataset}_sub-{subject:02d}_seed-{seed}_relevance-{k}_{split}", "dataset": dataset,
                 "subject": subject, "seed": seed, "split": split, "condition": f"relevance-{k}",
                 "channels": sorted(order[split][:k])}
                for k in spec["counts"]
                for seed in seeds
            )  # fmt: skip
    return jobs


def create_plan() -> None:
    ranked = {("bnci2a", s): orders("bnci2a", s) for s in data.SUBJECTS}
    ranked |= {("lee2019", s): orders("lee2019", s) for s in lee2019.SUBJECTS}
    plan = STUDY.create(
        plan_jobs(ranked),
        purpose="Trace accuracy over the electrode count at several amounts of training data, to test whether the best "
        "count moves with the data, as a bias-variance balance predicts.",
        claim_boundary="Exploratory for BCI IV 2a: planned after its electrode-count curve was known. For Lee2019 the "
        "predictions were fixed before any of these Lee2019 models was scored on session 2; Lee2019's final-model "
        "curve from the replication study was known. Two predictions, no correction for multiplicity. It explains a "
        "result; it does not change the frozen 2a configuration.",
        lee2019_new_models_scored_before_plan=False,
        predictions=PREDICTIONS,
        analysis=H,
        datasets={
            name: {key: list(v) if isinstance(v, tuple) else v for key, v in spec.items()}
            for name, spec in DATASETS.items()
        },
        combined_with="scripts/overfit_study.py records for full-22 and relevance-16 (2a), full-62 and relevance-45 "
        "(Lee2019) at the same splits",
        epochs=frozen_config()["epochs"],
        orders={f"{d}_{s}": o for (d, s), o in ranked.items()},
    )
    counts: dict[str, int] = {}
    for job in plan["jobs"]:
        key = f"{job['dataset']} {job['split'].split('-')[0]}"
        counts[key] = counts.get(key, 0) + 1
    print(f"{len(plan['jobs'])} jobs: {counts}")


def train_job(job: dict, epochs: int) -> None:
    X, y, runs = day1(job["dataset"], job["subject"])
    train, held = overfit.split_masks(runs, job["split"], job["dataset"])
    X_fit_raw = X[train][:, job["channels"]]
    scaler = ChannelStandardizer().fit(X_fit_raw)
    cfg = TrainConfig(epochs=epochs, seed=job["seed"])
    set_seed(cfg.seed)
    model = build_eegnet(len(job["channels"]), n_classes=DATASETS[job["dataset"]]["classes"])
    device = device_for(job["subject"])
    X_fit = scaler.transform(X_fit_raw)
    result = fit(model, X_fit, y[train], cfg, device=device)
    path = STUDY.model(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"dataset": job["dataset"], "subject": job["subject"], "seed": job["seed"], "split": job["split"],
            "condition": job["condition"], "epochs": epochs, "device": device}  # fmt: skip
    replication.save(path, model, job["channels"], scaler, **meta)
    record = {**meta, "name": job["name"], "n_train": int(train.sum()), "train_time_s": result["train_time_s"],
              "train_acc": scores(predict(model, X_fit), y[train])["acc"], "model_sha256": sha256(path),
              "protocol_sha256": sha256(STUDY.protocol_path)}  # fmt: skip
    if held is not None:
        logits = predict(model, scaler.transform(X[held][:, job["channels"]]))
        record |= {"n_held": int(held.sum()), "within_day": scores(logits, y[held]),
                   "within_day_pred": logits.argmax(1).tolist()}  # fmt: skip
    write_json(STUDY.record("runs", job), record)


def evaluate_job(job: dict) -> None:
    torch.set_num_threads(1)
    path = STUDY.model(job)
    model, scaler, bundle = replication.load(path)
    if bundle["channels"] != job["channels"]:
        raise RuntimeError(f"Bundle channels differ from the plan: {path}")
    if job["dataset"] == "bnci2a":
        ses = data.load_session(job["subject"], 2, final_evaluation=True)
        clean, extra = ~ses["artifact"], {"artifact": ses["artifact"].astype(int).tolist()}
    else:
        ses = lee2019.load_session(job["subject"], 2, final_evaluation=True)
        clean, extra = np.ones(len(ses["y"]), bool), {"run": ses["run"].tolist()}
    y = ses["y"].astype(np.int64)
    logits = predict(model, scaler.transform(ses["X"][:, job["channels"]]))
    write_json(STUDY.record("test", job), {
        **job, "model_sha256": sha256(path), "protocol_sha256": sha256(STUDY.protocol_path),
        "primary": scores(logits[clean], y[clean]), "secondary": scores(logits, y),
        "pred": logits.argmax(1).tolist(), "y": y.tolist(), **extra,
    })  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "train", "evaluate"], required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--shard", help="i/n: run only array task i of n (0-based)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "plan":
        create_plan()
        return
    plan = STUDY.load()
    jobs = [j for j in shard(plan["jobs"], args.shard) if j["dataset"] in args.datasets]
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
        require_files([STUDY.model(j) for j in jobs], "electrode-data evaluation inputs (run --stage train first)")
        pending = STUDY.pending(jobs, "test", lambda j: [STUDY.model(j)])
        run_pool(evaluate_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
