"""Control for the selection study: the reference models retrained on the same hardware (post-test).

scripts/selection_study.py trained its new methods on Longleaf CPUs, but took relevance-8, -10,
-12, -16 and full-22 (seeds 0-2) from the archived runs, which the M4 trained on its GPU (MPS)
for participants 1-6 and on its CPU for 7-9. Found on 2026-11-21: the region study's all-22
model (Longleaf CPU, seed 0) agrees with the archived full-22 seed-0 model on 97-100% of
session-2 trials for participants 7-9, but on only 50-94% for 1-6, and scores 2.4 points higher
for them on average. Training hardware therefore acts like another seed, and may shift accuracy,
so comparisons between new and archived models mix two sources of difference.

This study retrains exactly those 135 reference models (same electrodes, recipe and seeds) on
Longleaf CPUs and scores them on session 2, so that every method can be compared with references
trained on the same hardware. Exploratory: planned after the test results were known.

Stages: plan, train, evaluate. Tables: scripts/analyze_selection_study.py (uses these records
when they exist).
"""

from __future__ import annotations

import argparse
import json

from mibci import data
from mibci.artifacts import require_files, sha256, write_json
from mibci.experiment import run_pool, train_run
from mibci.layout import FINAL_DIR, frozen_config
from mibci.model import load_bundle
from mibci.study import Study, shard
from mibci.train import TrainConfig, predict, scores

STUDY = Study("selection_control_study", "scripts/selection_control_study.py", sources=["src/mibci/study.py"])
COUNTS = (8, 10, 12, 16)
SEEDS = (0, 1, 2)


def plan_jobs() -> list[dict]:
    jobs = []
    for subject in data.SUBJECTS:
        order = json.loads((FINAL_DIR / "rankings" / f"sub-{subject:02d}_seed-0.json").read_text())["orders"]
        subsets = {f"relevance-{k}": sorted(order["relevance"][:k]) for k in COUNTS}
        subsets["full-22"] = list(range(len(data.EEG_CHANNELS)))
        jobs.extend(
            {"name": f"sub-{subject:02d}_seed-{seed}_{condition}-longleaf", "subject": subject, "seed": seed,
             "condition": condition, "channels": channels}
            for condition, channels in subsets.items()
            for seed in SEEDS
        )  # fmt: skip
    return jobs


def create_plan() -> None:
    plan = STUDY.create(
        plan_jobs(),
        purpose="Retrain the selection study's archived reference models on the hardware its new methods used, so "
        "that method differences are not mixed with training-hardware differences.",
        claim_boundary="Exploratory, planned after the main study's test results were known. A hardware control for "
        "the post-test selection and region studies; it does not change any archived record or the frozen "
        "configuration.",
        counts=list(COUNTS),
        seeds=list(SEEDS),
        epochs=frozen_config()["epochs"],
        archived_devices="results/runs records: MPS for participants 1-6, CPU for 7-9 (mibci.experiment.device_for)",
    )
    print(f"{len(plan['jobs'])} jobs in the plan")


def train_job(job: dict, epochs: int) -> None:
    path = STUDY.model(job)
    record = train_run(job["subject"], "final", job["name"], job["channels"],
                       TrainConfig(epochs=epochs, seed=job["seed"]), path)  # fmt: skip
    record.update(condition=job["condition"], model_sha256=sha256(path), protocol_sha256=sha256(STUDY.protocol_path))
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
    parser.add_argument("--stage", choices=["plan", "train", "evaluate"], required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=data.SUBJECTS)
    parser.add_argument("--shard", help="i/n: run only array task i of n (0-based)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
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
        require_files([STUDY.model(j) for j in jobs], "selection-control evaluation inputs (run --stage train first)")
        pending = STUDY.pending(jobs, "test", lambda j: [STUDY.model(j)])
        run_pool(evaluate_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
