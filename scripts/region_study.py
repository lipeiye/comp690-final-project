"""Post-test study: choose electrodes by functional brain region (added after the midterm review).

Review question: would grouping electrodes by functional brain region reduce the montage
better than the per-electrode relevance and redundancy rule? The 22 electrodes form nine
fine regions (mibci.regions.FINE: premotor, motor and somatosensory rows of each
hemisphere, the front and centre of the midline, parietal). Every non-empty set of regions,
511 in all, is a candidate montage of 2 to 22 electrodes, and every one is trained:

  cv      each region set x participant x validation fold of the frozen protocol (session-1
          runs held out in pairs). Test-free: used for choosing montages and for the region
          Shapley values.
  final   each region set x participant, trained on all of session 1 (seed 0), then scored
          on session 2 in --stage evaluate: the exploratory test landscape.

Everything else is the frozen recipe: artifact-free trials, the frozen epoch budget, EEGNet-8,2,
per-channel standardisation fit on the training partition. The test results of the main study
were known when this was planned, so every session-2 number here is exploratory.

Stages: plan, train (--part cv|final|all), evaluate. Use --shard i/n to split a stage across
SLURM array tasks. Tables and figures: scripts/analyze_region_study.py.
"""

from __future__ import annotations

import argparse
import json

from mibci import data, regions
from mibci.artifacts import require_files, sha256, write_json
from mibci.experiment import device_for, run_pool, train_run
from mibci.model import load_bundle
from mibci.study import Study, shard
from mibci.train import TrainConfig, predict, scores

STUDY = Study("region_study", "scripts/region_study.py", sources=["src/mibci/regions.py", "src/mibci/study.py"])
SEED = 0
FOLDS = (0, 1, 2)


def region_jobs() -> list[dict]:
    jobs = []
    names = list(regions.FINE)
    for mask in range(1, 2 ** len(names)):
        combo = tuple(n for i, n in enumerate(names) if mask >> i & 1)
        channels = regions.coalition_channels(regions.FINE, combo)
        for subject in data.SUBJECTS:
            common = {"subject": subject, "seed": SEED, "mask": mask, "regions": list(combo), "channels": channels}
            for fold in FOLDS:
                name = f"sub-{subject:02d}_fold-{fold}_regions-{mask:03x}"
                jobs.append({"name": name, "part": "cv", "split": f"fold-{fold}", **common})
            name = f"sub-{subject:02d}_seed-{SEED}_regions-{mask:03x}"
            jobs.append({"name": name, "part": "final", "split": "final", **common})
    return jobs


def create_plan() -> None:
    frozen = json.loads((STUDY.out.parents[1] / "frozen_config.json").read_text())
    plan = STUDY.create(
        region_jobs(),
        purpose="Decode from sets of functional scalp regions; find which regions carry motor imagery and how "
        "small a region-based montage can be.",
        claim_boundary="Exploratory, planned after the main study's test results were known. Session-1 "
        "cross-validation numbers are test-free; session-2 numbers are not a prospective test and do not "
        "change the frozen final configuration.",
        partition=dict(regions.FINE),
        epochs=frozen["epochs"],
        batch_size=64,
        lr=1e-3,
        seed=SEED,
        folds=list(FOLDS),
        primary="artifact-free session-2 trials (final part) or held-out session-1 runs (cv part)",
    )
    parts = {p: sum(j["part"] == p for j in plan["jobs"]) for p in ("cv", "final")}
    print(f"{len(plan['jobs'])} jobs in the plan: {parts}")


def train_job(job: dict, epochs: int) -> None:
    path = STUDY.model(job)
    record = train_run(
        job["subject"], job["split"], job["name"], job["channels"], TrainConfig(epochs=epochs, seed=job["seed"]), path
    )
    record.update(
        part=job["part"],
        mask=job["mask"],
        regions=job["regions"],
        model_sha256=sha256(path),
        protocol_sha256=sha256(STUDY.protocol_path),
    )
    write_json(STUDY.record("runs", job), record)


def evaluate_job(job: dict) -> None:
    """Score a final model on session 2 (FP32, PyTorch): all trials and artifact-free trials."""
    path = STUDY.model(job)
    model, scaler, bundle = load_bundle(path)
    if bundle["channels"] != job["channels"]:
        raise RuntimeError(f"Bundle channels differ from the plan: {path}")
    ses = data.load_session(job["subject"], 2, final_evaluation=True)
    logits = predict(model, scaler.transform(ses["X"][:, bundle["channels"]]))
    clean = ~ses["artifact"]
    write_json(STUDY.record("test", job), {
        "name": job["name"], "subject": job["subject"], "seed": job["seed"], "mask": job["mask"],
        "regions": job["regions"], "channels": job["channels"], "model_sha256": sha256(path),
        "protocol_sha256": sha256(STUDY.protocol_path), "n_trials": len(ses["y"]), "n_clean": int(clean.sum()),
        "primary": scores(logits[clean], ses["y"][clean]), "secondary": scores(logits, ses["y"]),
        "pred": logits.argmax(1).tolist(), "y": ses["y"].tolist(), "artifact": ses["artifact"].astype(int).tolist(),
    })  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "train", "evaluate"], required=True)
    parser.add_argument("--part", choices=["cv", "final", "all"], default="all", help="train stage only")
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
        jobs = [j for j in jobs if args.part in ("all", j["part"])]
        STUDY.log_environment(
            f"train-{args.part}-{(args.shard or 'all').replace('/', 'of')}",
            device_assignment={str(s): device_for(s) for s in args.subjects},
        )
        pending = STUDY.pending(
            jobs,
            "runs",
            lambda j: [STUDY.model(j)],
            lambda j: {"subject": j["subject"], "split": j["split"], "epochs": plan["epochs"]},
        )
        run_pool(train_job, [(j["name"], {"job": j, "epochs": plan["epochs"]}) for j in pending], args.workers, 1)
    else:
        jobs = [j for j in jobs if j["part"] == "final"]
        require_files([STUDY.model(j) for j in jobs], "region-study evaluation inputs (train --part final first)")
        pending = STUDY.pending(jobs, "test", lambda j: [STUDY.model(j)])
        run_pool(evaluate_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
