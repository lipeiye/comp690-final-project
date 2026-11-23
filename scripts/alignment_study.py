"""Post-test study: unsupervised day-to-day alignment (Euclidean alignment, added after the midterm).

Observation that motivates it: the decoders lose about 29 points from the training day to
the next day, and part of the shift is scale and spatial mixing (participant 2's day-2
signals are 1.5 times larger; scaling them back changed that participant's accuracy by
several points). Euclidean alignment (mibci.alignment) whitens each day with its own mean
spatial covariance and needs no labels.

Models: full-22, relevance-16 and relevance-8 (the archived per-participant subsets) x 9
participants x seeds 0-2, trained with the frozen recipe on Euclidean-aligned session 1.
Scored on session 2 three ways:

  ea-all        aligned with all session-2 trials (unlabeled, offline)
  ea-first-run  aligned with session-2 run 1 only (48 unlabeled trials); scored on runs 2-6,
                which an online system could do after a short warm-up
  none          not aligned (shows how much the aligned model depends on alignment)

The unaligned archived models are the baseline, scored on the same trials. Exploratory:
the main study's test results were known when this was planned.

Stages: plan, train, evaluate. Tables: scripts/analyze_alignment_study.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from mibci import alignment, data
from mibci.artifacts import require_files, sha256, write_json
from mibci.experiment import device_for, run_pool, session1
from mibci.layout import FINAL_DIR, frozen_config
from mibci.model import build_eegnet, load_bundle, save_bundle
from mibci.paths import RESULTS_DIR
from mibci.study import Study, shard
from mibci.train import ChannelStandardizer, TrainConfig, fit, predict, scores, set_seed

STUDY = Study("alignment_study", "scripts/alignment_study.py", sources=["src/mibci/alignment.py", "src/mibci/study.py"])
CONDITIONS = ("full-22", "relevance-16", "relevance-8")
SEEDS = (0, 1, 2)


def plan_jobs() -> list[dict]:
    jobs = []
    for subject in data.SUBJECTS:
        subsets = json.loads((FINAL_DIR / "rankings" / f"sub-{subject:02d}_seed-0.json").read_text())["subsets"]
        jobs.extend(
            {"name": f"sub-{subject:02d}_seed-{seed}_{condition}-ea", "subject": subject, "seed": seed,
             "condition": condition, "channels": subsets[condition]}
            for condition in CONDITIONS
            for seed in SEEDS
        )  # fmt: skip
    return jobs


def create_plan() -> None:
    plan = STUDY.create(
        plan_jobs(),
        purpose="Does unsupervised Euclidean alignment of each recording day close part of the day-to-day gap, "
        "and does it change the 16-versus-22-electrode comparison?",
        claim_boundary="Exploratory, planned after the main study's test results were known. Uses unlabeled "
        "session-2 trials for alignment, which the frozen protocol does not.",
        conditions=list(CONDITIONS),
        seeds=list(SEEDS),
        epochs=frozen_config()["epochs"],
        alignment="Euclidean alignment per recording day (He and Wu 2019), then per-channel standardization "
        "fit on the aligned training day",
    )
    print(f"{len(plan['jobs'])} jobs in the plan")


def train_job(job: dict, epochs: int) -> None:
    """The frozen recipe (as experiment.train_run) on Euclidean-aligned session-1 trials."""
    X, y, train, _ = session1(job["subject"], "final")
    X = alignment.aligned(X[:, job["channels"]][train])
    scaler = ChannelStandardizer().fit(X)
    cfg = TrainConfig(epochs=epochs, seed=job["seed"])
    set_seed(cfg.seed)
    model = build_eegnet(len(job["channels"]))
    device = device_for(job["subject"])
    result = fit(model, scaler.transform(X), y[train], cfg, device=device)
    path = STUDY.model(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"subject": job["subject"], "split": "final", "condition": job["condition"], "seed": job["seed"],
            "epochs": epochs, "device": device, "alignment": "euclidean"}  # fmt: skip
    save_bundle(path, model, job["channels"], scaler, **meta)
    write_json(STUDY.record("runs", job), {
        **meta, "name": job["name"], "channels": [data.EEG_CHANNELS[c] for c in job["channels"]],
        "n_train": int(train.sum()), "train_time_s": result["train_time_s"], "model_sha256": sha256(path),
        "protocol_sha256": sha256(STUDY.protocol_path),
    })  # fmt: skip


def evaluate_job(job: dict) -> None:
    path = STUDY.model(job)
    model, scaler, bundle = load_bundle(path)
    ses = data.load_session(job["subject"], 2, final_evaluation=True)
    X, y, clean, later = ses["X"][:, bundle["channels"]], ses["y"], ~ses["artifact"], ses["run"] > 0
    inputs = {
        "ea-all": alignment.aligned(X),
        "ea-first-run": alignment.aligned(X, reference=X[ses["run"] == 0]),
        "none": X,
    }
    scored = {"ea-all": clean, "ea-first-run": clean & later, "none": clean}
    logits = {name: predict(model, scaler.transform(x)) for name, x in inputs.items()}
    preds = {name: lg.argmax(1) for name, lg in logits.items()}
    stem = f"sub-{job['subject']:02d}_seed-{job['seed']}_{job['condition']}"
    baseline = np.array(json.loads((RESULTS_DIR / "test" / f"{stem}.json").read_text())["pred"]["fp32"])
    write_json(STUDY.record("test", job), {
        **job, "model_sha256": sha256(path), "protocol_sha256": sha256(STUDY.protocol_path),
        "acc": {name: float((preds[name][mask] == y[mask]).mean()) for name, mask in scored.items()},
        "baseline_acc": {"all_clean": float((baseline[clean] == y[clean]).mean()),
                         "clean_runs_2_to_6": float((baseline[clean & later] == y[clean & later]).mean())},
        "primary_ea_all": scores(logits["ea-all"][clean], y[clean]),
        "pred": {name: p.tolist() for name, p in preds.items()}, "y": y.tolist(),
        "artifact": ses["artifact"].astype(int).tolist(), "run": ses["run"].tolist(),
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
        require_files([STUDY.model(j) for j in jobs], "alignment-study evaluation inputs (run --stage train first)")
        pending = STUDY.pending(jobs, "test", lambda j: [STUDY.model(j)])
        run_pool(evaluate_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
