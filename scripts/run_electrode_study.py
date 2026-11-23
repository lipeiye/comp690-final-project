"""Supplementary, post-test electrode-count study (added after the midterm review).

Review question: why do 16 selected electrodes score higher than all 22 on the
held-out day? The main run had three training seeds per model, and one run moves
about three accuracy points with the seed alone. This study adds, without touching
any historical record:

  seeds      seeds 3-9 for full-22, relevance-16/8 and (where the subset differs)
             proposed-16/8, so each of these conditions has ten seeds;
  sweep      relevance-ranked subsets of 20, 18, 14, 12 and 10 electrodes (seeds 0-2),
             to see accuracy as a function of electrode count;
  control    the 16 least relevant electrodes (the six most relevant removed);
  repro      two archived runs trained again, to check that this pipeline still
             reproduces archived predictions on this machine.

Everything else is the frozen recipe: all artifact-free session-1 trials, the frozen
epoch budget, the archived per-participant relevance order, the original per-subject
device allocation. Session 2 is scored only in `--stage evaluate`. The test results
were already known when this was planned, so every number from here is exploratory
and cannot be used to re-select the historical final configuration.

Stages: plan, train, evaluate. Tables come from scripts/analyze_electrode_study.py,
which is not pinned by the protocol and may change without invalidating the runs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from mibci import data, deploy
from mibci.artifacts import completed, model_exports, require_files, sha256, write_json
from mibci.env import environment
from mibci.experiment import MODELS_DIR, RESULTS_DIR, device_for, run_pool, session1, train_run
from mibci.model import load_bundle
from mibci.train import TrainConfig, predict, scores

OUT = RESULTS_DIR / "supplementary/electrode_count"
MODELS = MODELS_DIR / "supplementary/electrode_count"
CORE = ["src/mibci/data.py", "src/mibci/model.py", "src/mibci/train.py", "src/mibci/experiment.py",
        "src/mibci/deploy.py", "scripts/run_electrode_study.py"]
NEW_SEEDS = list(range(3, 10))
SWEEP_COUNTS = (20, 18, 14, 12, 10)
REPRO = ((1, 0, "relevance-8"), (7, 0, "relevance-8"))  # one MPS subject, one CPU subject


def load_plan() -> dict:
    plan = json.loads((OUT / "protocol.json").read_text())
    if sha256(RESULTS_DIR / "frozen_config.json") != plan["historical_frozen_config_sha256"]:
        raise RuntimeError("Historical frozen configuration changed since this study was planned")
    for relative, digest in plan["source_code_sha256"].items():
        if sha256(data.PROJECT_ROOT / relative) != digest:
            raise RuntimeError(f"Pinned source changed: {relative}; start a new supplementary directory")
    for relative, digest in plan["ranking_sha256"].items():
        if sha256(RESULTS_DIR / relative) != digest:
            raise RuntimeError(f"Historical electrode ranking changed: {relative}")
    return plan


def create_plan() -> None:
    if (OUT / "protocol.json").exists():
        load_plan()
        print("Existing electrode-count protocol retained.")
        return
    frozen_path = RESULTS_DIR / "frozen_config.json"
    frozen = json.loads(frozen_path.read_text())
    jobs, ranking_hashes, shared = [], {}, {}

    def add(subject, seed, condition, channels, wave):
        jobs.append({"name": f"sub-{subject:02d}_seed-{seed}_{condition}", "subject": subject, "seed": seed,
                     "condition": condition, "channels": sorted(int(c) for c in channels), "wave": wave})

    for subject in data.SUBJECTS:
        # The relevance and redundancy-aware orders depend on the data only, not on the seed.
        relative = f"final/rankings/sub-{subject:02d}_seed-0.json"
        ranking = json.loads((RESULTS_DIR / relative).read_text())
        ranking_hashes[relative] = sha256(RESULTS_DIR / relative)
        order, subsets = ranking["orders"]["relevance"], ranking["subsets"]
        for other in frozen["seeds"][1:]:
            again = json.loads((RESULTS_DIR / f"final/rankings/sub-{subject:02d}_seed-{other}.json").read_text())
            if again["orders"]["relevance"] != order or again["orders"]["proposed"] != ranking["orders"]["proposed"]:
                raise RuntimeError(f"Data-driven orders differ between seeds for subject {subject}")
        for k in frozen["channel_counts"]:
            if sorted(order[:k]) != subsets[f"relevance-{k}"]:
                raise RuntimeError(f"Archived relevance-{k} subset is not the top {k} of the relevance order")
        for seed in NEW_SEEDS:
            add(subject, seed, "full-22", range(len(data.EEG_CHANNELS)), "seeds")
            for k in frozen["channel_counts"]:
                add(subject, seed, f"relevance-{k}", subsets[f"relevance-{k}"], "seeds")
                if subsets[f"proposed-{k}"] != subsets[f"relevance-{k}"]:
                    add(subject, seed, f"proposed-{k}", subsets[f"proposed-{k}"], "seeds")
        for k in frozen["channel_counts"]:
            if subsets[f"proposed-{k}"] == subsets[f"relevance-{k}"]:
                shared.setdefault(f"proposed-{k}", []).append(subject)
        for seed in frozen["seeds"]:
            for k in SWEEP_COUNTS:
                add(subject, seed, f"relevance-{k}", order[:k], "sweep")
            add(subject, seed, "leastrelevant-16", order[-16:], "control")
    for subject, seed, condition in REPRO:
        subsets = json.loads((RESULTS_DIR / f"final/rankings/sub-{subject:02d}_seed-{seed}.json").read_text())["subsets"]
        add(subject, seed, condition, subsets[condition], "repro")
    order_of = {"repro": 0, "seeds": 1, "sweep": 2, "control": 3}
    jobs.sort(key=lambda j: (order_of[j["wave"]], j["seed"], j["condition"], j["subject"]))

    write_json(OUT / "protocol.json", {
        "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "status": "supplementary_post_test",
        "historical_test_results_already_known": True,
        "purpose": "Check whether the 16-versus-22 electrode difference survives more training seeds, "
                   "and describe accuracy as a function of electrode count",
        "selection": "All nine participants; channel subsets from the archived session-1 relevance order; "
                     "no new selection and no change to the historical final configuration",
        "epochs": frozen["epochs"], "batch_size": 64, "lr": 1e-3,
        "historical_seeds": frozen["seeds"], "new_seeds": NEW_SEEDS, "sweep_counts": list(SWEEP_COUNTS),
        "same_subset_as_relevance": shared,
        "same_subset_note": "For these participants the redundancy-aware subset equals the relevance subset, "
                            "so no separate model is trained; the analysis reuses the relevance runs.",
        "primary": "artifact-free session-2 trials; average seeds within participant, then the nine participants",
        "secondary": "all session-2 trials",
        "bootstrap": {"resamples": 10000, "seed": 690, "level": 0.95},
        "claim_boundary": "Exploratory, planned after the test results were known. Not a new prospective test "
                          "and not a basis for re-selecting the frozen final configuration.",
        "historical_frozen_config_sha256": sha256(frozen_path),
        "source_code_sha256": {p: sha256(data.PROJECT_ROOT / p) for p in CORE},
        "ranking_sha256": ranking_hashes,
        "jobs": jobs,
    })
    waves = {w: sum(j["wave"] == w for j in jobs) for w in order_of}
    print(f"Planned {len(jobs)} training runs: {waves}")


def bundle_path(job: dict) -> Path:
    return MODELS / job["wave"] / f"{job['name']}.pt"


def record_path(job: dict, kind: str) -> Path:
    return OUT / kind / job["wave"] / f"{job['name']}.json"


def train_job(job: dict, epochs: int) -> None:
    path = bundle_path(job)
    record = train_run(job["subject"], "final", job["condition"], job["channels"],
                       TrainConfig(epochs=epochs, seed=job["seed"]), path)
    record.update(wave=job["wave"], model_sha256=sha256(path), protocol_sha256=sha256(OUT / "protocol.json"))
    write_json(record_path(job, "runs"), record)


def evaluate_job(job: dict) -> None:
    """Export, quantize with session-1 calibration, then score session 1 (training fit) and session 2."""
    path = bundle_path(job)
    model, scaler, b = load_bundle(path)
    if b["channels"] != job["channels"] or b["seed"] != job["seed"]:
        raise RuntimeError(f"Bundle metadata differs from protocol: {path}")
    X, y, tr, _ = session1(job["subject"], "final")
    train_inputs = scaler.transform(X[tr][:, b["channels"]])
    fp32, int8 = model_exports(path)
    deploy.export_onnx(model, fp32)
    deploy.quantize_int8(fp32, int8, train_inputs)
    reference = predict(model, train_inputs[:32])
    exported = deploy.logits(deploy.session(fp32), train_inputs[:32])
    error = float(np.abs(reference - exported).max())
    if not np.allclose(reference, exported, atol=1e-5, rtol=1e-4):
        raise RuntimeError(f"FP32 export failed tolerance for {path}: {error}")
    sessions = {"fp32": deploy.session(fp32), "int8": deploy.session(int8)}
    ses = data.load_session(job["subject"], 2, final_evaluation=True)
    inputs = scaler.transform(ses["X"][:, b["channels"]])
    clean = ~ses["artifact"]
    outputs = {p: deploy.logits(s, inputs) for p, s in sessions.items()}
    historical_path = RESULTS_DIR / "test" / f"{job['name']}.json"
    historical = json.loads(historical_path.read_text()) if job["wave"] == "repro" else None
    write_json(record_path(job, "test"), {
        **job, "channel_names": [data.EEG_CHANNELS[c] for c in job["channels"]],
        "model_sha256": sha256(path), "protocol_sha256": sha256(OUT / "protocol.json"),
        "export_max_abs_diff": error, "n_train": int(tr.sum()), "n_trials": len(ses["y"]), "n_clean": int(clean.sum()),
        "train": {p: scores(deploy.logits(s, train_inputs), y[tr]) for p, s in sessions.items()},
        "primary": {p: scores(lg[clean], ses["y"][clean]) for p, lg in outputs.items()},
        "secondary": {p: scores(lg, ses["y"]) for p, lg in outputs.items()},
        "pred": {p: lg.argmax(1).tolist() for p, lg in outputs.items()},
        "y": ses["y"].tolist(), "artifact": ses["artifact"].astype(int).tolist(),
        "historical_prediction_agreement": {
            p: float((lg.argmax(1) == np.array(historical["pred"][p])).mean()) for p, lg in outputs.items()
        } if historical else None,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "train", "evaluate"], required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=data.SUBJECTS)
    parser.add_argument("--waves", nargs="+", default=["repro", "seeds", "sweep", "control"])
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not set(args.subjects) <= set(data.SUBJECTS):
        parser.error("subjects must be in 1..9")
    if args.stage == "plan":
        create_plan()
        return
    plan = load_plan()
    selected = [j for j in plan["jobs"] if j["subject"] in args.subjects and j["wave"] in args.waves]
    if args.stage == "train":
        write_json(OUT / "environments" / ("subjects-" + "-".join(map(str, args.subjects)) + ".json"),
                   {**environment(), "device_assignment": {str(s): device_for(s) for s in args.subjects}})
        jobs = [(j["name"], {"job": j, "epochs": plan["epochs"]}) for j in selected
                if not completed(record_path(j, "runs"), [bundle_path(j)],
                                 {"subject": j["subject"], "seed": j["seed"], "condition": j["condition"],
                                  "epochs": plan["epochs"]})]
        run_pool(train_job, jobs, args.workers, threads=1)
    else:
        require_files([bundle_path(j) for j in selected], "electrode-count evaluation inputs")
        jobs = [(j["name"], {"job": j}) for j in selected
                if not completed(record_path(j, "test"), [bundle_path(j), *model_exports(bundle_path(j))],
                                 {"name": j["name"]})]
        run_pool(evaluate_job, jobs, args.workers, threads=1)


if __name__ == "__main__":
    main()
