"""Post-test study: does a mixed precision chosen on the training day hold on the test day?

The bit-width study (scripts/quant_study.py) rounded one tensor at a time to 4 bits on each
model's own session-1 trials. Most tensors barely noticed; a few (the input, the temporal
convolution output, the spatial convolution weights) lost many points. This study turns that
session-1 sensitivity into mixed-precision maps and scores them on session 2, without
retraining, for every archived final model (17 conditions x 9 participants x 3 seeds):

  map       each weight layer and each activation tensor gets 8 bits if rounding it alone
            to 4 bits cost more than T points of session-1 accuracy, else 4 bits
  pooled    T applied to the change averaged over participants, the way
            scripts/analyze_quant_study.py reads its rule: one map for every model
  own       T applied to each model's own session-1 change: a per-user map that needs
            nothing but that user's training day
  T         2 points (the bit-width analysis's threshold) and 0.5 points

Calibration is the model's own session-1 trials, as in the bit-width study; float/float and
8/8 are scored again as a check against that study's records.

Exploratory: planned on 2026-11-21, after the bit-width study's session-2 results were known,
its session-2 sensitivity included. The maps use session-1 numbers only, but the thresholds
were picked with the session-2 results in view.

Stages: plan (reads the bit-width study's records), run. Tables: scripts/analyze_quant_rule_study.py.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from mibci import data, fakequant
from mibci.artifacts import require_files, sha256, write_json
from mibci.experiment import run_pool, session1
from mibci.layout import FINAL_MODELS
from mibci.model import load_bundle
from mibci.paths import RESULTS_DIR
from mibci.study import Study, shard
from mibci.train import scores

STUDY = Study(
    "quant_rule_study",
    "scripts/quant_rule_study.py",
    sources=["src/mibci/fakequant.py", "src/mibci/study.py", "src/mibci/layout.py"],
)
SOURCE = RESULTS_DIR / "supplementary" / "quant_study"
THRESHOLDS = (2.0, 0.5)
REFERENCES = {"float/float": (None, None), "8/8": (8, 8), "4/8": (4, 8)}
TOLERANCE = 1e-9  # a loss of exactly T points (whole trials) does not count as more than T


def session1_changes(record: dict) -> dict[str, float]:
    """Session-1 accuracy change (pp) when one tensor alone is rounded to 4 bits."""
    s = record["sensitivity"]
    return {key: 100 * (acc - s["session1_float_acc"]) for key, acc in s["session1"].items() if key.endswith("@4")}


def precision_map(changes: dict[str, float], threshold: float) -> dict[str, dict[str, int]]:
    def bits(kind: str, name: str) -> int:
        return 8 if changes[f"{kind}:{name}@4"] < -(threshold + TOLERANCE) else 4

    return {
        "weights": {name: bits("weights", name) for name in fakequant.WEIGHTED},
        "activations": {name: bits("activations", name) for name in fakequant.POINTS},
    }


def pooled_changes(records: list[dict]) -> dict[str, float]:
    """Seeds and conditions averaged within each participant, then the participants averaged."""
    by: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        for key, change in session1_changes(record).items():
            by[key][record["subject"]].append(change)
    return {key: float(np.mean([np.mean(v) for v in subjects.values()])) for key, subjects in by.items()}


def source_records() -> list[dict]:
    plan = json.loads((SOURCE / "protocol.json").read_text())
    digest = sha256(SOURCE / "protocol.json")
    jobs = [j for j in plan["jobs"] if j["kind"] == "ptq"]
    paths = [SOURCE / "ptq" / f"{j['name']}.json" for j in jobs]
    require_files(paths, "bit-width study PTQ records (scripts/quant_study.py --stage run --kind ptq)")
    records = [json.loads(p.read_text()) for p in paths]
    if any(r["protocol_sha256"] != digest for r in records):
        raise SystemExit("A bit-width study record does not belong to its plan")
    return records


def create_plan() -> None:
    records = source_records()
    pooled = pooled_changes(records)
    shared = {f"pooled-{t:g}": precision_map(pooled, t) for t in THRESHOLDS}
    jobs = [
        {"name": r["name"], "subject": r["subject"], "seed": r["seed"], "condition": r["condition"],
         "maps": shared | {f"own-{t:g}": precision_map(session1_changes(r), t) for t in THRESHOLDS}}
        for r in records
    ]  # fmt: skip
    plan = STUDY.create(
        jobs,
        purpose="Turn single-tensor session-1 sensitivity into mixed-precision maps (one pooled over participants, "
        "one per model) and see whether they keep accuracy on session 2 without retraining.",
        claim_boundary="Exploratory, planned after the bit-width study's session-2 results were known. The maps "
        "are computed from session-1 numbers only, but the thresholds were chosen with session-2 results in view. "
        "Bit widths below 8 are simulated integer grids; nothing is timed.",
        thresholds_pp=list(THRESHOLDS),
        rule="8 bits for a weight layer or activation tensor whose 4-bit rounding alone cost more than the "
        "threshold in session-1 accuracy; otherwise 4 bits",
        pooled_session1_changes_pp=pooled,
        pooled_maps=shared,
        references={k: {"weights": w, "activations": a} for k, (w, a) in REFERENCES.items()},
        source_protocol_sha256=sha256(SOURCE / "protocol.json"),
        calibration="the model's own artifact-free session-1 trials, as in the bit-width study",
    )
    print(f"{len(plan['jobs'])} jobs; pooled maps: {json.dumps(plan['pooled_maps'])}")


def weight_bytes(folded: fakequant.Folded, weight_bits) -> float:
    """Weights at their width plus 32-bit biases (BatchNorm folded, as deployed); scales not counted."""
    total = 0.0
    for name in fakequant.WEIGHTED:
        layer = getattr(folded, name)
        total += layer.weight.numel() * (fakequant.bits_of(weight_bits, name) or 32) / 8
        total += 4 * layer.bias.numel() if layer.bias is not None else 0
    return total


def run_job(job: dict) -> None:
    torch.set_num_threads(1)
    model, scaler, bundle = load_bundle(FINAL_MODELS / f"{job['name']}.pt")
    X, y, train, _ = session1(job["subject"], "final")
    calibration, y_train = scaler.transform(X[train][:, bundle["channels"]]), y[train]
    ses = data.load_session(job["subject"], 2, final_evaluation=True)
    X_test, clean = scaler.transform(ses["X"][:, bundle["channels"]]), ~ses["artifact"]
    folded = fakequant.Folded(model)
    ranges = fakequant.calibrate(folded, calibration)
    settings = {k: (w, a) for k, (w, a) in REFERENCES.items()}
    settings |= {k: (m["weights"], m["activations"]) for k, m in job["maps"].items()}
    out = {}
    for key, (wb, ab) in settings.items():
        network = fakequant.quantized(folded, ranges, wb, ab)
        test = fakequant.logits(network, X_test)
        train_pred = fakequant.logits(network, calibration).argmax(1)
        out[key] = {
            "primary": scores(test[clean], ses["y"][clean]),
            "secondary_acc": float((test.argmax(1) == ses["y"]).mean()),
            "session1_acc": float((train_pred == y_train).mean()),
            "weight_bytes": weight_bytes(folded, wb),
            "bops": fakequant.bit_operations(model, wb, ab)["bops"],
            "pred": test.argmax(1).tolist(),
        }
    source = json.loads((SOURCE / "ptq" / f"{job['name']}.json").read_text())
    check = {k: out[k]["primary"]["acc"] - source["grid"][k]["primary_acc"] for k in REFERENCES}
    write_json(STUDY.record("runs", job), {
        **job, "settings": out, "y": ses["y"].tolist(), "artifact": ses["artifact"].astype(int).tolist(),
        "difference_from_bit_width_study": check, "protocol_sha256": sha256(STUDY.protocol_path),
    })  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "run"], required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=data.SUBJECTS)
    parser.add_argument("--shard", help="i/n: run only array task i of n (0-based)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "plan":
        create_plan()
        return
    plan = STUDY.load()
    jobs = [j for j in shard(plan["jobs"], args.shard) if j["subject"] in args.subjects]
    require_files([FINAL_MODELS / f"{j['name']}.pt" for j in jobs], "quant-rule inputs (the archived final models)")
    pending = STUDY.pending(jobs, "runs")
    run_pool(run_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
