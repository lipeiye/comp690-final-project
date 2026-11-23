"""Post-test study: how many bits does the decoder need, independent of the hardware?

Review question: INT8 looked disappointing on the M4 (slower than FP32), and INT16 was
suggested. Speed belongs to a runtime and a chip; the number of bits a decoder needs does
not. This study measures it with mibci.fakequant, which rounds a trained EEGNet's weights
and activations to b-bit grids (the deployed function with BatchNorm folded, as exported):

  ptq   every archived final model (17 conditions x 9 participants x 3 seeds), no training:
        a grid of weight bits {float, 16, 8, 6, 5, 4, 3, 2} x activation bits {float, 16, 8,
        6, 5, 4, 3}, three mixed-precision settings (MIXED), and single-layer sensitivity
        (one layer or one activation tensor at low precision, the rest floating point),
        measured on the model's own session-1 trials as well as on session 2, so that a
        precision rule can be chosen without the test day. Calibration: the model's own
        session-1 trials. The simulated W8A8 predictions are compared with the archived ONNX
        Runtime INT8 ones.
  qat   four architectures (full-22, relevance-16, the frozen finalist relevance-16-prune50,
        relevance-8) x 9 participants x 3 seeds: 100 epochs of quantization-aware fine-tuning
        from the archived model (Adam, lr 1e-4, as the pruning fine-tune) at W8A8, W4A8, W4A4,
        W4A4 with 8-bit early activations, W3A8, W2A8, and the same 100 epochs without
        quantization as the matched control.

Session 2 is scored only inside the jobs and only for reporting; the main study's test
results were known when this was planned, so everything is exploratory.

Stages: plan, run (--kind ptq|qat). Tables: scripts/analyze_quant_study.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mibci import data, fakequant
from mibci.artifacts import require_files, sha256, write_json
from mibci.experiment import run_pool, session1
from mibci.layout import FINAL_MODELS, frozen_config
from mibci.model import load_bundle
from mibci.paths import RESULTS_DIR
from mibci.study import Study, shard
from mibci.train import scores

STUDY = Study(
    "quant_study",
    "scripts/quant_study.py",
    sources=["src/mibci/fakequant.py", "src/mibci/costs.py", "src/mibci/study.py", "src/mibci/layout.py"],
)
WEIGHT_BITS = (None, 16, 8, 6, 5, 4, 3, 2)
ACTIVATION_BITS = (None, 16, 8, 6, 5, 4, 3)
# activations right after the input and the temporal convolution carry the raw-signal range
EARLY8 = {
    "input": 8,
    "conv_t": 8,
    **dict.fromkeys(("conv_s", "elu1", "pool1", "conv_d", "conv_p", "elu2", "pool2", "logits"), 4),
}
SPATIAL4_REST2 = {"conv_t": 2, "conv_s": 4, "conv_d": 2, "conv_p": 2, "fc": 2}
MIXED = {"mixed:w4/a4-early8": (4, EARLY8), "mixed:w2-spatial4/a8": (SPATIAL4_REST2, 8),
         "mixed:w2-spatial4/a4-early8": (SPATIAL4_REST2, EARLY8)}  # fmt: skip
QAT_CONDITIONS = ("full-22", "relevance-16", "relevance-16-prune50", "relevance-8")
QAT_VARIANTS = {"fp32-ft": (None, None), "w8a8": (8, 8), "w4a8": (4, 8), "w4a4": (4, 4), "w4a4-early8": (4, EARLY8),
                "w3a8": (3, 8), "w2a8": (2, 8)}  # fmt: skip
QAT_EPOCHS, QAT_LR = 100, 1e-4
KEPT_PREDICTIONS = ("float/float", "8/8", "4/8", "4/4", "8/16")


def label(bits) -> str:
    return "float" if bits is None else str(bits) if isinstance(bits, int) else "mixed"


def plan_jobs() -> list[dict]:
    frozen = frozen_config()
    conditions = sorted({p.stem.split("_", 2)[2] for p in FINAL_MODELS.glob("sub-*_seed-*.pt")})
    jobs = []
    for subject in data.SUBJECTS:
        for seed in frozen["seeds"]:
            for condition in conditions:
                stem = f"sub-{subject:02d}_seed-{seed}_{condition}"
                jobs.append({"name": stem, "kind": "ptq", "subject": subject, "seed": seed, "condition": condition})
                if condition in QAT_CONDITIONS:
                    for variant, (wb, ab) in QAT_VARIANTS.items():
                        jobs.append({"name": f"{stem}__{variant}", "kind": "qat", "subject": subject, "seed": seed,
                                     "condition": condition, "variant": variant, "weight_bits": wb,
                                     "activation_bits": ab})  # fmt: skip
    return jobs


def create_plan() -> None:
    plan = STUDY.create(
        plan_jobs(),
        purpose="Accuracy as a function of weight and activation bit width, without any runtime or chip in the "
        "loop; and whether quantization-aware fine-tuning recovers low bit widths.",
        claim_boundary="Exploratory, planned after the main study's test results were known. Bit widths below 8 "
        "are simulated (integer grids in floating point); no deployed low-bit kernel is timed here.",
        weight_bits=[label(b) for b in WEIGHT_BITS],
        activation_bits=[label(b) for b in ACTIVATION_BITS],
        mixed={k: {"weights": w, "activations": a} for k, (w, a) in MIXED.items()},
        qat={
            "conditions": list(QAT_CONDITIONS),
            "variants": {k: {"weights": w, "activations": a} for k, (w, a) in QAT_VARIANTS.items()},
            "epochs": QAT_EPOCHS,
            "lr": QAT_LR,
            "batch_size": 64,
        },
        calibration="the model's own artifact-free session-1 trials, normalized with its training statistics",
    )
    kinds = {k: sum(j["kind"] == k for j in plan["jobs"]) for k in ("ptq", "qat")}
    print(f"{len(plan['jobs'])} jobs in the plan: {kinds}")


def archived_bundle(job: dict):
    return FINAL_MODELS / f"sub-{job['subject']:02d}_seed-{job['seed']}_{job['condition']}.pt"


def inputs(job: dict):
    model, scaler, bundle = load_bundle(archived_bundle(job))
    X, y, train, _ = session1(job["subject"], "final")
    calibration = scaler.transform(X[train][:, bundle["channels"]])
    ses = data.load_session(job["subject"], 2, final_evaluation=True)
    return model, calibration, y[train], scaler.transform(ses["X"][:, bundle["channels"]]), ses


def evaluate(network, X_test, ses) -> tuple[dict, np.ndarray]:
    logits = fakequant.logits(network, X_test)
    clean = ~ses["artifact"]
    return {"primary": scores(logits[clean], ses["y"][clean]), "secondary": scores(logits, ses["y"])}, logits


def sensitivity(folded, ranges, X, y, clean) -> dict[str, float]:
    """Accuracy with one layer's weights (4 or 2 bits) or one activation tensor (4 bits) rounded."""
    settings = {f"weights:{layer}@{bits}": ({layer: bits}, None) for layer in fakequant.WEIGHTED for bits in (4, 2)}
    settings |= {f"activations:{point}@4": (None, {point: 4}) for point in fakequant.POINTS}
    out = {}
    for key, (wb, ab) in settings.items():
        pred = fakequant.logits(fakequant.quantized(folded, ranges, wb, ab), X).argmax(1)
        out[key] = float((pred[clean] == y[clean]).mean())
    return out


def ptq_job(job: dict) -> None:
    torch.set_num_threads(1)
    model, calibration, y_train, X_test, ses = inputs(job)
    folded = fakequant.Folded(model)
    ranges = fakequant.calibrate(folded, calibration)
    grid, kept = {}, {}
    for wb in WEIGHT_BITS:
        for ab in ACTIVATION_BITS:
            key = f"{label(wb)}/{label(ab)}"
            result, logits = evaluate(fakequant.quantized(folded, ranges, wb, ab), X_test, ses)
            grid[key] = {"primary_acc": result["primary"]["acc"], "primary_f1": result["primary"]["macro_f1"],
                         "secondary_acc": result["secondary"]["acc"]}  # fmt: skip
            if key in KEPT_PREDICTIONS:
                kept[key] = logits.argmax(1).tolist()
    for key, (wb, ab) in MIXED.items():
        result, logits = evaluate(fakequant.quantized(folded, ranges, wb, ab), X_test, ses)
        grid[key] = {"primary_acc": result["primary"]["acc"], "primary_f1": result["primary"]["macro_f1"],
                     "secondary_acc": result["secondary"]["acc"]}  # fmt: skip
        kept[key] = logits.argmax(1).tolist()
    float_train = fakequant.logits(folded, calibration).argmax(1)
    every = np.ones(len(y_train), bool)
    sens = {
        "session1": sensitivity(folded, ranges, calibration, y_train, every),
        "session1_float_acc": float((float_train == y_train).mean()),
        "session2": sensitivity(folded, ranges, X_test, ses["y"], ~ses["artifact"]),
    }
    archived = json.loads((RESULTS_DIR / "test" / f"{job['name']}.json").read_text())
    agreement = {p: float(np.mean(np.array(kept["8/8"]) == np.array(archived["pred"][p]))) for p in ("fp32", "int8")}
    write_json(STUDY.record("ptq", job), {
        **job, "grid": grid, "sensitivity": sens, "pred": kept, "y": ses["y"].tolist(),
        "artifact": ses["artifact"].astype(int).tolist(), "simulated_w8a8_agreement_with_archived": agreement,
        "archived_primary_acc": {p: archived["primary"][p]["acc"] for p in ("fp32", "int8")},
        "protocol_sha256": sha256(STUDY.protocol_path),
    })  # fmt: skip


def qat_job(job: dict) -> None:
    torch.set_num_threads(1)
    model, calibration, y_train, X_test, ses = inputs(job)
    folded = fakequant.Folded(model)
    ranges = fakequant.calibrate(folded, calibration)
    start = fakequant.quantized(folded, ranges, job["weight_bits"], job["activation_bits"])
    if job["weight_bits"] is None and job["activation_bits"] is None:
        start.ranges = None  # the matched control fine-tunes in floating point
    before, _ = evaluate(start, X_test, ses)
    tuned = fakequant.finetune(start, calibration, y_train.astype(np.int64), epochs=QAT_EPOCHS, lr=QAT_LR,
                               seed=job["seed"])  # fmt: skip
    after, logits = evaluate(tuned, X_test, ses)
    train_fit = scores(fakequant.logits(tuned, calibration), y_train)
    write_json(STUDY.record("qat", job), {
        **job, "before": before, "after": after, "train_after": train_fit, "pred": logits.argmax(1).tolist(),
        "y": ses["y"].tolist(), "artifact": ses["artifact"].astype(int).tolist(),
        "protocol_sha256": sha256(STUDY.protocol_path),
    })  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "run"], required=True)
    parser.add_argument("--kind", choices=["ptq", "qat"], help="run stage: which jobs")
    parser.add_argument("--subjects", type=int, nargs="+", default=data.SUBJECTS)
    parser.add_argument("--shard", help="i/n: run only array task i of n (0-based)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "plan":
        create_plan()
        return
    if not args.kind:
        parser.error("--stage run needs --kind ptq or --kind qat")
    plan = STUDY.load()
    jobs = [j for j in shard(plan["jobs"], args.shard) if j["kind"] == args.kind and j["subject"] in args.subjects]
    bundles = {archived_bundle(j) for j in jobs}
    require_files(sorted(bundles), "quant-study inputs (the archived final models)")
    pending = STUDY.pending(jobs, args.kind)
    run_pool(ptq_job if args.kind == "ptq" else qat_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
