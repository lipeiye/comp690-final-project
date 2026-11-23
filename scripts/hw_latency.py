"""Time the same INT8 deployment variants on whatever machine runs this (post-test, exploratory).

Review question: is the slow INT8 model on the M4 a property of Apple's chip, or of the
software path that ONNX Runtime takes for EEGNet's grouped spatial convolution? Run this
script on several CPUs (the M4, and Longleaf nodes with different x86 integer extensions)
and compare: if the original INT8 graph is slow everywhere and the rewritten graphs are
fast everywhere, the cause is the kernel, not the chip.

For each of four architectures (full-22, relevance-16, the frozen finalist
relevance-16-prune50, relevance-8) of participant 1, seed 0 (as in benchmark.py), the
variants are rebuilt from the archived FP32 graph with mibci.timing (calibration: the
model's session-1 trials) and timed at batch 1 in interleaved rounds, single-threaded.
Writes results/supplementary/hw_latency/<host>.json with the CPU model, its integer
extensions (AVX2, AVX-512 VNNI, AVX-VNNI, AMX on x86; DotProd and I8MM on Arm) and the
ONNX Runtime version. Refuses to overwrite a result.
"""

from __future__ import annotations

import argparse
import os
import socket
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import onnxruntime as ort

from mibci.artifacts import model_exports, write_json
from mibci.env import environment
from mibci.experiment import session1
from mibci.layout import FINAL_MODELS
from mibci.model import load_bundle
from mibci.paths import RESULTS_DIR
from mibci.timing import build_variants, cpu, timed_rounds

OUT = RESULTS_DIR / "supplementary/hw_latency"
ARCHITECTURES = ("full-22", "relevance-16", "relevance-16-prune50", "relevance-8")
VARIANTS = ("int16", "int8-mixed", "int8-dense", "fp32-spatial-first", "int8-spatial-first")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--label", default=socket.gethostname().split(".")[0], help="name of the output file")
    args = parser.parse_args()
    out = OUT / f"{args.label}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; keep it and use another --label")
    rows = []
    with tempfile.TemporaryDirectory() as folder:
        for condition in ARCHITECTURES:
            bundle = FINAL_MODELS / f"sub-01_seed-0_{condition}.pt"
            _, scaler, b = load_bundle(bundle)
            X, _, train, _ = session1(1, "final")
            inputs = scaler.transform(X[train][:, b["channels"]])
            fp32, int8 = model_exports(bundle)
            paths = {"fp32": fp32, "int8": int8, **build_variants(fp32, Path(folder), bundle.stem, inputs, VARIANTS)}
            timing = timed_rounds(paths, inputs[:64], args.rounds, args.runs, warmup=200)
            rows += [{"condition": condition, "variant": name, **t} for name, t in timing.items()]
            print(f"{condition}: " + ", ".join(f"{n} {t['median_ms'] * 1000:.0f} us" for n, t in timing.items()))
    write_json(out, {
        "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "status": "supplementary_post_test",
        "host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),  # noqa: TID251  provenance only
        "cpu": cpu(), "onnxruntime": ort.__version__, "environment": environment(),
        "load_average": os.getloadavg(), "threads": 1, "batch": 1, "rows": rows,
        "method": "variants rebuilt from the archived FP32 graph of participant 1, seed 0; interleaved rounds; "
                  "median of the per-round medians",
    })  # fmt: skip


if __name__ == "__main__":
    main()
