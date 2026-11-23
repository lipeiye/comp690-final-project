"""Post-test study: for which convolution shapes is ONNX Runtime's INT8 faster than FP32, on which CPUs?

EEGNet's INT8 graph was slower than FP32 on every CPU we timed because its spatial layer (8 groups of 2
filters spanning every electrode) takes ONNX Runtime's general grouped path (scripts/ort_kernel_study.py).
This study maps the fast and slow paths beyond that one layer (part A of the plan's WP7): single
convolutions (mibci.int8_paths) with G groups of M filters (G * M <= 32), a 22 x 1 spatial or a 1 x 64
temporal kernel, on an EEG-shaped input, exported and quantized exactly as the decoders were. For each
shape the stage `map` checks which operators ONNX Runtime executes after its graph optimizations
(QLinearConv for INT8, as in the decoders' graphs), then times FP32 and INT8 at batch 1, one thread, in
interleaved rounds (mibci.timing), and records the time per multiply-accumulate.

Run `map` on each machine: the M4 Mac mini (when its load is low; the load is recorded) and one node
of each Longleaf CPU type (amd&avx512 = AMD EPYC 9654 with AVX-512 VNNI; intel&avx512 = Intel Xeon Gold
6140; amd = EPYC 75F3/7702; intel = Xeon E5-2680 v3), each writing map/<label>.json; a result is never
overwritten. Exploratory: no participant data are used. Tables: scripts/analyze_int8_paths_study.py.
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

from mibci import deploy, int8_paths
from mibci.artifacts import sha256, write_json
from mibci.env import environment
from mibci.study import Study
from mibci.timing import cpu, timed_rounds

STUDY = Study(
    "int8_paths_study",
    "scripts/int8_paths_study.py",
    sources=["src/mibci/int8_paths.py", "src/mibci/deploy.py", "src/mibci/timing.py", "src/mibci/study.py"],
)
TIMING = {"rounds": 7, "runs": 200, "warmup": 50, "calibration_windows": 32, "timing_windows": 8}


def create_plan() -> None:
    plan = STUDY.create(
        int8_paths.grid(),
        purpose="Map which grouped-convolution shapes ONNX Runtime runs faster in INT8 than in FP32, and on which "
        "CPUs, to show whether EEGNet's slow layer is one case of a general slow path.",
        claim_boundary="Exploratory and synthetic: random weights and inputs, no EEG. Timings are batch-1, "
        "single-threaded medians of interleaved rounds on shared machines (the load is recorded); they are not "
        "the decoders' end-to-end latency.",
        timing=TIMING,
        kernels={k: {"size": list(v[0]), "padding": list(v[1])} for k, v in int8_paths.KERNELS.items()},
        input_shape=["G", int8_paths.N_CHANS, int8_paths.N_TIMES],
        quantization="mibci.deploy.quantize_int8: QDQ, per-channel int8 weights, int8 activations, min-max",
    )
    print(f"{len(plan['jobs'])} shapes")


def map_machine(label: str) -> None:
    plan = STUDY.load()
    out = STUDY.out / "map" / f"{label}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; keep it and use another --label")
    started = datetime.now(ZoneInfo("America/New_York")).isoformat()
    load_before = os.getloadavg()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for shape in plan["jobs"]:
            fp32, int8 = Path(tmp) / f"{shape['name']}.onnx", Path(tmp) / f"{shape['name']}_int8.onnx"
            calibration = int8_paths.inputs(shape, TIMING["calibration_windows"], seed=1)
            int8_paths.export(shape, fp32, int8, calibration)
            timed = timed_rounds({"fp32": fp32, "int8": int8}, int8_paths.inputs(shape, TIMING["timing_windows"], 2),
                                 TIMING["rounds"], TIMING["runs"], TIMING["warmup"])  # fmt: skip
            macs = int8_paths.macs(shape)
            rows.append({
                **shape, "macs": macs, "ops_fp32": deploy.executed_ops(fp32), "ops_int8": deploy.executed_ops(int8),
                "fp32_ms": timed["fp32"]["median_ms"], "int8_ms": timed["int8"]["median_ms"],
                "fp32_ns_per_mac": 1e6 * timed["fp32"]["median_ms"] / macs,
                "int8_ns_per_mac": 1e6 * timed["int8"]["median_ms"] / macs,
                "int8_over_fp32": timed["int8"]["median_ms"] / timed["fp32"]["median_ms"], "timing": timed,
            })  # fmt: skip
            print(f"{shape['name']}: INT8/FP32 {rows[-1]['int8_over_fp32']:.2f}", flush=True)
    write_json(out, {
        "label": label, "host": socket.gethostname(), "cpu": cpu(), "onnxruntime": ort.__version__,
        "started": started, "load_before": list(load_before), "load_after": list(os.getloadavg()),
        "environment": environment(), "protocol_sha256": sha256(STUDY.protocol_path), "shapes": rows,
    })  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "map"], required=True)
    parser.add_argument("--label", default=socket.gethostname().split(".")[0], help="name of the output file")
    args = parser.parse_args()
    if args.stage == "plan":
        create_plan()
    else:
        map_machine(args.label)


if __name__ == "__main__":
    main()
