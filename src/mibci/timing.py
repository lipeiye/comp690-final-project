"""Build deployment variants of an exported EEGNet and time them in interleaved rounds.

Moved unchanged from scripts/int8_study.py (where it produced results/supplementary/
int8_study) so that the cross-hardware timing in scripts/hw_latency.py runs the same code.
"""

from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from mibci import deploy, variants


def build_variants(fp32: Path, folder: Path, stem: str, calibration: np.ndarray, names) -> dict[str, Path]:
    """Write the requested variants of one archived FP32 graph into `folder`."""

    def path(name: str) -> Path:
        return folder / f"{stem}__{name}.onnx"

    made = {}
    if {"int8-dense"} & set(names):
        dense = path("fp32-dense")
        variants.spatial_dense(fp32, dense)
        variants.quantize(dense, path("int8-dense"), calibration)
        made["int8-dense"] = path("int8-dense")
    if {"fp32-spatial-first", "int8-spatial-first"} & set(names):
        variants.spatial_first(fp32, path("fp32-spatial-first"))
        made["fp32-spatial-first"] = path("fp32-spatial-first")
        if "int8-spatial-first" in names:
            variants.quantize(path("fp32-spatial-first"), path("int8-spatial-first"), calibration)
            made["int8-spatial-first"] = path("int8-spatial-first")
    settings = {
        "int8-mixed": {"spatial_fp32": True},
        "int16": {"activation_bits": 16},
        "int16-w16": {"activation_bits": 16, "weight_bits": 16},
        "int8-u8act": {"unsigned_activations": True},
        "int8-pertensor": {"per_channel": False},
        "int8-qoperator": {"qoperator": True},
    }
    for name, options in settings.items():
        if name in names:
            variants.quantize(fp32, path(name), calibration, **options)
            made[name] = path(name)
    return made


def power_state() -> dict:
    if platform.system() != "Darwin":
        return {"source": "unknown"}
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=False).stdout
    return {"source": "AC" if "AC Power" in batt else ("battery" if "Battery Power" in batt else "unknown")}


def timed_rounds(paths: dict[str, Path], inputs: np.ndarray, rounds: int, runs: int, warmup: int) -> dict:
    """Median latency per round for every graph. Rounds alternate direction so that slow
    drift in machine load falls on all graphs alike."""
    sessions = {name: deploy.session(path) for name, path in paths.items()}
    feeds = [{deploy.INPUT: inputs[i : i + 1]} for i in range(len(inputs))]
    for sess in sessions.values():
        for i in range(warmup):
            sess.run(None, feeds[i % len(feeds)])
    medians = {name: [] for name in sessions}
    for r in range(rounds):
        for name in list(sessions) if r % 2 == 0 else list(sessions)[::-1]:
            sess, times = sessions[name], np.empty(runs)
            for i in range(runs):
                start = time.perf_counter_ns()
                sess.run(None, feeds[i % len(feeds)])
                times[i] = (time.perf_counter_ns() - start) / 1e6
            medians[name].append(float(np.median(times)))
    return {
        name: {
            "median_ms": float(np.median(m)),
            "best_round_ms": float(np.min(m)),
            "worst_round_ms": float(np.max(m)),
            "rounds": rounds,
            "runs_per_round": runs,
            "warmup": warmup,
        }
        for name, m in medians.items()
    }


X86_FLAGS = ("avx2", "avx512f", "avx512_vnni", "avx_vnni", "amx_int8", "amx_tile")
ARM_FEATURES = ("hw.optional.arm.FEAT_DotProd", "hw.optional.arm.FEAT_I8MM")


def cpu() -> dict:
    """CPU model and the integer-arithmetic extensions it reports."""
    if platform.system() == "Darwin":

        def sysctl(key):
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=False).stdout.strip()

        return {"model": sysctl("machdep.cpu.brand_string"), "features": {k: sysctl(k) for k in ARM_FEATURES}}
    info = Path("/proc/cpuinfo").read_text()
    model = next((line.split(":", 1)[1].strip() for line in info.splitlines() if line.startswith("model name")), "")
    # x86 lists "flags"; Arm lists "Features" (asimddp = DotProd, i8mm = I8MM)
    flags = next(
        (line.split(":", 1)[1].split() for line in info.splitlines() if line.startswith(("flags", "Features"))), []
    )
    wanted = (*X86_FLAGS, "asimddp", "i8mm")
    return {"model": model, "features": {f: f in flags for f in wanted}}
