"""Efficiency of every decoder architecture on this computer's CPU (ONNX Runtime).

One final model per architecture (electrode count x pruning), in FP32 and INT8:
batch-1, single-thread latency (100 warm-up calls, then 1000 timed calls cycling
through a fixed set of real session-1 trials; median and 95th percentile), file
size, parameter bytes, MACs, input and peak activation bytes under a layer-by-layer
schedule, the op types ONNX Runtime executes, and a per-node time profile.
Preprocessing (causal band-pass of one 2-s window plus normalization) is timed
separately. Run on an otherwise idle machine connected to AC power.
"""

import json
import platform
import subprocess
import time

import numpy as np
import onnxruntime as ort
from scipy.signal import sosfilt

from mibci import data, deploy
from mibci.costs import cost_summary, layer_costs
from mibci.env import environment
from mibci.experiment import MODELS_DIR, RESULTS_DIR, session1, write_json
from mibci.model import load_bundle

OUT = RESULTS_DIR / "benchmark"
SUBJECT, SEED = 1, 0
N_INPUTS = 100


def power_state() -> dict:
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    settings = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
    low_power = [ln.split()[-1] for ln in settings.splitlines() if "lowpowermode" in ln]
    return {"source": "AC" if "AC Power" in batt else "battery", "low_power_mode": low_power[0] if low_power else "n/a"}


def preprocessing_ms(n_chans: int, runs: int = 1000) -> dict:
    """Causal filtering of one 2-s chunk (filter state carried over) plus z-scoring."""
    rng = np.random.default_rng(0)
    chunk = rng.standard_normal((data.N_TIMES, n_chans)) * 10
    sos = data.bandpass_sos()
    zi = np.zeros((sos.shape[0], 2, n_chans))
    mean, std = np.zeros((n_chans, 1)), np.ones((n_chans, 1))
    times = np.empty(runs)
    for i in range(runs):
        start = time.perf_counter_ns()
        out, zi = sosfilt(sos, chunk, axis=0, zi=zi)
        ((out.T - mean) / std).astype(np.float32)[None]
        times[i] = (time.perf_counter_ns() - start) / 1e6
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def main() -> None:
    frozen = json.loads((RESULTS_DIR / "frozen_config.json").read_text())
    base = frozen["pruning"]["bases"][1]
    # One model per architecture: 22, 16 and 8 electrodes, plus the pruned variants.
    conditions = ["full-22", "proposed-16", "proposed-8", base] + [
        f"{b}-prune{p}" for b in ("full-22", base) for p in (25, 50)]
    X, _, train_mask, _ = session1(SUBJECT, "final")

    rows, profiles, layers = [], {}, {}
    for condition in dict.fromkeys(conditions):  # de-duplicated, order kept
        path = MODELS_DIR / "final" / f"sub-{SUBJECT:02d}_seed-{SEED}_{condition}.pt"
        model, scaler, b = load_bundle(path)
        inputs = scaler.transform(X[train_mask][:N_INPUTS][:, b["channels"]])
        costs = cost_summary(model)
        layers[condition] = layer_costs(model)
        for precision, onnx_path in (("fp32", path.with_suffix(".onnx")),
                                     ("int8", path.with_name(f"{path.stem}_int8.onnx"))):
            name = f"{condition} {precision}"
            rows.append({
                "condition": condition, "precision": precision, "electrodes": len(b["channels"]),
                "F2": b["F2"], "params": costs["params"], "macs": costs["macs"],
                "input_bytes": costs["input_bytes"][precision],
                "peak_activation_bytes": costs["peak_activation_bytes"][precision],
                **deploy.size_bytes(onnx_path),
                **{f"latency_{k}": v for k, v in deploy.latency(onnx_path, inputs).items()},
                "executed_ops": deploy.executed_ops(onnx_path),
            })
            profiles[name] = deploy.op_profile(onnx_path, inputs)
            print(f"{name:>24}: median {rows[-1]['latency_median_ms']:.3f} ms, p95 {rows[-1]['latency_p95_ms']:.3f} ms",
                  flush=True)

    write_json(OUT / "benchmark.json", {
        "environment": {**environment(), "onnxruntime": ort.__version__, "machine": platform.machine(),
                        "power": power_state(), "ort_threads": {"intra_op": 1, "inter_op": 1}},
        "models": f"final models of subject {SUBJECT}, seed {SEED}",
        "rows": rows,
        "preprocessing": {str(n): preprocessing_ms(n) for n in sorted({r["electrodes"] for r in rows})},
        "op_profile_us": profiles,
        "layer_costs": layers,
    })


if __name__ == "__main__":
    main()
