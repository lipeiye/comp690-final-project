"""Does Apple's CoreML backend help this decoder on the M4? (supplementary timing probe)

The midterm review asked whether something specific to the machine could make inference
faster. Besides the CPU, the M4 has a GPU and a Neural Engine, which ONNX Runtime reaches
through its CoreML execution provider. This script times the archived participant-1,
seed-0, 22-electrode model in four graphs (FP32 and INT8, each as exported and with the
first two layers swapped) on the plain CPU provider and on CoreML, either restricted to
the CPU or free to use any compute unit. Rounds are interleaved as in
`scripts/int8_study.py --stage benchmark`, and the system load is recorded.

CoreML manages its own threads, so its timings are not single-thread numbers. They answer
"is this faster on this machine", not "is the kernel faster per core". Nodes that CoreML
does not take fall back to the CPU provider; the record says how many each one ran.

Writes results/supplementary/int8_study/coreml_probe.json and refuses to overwrite it.
Run `scripts/int8_study.py --stage benchmark` first: it leaves the swapped graphs in
data.nosync/models/supplementary/int8_study/.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import onnxruntime as ort

from mibci import deploy
from mibci.artifacts import require_files, write_json
from mibci.env import environment
from mibci.experiment import MODELS_DIR, RESULTS_DIR, session1
from mibci.model import load_bundle

OUT = RESULTS_DIR / "supplementary/int8_study/coreml_probe.json"
SUBJECT = 1
STEM = f"sub-{SUBJECT:02d}_seed-0_full-22"
BACKENDS = {
    "cpu": None,
    "coreml-cpu-only": {"MLComputeUnits": "CPUOnly", "ModelFormat": "MLProgram"},
    "coreml-any-unit": {"MLComputeUnits": "ALL", "ModelFormat": "MLProgram"},
}


def make_session(path: Path, backend: str, profile_prefix: str | None = None) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile_prefix:
        opts.enable_profiling, opts.profile_file_prefix = True, profile_prefix
    providers = ["CPUExecutionProvider"]
    if BACKENDS[backend]:
        providers.insert(0, ("CoreMLExecutionProvider", BACKENDS[backend]))
    return ort.InferenceSession(str(path), opts, providers=providers)


def nodes_by_provider(path: Path, backend: str, feed: dict) -> dict[str, int]:
    """How many graph nodes each provider executed in one call."""
    with tempfile.TemporaryDirectory() as tmp:
        sess = make_session(path, backend, str(Path(tmp) / "profile"))
        sess.run(None, feed)
        events = json.loads(Path(sess.end_profiling()).read_text())
    names = {
        (e["args"].get("provider", "unknown"), e["name"])
        for e in events
        if e.get("cat") == "Node" and e["name"].endswith("_kernel_time")
    }
    return dict(collections.Counter(provider for provider, _ in names))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--runs", type=int, default=500)
    args = parser.parse_args()
    if OUT.exists():
        raise SystemExit(f"{OUT} exists. Keep the measurement; move it aside deliberately to measure again.")
    if "CoreMLExecutionProvider" not in ort.get_available_providers():
        raise SystemExit("This ONNX Runtime build has no CoreML execution provider.")
    ort.set_default_logger_severity(3)  # CoreML warns about every node it leaves to the CPU provider
    final, kept = MODELS_DIR / "final", MODELS_DIR / "supplementary/int8_study"
    graphs = {
        "fp32": final / f"{STEM}.onnx",
        "int8": final / f"{STEM}_int8.onnx",
        "fp32-spatial-first": kept / f"{STEM}__fp32-spatial-first.onnx",
        "int8-spatial-first": kept / f"{STEM}__int8-spatial-first.onnx",
    }
    require_files(list(graphs.values()), "CoreML probe (run scripts/int8_study.py --stage benchmark first)")
    _, scaler, b = load_bundle(final / f"{STEM}.pt")
    X, _, train_mask, _ = session1(SUBJECT, "final")
    inputs = scaler.transform(X[train_mask][:, b["channels"]])[:100]
    feeds = [{deploy.INPUT: inputs[i : i + 1]} for i in range(len(inputs))]

    load_before = os.getloadavg()
    sessions = {(g, backend): make_session(path, backend) for g, path in graphs.items() for backend in BACKENDS}
    for sess in sessions.values():
        for i in range(200):
            sess.run(None, feeds[i % len(feeds)])
    medians = {key: [] for key in sessions}
    for r in range(args.rounds):
        for key in list(sessions) if r % 2 == 0 else list(sessions)[::-1]:
            sess, times = sessions[key], np.empty(args.runs)
            for i in range(args.runs):
                start = time.perf_counter_ns()
                sess.run(None, feeds[i % len(feeds)])
                times[i] = (time.perf_counter_ns() - start) / 1e6
            medians[key].append(float(np.median(times)))
    rows = []
    for (g, backend), m in medians.items():
        reference = np.concatenate([sessions[(g, "cpu")].run(None, f)[0] for f in feeds])
        output = np.concatenate([sessions[(g, backend)].run(None, f)[0] for f in feeds])
        rows.append(
            {
                "graph": g,
                "backend": backend,
                "median_ms": float(np.median(m)),
                "best_round_ms": float(np.min(m)),
                "worst_round_ms": float(np.max(m)),
                "rounds": args.rounds,
                "runs_per_round": args.runs,
                "max_abs_logit_diff_vs_cpu_provider": float(np.abs(output - reference).max()),
                "nodes_by_provider": nodes_by_provider(graphs[g], backend, feeds[0]),
            }
        )
        print(f"{g:20s} {backend:16s} {rows[-1]['median_ms']:.3f} ms  {rows[-1]['nodes_by_provider']}", flush=True)

    def sysctl(key: str) -> str:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=False).stdout.strip()

    write_json(
        OUT,
        {
            "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "status": "supplementary_post_test",
            "environment": {
                **environment(),
                "onnxruntime": ort.__version__,
                "machine": platform.machine(),
                "hw_model": sysctl("hw.model"),
                "load_average_before": load_before,
                "load_average_after": os.getloadavg(),
            },
            "model": f"archived {STEM}; swapped graphs from scripts/int8_study.py",
            "backends": BACKENDS,
            "method": "Batch 1. All sessions are created together and timed in interleaved rounds; the value is the "
            "median of the per-round medians. The CPU provider uses one thread; CoreML manages its own threads.",
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()
