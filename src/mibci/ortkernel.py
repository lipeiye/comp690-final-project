"""The missing INT8 kernel for EEGNet's grouped spatial convolution, plugged into ONNX Runtime.

ONNX Runtime runs the QDQ INT8 graph as NHWC QLinearConv nodes (com.microsoft, channels_last=1)
between two transposes. The spatial convolution (8 groups of 2 filters over all electrodes) gets
the generic grouped path, which is why the INT8 graph was slower than FP32 (see
scripts/int8_study.py). kernels/ort_grouped_conv.c implements that one layer as a custom operator
(domain "mibci"); this module builds it, swaps it into ONNX Runtime's own optimized graph and
opens sessions with it, so that everything else runs exactly as before:

  optimized  ONNX Runtime's optimized form of a stored graph (what its CPU provider executes)
  with_kernel  the same graph with the grouped QLinearConv and the transpose after it replaced by
             mibci.QLinearGroupedConvNhwc, which reads NHWC and writes NCHW
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

from mibci.paths import PROJECT_ROOT

SOURCE = PROJECT_ROOT / "kernels" / "ort_grouped_conv.c"
DOMAIN, OP = "mibci", "QLinearGroupedConvNhwc"
FLAGS = ["-O3", "-std=c11", "-Wall", "-Wextra", "-shared", "-fPIC"]


def build(headers: Path, out: Path) -> Path:
    """Compile the operator library against ONNX Runtime's C API headers (directory `headers`)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["cc", *FLAGS, f"-I{headers}", "-o", str(out), str(SOURCE)]
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"kernel build failed:\n{done.stdout}{done.stderr}")
    return out


def _options(level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL) -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = level
    return opts


def optimized(path: Path, out: Path) -> Path:
    """ONNX Runtime's optimized graph of `path`, as its CPU provider will execute it on this machine."""
    opts = _options()
    opts.optimized_model_filepath = str(out)
    ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    return out


def with_kernel(optimized_path: Path, out: Path) -> dict:
    """Replace the grouped spatial QLinearConv (+ its NHWC->NCHW transpose) by the custom operator."""
    model = onnx.load(str(optimized_path))
    nodes = list(model.graph.node)
    consumers = {name: [n for n in nodes if name in n.input] for n in nodes for name in n.output}
    target = None
    for node in nodes:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        grouped = node.op_type == "QLinearConv" and attrs.get("channels_last") == 1 and attrs.get("group", 1) > 1
        if grouped and attrs["kernel_shape"][1] == 1 and attrs["kernel_shape"][0] > 1:  # all electrodes, 1 sample
            target = (node, attrs)
    if target is None:
        raise ValueError(f"{optimized_path}: no grouped spatial QLinearConv in channels-last form")
    conv, attrs = target
    after = consumers[conv.output[0]]
    if len(after) != 1 or after[0].op_type != "Transpose":
        raise ValueError("expected a single NHWC->NCHW Transpose after the grouped convolution")
    transpose = after[0]
    perm = next(helper.get_attribute_value(a) for a in transpose.attribute if a.name == "perm")
    if list(perm) != [0, 3, 1, 2] or any(attrs.get(k, [0] * 4) != [0] * 4 for k in ("pads",)):
        raise ValueError(f"unexpected transpose {perm} or padding {attrs.get('pads')}")
    custom = helper.make_node(OP, list(conv.input), [transpose.output[0]], name="spatial_grouped_int8", domain=DOMAIN,
                              group=attrs["group"])  # fmt: skip
    position = nodes.index(conv)
    kept = [n for n in nodes if n is not conv and n is not transpose]
    del model.graph.node[:]
    model.graph.node.extend([*kept[:position], custom, *kept[position:]])
    if not any(o.domain == DOMAIN for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid(DOMAIN, 1))
    onnx.save(model, str(out))
    return {"replaced": [conv.name, transpose.name], "group": attrs["group"], "kernel_shape": attrs["kernel_shape"]}


def session(path: Path, library: Path | None = None, threads: int = 1) -> ort.InferenceSession:
    """Single-thread CPU session (as mibci.deploy.session), with the operator library registered if given."""
    opts = _options()
    opts.intra_op_num_threads = threads
    if library is not None:
        opts.register_custom_ops_library(str(library))
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def timed(sessions: dict[str, ort.InferenceSession], inputs: np.ndarray, rounds: int, runs: int, warmup: int) -> dict:
    """mibci.timing.timed_rounds for ready sessions: batch 1, interleaved rounds that alternate direction,
    the median of the per-round medians."""
    feeds = [inputs[i : i + 1] for i in range(len(inputs))]
    name_of = {name: s.get_inputs()[0].name for name, s in sessions.items()}
    for name, sess in sessions.items():
        for i in range(warmup):
            sess.run(None, {name_of[name]: feeds[i % len(feeds)]})
    medians: dict[str, list[float]] = {name: [] for name in sessions}
    for r in range(rounds):
        for name in list(sessions) if r % 2 == 0 else list(sessions)[::-1]:
            sess, times = sessions[name], np.empty(runs)
            for i in range(runs):
                start = time.perf_counter_ns()
                sess.run(None, {name_of[name]: feeds[i % len(feeds)]})
                times[i] = (time.perf_counter_ns() - start) / 1e6
            medians[name].append(float(np.median(times)))
    return {
        name: {"median_ms": float(np.median(m)), "best_round_ms": float(np.min(m)), "worst_round_ms": float(np.max(m)),
               "round_medians_ms": m}
        for name, m in medians.items()
    }  # fmt: skip


def profile(path: Path, library: Path | None, inputs: np.ndarray, runs: int = 300) -> dict[str, float]:
    """Mean microseconds per inference in each executed node (ONNX Runtime's profiler), as mibci.deploy.op_profile."""
    with tempfile.TemporaryDirectory() as tmp:
        opts = _options()
        opts.enable_profiling = True
        opts.profile_file_prefix = str(Path(tmp) / "prof")
        if library is not None:
            opts.register_custom_ops_library(str(library))
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        for i in range(runs):
            sess.run(None, {name: inputs[i % len(inputs)][None]})
        events = json.loads(Path(sess.end_profiling()).read_text())
    per_node: dict[str, float] = defaultdict(float)
    for e in events:
        if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
            per_node[f"{e['args']['op_name']}:{e['name'].removesuffix('_kernel_time')}"] += e["dur"]
    return {k: v / runs for k, v in per_node.items()}
