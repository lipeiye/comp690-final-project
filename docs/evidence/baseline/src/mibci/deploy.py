"""ONNX export, static INT8 post-training quantization, and ONNX Runtime inference on CPU."""

from __future__ import annotations

import collections
import contextlib
import io
import json
import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from mibci.data import N_TIMES
from mibci.model import deploy_copy

OPSET = 18
INPUT = "eeg"
ort.set_default_logger_severity(3)  # errors only; optimized-graph dumps otherwise warn every time
logging.getLogger("torch.onnx").setLevel(logging.ERROR)  # "torchvision is not installed" notices


def export_onnx(model: torch.nn.Module, path: Path) -> None:
    """FP32 graph of the eval-mode model; the exporter's optimizer folds BatchNorm into Conv."""
    model = deploy_copy(model)
    x = torch.zeros(1, model.conv_spatial.kernel_size[0], N_TIMES)
    with contextlib.redirect_stdout(io.StringIO()):  # the exporter prints progress lines
        torch.onnx.export(model, (x,), path, input_names=[INPUT], output_names=["logits"],
                          dynamic_shapes=({0: torch.export.Dim("batch")},), opset_version=OPSET,
                          dynamo=True, optimize=True, external_data=False, verbose=False)


class _Calibration(CalibrationDataReader):
    def __init__(self, X: np.ndarray):
        self._batches = iter({INPUT: X[i : i + 1]} for i in range(len(X)))

    def get_next(self):
        return next(self._batches, None)


def quantize_int8(fp32_path: Path, int8_path: Path, X_calib: np.ndarray) -> None:
    """Static INT8 PTQ: QDQ format, signed 8-bit activations and per-channel weights,
    min-max calibration on normalized training-partition trials."""
    with tempfile.TemporaryDirectory() as tmp:
        prepped = Path(tmp) / "prepped.onnx"
        quant_pre_process(str(fp32_path), str(prepped))
        quantize_static(str(prepped), str(int8_path), _Calibration(X_calib), quant_format=QuantFormat.QDQ,
                        per_channel=True, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax)


def session(path: Path, threads: int = 1, profile_prefix: str | None = None) -> ort.InferenceSession:
    """CPU session with fixed threading: one intra-op thread and sequential execution by default."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile_prefix:
        opts.enable_profiling = True
        opts.profile_file_prefix = profile_prefix
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def logits(sess: ort.InferenceSession, X: np.ndarray, batch: int = 256) -> np.ndarray:
    return np.concatenate([sess.run(None, {INPUT: X[i : i + batch]})[0] for i in range(0, len(X), batch)])


def executed_ops(path: Path) -> dict[str, int]:
    """Op types left after ONNX Runtime's graph optimizations, i.e. the kernels that run.
    QLinear*/QGemm ops are integer kernels; the rest run in floating point."""
    with tempfile.TemporaryDirectory() as tmp:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.optimized_model_filepath = str(Path(tmp) / "optimized.onnx")
        ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        graph = onnx.load(opts.optimized_model_filepath).graph
    return dict(collections.Counter(n.op_type for n in graph.node))


def latency(path: Path, X: np.ndarray, warmup: int = 100, runs: int = 1000) -> dict:
    """Batch-1, single-thread latency over a fixed cycle of representative inputs."""
    sess = session(path)
    inputs = [X[i : i + 1] for i in range(len(X))]
    for i in range(warmup):
        sess.run(None, {INPUT: inputs[i % len(inputs)]})
    times = np.empty(runs)
    for i in range(runs):
        start = time.perf_counter_ns()
        sess.run(None, {INPUT: inputs[i % len(inputs)]})
        times[i] = (time.perf_counter_ns() - start) / 1e6
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
            "mean_ms": float(times.mean()), "warmup": warmup, "runs": runs}


def op_profile(path: Path, X: np.ndarray, runs: int = 300) -> dict[str, float]:
    """Mean time per inference spent in each executed node (microseconds), from ORT profiling."""
    with tempfile.TemporaryDirectory() as tmp:
        sess = session(path, profile_prefix=str(Path(tmp) / "prof"))
        for i in range(runs):
            sess.run(None, {INPUT: X[i % len(X)][None]})
        events = json.loads(Path(sess.end_profiling()).read_text())
    per_node = collections.defaultdict(float)
    for e in events:
        if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"):
            per_node[f"{e['args']['op_name']}:{e['name'].removesuffix('_kernel_time')}"] += e["dur"]
    return {k: v / runs for k, v in per_node.items()}


def size_bytes(path: Path) -> dict[str, int]:
    """Serialized file size and the bytes held in initializers (weights, scales, zero points)."""
    model = onnx.load(str(path))
    params = sum(onnx.numpy_helper.to_array(t).nbytes for t in model.graph.initializer)
    return {"file_bytes": Path(path).stat().st_size, "parameter_bytes": int(params)}
