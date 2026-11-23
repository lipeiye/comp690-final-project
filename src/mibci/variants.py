"""Deployment variants of a trained EEGNet (added after the midterm review).

The exported graph keeps Braindecode's layer order: a temporal convolution over every
electrode, then a grouped spatial convolution (8 groups, 2 filters each). ONNX Runtime's
INT8 kernel for that grouped layer is slow on the CPU, and the temporal layer holds
almost all multiply-accumulates. This module provides, without retraining:

spatial_dense   The grouped spatial convolution becomes one ordinary convolution whose
                weights are zero outside each group's block. Same numbers in, same
                numbers out; only the kernel that ONNX Runtime picks changes.
spatial_first   The temporal and spatial convolutions are both linear, with nothing but
                (already folded) BatchNorm between them, so their order can be swapped:
                16 spatial filters first, then a depthwise temporal convolution. The
                function is the same up to float rounding, and the temporal filters run
                on 16 signals instead of 8 x n_electrodes. MI-BMInet (Wang et al., 2024)
                builds this order into its architecture; here it is applied to an
                already trained EEGNet.
quantize        Static post-training quantization with a chosen activation and weight
                width, optionally leaving the spatial convolution in FP32.

Calibration data must come from the model's own training partition, as everywhere else.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

INPUT = "eeg"
TEMPORAL, SPATIAL = "conv_temporal.weight", "conv_spatial.weight"
_TYPES = {8: QuantType.QInt8, 16: QuantType.QInt16}


def _conv(graph, weight_name: str):
    for node in graph.node:
        if node.op_type == "Conv" and node.input[1] == weight_name:
            return node
    raise ValueError(f"No Conv node uses {weight_name}; expected an FP32 EEGNet export")


def _set_attr(node, key: str, value) -> None:
    for attr in list(node.attribute):
        if attr.name == key:
            node.attribute.remove(attr)
    node.attribute.append(helper.make_attribute(key, value))


def _first_layers(model):
    graph = model.graph
    init = {i.name: i for i in graph.initializer}
    temporal, spatial = _conv(graph, TEMPORAL), _conv(graph, SPATIAL)
    if spatial.input[0] != temporal.output[0]:
        raise ValueError("The spatial convolution does not read the temporal convolution's output")
    w_t, w_s = numpy_helper.to_array(init[TEMPORAL]), numpy_helper.to_array(init[SPATIAL])
    f1, depth = w_t.shape[0], w_s.shape[0] // w_t.shape[0]
    if w_t.shape[1:3] != (1, 1) or w_s.shape[1] != 1 or w_s.shape[3] != 1 or w_s.shape[0] != f1 * depth:
        raise ValueError(f"Unexpected first-layer shapes: temporal {w_t.shape}, spatial {w_s.shape}")

    def bias(node, size):
        return numpy_helper.to_array(init[node.input[2]]) if len(node.input) > 2 else np.zeros(size, np.float32)

    return init, temporal, spatial, w_t, bias(temporal, f1), w_s, bias(spatial, f1 * depth), depth


def _put(model, init, name: str, value: np.ndarray) -> None:
    tensor = numpy_helper.from_array(np.ascontiguousarray(value, dtype=np.float32), name)
    if name in init:
        init[name].CopyFrom(tensor)
    else:
        model.graph.initializer.append(tensor)


def spatial_dense(fp32_path: Path, out_path: Path) -> None:
    """Rewrite the grouped spatial convolution as an equivalent dense convolution."""
    model = onnx.load(str(fp32_path))
    init, _, spatial, w_t, _, w_s, _, depth = _first_layers(model)
    dense = np.zeros((w_s.shape[0], w_t.shape[0], w_s.shape[2], 1), np.float32)
    for out_channel in range(w_s.shape[0]):
        dense[out_channel, out_channel // depth] = w_s[out_channel, 0]
    _put(model, init, SPATIAL, dense)
    _set_attr(spatial, "group", 1)
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))


def spatial_first(fp32_path: Path, out_path: Path) -> None:
    """Swap the first two layers: spatial filters on the raw input, then a depthwise temporal convolution.

    With temporal weights w_t[f, k] and bias b_t[f], spatial weights w_s[o, c] and bias b_s[o]
    (output o uses temporal filter f = o // depth):
        original   y[o, t] = sum_c w_s[o, c] * (sum_k w_t[f, k] * x[c, t + k] + b_t[f]) + b_s[o]
        rewritten  y[o, t] = sum_k w_t[f, k] * (sum_c w_s[o, c] * x[c, t + k]) + b_t[f] * sum_c w_s[o, c] + b_s[o]
    Zero padding in time commutes with the spatial sum, so the two are equal.
    """
    model = onnx.load(str(fp32_path))
    graph = model.graph
    init, temporal, spatial, w_t, b_t, w_s, b_s, depth = _first_layers(model)
    out_channels = w_s.shape[0]
    source, result = temporal.input[0], spatial.output[0]
    middle = "spatial_first_output"

    del spatial.input[2:]
    spatial.input[0], spatial.output[0] = source, middle
    _set_attr(spatial, "group", 1)

    bias_name = temporal.input[2] if len(temporal.input) > 2 else "conv_temporal.spatial_first_bias"
    _put(model, init, TEMPORAL, np.repeat(w_t, depth, axis=0))
    _put(model, init, bias_name, np.repeat(b_t, depth) * w_s[:, 0, :, 0].sum(axis=1) + b_s)
    del temporal.input[2:]
    temporal.input.append(bias_name)
    temporal.input[0], temporal.output[0] = middle, result
    _set_attr(temporal, "group", out_channels)

    nodes = list(graph.node)
    position = min(nodes.index(temporal), nodes.index(spatial))
    rest = [n for n in nodes if n is not temporal and n is not spatial]
    del graph.node[:]
    graph.node.extend([*rest[:position], spatial, temporal, *rest[position:]])
    used = {name for node in graph.node for name in node.input}
    for tensor in [t for t in graph.initializer if t.name not in used]:
        graph.initializer.remove(tensor)
    del graph.value_info[:]
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))


class _Calibration(CalibrationDataReader):
    def __init__(self, X: np.ndarray):
        self._batches = iter({INPUT: X[i : i + 1]} for i in range(len(X)))

    def get_next(self):
        return next(self._batches, None)


def quantize(  # noqa: PLR0913  public API; each keyword changes one quantization setting
    fp32_path: Path,
    out_path: Path,
    X_calib: np.ndarray,
    activation_bits: int = 8,
    weight_bits: int = 8,
    spatial_fp32: bool = False,
    unsigned_activations: bool = False,
    per_channel: bool = True,
    qoperator: bool = False,
    reduce_range: bool = False,
) -> None:
    """Static PTQ with min-max calibration. The defaults are the project's settings (QDQ, signed
    8-bit activations, per-channel signed 8-bit weights); the arguments change one setting each.
    `spatial_fp32` leaves the spatial convolution out of quantization. `reduce_range` stores
    weights in 7 bits, ONNX Runtime's documented remedy for 16-bit saturation in its x86
    kernels on CPUs without VNNI.

    16-bit QDQ needs opset 21, so the quantizer upgrades the graph and says so in a warning.
    """
    exclude = [_conv(onnx.load(str(fp32_path)).graph, SPATIAL).name] if spatial_fp32 else []
    activation = QuantType.QUInt8 if unsigned_activations and activation_bits == 8 else _TYPES[activation_bits]
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.ERROR)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            prepped = Path(tmp) / "prepped.onnx"
            quant_pre_process(str(fp32_path), str(prepped))
            quantize_static(
                str(prepped),
                str(out_path),
                _Calibration(X_calib),
                quant_format=QuantFormat.QOperator if qoperator else QuantFormat.QDQ,
                per_channel=per_channel,
                activation_type=activation,
                weight_type=_TYPES[weight_bits],
                calibrate_method=CalibrationMethod.MinMax,
                nodes_to_exclude=exclude,
                reduce_range=reduce_range,
            )
    finally:
        root.setLevel(level)


def graph_costs(fp32_path: Path) -> dict:
    """Analytical costs of an FP32 graph for one window (batch 1), in the conventions of costs.py:
    MACs of convolutions and the dense layer, weight and bias values, and the largest
    input + output pair among convolution, dense, pooling and ELU nodes."""
    model = onnx.load(str(fp32_path))
    for value in model.graph.input:
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param:
                dim.dim_value = 1
    model = shape_inference.infer_shapes(model, strict_mode=True)
    graph = model.graph
    init = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    shapes = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in list(graph.input) + list(graph.value_info) + list(graph.output)
    }

    def elems(name):
        return int(np.prod(shapes[name]))

    macs, params, peak, layers = 0, 0, 0, []
    for node in graph.node:
        if node.op_type not in ("Conv", "Gemm", "AveragePool", "Elu"):
            continue
        node_macs = 0
        if node.op_type == "Conv":
            weight = init[node.input[1]]
            node_macs = elems(node.output[0]) * int(np.prod(weight.shape[1:]))
        elif node.op_type == "Gemm":
            node_macs = int(init[node.input[1]].size)
        if node.op_type in ("Conv", "Gemm"):
            params += sum(int(init[name].size) for name in node.input[1:])
        macs += node_macs
        peak = max(peak, elems(node.input[0]) + elems(node.output[0]))
        layers.append(
            {
                "op": node.op_type,
                "name": node.name,
                "out_shape": shapes[node.output[0]][1:],
                "in_elems": elems(node.input[0]),
                "out_elems": elems(node.output[0]),
                "macs": node_macs,
            }
        )
    return {"macs": macs, "params": params, "peak_activation_elems": peak, "layers": layers}
