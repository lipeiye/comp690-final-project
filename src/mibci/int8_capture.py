"""Every integer tensor of an INT8 graph as this machine's ONNX Runtime computes it (scripts/int8_capture_study.py).

scripts/int8_layers_study.py added the QDQ graph's int8 tensors as graph outputs. On x86, ONNX Runtime first
rewrites an S8S8 QDQ graph as U8S8 (activations shifted by 128 to uint8) and then fuses it into integer kernels;
with the int8 tensors pinned as graph outputs that rewrite cannot happen, so the capture ran another path, and
the VNNI and non-VNNI machines agreed exactly. Here the graph is optimized on the machine first, as a normal
session does, and the integer tensors are read from that optimized graph with further optimization switched
off; `capture` also returns a normal session's logits so the two can be checked for equality.

optimized        the graph after ONNX Runtime's optimizations on this machine (ORT_ENABLE_ALL)
integer_tensors  the integer tensors that nodes of the optimized graph produce, in node order, with their
                 element type (QuantizeLinear, QLinear*, QGemm and layout ops applied to their outputs)
producers        for each integer tensor made by a QLinearConv or QGemm: that node, to inspect the layer
labels           a readable layer name for each integer tensor
capture          those tensors and the logits for windows X from the optimized graph (no further optimization),
                 and the logits of a normal session on the original graph
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto

from mibci import deploy

# where an op's output zero point (and so its output element type) sits among its inputs
OUTPUT_ZERO_POINT = {
    "QLinearConv": 7,
    "QGemm": 8,
    "QLinearMatMul": 7,
    "QLinearAdd": 7,
    "QLinearMul": 7,
    "QLinearAveragePool": 4,
    "QLinearGlobalAveragePool": 4,
    "QLinearLeakyRelu": 4,
    "QLinearSigmoid": 4,
    "QLinearSoftmax": 4,
}
SAME_AS_INPUT = {"Transpose", "Reshape", "Squeeze", "Unsqueeze", "Flatten", "MaxPool", "Identity"}
INTEGER = {TensorProto.INT8, TensorProto.UINT8}


def _session(model: onnx.ModelProto, level: ort.GraphOptimizationLevel, save: Path | None = None):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = level
    if save is not None:
        opts.optimized_model_filepath = str(save)
    return ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])


def optimized(model: onnx.ModelProto) -> onnx.ModelProto:
    """The graph a normal session on this machine runs (mibci.deploy.session's optimization level)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "optimized.onnx"
        _session(model, ort.GraphOptimizationLevel.ORT_ENABLE_ALL, path)
        return onnx.load(str(path))


def integer_tensors(model: onnx.ModelProto) -> dict[str, int]:
    """Tensor name -> element type, for the integer tensors the graph's nodes produce, in node order."""
    types = {i.name: i.data_type for i in model.graph.initializer}
    out = {}
    for node in model.graph.node:
        kind = None
        if node.op_type == "QuantizeLinear":
            kind = types.get(node.input[2], TensorProto.UINT8) if len(node.input) > 2 and node.input[2] else None
            kind = TensorProto.UINT8 if kind is None else kind
        elif node.op_type in OUTPUT_ZERO_POINT:
            at = OUTPUT_ZERO_POINT[node.op_type]
            kind = types.get(node.input[at]) if len(node.input) > at and node.input[at] else None
        elif node.op_type in SAME_AS_INPUT and node.input:
            kind = types.get(node.input[0])
        if kind in INTEGER:
            types[node.output[0]] = kind
            out[node.output[0]] = kind
    return out


def producers(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    """Integer tensor -> the QLinearConv or QGemm node that computes it."""
    ints = integer_tensors(model)
    return {n.output[0]: n for n in model.graph.node if n.op_type in ("QLinearConv", "QGemm") and n.output[0] in ints}


LAYERS = {
    "conv_temporal": "temporal conv",
    "conv_spatial": "spatial conv",
    "conv_separable_depth": "depthwise conv",
    "conv_separable_point": "pointwise conv",
    "final_layer": "dense (logits)",
}
QDQ_NAMES = {
    "eeg_q_to_dq": "input",
    "view_1_QuantizeLinear_Output": "input",
    "getitem_QuantizeLinear_Output": "temporal conv",
    "getitem_3_QuantizeLinear_Output": "spatial conv",
    "avg_pool2d_QuantizeLinear_Output": "pool 1",
    "conv2d_2_QuantizeLinear_Output": "depthwise conv",
    "getitem_6_QuantizeLinear_Output": "pointwise conv",
    "view_2_QuantizeLinear_Output": "pool 2",
    "avg_pool2d_1_q_to_dq": "pool 2",
    "logits_QuantizeLinear_Output": "dense (logits)",
}


def labels(model: onnx.ModelProto) -> dict[str, str]:
    """A readable name for each integer tensor: the EEGNet layer that computes it (from the weights it uses or
    the QDQ graph's tensor name), '(layout)' for transposes and reshapes of another tensor."""
    ints, out = integer_tensors(model), {}
    for node in model.graph.node:
        name = node.output[0]
        if name not in ints:
            continue
        if node.op_type in ("QLinearConv", "QGemm"):
            out[name] = next((v for k, v in LAYERS.items() if node.input[3].startswith(k)), node.input[3])
        elif node.op_type in SAME_AS_INPUT:
            out[name] = f"{out.get(node.input[0], node.input[0]).removesuffix(' (layout)')} (layout)"
        else:
            out[name] = QDQ_NAMES.get(name, name)
    return out


def capture(model: onnx.ModelProto, X: np.ndarray, output: str = "logits") -> dict:
    """{'tensors': name -> array, 'optimized': the optimized graph, 'plain': a normal session's logits}."""
    opt = optimized(model)
    ints = integer_tensors(opt)
    probe = copy.deepcopy(opt)
    present = {o.name for o in probe.graph.output}
    probe.graph.output.extend(
        onnx.helper.make_tensor_value_info(name, kind, None) for name, kind in ints.items() if name not in present
    )
    sess = _session(probe, ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    wanted = [*ints, output]
    tensors = dict(zip(wanted, sess.run(wanted, {deploy.INPUT: X}), strict=True))
    plain = _session(model, ort.GraphOptimizationLevel.ORT_ENABLE_ALL).run([output], {deploy.INPUT: X})[0]
    return {"tensors": tensors, "optimized": opt, "plain": plain}
