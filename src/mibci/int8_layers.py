"""Where the same INT8 graph computes differently on x86 CPUs without VNNI (scripts/int8_layers_study.py).

scripts/int8_portability.py found that the archived INT8 graphs give the same predictions on the Apple M4
and on AVX-512 VNNI CPUs, but different ones (457 of 459 models, -4 points) on x86 CPUs without VNNI.
ONNX Runtime's documentation explains why this can happen: without VNNI its 8-bit matrix kernel adds two
adjacent 8-bit products in a saturating 16-bit register (x86 VPMADDUBSW) before widening to 32 bits.
This module looks inside the graphs:

quantized_outputs  the int8 tensors of a QDQ graph (the outputs of its QuantizeLinear nodes), in order:
                   for EEGNet the input, the temporal and spatial convolutions, the first pooling, the
                   depthwise and pointwise convolutions, the second pooling and the logits
capture            all of them for a set of windows, by adding them as graph outputs (ONNX Runtime still
                   fuses every layer into its integer kernel; the tensors are its own outputs)
pair_saturation    for one convolution, which outputs have a pair of adjacent products (along the
                   reduction axis in the order (kh, kw, input channel), the activations shifted to
                   unsigned 8 bits as the x86 kernel takes them) whose sum leaves the int16 range
seven_bit          a copy of the graph with one weighted layer's weights on a 7-bit grid (the scale
                   doubled, the integers halved), the per-layer form of ONNX Runtime's reduce_range
"""

from __future__ import annotations

import copy

import numpy as np
import onnx
import onnxruntime as ort
from numpy.lib.stride_tricks import sliding_window_view
from onnx import numpy_helper

from mibci import deploy

INT16 = (-32768, 32767)
WEIGHTED = ("conv_temporal", "conv_spatial", "conv_separable_depth", "conv_separable_point", "final_layer")
# braindecode wraps the dense layer in its max-norm constraint, so the graph names its weights after the wrapper
PREFIX = {"final_layer": "final_layer.linearconstraint"}


def quantized_outputs(model: onnx.ModelProto) -> list[str]:
    weights = {i.name for i in model.graph.initializer}
    return [n.output[0] for n in model.graph.node if n.op_type == "QuantizeLinear" and n.input[0] not in weights]


def with_outputs(model: onnx.ModelProto, names: list[str]) -> onnx.ModelProto:
    out = copy.deepcopy(model)
    present = {o.name for o in out.graph.output}
    for name in names:
        if name not in present:
            out.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.INT8, None))
    return out


def capture(model: onnx.ModelProto, X: np.ndarray) -> dict[str, np.ndarray]:
    """Every int8 tensor of the graph and the logits, for windows X, one ONNX Runtime thread."""
    names = quantized_outputs(model)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    graph = with_outputs(model, names).SerializeToString()
    sess = ort.InferenceSession(graph, opts, providers=["CPUExecutionProvider"])
    wanted = [*names, "logits"]
    return dict(zip(wanted, sess.run(wanted, {deploy.INPUT: X}), strict=True))


def initializer(model: onnx.ModelProto, name: str) -> np.ndarray:
    return numpy_helper.to_array(next(i for i in model.graph.initializer if i.name == name))


def pair_saturation(
    x: np.ndarray, x_zero: int, w: np.ndarray, groups: int, pads: tuple[int, int, int, int]
) -> np.ndarray:
    """Boolean mask over the convolution's outputs (N, M, Ho, Wo): some adjacent pair of products, in the x86
    kernel's form (activations + 128 as uint8, int8 weights), sums outside the int16 range.

    x: int8 input (N, C, H, W); w: int8 weights (M, C / groups, KH, KW); pads: top, left, bottom, right."""
    u = x.astype(np.int32) + 128
    top, left, bottom, right = pads
    u = np.pad(u, ((0, 0), (0, 0), (top, bottom), (left, right)), constant_values=int(x_zero) + 128)
    m_out, c_group, kh, kw = w.shape
    windows = sliding_window_view(u, (kh, kw), axis=(2, 3))  # (N, C, Ho, Wo, KH, KW)
    n, _, ho, wo = windows.shape[:4]
    per_group = m_out // groups
    mask = np.zeros((n, m_out, ho, wo), bool)
    for m in range(m_out):
        g = m // per_group
        a = windows[:, g * c_group : (g + 1) * c_group]  # (N, Cg, Ho, Wo, KH, KW)
        a = a.transpose(0, 2, 3, 4, 5, 1).reshape(n, ho, wo, -1)  # reduction order (kh, kw, c)
        k = w[m].transpose(1, 2, 0).reshape(-1).astype(np.int32)  # the same order
        products = a * k
        if products.shape[-1] % 2:
            products = np.concatenate([products, np.zeros((*products.shape[:-1], 1), np.int32)], axis=-1)
        pairs = products[..., 0::2] + products[..., 1::2]
        mask[:, m] = ((pairs < INT16[0]) | (pairs > INT16[1])).any(axis=-1)
    return mask


def seven_bit(model: onnx.ModelProto, layer: str) -> onnx.ModelProto:
    """A copy whose `layer` weights lie on a 7-bit grid: integers halved (rounded half away from zero, as
    ONNX Runtime's quantizer rounds), clipped to [-64, 63], the scale doubled; nothing else changes."""
    out = copy.deepcopy(model)
    inits = {i.name: i for i in out.graph.initializer}
    prefix = PREFIX.get(layer, layer)
    q, scale = inits[f"{prefix}.weight_quantized"], inits[f"{prefix}.weight_scale"]
    w = numpy_helper.to_array(q).astype(np.float64)
    halved = np.clip(np.sign(w) * np.floor(np.abs(w) / 2 + 0.5), -64, 63).astype(np.int8)
    q.CopyFrom(numpy_helper.from_array(halved, q.name))
    scale.CopyFrom(numpy_helper.from_array((numpy_helper.to_array(scale) * 2).astype(np.float32), scale.name))
    return out
