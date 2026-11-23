"""Hardware-neutral quantization of a trained EEGNet: simulate b-bit integer inference in PyTorch.

How fast INT8 runs depends on the runtime and the chip; how many bits the decoder needs
does not. This module answers the second question for any bit width, not only the 8 and
16 bits that ONNX Runtime offers, by rounding weights and activations to integer grids:

  fold        BatchNorm folded into the preceding convolution, as the ONNX exporter does,
              so the simulated network is the deployed one (same function in eval mode).
  weights     symmetric per-output-channel grid, 2^(b-1) - 1 levels on each side of zero
              (ONNX Runtime's per-channel signed weights).
  activations asymmetric per-tensor grid with 2^b levels over the min-max range seen on
              calibration data, zero included (ONNX Runtime's MinMax calibration), at the
              network input and after every layer, as in the QDQ graph. Biases stay in
              floating point (ONNX Runtime keeps them as 32-bit integers).
  qat         quantization-aware fine-tuning: the same rounding with a straight-through
              gradient, activation ranges tracked as a moving average while training.

A bit width is an int, None (floating point), or a mapping from layer or tap name to bits
(names missing from the mapping stay floating point). `bit_operations` counts MACs x weight
bits x activation bits, the usual hardware-neutral measure of compute.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mibci.costs import layer_costs
from mibci.model import deploy_copy

WEIGHTED = ("conv_t", "conv_s", "conv_d", "conv_p", "fc")
POINTS = ("input", "conv_t", "conv_s", "elu1", "pool1", "conv_d", "conv_p", "elu2", "pool2", "logits")
# Braindecode layer name (for MAC counts) -> (name here, tap that feeds the layer)
BRAINDECODE = {
    "conv_temporal": ("conv_t", "input"),
    "conv_spatial": ("conv_s", "conv_t"),
    "conv_separable_depth": ("conv_d", "pool1"),
    "conv_separable_point": ("conv_p", "conv_d"),
    "final_layer.linearconstraint": ("fc", "pool2"),
}
Bits = int | Mapping | None


def bits_of(spec: Bits, name: str) -> int | None:
    return spec.get(name) if isinstance(spec, Mapping) else spec


def _round(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest integer; the gradient passes straight through."""
    return x + (torch.round(x) - x).detach()


def quantize_weight(w: torch.Tensor, bits: int | None) -> torch.Tensor:
    if bits is None:
        return w
    levels = 2 ** (bits - 1) - 1
    scale = w.detach().abs().amax(dim=tuple(range(1, w.dim())), keepdim=True).clamp_min(1e-12) / levels
    return torch.clamp(_round(w / scale), -levels, levels) * scale


def quantize_activation(x: torch.Tensor, low: float, high: float, bits: int | None) -> torch.Tensor:
    if bits is None:
        return x
    low, high = min(low, 0.0), max(high, 0.0)
    scale = max(high - low, 1e-12) / (2**bits - 1)
    zero = round(-low / scale)
    return (torch.clamp(_round(x / scale) + zero, 0, 2**bits - 1) - zero) * scale


def _fold(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    scale = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
    folded = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, groups=conv.groups
    )
    folded.weight.data = conv.weight.detach() * scale[:, None, None, None]
    folded.bias.data = bn.bias.detach() - bn.running_mean * scale
    return folded


class Folded(nn.Module):
    """A trained EEGNet with BatchNorm folded in, computing with optionally rounded weights and
    activations. Without bit widths it is the same function as the original in eval mode."""

    def __init__(self, model: nn.Module):
        super().__init__()
        m = deploy_copy(model)
        self.prepare = nn.Sequential(m.ensuredims, m.dimshuffle)
        self.conv_t = _fold(m.conv_temporal, m.bnorm_temporal)
        self.conv_s = _fold(m.conv_spatial, m.bnorm_1)
        self.pool1, self.drop1 = m.pool_1, m.drop_1
        self.conv_d = copy.deepcopy(m.conv_separable_depth)
        self.conv_p = _fold(m.conv_separable_point, m.bnorm_2)
        self.pool2, self.drop2 = m.pool_2, m.drop_2
        dense = m.final_layer.linearconstraint
        self.fc = nn.Linear(dense.in_features, dense.out_features)
        self.fc.weight.data, self.fc.bias.data = dense.weight.detach().clone(), dense.bias.detach().clone()
        self.weight_bits: Bits = None
        self.activation_bits: Bits = None
        self.ranges: dict[str, list[float]] | None = None
        self.track_ranges = False  # moving-average ranges while training (QAT)
        self.momentum = 0.99
        self.eval()

    def _tap(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if self.ranges is None:
            return x
        if self.training and self.track_ranges:
            low, high = self.ranges[name]
            m = self.momentum
            seen = x.detach()
            self.ranges[name] = [m * low + (1 - m) * float(seen.min()), m * high + (1 - m) * float(seen.max())]
        return quantize_activation(x, *self.ranges[name], bits_of(self.activation_bits, name))

    def _conv(self, name: str, x: torch.Tensor) -> torch.Tensor:
        layer = getattr(self, name)
        w = quantize_weight(layer.weight, bits_of(self.weight_bits, name))
        return F.conv2d(x, w, layer.bias, layer.stride, layer.padding, layer.dilation, layer.groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._tap("input", x)
        x = self._tap("conv_t", self._conv("conv_t", self.prepare(x)))
        x = self._tap("conv_s", self._conv("conv_s", x))
        x = self._tap("elu1", F.elu(x))
        x = self.drop1(self._tap("pool1", self.pool1(x)))
        x = self._tap("conv_d", self._conv("conv_d", x))
        x = self._tap("conv_p", self._conv("conv_p", x))
        x = self._tap("elu2", F.elu(x))
        x = self.drop2(self._tap("pool2", self.pool2(x))).flatten(1)
        w = quantize_weight(self.fc.weight, bits_of(self.weight_bits, "fc"))
        return self._tap("logits", F.linear(x, w, self.fc.bias))


@torch.no_grad()
def calibrate(folded: Folded, X: np.ndarray, batch: int = 256) -> dict[str, list[float]]:
    """Min and max of every tapped tensor of the floating-point network over the calibration trials."""
    plain = copy.deepcopy(folded)
    plain.weight_bits = plain.activation_bits = None
    plain.ranges = {name: [np.inf, -np.inf] for name in POINTS}
    plain.eval()

    def record(name, x):
        low, high = plain.ranges[name]
        plain.ranges[name] = [min(low, float(x.min())), max(high, float(x.max()))]
        return x

    plain._tap = record
    for i in range(0, len(X), batch):
        plain(torch.from_numpy(X[i : i + batch]))
    return plain.ranges


def quantized(folded: Folded, ranges: dict, weight_bits: Bits = 8, activation_bits: Bits = 8) -> Folded:
    """A copy that computes with rounded weights and activations (post-training quantization)."""
    q = copy.deepcopy(folded)
    q.weight_bits, q.activation_bits = weight_bits, activation_bits
    q.ranges = {k: list(v) for k, v in ranges.items()}
    return q.eval()


@torch.no_grad()
def logits(model: nn.Module, X: np.ndarray, batch: int = 512) -> np.ndarray:
    model.eval()
    return torch.cat([model(torch.from_numpy(X[i : i + batch])) for i in range(0, len(X), batch)]).numpy()


def finetune(
    folded: Folded, X: np.ndarray, y: np.ndarray, *, epochs: int, lr: float, seed: int, batch_size: int = 64
) -> Folded:
    """Fine-tune a (possibly quantized) folded network with Adam on training trials. With bit widths
    set this is quantization-aware training; with none it is the matched floating-point control."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = copy.deepcopy(folded)
    model.track_ranges = model.ranges is not None
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X_t, y_t = torch.from_numpy(X), torch.from_numpy(y)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(X_t), generator=gen)
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            loss = F.cross_entropy(model(X_t[idx]), y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.track_ranges = False
    return model.eval()


def bit_operations(model: nn.Module, weight_bits: Bits, activation_bits: Bits) -> dict:
    """MACs x weight bits x input-activation bits per layer of the original EEGNet; floating
    point counts as 32 bits."""
    rows, total = [], 0
    for row in layer_costs(model):
        if row["layer"] not in BRAINDECODE:
            continue
        name, source = BRAINDECODE[row["layer"]]
        w, a = bits_of(weight_bits, name) or 32, bits_of(activation_bits, source) or 32
        rows.append({"layer": name, "macs": row["macs"], "weight_bits": w, "activation_bits": a,
                     "bops": row["macs"] * w * a})  # fmt: skip
        total += row["macs"] * w * a
    return {"bops": total, "layers": rows}
