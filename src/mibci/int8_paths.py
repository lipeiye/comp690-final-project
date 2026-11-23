"""Single-convolution models that map where ONNX Runtime's INT8 convolution is fast (scripts/int8_paths_study.py).

EEGNet's INT8 graph was slower than FP32 because one layer, the spatial convolution with 8 groups of
2 filters spanning every electrode, fell to ONNX Runtime's general grouped path (scripts/ort_kernel_study.py).
To see which shapes are affected beyond that one layer, `OneConv` is a single convolution with a chosen
number of groups G, filters per group M (one input channel per group) and kernel orientation, applied to an
EEG-shaped input (G channels x 22 electrodes x 500 samples):

  spatial    a 22 x 1 kernel spanning every electrode, as EEGNet's spatial layer (G = 8, M = 2) and the
             swapped network's first layer (G = 1, M = 16)
  temporal   a 1 x 64 kernel along time with 32 samples of zero padding on each side, as EEGNet's temporal
             layer (G = 1, M = 8) and, depthwise (M = 1), the swapped network's temporal layer

Graphs are exported and quantized exactly as the decoders were (mibci.deploy: opset 18, static QDQ,
per-channel int8 weights, min-max calibration), so each runs as the same kind of QLinearConv node.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mibci import deploy

N_CHANS, N_TIMES = 22, 500
KERNELS = {"spatial": ((N_CHANS, 1), (0, 0)), "temporal": ((1, 64), (0, 32))}
MAX_OUTPUTS = 32  # G * M; larger grouped layers do not occur in EEGNet-sized networks


def grid() -> list[dict]:
    """Every (kernel, G, M) with G in 1..16, M in 1..16 (powers of two) and G * M <= MAX_OUTPUTS."""
    sizes = (1, 2, 4, 8, 16)
    return [
        {"name": f"{kernel}_g{g}_m{m}", "kernel": kernel, "groups": g, "per_group": m}
        for kernel in KERNELS
        for g in sizes
        for m in sizes
        if g * m <= MAX_OUTPUTS
    ]


class OneConv(nn.Module):
    def __init__(self, kernel: str, groups: int, per_group: int, seed: int = 0):
        super().__init__()
        size, padding = KERNELS[kernel]
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(groups, groups * per_group, size, groups=groups, padding=padding, bias=False)

    def forward(self, x):
        return self.conv(x)


def macs(shape: dict) -> int:
    """Multiply-accumulates of one window."""
    (kh, kw), (ph, pw) = KERNELS[shape["kernel"]]
    out_h, out_w = N_CHANS + 2 * ph - kh + 1, N_TIMES + 2 * pw - kw + 1
    return shape["groups"] * shape["per_group"] * out_h * out_w * kh * kw


def inputs(shape: dict, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, shape["groups"], N_CHANS, N_TIMES)).astype(np.float32)


def export(shape: dict, fp32: Path, int8: Path, calibration: np.ndarray) -> None:
    """FP32 and INT8 graphs of one shape, as mibci.deploy exports and quantizes the decoders."""
    model = OneConv(shape["kernel"], shape["groups"], shape["per_group"]).eval()
    x = torch.from_numpy(calibration[:1])
    with contextlib.redirect_stdout(io.StringIO()):
        torch.onnx.export(model, (x,), fp32, input_names=[deploy.INPUT], output_names=["out"],
                          opset_version=deploy.OPSET, dynamo=True, optimize=True, external_data=False,
                          verbose=False)  # fmt: skip
    deploy.quantize_int8(fp32, int8, calibration)
