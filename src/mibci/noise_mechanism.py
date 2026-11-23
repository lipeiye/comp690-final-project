"""Why an input that carries nothing costs EEGNet accuracy (shared code of scripts/noise_mechanism_study.py).

The input-cost study (scripts/input_cost_study.py) found that every input of band-limited noise added
to 16 electrodes costs about 0.9 percentage points on BCI IV 2a, on held-out runs of the training day
as much as on the next day. Two things are measured here.

Mechanism. After training, how much of each spatial filter's output is noise (`noise_share`, also at
initialization)? And is the trained network worse because its spatial filters still pass the noise
through, or because it learned less from the electrodes? `without_noise_path` sets the spatial
weights of the noise inputs to zero and keeps every other weight; `recalibrate` then re-estimates the
batch norms after the spatial filters on the training trials, so that later layers see features of
the scale they were trained on. If accuracy comes back, the noise passing through was the cost.

Noise draws. Each study drew one noise sample per participant and day, shared by every seed.
`appended_noise` returns draw 0, exactly the input-cost study's, or an independent further draw.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from scipy.signal import sosfilt
from torch import nn

from mibci import overfit
from mibci.data import SFREQ, bandpass_sos

DRAW_SEED = 693  # draws d >= 1: numpy seed [DRAW_SEED, dataset code, subject, day, d]


def band_gain() -> float:
    """Standard deviation of unit white noise after the preprocessing band-pass (impulse response)."""
    impulse = np.zeros(20 * SFREQ)
    impulse[0] = 1.0
    return float(np.sqrt((sosfilt(bandpass_sos(), impulse) ** 2).sum()))


def appended_noise(
    n_trials: int, n_times: int, added: int, largest: int, dataset: str, subject: int, day: int, draw: int
) -> np.ndarray:
    """Noise inputs (trials, added, times) of unit variance after the band-pass, nested: the first
    `added` of `largest`. Draw 0 is the input-cost study's (mibci.overfit.noise_channels on `largest`
    inputs); draw d >= 1 is made the same way from its own seed."""
    if draw == 0:
        zeros = np.zeros((n_trials, largest, n_times), np.float32)
        return overfit.noise_channels(zeros, list(range(largest)), dataset, subject, day)[:, :added]
    rng = np.random.default_rng([DRAW_SEED, overfit.DATASETS[dataset], subject, day, draw])
    warm = 2 * SFREQ
    white = rng.standard_normal((n_trials, largest, warm + n_times))
    noise = sosfilt(bandpass_sos(), white, axis=-1)[..., warm:] / band_gain()
    return noise[:, :added].astype(np.float32)


def spatial_input(model: nn.Module, X: np.ndarray) -> torch.Tensor:
    """What the spatial filters see: the temporal stage's output, (trials, F1, inputs, times)."""
    h = torch.from_numpy(X)
    for name, module in model.named_children():
        if name == "conv_spatial":
            return h
        h = module(h)
    raise ValueError("the network has no conv_spatial layer")


@torch.no_grad()
def noise_share(model: nn.Module, X: np.ndarray, noise: list[int]) -> np.ndarray:
    """Per spatial filter: the variance of the part of its output that comes from the `noise` inputs,
    over that variance plus the variance of the part from the other inputs (eval mode, inputs X)."""
    model.eval()
    h = spatial_input(model, X)
    w = model.conv_spatial.weight[:, 0, :, 0]  # effective weights, max-norm applied: (filters, inputs)
    per_map = w.shape[0] // h.shape[1]
    mask = torch.zeros(w.shape[1])
    mask[noise] = 1.0
    shares = []
    for f in range(w.shape[0]):
        x = h[:, f // per_map]
        from_noise = torch.einsum("c,nct->nt", w[f] * mask, x).var()
        from_rest = torch.einsum("c,nct->nt", w[f] * (1 - mask), x).var()
        shares.append(float(from_noise / (from_noise + from_rest)))
    return np.array(shares)


def weight_rms(model: nn.Module, inputs: list[int]) -> float:
    """Root mean square of the effective spatial weights on the given inputs."""
    w = model.conv_spatial.weight.detach()[:, 0, :, 0]
    return float(w[:, inputs].pow(2).mean().sqrt())


def without_noise_path(model: nn.Module, noise: list[int]) -> nn.Module:
    """A copy whose spatial filters give the noise inputs zero weight; every other weight unchanged.

    The effective weights (max-norm applied) are written back as the parametrization's original;
    zeroing entries only lowers a filter's norm, so the max-norm leaves the others as they were."""
    out = copy.deepcopy(model)
    with torch.no_grad():
        w = out.conv_spatial.weight.detach().clone()
        w[:, :, noise, :] = 0.0
        out.conv_spatial.parametrizations.weight.original.copy_(w)
    return out.eval()


@torch.no_grad()
def recalibrate(model: nn.Module, X_train: np.ndarray) -> nn.Module:
    """A copy whose batch norms after the spatial filters (bnorm_1, bnorm_2) are re-estimated on the
    training trials in one pass with dropout off. The temporal batch norm, whose input is unchanged,
    keeps its statistics."""
    out = copy.deepcopy(model).eval()
    for bn in (out.bnorm_1, out.bnorm_2):
        bn.reset_running_stats()
        bn.momentum = None  # one cumulative pass: the statistics of the whole training set
        bn.train()
    out(torch.from_numpy(X_train))
    return out.eval()
