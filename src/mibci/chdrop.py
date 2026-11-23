"""Why channel dropout costs EEGNet so much (shared code of scripts/chdrop_study.py).

scripts/overfit_study.py trained EEGNet with channel dropout (mibci.overfit.fit_channel_dropout:
each training example loses each electrode with probability 0.2) and found it lowered next-day
accuracy by 11 points on BCI IV 2a and 2.7 on Lee2019. The explanation tested here: while some
electrodes are dropped, a spatial filter cannot cancel what the electrodes share, so its output
varies far more in training than at test, when every electrode is present. The batch norm after the
spatial filters stores the training variance and divides by it at test, which shrinks the signal.

variance_ratios   per batch-norm layer and channel, the stored running variance over the variance of
                  the layer's input when the network runs as at test (eval mode, every electrode)
with_broken       standardized inputs with some electrodes at their training mean (0), the same as a
                  dropped electrode in training, to measure how a model copes with a failed electrode
failure_sets      fixed random sets of electrodes to fail together

The batch norms are re-estimated with mibci.noise_mechanism.recalibrate (pinned; it re-estimates the
two batch norms after the spatial filters, which absorb any scale the temporal one passes on).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

BATCH_NORMS = ("bnorm_temporal", "bnorm_1", "bnorm_2")


@torch.no_grad()
def variance_ratios(model: nn.Module, X: np.ndarray) -> dict[str, list[float]]:
    """Stored running variance over the eval-mode variance of each batch norm's input on X, per channel."""
    model.eval()
    inputs: dict[str, torch.Tensor] = {}
    hooks = [
        getattr(model, name).register_forward_hook(lambda _m, args, _out, name=name: inputs.__setitem__(name, args[0]))
        for name in BATCH_NORMS
    ]
    try:
        model(torch.from_numpy(X))
    finally:
        for hook in hooks:
            hook.remove()
    out = {}
    for name in BATCH_NORMS:
        actual = inputs[name].var(dim=(0, 2, 3), unbiased=True)
        out[name] = (getattr(model, name).running_var / actual).tolist()
    return out


def with_broken(X: np.ndarray, electrodes) -> np.ndarray:
    """X (standardized inputs) with the given input positions at their training mean, 0."""
    out = X.copy()
    out[:, list(electrodes)] = 0.0
    return out


def failure_sets(n_inputs: int, size: int, count: int, seed: int) -> list[list[int]]:
    """`count` sets of `size` input positions, drawn without replacement within a set from a fixed seed."""
    rng = np.random.default_rng(seed)
    return [sorted(int(i) for i in rng.choice(n_inputs, size, replace=False)) for _ in range(count)]
