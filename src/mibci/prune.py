"""Structured pruning of EEGNet's final pointwise convolution."""

from __future__ import annotations

import torch
from braindecode.models import EEGNet

from mibci.model import build_eegnet

FRACTIONS = (0.25, 0.5)
FINETUNE_EPOCHS = 100
FINETUNE_LR = 1e-4  # one tenth of the training rate, fixed before any pruning result


def prune_pointwise(model: EEGNet, fraction: float) -> EEGNet:
    """Drop the `fraction` of conv_separable_point output filters with the smallest L1 norm.

    The matching BatchNorm channels and classifier inputs are sliced away too, so the
    returned model is genuinely smaller (no zero masks).
    """
    w = model.conv_separable_point.weight.detach()  # (F2, F1*D, 1, 1)
    n_keep = w.shape[0] - round(fraction * w.shape[0])
    keep = torch.argsort(w.abs().sum(dim=(1, 2, 3)), descending=True)[:n_keep].sort().values

    pruned = build_eegnet(model.conv_spatial.kernel_size[0], F2=n_keep)
    state, new = model.state_dict(), pruned.state_dict()
    for key in new:
        if key.startswith(("conv_separable_point.", "bnorm_2.")) and key != "bnorm_2.num_batches_tracked":
            new[key] = state[key][keep]
        elif not key.startswith("final_layer.linearconstraint.parametrizations.weight"):
            new[key] = state[key]
    # The classifier sees (F2 x time) features flattened filter-major. Copy the max-norm
    # constrained weights it actually used, restricted to the kept filters.
    dense = model.final_layer.linearconstraint.weight.detach()
    dense = dense.reshape(dense.shape[0], -1, model.final_conv_length)[:, keep].reshape(dense.shape[0], -1)
    new["final_layer.linearconstraint.parametrizations.weight.original"] = dense
    pruned.load_state_dict(new)
    return pruned
