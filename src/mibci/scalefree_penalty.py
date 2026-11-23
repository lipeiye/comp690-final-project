"""EEGNet training with a penalty on the inputs that no rescaling of a spatial filter can lower.

scripts/input_penalty_study.py added a group-lasso penalty on each input's spatial weights and found
that it does not switch inputs off: a batch norm follows the spatial filters and standardizes each
filter's output, so shrinking every weight of a filter together leaves the network's output as it was,
and that is the cheapest way to lower a group lasso. The penalty here cannot be lowered that way.
Each spatial filter's weight vector over the inputs is first scaled to unit length; g_i is input i's
norm over all filters of those unit vectors, and the penalty is lam * sum_i g_i. Because
sum_i g_i^2 equals the number of filters whatever the weights are, the only way to lower sum_i g_i
is to put each filter's weight on fewer inputs.

With lam = 0, `fit_scalefree` is mibci.train.fit step for step: the same batch order, optimizer,
updates and random draws (tests/test_scalefree_penalty.py checks that the weights come out identical).
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from mibci.train import TrainConfig

EPS = 1e-12  # keeps the norms differentiable at zero


def normalized_norms(weight: torch.Tensor) -> torch.Tensor:
    """g_i: each input's norm, over all filters, of the filters' unit-length weight vectors.

    weight is the effective spatial weight (filters, 1, inputs, 1)."""
    w = weight[:, 0, :, 0]
    unit = w / (w.pow(2).sum(dim=1, keepdim=True) + EPS).sqrt()
    return (unit.pow(2).sum(dim=0) + EPS).sqrt()


def penalty(weight: torch.Tensor) -> torch.Tensor:
    """sum_i g_i, between sqrt(filters) (every filter on one input) and sqrt(filters * inputs) (spread evenly)."""
    return normalized_norms(weight).sum()


def input_norms(model: nn.Module) -> torch.Tensor:
    """g_i of a network, from its effective spatial weights (max-norm applied)."""
    return normalized_norms(model.conv_spatial.weight)


def fit_scalefree(
    model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, cfg: TrainConfig, lam: float, device: str = "cpu"
) -> dict:
    """mibci.train.fit (fixed budget, no validation) with lam * penalty(spatial weights) added to each batch loss."""
    gen = torch.Generator().manual_seed(cfg.seed)  # batch order is drawn on the CPU for every device
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    X_t, y_t = torch.from_numpy(X_train).to(device), torch.from_numpy(y_train).to(device)
    history: dict[str, list[float]] = {"train_loss": [], "penalty": []}
    start = time.perf_counter()
    for _ in range(cfg.epochs):
        model.train()
        order = torch.randperm(len(X_t), generator=gen).to(device)
        total = torch.zeros((), device=device)
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i : i + cfg.batch_size]
            loss = nn.functional.cross_entropy(model(X_t[idx]), y_t[idx])
            objective = loss + lam * penalty(model.conv_spatial.weight) if lam else loss
            opt.zero_grad()
            objective.backward()
            opt.step()
            total += loss.detach() * len(idx)
        history["train_loss"].append(total.item() / len(order))
        with torch.no_grad():
            history["penalty"].append(float(penalty(model.conv_spatial.weight)))
    model.to("cpu")
    return {"history": history, "train_time_s": time.perf_counter() - start}
