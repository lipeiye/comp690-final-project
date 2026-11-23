"""EEGNet training with a group-lasso penalty on the spatial weights, one group per input.

scripts/noise_mechanism_study.py found that EEGNet trained as frozen (Adam, no weight decay, 475
epochs) does not learn to switch off inputs that carry nothing: after training, noise inputs make up
about as much of each spatial filter's output as at initialization, and the network learns less from
the electrodes. Nothing in that training pulls an input's weights toward zero. The penalty here,
lam times the sum over inputs of the L2 norm of that input's spatial weights across all filters, does:
an input keeps its weights only if the data keep pulling them back, so the network can drop inputs it
cannot use (and the remaining norms rank the inputs it uses).

With lam = 0, `fit_penalized` is mibci.train.fit step for step: the same batch order, optimizer,
updates and random draws (tests/test_input_penalty.py checks that the weights come out identical).
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from mibci.train import TrainConfig

EPS = 1e-12  # keeps the norm differentiable at zero


def input_norms(model: nn.Module) -> torch.Tensor:
    """Each input's L2 norm of effective spatial weights (max-norm applied) over all spatial filters."""
    w = model.conv_spatial.weight  # (filters, 1, inputs, 1)
    return (w.pow(2).sum(dim=(0, 1, 3)) + EPS).sqrt()


def fit_penalized(
    model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, cfg: TrainConfig, lam: float, device: str = "cpu"
) -> dict:
    """mibci.train.fit (fixed budget, no validation) with lam * sum(input_norms) added to each batch loss."""
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
            objective = loss + lam * input_norms(model).sum() if lam else loss
            opt.zero_grad()
            objective.backward()
            opt.step()
            total += loss.detach() * len(idx)
        history["train_loss"].append(total.item() / len(order))
        with torch.no_grad():
            history["penalty"].append(float(input_norms(model).sum()))
    model.to("cpu")
    return {"history": history, "train_time_s": time.perf_counter() - start}
