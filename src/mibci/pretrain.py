"""Pretraining on other people's training days (shared code of scripts/pretrain_study.py).

Every earlier study trained each person's network on that person's first day alone, and more of
one person's data did not shrink the cost of extra inputs. Pretraining pools other people instead:

group_electrodes   the k electrodes with the best mean rank in the relevance orders of the
                   pretraining participants (the target is never looked at), so that everyone in
                   one pretraining run has the same inputs
pooled             each person's trials standardized with that person's own training-day mean and
                   standard deviation (mibci.train.ChannelStandardizer), then concatenated
budget_epoch       the project's epoch-budget rule (scripts/choose_budget.py): average the accuracy
                   curves, smooth them with a centred moving average, take the earliest checkpoint
                   within the tolerance of the smoothed maximum (tests/test_pretrain.py reproduces
                   the frozen 475 epochs from the original validation curves)
lee_group          Lee2019 participants in six groups of nine by number (1-9, 10-18, ...)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mibci.train import ChannelStandardizer

LEE_GROUP_SIZE = 9


def lee_group(subject: int) -> int:
    return (subject - 1) // LEE_GROUP_SIZE


def group_electrodes(orders: Sequence[Sequence[int]], n_electrodes: int, k: int) -> list[int]:
    """The k electrodes with the smallest mean position over the given relevance orders (ties: lower index),
    returned in that order (best first)."""
    position = np.zeros(n_electrodes)
    for order in orders:
        if sorted(order) != list(range(n_electrodes)):
            raise ValueError("every order must rank all electrodes once")
        position[np.asarray(order)] += np.arange(n_electrodes)
    mean = position / len(orders)
    ranked = sorted(range(n_electrodes), key=lambda c: (mean[c], c))
    return [int(c) for c in ranked[:k]]


def standardized(X_train: np.ndarray, X: np.ndarray | None = None) -> tuple[np.ndarray, ChannelStandardizer]:
    """X (default: X_train) standardized with the statistics of X_train, and the scaler."""
    scaler = ChannelStandardizer().fit(X_train)
    return scaler.transform(X_train if X is None else X), scaler


def pooled(people: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate people's (inputs, labels), each person standardized with their own statistics."""
    X = np.concatenate([standardized(inputs)[0] for inputs, _ in people])
    y = np.concatenate([labels for _, labels in people]).astype(np.int64)
    return X, y


def budget_epoch(curves: np.ndarray, checkpoints: Sequence[int], window: int, tolerance: float) -> dict:
    """The earliest checkpoint at which the smoothed mean curve is within `tolerance` of its maximum.

    curves: (runs, epochs) accuracy after each epoch 1..E; checkpoints: epochs that were saved."""
    mean = np.asarray(curves, float).mean(0)
    smooth = pd.Series(mean).rolling(window, center=True, min_periods=1).mean().to_numpy()
    epochs = np.arange(1, len(mean) + 1)
    saved = np.isin(epochs, np.asarray(checkpoints))
    candidates = epochs[saved & (smooth >= smooth.max() - tolerance)]
    budget = int(candidates[0])
    return {
        "epochs": budget,
        "smoothed_max": float(smooth.max()),
        "smoothed_max_epoch": int(epochs[smooth.argmax()]),
        "smoothed_at_budget": float(smooth[budget - 1]),
        "smoothed": [round(float(v), 6) for v in smooth],
    }


def save_checkpoints(path: Path, states: dict[int, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({int(epoch): state for epoch, state in states.items()}, path)


def load_checkpoint(path: Path, epoch: int) -> dict:
    states = torch.load(path, weights_only=True)  # noqa: TID251  a dict of state dicts, not a model bundle
    return states[int(epoch)]
