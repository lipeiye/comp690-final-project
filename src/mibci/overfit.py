"""Inputs, splits and training for the overfitting study (scripts/overfit_study.py, post-test).

The study tests the explanation offered for why 16 of the 22 BCI IV 2a electrodes decode the
next day slightly better than all 22: the dropped electrodes carry little information, and with
a few hundred training trials the network fits patterns in them that do not carry over. Each
helper isolates one prediction of that explanation:

noise_channels       band-limited Gaussian noise in place of the dropped electrodes: inputs at
                     the same positions that carry nothing and do not change between days
shuffled_channels    the dropped electrodes' own signals, every trial given another trial's from
                     the same day: real EEG with its day-to-day changes, but unrelated to the
                     trial's class and to the kept electrodes
fit_channel_dropout  mibci.train.fit with channel dropout (each training example loses each
                     electrode with probability p, drawn again for every batch)
split_masks          training on whole runs of the training day, so that the held-out runs of
                     that day and the next day both score the same model

Random draws for inputs depend only on the dataset, participant and day, so every seed sees the
same inputs, as with a real recording; dropout masks come from their own generator seeded by
the training seed, never from the global state that EEGNet's own dropout layers use.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy.signal import sosfilt
from torch import nn

from mibci import selection, splits
from mibci.data import SFREQ, bandpass_sos
from mibci.train import TrainConfig

NOISE_SEED, SHUFFLE_SEED, MASK_SEED = 690, 691, 692
DATASETS = {"bnci2a": 1, "lee2019": 2}
DROPOUT_P = 0.2  # Braindecode ChannelsDropout default, fixed before any result of this study


def _noise_gain() -> float:
    """Standard deviation of unit white noise after the band-pass (from the impulse response)."""
    impulse = np.zeros(20 * SFREQ)
    impulse[0] = 1.0
    return float(np.sqrt((sosfilt(bandpass_sos(), impulse) ** 2).sum()))


def noise_channels(X: np.ndarray, replaced: list[int], dataset: str, subject: int, day: int) -> np.ndarray:
    """X with the `replaced` electrodes overwritten by band-limited Gaussian noise of unit variance.

    White noise goes through the preprocessing band-pass (causal 4th-order Butterworth, 4-38 Hz at
    250 Hz; both datasets use this band) after a 2-s warm-up. The scale is irrelevant to the
    network, which standardizes each electrode with training-day statistics, and is the same on
    both days, so these inputs carry neither class information nor a day-to-day change."""
    rng = np.random.default_rng([NOISE_SEED, DATASETS[dataset], subject, day])
    n_trials, _, n_times = X.shape
    warm = 2 * SFREQ
    white = rng.standard_normal((n_trials, len(replaced), warm + n_times))
    out = X.copy()
    out[:, replaced] = (sosfilt(bandpass_sos(), white, axis=-1)[..., warm:] / _noise_gain()).astype(X.dtype)
    return out


def shuffled_channels(X: np.ndarray, replaced: list[int], dataset: str, subject: int, day: int) -> np.ndarray:
    """X with the `replaced` electrodes copied from another trial of the same set of trials.

    The trials are deranged (no trial keeps its own signals) by a permutation that depends only on
    dataset, participant, day and the number of trials."""
    rng = np.random.default_rng([SHUFFLE_SEED, DATASETS[dataset], subject, day])
    order = rng.permutation(len(X))
    source = np.empty(len(X), dtype=int)
    source[order] = np.roll(order, 1)
    out = X.copy()
    out[:, replaced] = X[source][:, replaced]
    return out


def replace(X: np.ndarray, job: dict, day: int) -> np.ndarray:
    """The model input of a job: X unchanged, or with the job's replaced electrodes swapped out."""
    if not job["replaced"]:
        return X
    make = {"noise": noise_channels, "shuffled": shuffled_channels}[job["replacement"]]
    return make(X, job["replaced"], job["dataset"], job["subject"], day)


def split_masks(runs: np.ndarray, split: str, dataset: str) -> tuple[np.ndarray, np.ndarray | None]:
    """(train, held-out) masks over the training day's trials.

    'final' trains on everything. BCI IV 2a: 'fold-k' holds out the validation runs of fold k
    (mibci.splits, two of six runs) and trains on the other four; 'rfold-k' trains on those two
    runs and holds out the four. Lee2019: 'fold-k' trains on run k (offline 0, online 1) and holds
    out the other run."""
    if split == "final":
        return np.ones(len(runs), bool), None
    kind, k = split.rsplit("-", 1)
    if dataset == "lee2019":
        if kind != "fold":
            raise ValueError(f"Lee2019 has only fold-0 and fold-1, not {split}")
        train = runs == int(k)
        return train, ~train
    held = np.isin(runs, splits.VAL_RUNS[int(k)])
    return (~held, held) if kind == "fold" else (held, ~held)


def relevance_order(X: np.ndarray, y: np.ndarray) -> list[int]:
    """The frozen relevance order (mibci.selection, penalty 0) for one training partition.

    An electrode that is exactly zero in some training trial (two Lee2019 participants) has
    log power -inf; as in the replication study, such electrodes go last and the others are
    ranked as usual."""
    with np.errstate(divide="ignore"):
        logpow = selection.band_log_power(X)
    usable = np.flatnonzero(np.isfinite(logpow).all(axis=(0, 2)))
    rel = selection.relevance(logpow[:, usable], y)
    ranked = selection.greedy_ranking(rel, np.zeros((len(rel), len(rel))), 0.0)
    return [int(usable[i]) for i in ranked] + [int(c) for c in np.setdiff1d(np.arange(X.shape[1]), usable)]


def fit_channel_dropout(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, cfg: TrainConfig,
                        p: float = DROPOUT_P, device: str = "cpu") -> dict:  # fmt: skip
    """mibci.train.fit (fixed budget, Adam, same batch order) with channel dropout on every batch.

    Inputs are standardized, so a dropped electrode is set to its training mean (zero), the
    usual form of channel dropout for EEG; nothing is rescaled and nothing is dropped at test."""
    gen = torch.Generator().manual_seed(cfg.seed)  # batch order exactly as in mibci.train.fit
    masks = torch.Generator().manual_seed(cfg.seed * 1000 + MASK_SEED)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    X_t, y_t = torch.from_numpy(X_train).to(device), torch.from_numpy(y_train).to(device)
    losses = []
    start = time.perf_counter()
    for _ in range(cfg.epochs):
        model.train()
        order = torch.randperm(len(X_t), generator=gen).to(device)
        total = torch.zeros((), device=device)
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i : i + cfg.batch_size]
            keep = (torch.rand(len(idx), X_t.shape[1], 1, generator=masks) >= p).to(device=device, dtype=X_t.dtype)
            loss = nn.functional.cross_entropy(model(X_t[idx] * keep), y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach() * len(idx)
        losses.append(total.item() / len(order))
    model.to("cpu")
    return {"history": {"train_loss": losses}, "train_time_s": time.perf_counter() - start}
