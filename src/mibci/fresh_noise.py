"""Noise inputs drawn afresh in every training epoch (scripts/fresh_noise_study.py).

scripts/input_cost_study.py appends band-limited noise inputs to the 16 most relevant electrodes, drawn once
per participant and day: in all 475 epochs a training trial comes with the same noise, which a network can
use to remember the trial's label. Here every epoch draws new noise for every training trial, so the noise
never repeats; the EEG, the number of epochs and everything else stay as in that study, and the inputs that
are scored (held-out runs, the next day) are that study's own (draw 0).

draw0         the input-cost study's inputs: the kept electrodes, then the first `added` of `largest` noise
              inputs of mibci.overfit.noise_channels (its draw for every trial of that day)
epoch_noise   fresh noise for one epoch, the same band-limited unit-variance construction, seeded by
              (dataset, participant, split, model seed, epoch)
unseen_noise  a draw no model of either study was trained on (mibci.noise_mechanism.appended_noise, draw
              UNSEEN_DRAW), to test the training trials with noise the network has not seen
fit_epochwise mibci.train.fit with the training inputs taken from a function of the epoch; given the same
              inputs in every epoch it is mibci.train.fit, bit for bit (tests/test_fresh_noise.py)
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
from scipy.signal import sosfilt
from torch import nn

from mibci import noise_mechanism, overfit
from mibci.data import SFREQ, bandpass_sos
from mibci.train import TrainConfig

FRESH_SEED = 694  # numpy seed [FRESH_SEED, dataset code, participant, split code, model seed, epoch]
UNSEEN_DRAW = 100
SPLIT_CODES = {"final": 0, "fold-0": 1, "fold-1": 2, "fold-2": 3}


def draw0(
    X: np.ndarray, channels: list[int], added: int, largest: int, dataset: str, subject: int, day: int
) -> np.ndarray:
    noise = overfit.noise_channels(
        np.zeros((len(X), largest, X.shape[2]), np.float32), list(range(largest)), dataset, subject, day
    )
    return np.concatenate([X[:, channels], noise[:, :added]], axis=1)


def epoch_noise(
    n_trials: int, n_times: int, added: int, dataset: str, subject: int, split: str, seed: int, epoch: int
) -> np.ndarray:
    rng = np.random.default_rng([FRESH_SEED, overfit.DATASETS[dataset], subject, SPLIT_CODES[split], seed, epoch])
    warm = 2 * SFREQ
    white = rng.standard_normal((n_trials, added, warm + n_times))
    return (sosfilt(bandpass_sos(), white, axis=-1)[..., warm:] / noise_mechanism.band_gain()).astype(np.float32)


def unseen_noise(n_trials: int, n_times: int, added: int, largest: int, dataset: str, subject: int) -> np.ndarray:
    """Training-day noise from a draw that no model was trained on (for every trial of the day)."""
    return noise_mechanism.appended_noise(n_trials, n_times, added, largest, dataset, subject, 1, UNSEEN_DRAW)


def fit_epochwise(
    model: nn.Module, inputs: Callable[[int], np.ndarray], y_train: np.ndarray, cfg: TrainConfig, device: str = "cpu"
) -> dict:
    """Train for cfg.epochs with inputs(epoch) as the training inputs of that epoch (epochs counted from 1)."""
    gen = torch.Generator().manual_seed(cfg.seed)  # batch order as in mibci.train.fit
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    y_t = torch.from_numpy(y_train).to(device)
    history: dict[str, list[float]] = {"train_loss": []}
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        X_t = torch.from_numpy(np.ascontiguousarray(inputs(epoch))).to(device)
        if len(X_t) != len(y_t):
            raise ValueError(f"epoch {epoch}: {len(X_t)} inputs for {len(y_t)} labels")
        model.train()
        order = torch.randperm(len(X_t), generator=gen).to(device)
        total = torch.zeros((), device=device)
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i : i + cfg.batch_size]
            loss = nn.functional.cross_entropy(model(X_t[idx]), y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach() * len(idx)
        history["train_loss"].append(total.item() / len(order))
    model.to("cpu")
    return {"history": history, "train_time_s": time.perf_counter() - start}
