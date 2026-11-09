"""Electrode selection: the redundancy-aware rule and its controls.

Every data-driven ranking is fit on training trials only: the training part of a
validation fold, or all of session 1 for the final models.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import welch
from scipy.stats import spearmanr

from mibci.data import EEG_CHANNELS, SFREQ

BANDS_HZ = ((8.0, 13.0), (13.0, 30.0))  # mu and beta
REDUNDANCY_PENALTY = 0.5
CHANNEL_COUNTS = (16, 8)
N_RANDOM = 3
RANDOM_SEED = 690  # fixed before any results; the random subsets are shared by all subjects
METHODS = ("proposed", "relevance", "weightnorm", "random1", "random2", "random3")
FULL = "full-22"


def band_log_power(X: np.ndarray) -> np.ndarray:
    """Log band power per trial, channel and band (trials x channels x bands), from Welch PSDs."""
    freqs, psd = welch(X, fs=SFREQ, nperseg=SFREQ, axis=-1)
    df = freqs[1] - freqs[0]
    power = [psd[..., (freqs >= lo) & (freqs < hi)].sum(-1) * df for lo, hi in BANDS_HZ]
    return np.log(np.stack(power, axis=-1))


def fisher_score(features: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Multiclass Fisher score of each feature: between-class over within-class variance."""
    classes = np.unique(y)
    mean = features.mean(0)
    between = sum((y == c).sum() * (features[y == c].mean(0) - mean) ** 2 for c in classes)
    within = sum((y == c).sum() * features[y == c].var(0) for c in classes)
    return between / within


def relevance(logpow: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean Fisher score over the two bands, min-max scaled to 0-1 across channels."""
    r = fisher_score(logpow, y).mean(-1)
    return (r - r.min()) / (r.max() - r.min())


def redundancy(logpow: np.ndarray) -> np.ndarray:
    """Channel x channel mean |Spearman rho| of log power across trials, averaged over bands."""
    rho = [np.abs(spearmanr(logpow[:, :, b]).statistic) for b in range(logpow.shape[-1])]
    return np.mean(rho, axis=0)


def greedy_ranking(rel: np.ndarray, red: np.ndarray, penalty: float) -> list[int]:
    """Start from the most relevant channel, then repeatedly add the channel that maximizes
    relevance - penalty * (mean redundancy with the channels already chosen).
    With penalty 0 this is simply the relevance order."""
    order = [int(np.argmax(rel))]
    while len(order) < len(rel):
        candidates = [c for c in range(len(rel)) if c not in order]
        scores = [rel[c] - penalty * red[c, order].mean() for c in candidates]
        order.append(candidates[int(np.argmax(scores))])
    return order


def weightnorm_ranking(full_model: torch.nn.Module) -> list[int]:
    """Channels ordered by the L2 norm of their learned EEGNet spatial weights.

    Inspired by MI-BMInet's spatial-filter-based selection; an adaptation, not a
    reproduction. The weights are read after the max-norm constraint is applied.
    """
    w = full_model.conv_spatial.weight.detach()  # (F1*D, 1, n_chans, 1)
    return torch.argsort(w[:, 0, :, 0].norm(dim=0), descending=True).tolist()


def random_subsets() -> dict[int, list[list[int]]]:
    rng = np.random.default_rng(RANDOM_SEED)
    return {
        k: [sorted(rng.choice(len(EEG_CHANNELS), size=k, replace=False).tolist()) for _ in range(N_RANDOM)]
        for k in CHANNEL_COUNTS
    }


def rank_channels(X_train: np.ndarray, y_train: np.ndarray, full_model: torch.nn.Module) -> dict:
    """Relevance scores and the three data-driven channel orders for one training partition."""
    logpow = band_log_power(X_train)
    rel, red = relevance(logpow, y_train), redundancy(logpow)
    return {
        "relevance_scores": rel.tolist(),
        "orders": {
            "proposed": greedy_ranking(rel, red, REDUNDANCY_PENALTY),
            "relevance": greedy_ranking(rel, red, 0.0),
            "weightnorm": weightnorm_ranking(full_model),
        },
    }


def condition_subsets(orders: dict[str, list[int]]) -> dict[str, list[int]]:
    """Channel indices (montage order) for every condition, keyed like 'proposed-8'."""
    subsets = {FULL: list(range(len(EEG_CHANNELS)))}
    randoms = random_subsets()
    for k in CHANNEL_COUNTS:
        for method, order in orders.items():
            subsets[f"{method}-{k}"] = sorted(order[:k])
        for i, subset in enumerate(randoms[k], start=1):
            subsets[f"random{i}-{k}"] = subset
    return subsets
