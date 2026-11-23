"""Variants of Euclidean alignment that separate two reasons it could hurt (scripts/alignment_mechanism_study.py).

Euclidean alignment (mibci.alignment) whitens each recording day with the inverse square root of its
mean spatial covariance. It helped on BCI IV 2a on average and cost 5.5 points with Lee2019's 62
electrodes. Two explanations were mixed together: the covariance of 62 electrodes from 100-200 trials is
estimated badly; or it is estimated well, but whitening blows up directions of tiny variance, mostly
noise, to the size of the useful ones. The variants:

ea           mibci.alignment exactly (tests/test_alignment_variants.py)
ea-shrink10  the covariance shrunk 10% toward the identity scaled to its mean eigenvalue, then whitened
ea-lw        the same with the Ledoit-Wolf coefficient (`ledoit_wolf_trials`)
ea-top       only the directions with eigenvalue at least 1% of the largest are whitened; the others
             are set to zero
ea-floor     eigenvalues below 1% of the largest are raised to 1% before whitening: the small directions
             are kept but not blown up (added after part A showed that 45-52 of Lee2019's 62 directions, and
             16-18 of 2a's 22, lie below 1%, so ea-top removes most of the space; the two together separate
             "blowing up noise" from "removing information")

`ledoit_wolf_trials` treats each trial's covariance as one observation of the day's mean covariance (the
day-level matrix alignment uses), so time samples that are correlated within a trial are not counted as
independent: alpha = min(1, sum_i ||C_i - R||_F^2 / (n^2 ||R - mu I||_F^2)), R the mean of the n trial
covariances C_i and mu its mean eigenvalue (Ledoit and Wolf, 2004). `spectrum` gives the numbers of
part A of the study: eigenvalues, condition number, directions below 1% of the largest, the coefficient,
and how far apart the whitening matrices of two interleaved halves of the trials are.
"""

from __future__ import annotations

import numpy as np

from mibci import alignment

VARIANTS = ("ea", "ea-shrink10", "ea-lw", "ea-top", "ea-floor")
SHRINK = 0.1
TOP_FLOOR = 0.01  # keep directions with eigenvalue >= this share of the largest


def trial_covariances(X: np.ndarray) -> np.ndarray:
    """(trials, channels, channels) covariance of each trial, mean removed per trial, as mibci.alignment does."""
    X = X - X.mean(-1, keepdims=True)
    return np.einsum("nct,ndt->ncd", X, X) / X.shape[2]


def ledoit_wolf_trials(X: np.ndarray) -> float:
    C = trial_covariances(X.astype(np.float64))
    R = C.mean(0)
    p = R.shape[0]
    mu = np.trace(R) / p
    delta = ((R - mu * np.eye(p)) ** 2).sum()
    beta = ((C - R) ** 2).sum() / len(C) ** 2
    return float(min(1.0, beta / delta))


def shrunk(R: np.ndarray, alpha: float) -> np.ndarray:
    p = R.shape[0]
    return (1 - alpha) * R + alpha * (np.trace(R) / p) * np.eye(p)


def top_whitener(R: np.ndarray, floor: float = TOP_FLOOR) -> np.ndarray:
    """R^(-1/2) on the directions with eigenvalue >= floor * the largest; zero on the others."""
    values, vectors = np.linalg.eigh(R)
    keep = values >= floor * values.max()
    V = vectors[:, keep]
    return (V / np.sqrt(values[keep])) @ V.T


def floor_whitener(R: np.ndarray, floor: float = TOP_FLOOR) -> np.ndarray:
    """R^(-1/2) with every eigenvalue raised to at least floor * the largest."""
    values, vectors = np.linalg.eigh(R)
    values = np.maximum(values, floor * values.max())
    return (vectors / np.sqrt(values)) @ vectors.T


def whitener(X: np.ndarray, variant: str) -> np.ndarray:
    """The alignment matrix of a variant, from the trials X (trials, channels, time)."""
    R = alignment.mean_covariance(X)
    if variant == "ea":
        return alignment.whitener(R)
    if variant == "ea-shrink10":
        return alignment.whitener(shrunk(R, SHRINK))
    if variant == "ea-lw":
        return alignment.whitener(shrunk(R, ledoit_wolf_trials(X)))
    if variant == "ea-top":
        return top_whitener(R)
    if variant == "ea-floor":
        return floor_whitener(R)
    raise ValueError(f"unknown variant {variant}")


def aligned(X: np.ndarray, variant: str, reference: np.ndarray | None = None) -> np.ndarray:
    """Trials aligned with the variant's matrix from `reference` (default: the trials themselves)."""
    return alignment.align(X, whitener(X if reference is None else reference, variant))


def spectrum(X: np.ndarray) -> dict:
    """Eigenvalues of the mean covariance and the measures of how well it is estimated."""
    R = alignment.mean_covariance(X.astype(np.float64))
    values = np.sort(np.linalg.eigvalsh(R))[::-1]
    even, odd = X[0::2], X[1::2]
    Wa, Wb = alignment.whitener(alignment.mean_covariance(even)), alignment.whitener(alignment.mean_covariance(odd))
    half = float(np.linalg.norm(Wa - Wb) / ((np.linalg.norm(Wa) + np.linalg.norm(Wb)) / 2))
    return {
        "eigenvalues": [float(v) for v in values],
        "condition_number": float(values[0] / values[-1]),
        "below_1pct_of_largest": int((values < TOP_FLOOR * values[0]).sum()),
        "ledoit_wolf": ledoit_wolf_trials(X),
        "split_half_whitener_difference": half,
        "trials": len(X),
    }
