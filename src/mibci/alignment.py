"""Euclidean alignment (He and Wu, 2019): give every recording day the same average covariance.

The day-to-day shift in this data is partly a change of scale and of spatial mixing
(participant 2's day-2 signals are 1.5 times larger). Euclidean alignment whitens each
day's trials with the inverse square root of that day's mean spatial covariance, so both
days look alike to the decoder. It needs no labels: the training day is aligned with its
own trials, the test day with its own unlabeled trials (all of them offline, or only the
first run, as an online system could do while the user warms up).
"""

from __future__ import annotations

import numpy as np


def mean_covariance(X: np.ndarray) -> np.ndarray:
    """Average spatial covariance of trials (trials x channels x time), mean removed per trial."""
    X = X - X.mean(-1, keepdims=True)
    return np.einsum("nct,ndt->cd", X, X) / (X.shape[0] * X.shape[2])


def whitener(R: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """R^(-1/2), with a tiny ridge relative to the mean eigenvalue."""
    values, vectors = np.linalg.eigh(R)
    values = np.maximum(values, ridge * values.mean())
    return (vectors / np.sqrt(values)) @ vectors.T


def align(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.einsum("cd,ndt->nct", W, X).astype(np.float32)


def aligned(X: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Trials aligned with the mean covariance of `reference` (default: the trials themselves)."""
    return align(X, whitener(mean_covariance(X if reference is None else reference)))
