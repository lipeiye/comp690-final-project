"""Electrode-selection methods that look at electrodes jointly (added after the midterm review).

The frozen selection rule (mibci.selection) scores each electrode on its own: band power,
Fisher score, and a penalty for electrodes whose band power rises and falls together. A
motor-imagery decoder, however, decodes through spatial filters, weighted sums of
electrodes; two correlated neighbours can be worth more together than either alone,
because their difference cancels activity they share. The methods here score subsets
through the spatial covariance, which is what spatial filters act on:

riemann_order     Barachant and Bonnet (2011): backward elimination that keeps the subset
                  whose class-mean covariance matrices stay farthest apart in the
                  affine-invariant Riemannian metric (pyriemann's ElectrodeSelection).
csp_order         one-vs-rest common spatial patterns; electrodes ranked by the size of their
                  pattern coefficients in the most discriminative filters.
wrapper_order     backward elimination scored by run-wise cross-validated accuracy of a fast
                  surrogate decoder (Riemannian tangent space + logistic regression).

Every order lists electrodes most important first, so `order[:k]` is the k-electrode subset.
All of them are fit on training trials only (session 1, or the training runs of a fold).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression

from mibci.data import SFREQ
from mibci.splits import VAL_RUNS

MU_BETA_HZ = (8.0, 30.0)


def mu_beta(X: np.ndarray) -> np.ndarray:
    """8-30 Hz zero-phase band-pass of each trial (offline selection only, never the decoder)."""
    sos = butter(4, MU_BETA_HZ, btype="bandpass", fs=SFREQ, output="sos")
    return sosfiltfilt(sos, X, axis=-1)


def covariances(X: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Trial covariance matrices (trials x channels x channels), slightly regularised."""
    X = X - X.mean(-1, keepdims=True)
    covs = np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]
    scale = np.trace(covs, axis1=1, axis2=2)[:, None, None] / covs.shape[1]
    return covs + ridge * scale * np.eye(covs.shape[1])


def _powm(C: np.ndarray, power: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(C)
    return (vectors * values**power) @ vectors.T


def _logm(C: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(C)
    return (vectors * np.log(values)) @ vectors.T


def _expm(C: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(C)
    return (vectors * np.exp(values)) @ vectors.T


def riemann_mean(covs: np.ndarray, tol: float = 1e-8, max_iter: int = 50) -> np.ndarray:
    """Geometric (Karcher) mean of SPD matrices under the affine-invariant metric."""
    mean = covs.mean(0)
    for _ in range(max_iter):
        root, inv_root = _powm(mean, 0.5), _powm(mean, -0.5)
        step = np.mean([_logm(inv_root @ c @ inv_root) for c in covs], axis=0)
        mean = root @ _expm(step) @ root
        if np.linalg.norm(step) < tol:
            break
    return mean


def riemann_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Affine-invariant distance: sqrt(sum log^2 of the generalised eigenvalues of (B, A))."""
    return float(np.sqrt(np.sum(np.log(eigh(B, A, eigvals_only=True)) ** 2)))


def riemann_order(X: np.ndarray, y: np.ndarray) -> list[int]:
    """Backward elimination keeping class-mean covariances farthest apart (Barachant and Bonnet 2011)."""
    covs = covariances(mu_beta(X))
    means = [riemann_mean(covs[y == c]) for c in np.unique(y)]
    keep, removed = list(range(X.shape[1])), []
    while len(keep) > 1:
        spread = []
        for drop in keep:
            sub = [c for c in keep if c != drop]
            grid = np.ix_(sub, sub)
            spread.append(
                sum(
                    riemann_distance(means[i][grid], means[j][grid])
                    for i in range(len(means))
                    for j in range(i + 1, len(means))
                )
            )
        removed.append(keep.pop(int(np.argmax(spread))))
    return keep + removed[::-1]


def csp_order(X: np.ndarray, y: np.ndarray, n_filters: int = 4) -> list[int]:
    """One-vs-rest CSP: rank electrodes by the |pattern| of each class's most discriminative
    filters, each filter weighted by how far its variance ratio is from chance,
    |log(lambda / (1 - lambda))|, so that filters fitting noise barely count."""
    covs = covariances(mu_beta(X))
    score = np.zeros(X.shape[1])
    for c in np.unique(y):
        target, rest = covs[y == c].mean(0), covs[y != c].mean(0)
        values, filters = eigh(target, target + rest)
        weight = np.abs(np.log(values / (1 - values)))
        chosen = np.argsort(-weight)[:n_filters]
        patterns = np.abs((target + rest) @ filters[:, chosen])  # forward model of each filter
        score += (patterns / patterns.max(0) * weight[chosen]).sum(1)
    return [int(i) for i in np.argsort(-score, kind="stable")]


def tangent_space(covs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Upper-triangular tangent vectors at `reference` (off-diagonals weighted by sqrt 2)."""
    inv_root = _powm(reference, -0.5)
    logs = np.array([_logm(inv_root @ c @ inv_root) for c in covs])
    rows, cols = np.triu_indices(covs.shape[1])
    weights = np.where(rows == cols, 1.0, np.sqrt(2.0))
    return logs[:, rows, cols] * weights


def surrogate_accuracy(covs: np.ndarray, y: np.ndarray, runs: np.ndarray, channels: list[int]) -> float:
    """Run-wise 3-fold accuracy of tangent space + logistic regression on a channel subset."""
    grid = np.ix_(range(len(covs)), channels, channels)
    sub = covs[grid]
    correct = 0
    for held_out in VAL_RUNS:
        test = np.isin(runs, held_out)
        reference = riemann_mean(sub[~test], tol=1e-6, max_iter=20)
        model = LogisticRegression(C=1.0, max_iter=2000)
        model.fit(tangent_space(sub[~test], reference), y[~test])
        correct += int((model.predict(tangent_space(sub[test], reference)) == y[test]).sum())
    return correct / len(y)


def wrapper_order(X: np.ndarray, y: np.ndarray, runs: np.ndarray) -> tuple[list[int], list[float]]:
    """Backward elimination by surrogate cross-validated accuracy; also returns the accuracy
    of each nested subset, from all electrodes down to two."""
    covs = covariances(mu_beta(X))
    keep, removed = list(range(X.shape[1])), []
    curve = [surrogate_accuracy(covs, y, runs, keep)]
    while len(keep) > 2:
        scores = [surrogate_accuracy(covs, y, runs, [c for c in keep if c != drop]) for drop in keep]
        best = int(np.argmax(scores))
        removed.append(keep.pop(best))
        curve.append(scores[best])
    # the last two cannot be told apart by a one-electrode decoder; keep their montage order
    return keep + removed[::-1], curve
