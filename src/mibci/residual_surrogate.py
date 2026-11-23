"""Dropped electrodes split into what the kept electrodes predict and the rest (residual_surrogate_study).

scripts/input_kind_study.py placed the real dropped electrodes on the axis of copies mixed with noise and read
their cost off that curve by linear interpolation, which can be too small (the curve rises steeply near zero)
or too large (the mixes' noise is independent across inputs, the dropped electrodes' unpredictable parts are
correlated). This module builds extra inputs that keep everything about the dropped electrodes except their
relation to the trial, so the cost and the information of the real electrodes can be measured directly.

fit          a least-squares prediction (with intercept) of every dropped electrode from the kept electrodes, every
             time sample of every trial one observation, fitted on the given (training) trials only; the same
             fit as mibci.input_kind.unexplained_fraction
predictable  the fit's prediction for any trials: (trials, dropped, times)
residual     the dropped electrodes minus their prediction
derangement  a seeded random permutation of n trials that moves every trial
extra_inputs real  the dropped electrodes themselves (prediction + this trial's own residual)
             pred  the prediction alone (residual zero)
             surr  the prediction + another trial's residual: the residuals of all dropped electrodes of a
                   trial move together to another trial of the same set (a derangement), keeping their size,
                   spectrum and correlation across electrodes but not their relation to the trial
model_input  the kept electrodes, then the extra inputs
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

KINDS = ("real", "pred", "surr")


@dataclass(frozen=True)
class Fit:
    kept: tuple[int, ...]
    dropped: tuple[int, ...]
    kept_mean: np.ndarray  # (kept,)
    dropped_mean: np.ndarray  # (dropped,)
    coef: np.ndarray  # (kept, dropped)

    def digest(self) -> str:
        """SHA-256 of the fitted numbers, to check that evaluation refits the same prediction."""
        h = hashlib.sha256()
        for a in (self.kept_mean, self.dropped_mean, self.coef):
            h.update(np.ascontiguousarray(a, np.float64).tobytes())
        return h.hexdigest()


def _rows(X: np.ndarray, electrodes) -> np.ndarray:
    """(trials * times, electrodes) in float64: every time sample of every trial is one observation."""
    picked = X[:, list(electrodes)].astype(np.float64)
    return picked.transpose(0, 2, 1).reshape(-1, len(electrodes))


def fit(X_train: np.ndarray, kept: list[int], dropped: list[int]) -> Fit:
    A, B = _rows(X_train, kept), _rows(X_train, dropped)
    a_mean, b_mean = A.mean(axis=0), B.mean(axis=0)
    coef, *_ = np.linalg.lstsq(A - a_mean, B - b_mean, rcond=None)
    return Fit(tuple(int(c) for c in kept), tuple(int(c) for c in dropped), a_mean, b_mean, coef)


def predictable(X: np.ndarray, f: Fit) -> np.ndarray:
    """(trials, dropped, times) float64: the part of each dropped electrode the kept ones predict."""
    n, _, t = X.shape
    P = (_rows(X, f.kept) - f.kept_mean) @ f.coef + f.dropped_mean
    return P.reshape(n, t, len(f.dropped)).transpose(0, 2, 1)


def residual(X: np.ndarray, f: Fit) -> np.ndarray:
    """(trials, dropped, times) float64: the dropped electrodes minus their prediction."""
    return X[:, list(f.dropped)].astype(np.float64) - predictable(X, f)


def derangement(n: int, key: str) -> np.ndarray:
    """A permutation of range(n) with no fixed point, drawn uniformly by rejection from a generator seeded by
    the SHA-256 of `key` (the same key always gives the same permutation)."""
    if n < 2:
        raise ValueError(f"no derangement of {n} trial(s)")
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if not (perm == np.arange(n)).any():
            return perm


def derangement_key(dataset: str, subject: int, split: str, part: str, draw: int) -> str:
    """One permutation per participant, split, set of trials ('train', 'held', 'day2') and draw; every seed of a
    condition shares it."""
    return f"residual_surrogate|{dataset}|{subject}|{split}|{part}|{draw}"


def extra_inputs(X: np.ndarray, kind: str, f: Fit, perm: np.ndarray | None = None) -> np.ndarray:
    """(trials, dropped, times) float32 extra inputs of one kind; `perm` (a derangement of the trials of X) only
    for 'surr'."""
    if kind == "real":
        return X[:, list(f.dropped)].copy()
    P = predictable(X, f)
    if kind == "pred":
        return P.astype(np.float32)
    if kind != "surr":
        raise ValueError(f"unknown kind {kind}")
    if perm is None or len(perm) != len(X):
        raise ValueError("'surr' needs a permutation of the trials")
    R = X[:, list(f.dropped)].astype(np.float64) - P
    return (P + R[perm]).astype(np.float32)


def model_input(X: np.ndarray, f: Fit, kind: str, perm: np.ndarray | None = None) -> np.ndarray:
    """The network input: the kept electrodes, then the extra inputs at the dropped electrodes' positions."""
    return np.concatenate([X[:, list(f.kept)], extra_inputs(X, kind, f, perm)], axis=1)
