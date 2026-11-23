"""Extra inputs that differ in how much independent fluctuation they bring (scripts/input_kind_study.py).

scripts/noise_mechanism_study.py offered an untested explanation of why an input that carries nothing
costs EEGNet accuracy: EEG electrodes are highly correlated, so a spatial filter can cancel what they
share, while noise inputs are independent of each other and of the electrodes, so their variance goes
straight into every filter's output. If that is right, the cost comes from fluctuation unrelated to the
other inputs, not from the number of inputs. The extra inputs built here separate the two:

dup      exact copies of kept electrodes (the kept electrodes in relevance order, from the most relevant,
         cycled): more inputs, but no new information and no new independent fluctuation
mixA     sqrt(1 - a) * the copied electrode, standardized with the training partition's mean and standard
         deviation, plus sqrt(a) * the input-cost study's noise input at the same position (a = A / 100).
         At a = 1 the extra input is that study's noise input exactly (tests/test_input_kind.py)
shared   every extra input is the same noise waveform, the first input of the input-cost draw: k inputs,
         one independent fluctuation

`unexplained_fraction` asks the same question of the real dropped electrodes: the share of a dropped
electrode's variance on held-out runs that a least-squares fit from the kept electrodes, fitted on the
training runs, cannot predict. A copy has 0, a mix about a, independent noise about 1.
"""

from __future__ import annotations

import numpy as np

KINDS = ("dup", "mix25", "mix50", "mix75", "shared")


def mix_fraction(kind: str) -> float | None:
    """a of a 'mixA' kind (mix50 -> 0.5); None for the other kinds."""
    return int(kind.removeprefix("mix")) / 100 if kind.startswith("mix") else None


def copied(order: list[int], added: int) -> list[int]:
    """The electrode behind each extra input: the kept electrodes in relevance order, cycled."""
    return [int(order[i % len(order)]) for i in range(added)]


def source_stats(X_train: np.ndarray, sources: list[int]) -> tuple[list[float], list[float]]:
    """Per extra input, the mean and standard deviation of its electrode over the training partition,
    reduced as mibci.train.ChannelStandardizer reduces (over trials and time, in the data's dtype)."""
    picked = X_train[:, sources]
    return picked.mean(axis=(0, 2)).tolist(), picked.std(axis=(0, 2)).tolist()


def extra_inputs(
    X: np.ndarray, kind: str, sources: list[int], noise: np.ndarray, mean: list[float], std: list[float]
) -> np.ndarray:
    """(trials, len(sources), times) extra inputs of one kind.

    noise is the input-cost study's draw for these trials and this day, (trials, len(sources), times);
    mean and std are source_stats of the training partition (the same numbers on both days)."""
    if noise.shape[1] != len(sources):
        raise ValueError(f"need {len(sources)} noise inputs, got {noise.shape[1]}")
    if kind == "dup":
        return X[:, sources].copy()
    if kind == "shared":
        return np.repeat(noise[:, :1], len(sources), axis=1)
    a = mix_fraction(kind)
    if a is None or not 0.0 <= a <= 1.0:
        raise ValueError(f"unknown kind {kind}")
    m = np.asarray(mean, np.float32)[None, :, None]
    s = np.asarray(std, np.float32)[None, :, None]
    z = (X[:, sources] - m) / s
    return (np.sqrt(1.0 - a) * z.astype(np.float64) + np.sqrt(a) * noise.astype(np.float64)).astype(np.float32)


def model_input(
    X: np.ndarray,
    channels: list[int],
    kind: str,
    sources: list[int],
    noise: np.ndarray,
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    """The network input: the kept electrodes (in the input-cost study's order), then the extra inputs."""
    return np.concatenate([X[:, channels], extra_inputs(X, kind, sources, noise, mean, std)], axis=1)


def unexplained_fraction(
    X: np.ndarray, kept: list[int], dropped: list[int], train: np.ndarray, held: np.ndarray
) -> np.ndarray:
    """Per dropped electrode: the variance on the held-out trials of what a least-squares prediction from
    the kept electrodes leaves, over the electrode's own held-out variance (1 - R^2 out of sample).

    Every time sample of every trial is one observation; the fit (with intercept) uses the training
    trials only, so the held-out runs of the training day measure it."""

    def rows(mask: np.ndarray, electrodes: list[int]) -> np.ndarray:
        picked = X[mask][:, electrodes].astype(np.float64)
        return picked.transpose(0, 2, 1).reshape(-1, len(electrodes))

    A, B = rows(train, kept), rows(train, dropped)
    a_mean, b_mean = A.mean(axis=0), B.mean(axis=0)
    coef, *_ = np.linalg.lstsq(A - a_mean, B - b_mean, rcond=None)
    A_held, B_held = rows(held, kept), rows(held, dropped)
    residual = (B_held - b_mean) - (A_held - a_mean) @ coef
    return residual.var(axis=0) / B_held.var(axis=0)
