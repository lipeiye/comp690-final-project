"""Participant-level statistics shared by every analysis.

The frozen plan (`results/frozen_config.json`, key "uncertainty") fixes the interval: a
percentile bootstrap over participants, 10,000 resamples, 95 %, seed 690. Each analysis
first averages seeds within a participant, so the resampled unit is always a person.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

FROZEN_BOOTSTRAP = {"resamples": 10000, "seed": 690, "level": 0.95}


def bootstrap_ci(diffs: np.ndarray, plan: Mapping = FROZEN_BOOTSTRAP) -> tuple[float, float]:
    """Percentile interval of the mean of `diffs` (one value per participant)."""
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(plan["seed"])
    means = diffs[rng.integers(0, len(diffs), size=(plan["resamples"], len(diffs)))].mean(1)
    tail = (1 - plan["level"]) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))
