"""Validation folds for model selection, built from session-1 runs only."""

import numpy as np

# Three folds, each holding out two contiguous runs of session 1. Whole runs stay on
# one side of the split, so validation sees the drift within a recording day.
VAL_RUNS = ((0, 1), (2, 3), (4, 5))


def fold_masks(runs: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean (train, val) masks over trials for one fold."""
    val = np.isin(runs, VAL_RUNS[fold])
    return ~val, val
