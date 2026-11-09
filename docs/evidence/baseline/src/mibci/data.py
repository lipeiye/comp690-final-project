"""BCI Competition IV 2a: download, causal preprocessing and a per-session cache.

Preprocessing follows Section 4 of the revised proposal: the 22 EEG channels at
250 Hz (EOG dropped, recorded reference kept), a causal 4-38 Hz band-pass that
restarts from zero state at every run, and one 2-s window per trial taken
0.5-2.5 s after cue onset.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfilt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Documents is synced by iCloud; the `.nosync` suffix keeps raw and cached data local.
DATA_DIR = PROJECT_ROOT / "data.nosync"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SFREQ = 250
EEG_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]  # fmt: skip
CLASSES = ["left_hand", "right_hand", "feet", "tongue"]
SUBJECTS = list(range(1, 10))
# Session 1 is the training day ("T" file); session 2 is the held-out test day ("E" file).
SESSION_FILES = {1: "T", 2: "E"}

BAND_HZ = (4.0, 38.0)
FILTER_ORDER = 4
CUE_ONSET_S = 2.0  # the cue appears 2 s after each trial-start marker
WINDOW_S = (0.5, 2.5)  # relative to cue onset
WIN_START = round((CUE_ONSET_S + WINDOW_S[0]) * SFREQ)
N_TIMES = round((WINDOW_S[1] - WINDOW_S[0]) * SFREQ)


def bandpass_sos() -> np.ndarray:
    return butter(FILTER_ORDER, BAND_HZ, btype="bandpass", fs=SFREQ, output="sos")


def download(subject: int) -> dict[int, Path]:
    """Fetch a subject's two .mat files through MOABB into RAW_DIR."""
    # Setting the path via the environment stops MOABB from writing ~/.mne config.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATASETS_BNCI_PATH"] = str(RAW_DIR)
    from moabb.datasets import BNCI2014_001

    paths = [Path(p) for p in BNCI2014_001().data_path(subject)]
    return {
        session: next(p for p in paths if p.stem.endswith(code))
        for session, code in SESSION_FILES.items()
    }


def preprocess_session(mat_path: Path) -> dict[str, np.ndarray]:
    """Filter each run causally and cut one window per trial.

    Returns X (trials x 22 x N_TIMES, microvolts), y (0-3), the run index (0-5),
    the expert artifact flag of each trial, and the earliest window start in
    seconds after a run begins (how long the filter had to settle).
    """
    runs = loadmat(mat_path, struct_as_record=False, squeeze_me=True)["data"]
    runs = [r for r in np.atleast_1d(runs) if np.size(r.trial) > 0]  # drop EOG calibration runs
    sos = bandpass_sos()
    X, y, run_idx, artifact, first_window = [], [], [], [], []
    for i, run in enumerate(runs):
        classes = [str(c).strip().lower().replace(" ", "_") for c in np.atleast_1d(run.classes)]
        if int(run.fs) != SFREQ or classes != CLASSES:
            raise ValueError(f"{mat_path.name} run {i}: fs={run.fs}, classes={classes}")
        eeg = np.asarray(run.X, dtype=np.float64)[:, : len(EEG_CHANNELS)]
        if np.isnan(eeg).any():
            raise ValueError(f"{mat_path.name} run {i}: {np.isnan(eeg).sum()} NaN samples")
        filtered = sosfilt(sos, eeg, axis=0)

        starts = np.atleast_1d(run.trial).astype(int) - 1 + WIN_START  # .mat indices are 1-based
        if starts.max() + N_TIMES > len(filtered):
            raise ValueError(f"{mat_path.name} run {i}: last window runs past the recording")
        windows = filtered[starts[:, None] + np.arange(N_TIMES)]  # trials x time x channels
        X.append(windows.transpose(0, 2, 1))
        y.append(np.atleast_1d(run.y).astype(int) - 1)
        run_idx.append(np.full(len(starts), i))
        artifact.append(np.atleast_1d(run.artifacts).astype(bool))
        first_window.append(starts.min() / SFREQ)

    return {
        "X": np.concatenate(X).astype(np.float32),
        "y": np.concatenate(y),
        "run": np.concatenate(run_idx),
        "artifact": np.concatenate(artifact),
        "first_window_s": np.float64(min(first_window)),
    }


def session_path(subject: int, session: int) -> Path:
    return PROCESSED_DIR / f"sub-{subject:02d}_ses-{session}.npz"


def load_session(subject: int, session: int, *, final_evaluation: bool = False) -> dict:
    """Load cached windows. Session 2 is the held-out test day and stays locked
    until the final evaluation, after every setting has been frozen."""
    if session == 2 and not final_evaluation:
        raise PermissionError("session 2 is held out; only the final evaluation may load it")
    with np.load(session_path(subject, session)) as f:
        return {k: f[k] for k in f.files}
