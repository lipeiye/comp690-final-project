"""Lee et al. (2019) OpenBMI motor imagery: the second two-day dataset, for a prospective replication.

54 participants, two sessions on different days, 62 EEG electrodes at 1000 Hz, left- versus
right-hand imagery: per session an offline run of 100 trials and an online run of 100
trials. Files come through MOABB (Lee2019_MI) from the GigaDB mirror, about 600 MB each.

Preprocessing mirrors mibci.data for BCI IV 2a so that the same EEGNet input results: a
causal 4th-order Butterworth band-pass 4-38 Hz on each run from zero state, then every
fourth sample (250 Hz; the band-pass is the anti-aliasing filter), then one 2-s window per
trial 0.5-2.5 s after the cue. There are no expert artifact flags in this dataset, so every
trial counts as clean. Session 2 is the held-out test day and stays locked as in mibci.data.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfilt

from mibci.paths import DATA_DIR

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed" / "lee2019"
SUBJECTS = list(range(1, 55))
CLASSES = ["left_hand", "right_hand"]
RAW_SFREQ, SFREQ = 1000, 250
DECIMATE = RAW_SFREQ // SFREQ
BAND_HZ = (4.0, 38.0)
WINDOW_S = (0.5, 2.5)
N_TIMES = round((WINDOW_S[1] - WINDOW_S[0]) * SFREQ)
RUNS = ("EEG_MI_train", "EEG_MI_test")  # offline run, then online run, in every session


def download(subject: int) -> dict[int, Path]:
    """Both session files of one participant into RAW_DIR (MOABB names the folder)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATASETS_LEE2019-MI_PATH"] = str(RAW_DIR)  # noqa: TID251  MOABB reads its path from here
    from moabb.datasets import Lee2019_MI

    paths = [Path(p) for p in Lee2019_MI().data_path(subject)]
    return {1: paths[0], 2: paths[1]}


def _run_windows(run, sos: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if int(np.squeeze(run["fs"])) != RAW_SFREQ:
        raise ValueError(f"expected {RAW_SFREQ} Hz, found {np.squeeze(run['fs'])}")
    channels = [str(np.squeeze(c)) for c in np.ravel(run["chan"])]
    eeg = np.asarray(run["x"], dtype=np.float64)  # samples x channels, microvolts
    filtered = sosfilt(sos, eeg, axis=0)[::DECIMATE]
    onsets = np.ravel(run["t"]).astype(int) // DECIMATE  # cue samples at 250 Hz
    starts = onsets + round(WINDOW_S[0] * SFREQ)
    if starts.max() + N_TIMES > len(filtered):
        raise ValueError("last window runs past the recording")
    windows = filtered[starts[:, None] + np.arange(N_TIMES)].transpose(0, 2, 1)
    labels = np.ravel(run["y_dec"]).astype(int)  # file codes: 1 right hand, 2 left hand
    y = np.where(labels == 1, CLASSES.index("right_hand"), CLASSES.index("left_hand"))
    return windows.astype(np.float32), y, channels


def preprocess_session(mat_path: Path) -> dict[str, np.ndarray]:
    mat = loadmat(mat_path, squeeze_me=False)
    sos = butter(4, BAND_HZ, btype="bandpass", fs=RAW_SFREQ, output="sos")
    X, y, run_index, names = [], [], [], None
    for i, key in enumerate(RUNS):
        windows, labels, channels = _run_windows(mat[key][0, 0], sos)
        if names is not None and channels != names:
            raise ValueError("channel order differs between runs")
        names = channels
        X.append(windows)
        y.append(labels)
        run_index.append(np.full(len(labels), i))
    y_all = np.concatenate(y)
    return {
        "X": np.concatenate(X),
        "y": y_all,
        "run": np.concatenate(run_index),
        "artifact": np.zeros(len(y_all), bool),
        "channels": np.array(names),
    }


def session_path(subject: int, session: int) -> Path:
    return PROCESSED_DIR / f"sub-{subject:02d}_ses-{session}.npz"


def load_session(subject: int, session: int, *, final_evaluation: bool = False) -> dict:
    """Cached windows; session 2 is the held-out day and opens only for the final evaluation."""
    if session == 2 and not final_evaluation:
        raise PermissionError("session 2 is held out; only the final evaluation may load it")
    with np.load(session_path(subject, session)) as f:  # noqa: TID251  this module is the loader
        return {k: f[k] for k in f.files}
