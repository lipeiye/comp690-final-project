"""One training run on session-1 data, and a process pool for running many of them."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from mibci import data, splits
from mibci.artifacts import write_json
from mibci.model import build_eegnet, save_bundle
from mibci.train import ChannelStandardizer, TrainConfig, fit, predict, scores, set_seed

RESULTS_DIR = Path(os.environ.get("MIBCI_RESULTS_DIR", data.PROJECT_ROOT / "results")).expanduser().resolve()
MODELS_DIR = data.DATA_DIR / "models"


def device_for(subject: int) -> str:
    """Preserve the original per-subject MPS/CPU allocation unless explicitly overridden.

    New runs may request cpu, mps or cuda with MIBCI_DEVICE. Backend changes must be
    recorded as independent reproductions; bitwise equality across devices is not promised.
    """
    requested = os.environ.get("MIBCI_DEVICE", "legacy")
    if requested not in {"legacy", "cpu", "mps", "cuda"}:
        raise ValueError("MIBCI_DEVICE must be legacy, cpu, mps or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("MIBCI_DEVICE=cuda requested, but CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MIBCI_DEVICE=mps requested, but MPS is unavailable")
    if requested != "legacy":
        return requested
    return "mps" if subject <= 6 and torch.backends.mps.is_available() else "cpu"


def session1(subject: int, split: str):
    """Artifact-free session-1 trials with (train, val) masks for `split`:
    'fold-0'..'fold-2', or 'final' to train on all of session 1 (val mask is None)."""
    ses = data.load_session(subject, 1)
    keep = ~ses["artifact"]
    X, y, runs = ses["X"][keep], ses["y"][keep], ses["run"][keep]
    if split == "final":
        return X, y, np.ones(len(y), bool), None
    train_mask, val_mask = splits.fold_masks(runs, int(split.removeprefix("fold-")))
    return X, y, train_mask, val_mask


def train_run(subject: int, split: str, condition: str, channels: list[int], cfg: TrainConfig,
              model_path: Path, track_val: bool = False, checkpoints=(), checkpoint_path: Path | None = None) -> dict:
    """Train EEGNet on one channel subset, save the model bundle, and score the validation runs."""
    X, y, tr, va = session1(subject, split)
    X = X[:, channels]
    scaler = ChannelStandardizer().fit(X[tr])
    X_val = scaler.transform(X[va]) if va is not None else None
    set_seed(cfg.seed)
    model = build_eegnet(len(channels))
    device = device_for(subject)
    res = fit(model, scaler.transform(X[tr]), y[tr], cfg,
              val=(X_val, y[va]) if track_val else None, checkpoints=checkpoints, device=device)

    meta = dict(subject=subject, split=split, condition=condition, seed=cfg.seed, epochs=cfg.epochs,
                device=device)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_bundle(model_path, model, channels, scaler, **meta)
    if checkpoint_path is not None:
        torch.save(res["checkpoints"], checkpoint_path)
    record = {
        **meta,
        "channels": [data.EEG_CHANNELS[c] for c in channels],
        "config": asdict(cfg),
        "n_train": int(tr.sum()),
        "train_time_s": res["train_time_s"],
    }
    if va is not None:
        record["n_val"] = int(va.sum())
        record["val"] = scores(predict(model, X_val), y[va])
    if track_val:
        record["history"] = res["history"]
    return record


def run_pool(fn, jobs: list[tuple[str, dict]], workers: int, threads: int) -> None:
    """Run fn(**kwargs) for every (name, kwargs) job across `workers` processes,
    each limited to `threads` intra-op threads."""
    if workers < 1 or threads < 1:
        raise ValueError("workers and threads must be positive")
    if not jobs:
        print("No pending jobs (all required artifacts checked).", flush=True)
        return
    start = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=torch.set_num_threads, initargs=(threads,)) as pool:
        futures = {pool.submit(fn, **kwargs): name for name, kwargs in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            elapsed = time.perf_counter() - start
            left = elapsed / i * (len(jobs) - i)
            print(f"[{i}/{len(jobs)}] {futures[future]} done, {elapsed / 60:.1f} min elapsed, "
                  f"~{left / 60:.0f} min left", flush=True)
