"""EEGNet training and scoring."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn


@dataclass
class TrainConfig:
    epochs: int = 500
    batch_size: int = 64
    lr: float = 1e-3  # Adam defaults, as in the EEGNet paper
    seed: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ChannelStandardizer:
    """Per-channel z-scoring; statistics come from the training partition only."""

    def fit(self, X: np.ndarray) -> ChannelStandardizer:
        self.mean = X.mean(axis=(0, 2), keepdims=True)
        self.std = X.std(axis=(0, 2), keepdims=True)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) / self.std).astype(np.float32)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Class logits in eval mode, computed on whatever device the model is on."""
    model.eval()
    device = next(model.parameters()).device
    batches = [model(torch.from_numpy(X[i : i + batch_size]).to(device)) for i in range(0, len(X), batch_size)]
    return torch.cat(batches).cpu().numpy()


def scores(logits: np.ndarray, y: np.ndarray) -> dict:
    pred = logits.argmax(1)
    cm = confusion_matrix(y, pred, labels=range(logits.shape[1]))
    return {
        "loss": float(nn.functional.cross_entropy(torch.from_numpy(logits), torch.from_numpy(y))),
        "acc": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "recall": (cm.diagonal() / cm.sum(1)).tolist(),
        "confusion": cm.tolist(),
    }


def fit(model, X_train, y_train, cfg: TrainConfig, val=None, checkpoints=(), device: str = "cpu") -> dict:
    """Train for a fixed budget of cfg.epochs (no early stopping).

    With val=(X_val, y_val), validation loss and accuracy are recorded after every
    epoch. Copies of the weights are kept at the epochs listed in `checkpoints`.
    The model is moved back to the CPU when training ends.
    """
    gen = torch.Generator().manual_seed(cfg.seed)  # batch order is drawn on the CPU for every device
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    X_t, y_t = torch.from_numpy(X_train).to(device), torch.from_numpy(y_train).to(device)
    if val is not None:
        X_v, y_v = torch.from_numpy(val[0]).to(device), torch.from_numpy(val[1]).to(device)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    saved = {}
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(len(X_t), generator=gen).to(device)
        total = torch.zeros((), device=device)
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i : i + cfg.batch_size]
            loss = nn.functional.cross_entropy(model(X_t[idx]), y_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach() * len(idx)
        history["train_loss"].append(total.item() / len(order))
        if val is not None:
            model.eval()
            with torch.no_grad():
                logits = model(X_v)
            history["val_loss"].append(nn.functional.cross_entropy(logits, y_v).item())
            history["val_acc"].append((logits.argmax(1) == y_v).float().mean().item())
        if epoch in checkpoints:
            saved[epoch] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.to("cpu")
    return {"history": history, "checkpoints": saved, "train_time_s": time.perf_counter() - start}
