"""Pilot: full 22-channel FP32 EEGNet, 3-fold run-grouped validation on one subject.

Development evidence only: uses the artifact-free trials of session 1 and never
loads session 2. Writes a JSON record and a learning-curve plot to results/pilot/.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mibci import data, splits, train
from mibci.env import environment
from mibci.model import EEGNET_KWARGS, build_eegnet


def plot_curves(folds: list[dict], path: Path) -> None:
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for k, fold in enumerate(folds):
        h = fold["history"]
        epochs = np.arange(1, len(h["val_acc"]) + 1)
        (line,) = ax_acc.plot(epochs, h["val_acc"], lw=1, label=f"fold {k} (val runs {fold['val_runs']})")
        ax_acc.axvline(fold["lowest_val_loss_epoch"], color=line.get_color(), ls=":", lw=1)
        ax_loss.plot(epochs, h["val_loss"], lw=1, color=line.get_color(), label=f"fold {k} val")
        ax_loss.plot(epochs, h["train_loss"], lw=1, ls="--", alpha=0.6, color=line.get_color(), label=f"fold {k} train")
    ax_acc.axhline(0.25, color="grey", lw=0.8, ls="--", label="chance")
    ax_acc.set(xlabel="epoch", ylabel="validation accuracy", ylim=(0, 1),
               title="Validation accuracy (dotted: lowest validation loss)")
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy", title="Training and validation loss")
    ax_acc.legend(fontsize=8)
    ax_loss.legend(fontsize=8, ncol=2)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=data.PROJECT_ROOT / "results" / "pilot")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    ses = data.load_session(args.subject, 1)
    keep = ~ses["artifact"]  # primary analysis: artifact-free trials
    X, y, runs = ses["X"][keep], ses["y"][keep], ses["run"][keep]
    cfg = train.TrainConfig(epochs=args.epochs, seed=args.seed)

    folds = []
    for k, val_runs in enumerate(splits.VAL_RUNS):
        tr, va = splits.fold_masks(runs, k)
        scaler = train.ChannelStandardizer().fit(X[tr])
        X_va = scaler.transform(X[va])
        train.set_seed(cfg.seed)
        model = build_eegnet(n_chans=X.shape[1])
        res = train.fit(model, scaler.transform(X[tr]), y[tr], cfg, val=(X_va, y[va]))
        h = res["history"]
        lowest = int(np.argmin(h["val_loss"]) + 1)
        final = train.scores(train.predict(model, X_va), y[va])
        folds.append({
            "val_runs": list(val_runs), "n_train": int(tr.sum()), "n_val": int(va.sum()),
            "lowest_val_loss_epoch": lowest, "val_acc_at_lowest_val_loss": h["val_acc"][lowest - 1],
            "final": final, "train_time_s": res["train_time_s"], "history": h,
        })
        print(
            f"fold {k} (val runs {val_runs}): lowest val loss at epoch {lowest} "
            f"(val acc {h['val_acc'][lowest - 1]:.3f}), val acc {final['acc']:.3f} at the last epoch "
            f"({res['train_time_s']:.0f} s)"
        )

    mean_val_loss = np.mean([f["history"]["val_loss"] for f in folds], axis=0)
    summary = {
        "val_acc_at_lowest_val_loss": float(np.mean([f["val_acc_at_lowest_val_loss"] for f in folds])),
        "val_acc_at_last_epoch": float(np.mean([f["final"]["acc"] for f in folds])),
        "val_macro_f1_at_last_epoch": float(np.mean([f["final"]["macro_f1"] for f in folds])),
        "epoch_of_lowest_mean_val_loss": int(mean_val_loss.argmin() + 1),
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    record = {
        "subject": args.subject,
        "condition": "all 22 channels, FP32",
        "trials": "session 1, artifact-free",
        "n_flagged_excluded": int((~keep).sum()),
        "config": asdict(cfg),
        "eegnet": EEGNET_KWARGS,
        "environment": environment(),
        "summary": summary,
        "folds": folds,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"sub-{args.subject:02d}_full22_fp32_seed{args.seed}"
    (args.out / f"{stem}.json").write_text(json.dumps(record, indent=1))
    plot_curves(folds, args.out / f"{stem}_curves.png")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
