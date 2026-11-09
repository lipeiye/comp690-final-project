"""Fix the training budget (epochs) from the 22-channel validation curves.

Rule, fixed before any channel-selection result exists: average validation accuracy
over all subjects and folds, smooth it with a centred 25-epoch moving average, and
take the first checkpoint epoch (a multiple of 25) at which the smoothed curve is
within 0.5 percentage points of its maximum. Writes results/validation/budget.json
and a plot of the curves.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mibci.experiment import write_json
from run_validation import CHECKPOINT_EVERY, VAL_DIR

SMOOTH_EPOCHS = 25
TOLERANCE = 0.005


def main() -> None:
    records = [json.loads(p.read_text()) for p in sorted((VAL_DIR / "full").glob("sub-*.json"))]
    curves = np.array([r["history"]["val_acc"] for r in records])
    mean = curves.mean(0)
    smooth = pd.Series(mean).rolling(SMOOTH_EPOCHS, center=True, min_periods=1).mean().to_numpy()
    epochs = np.arange(1, len(mean) + 1)
    candidates = epochs[(epochs % CHECKPOINT_EVERY == 0) & (smooth >= smooth.max() - TOLERANCE)]
    budget = int(candidates[0])

    write_json(VAL_DIR / "budget.json", {
        "epochs": budget,
        "rule": __doc__.split("\n\n")[1].replace("\n", " "),
        "n_runs": len(records),
        "smoothed_max_val_acc": float(smooth.max()),
        "smoothed_max_epoch": int(epochs[smooth.argmax()]),
        "smoothed_val_acc_at_budget": float(smooth[budget - 1]),
    })

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    by_subject = {}
    for r, curve in zip(records, curves):
        by_subject.setdefault(r["subject"], []).append(curve)
    for subject, cs in by_subject.items():
        ax.plot(epochs, np.mean(cs, 0), lw=0.7, alpha=0.5, label=f"S{subject}")
    ax.plot(epochs, smooth, color="black", lw=2, label="mean (smoothed)")
    ax.axvline(budget, color="black", ls="--", lw=1)
    ax.text(budget, 0.27, f" budget = {budget} epochs", fontsize=9)
    ax.axhline(0.25, color="grey", lw=0.8, ls=":")
    ax.set(xlabel="epoch", ylabel="validation accuracy (session 1)", ylim=(0.2, 1),
           title="22-channel EEGNet, 3-fold run-grouped validation")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(VAL_DIR / "budget.png", dpi=130)
    print(f"budget: {budget} epochs (smoothed mean val acc {smooth[budget - 1]:.3f}, max {smooth.max():.3f})")


if __name__ == "__main__":
    main()
