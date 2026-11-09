"""Structured pruning of the 22-channel model and one validation-selected reduced model.

Each base model (a validation fold, or a final seed) loses 25% or 50% of the output
filters of its final pointwise convolution by L1 norm, and is then fine-tuned for a
fixed 100 epochs at lr 1e-4 on its own training partition. The channel-reduced base
is the condition named in results/validation/pruning_base.json.
"""

import argparse
import json
from pathlib import Path

from mibci.experiment import MODELS_DIR, RESULTS_DIR, device_for, run_pool, session1, write_json
from mibci.model import load_bundle, save_bundle
from mibci.prune import FINETUNE_EPOCHS, FINETUNE_LR, FRACTIONS, prune_pointwise
from mibci.selection import FULL
from mibci.train import TrainConfig, fit, predict, scores, set_seed


def prune_job(base_path: str, out_path: str, record_path: str, fraction: float) -> None:
    model, scaler, b = load_bundle(Path(base_path))
    X, y, tr, va = session1(b["subject"], b["split"])
    X = X[:, b["channels"]]
    X_tr = scaler.transform(X[tr])
    X_va = scaler.transform(X[va]) if va is not None else None

    pruned = prune_pointwise(model, fraction)
    before = scores(predict(pruned, X_va), y[va]) if va is not None else None
    cfg = TrainConfig(epochs=FINETUNE_EPOCHS, lr=FINETUNE_LR, seed=b["seed"])
    set_seed(cfg.seed)
    device = device_for(b["subject"])
    res = fit(pruned, X_tr, y[tr], cfg, device=device)

    condition = f"{b['condition']}-prune{round(100 * fraction)}"
    meta = dict(subject=b["subject"], split=b["split"], condition=condition, seed=b["seed"],
                epochs=b["epochs"], device=device)
    save_bundle(Path(out_path), pruned, b["channels"], scaler, **meta)
    record = {**meta, "base": b["condition"], "fraction": fraction, "kept_filters": pruned.F2,
              "finetune": {"epochs": cfg.epochs, "lr": cfg.lr}, "n_train": int(tr.sum()),
              "train_time_s": res["train_time_s"]}
    if va is not None:
        record.update(n_val=int(va.sum()), val_before_finetune=before, val=scores(predict(pruned, X_va), y[va]))
    write_json(Path(record_path), record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["validation", "final"], required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    reduced = json.loads((RESULTS_DIR / "validation" / "pruning_base.json").read_text())["condition"]
    jobs = []
    for subject in args.subjects:
        for base in (FULL, reduced):
            for base_path in sorted((MODELS_DIR / args.phase).glob(f"sub-{subject:02d}_*_{base}.pt")):
                for fraction in FRACTIONS:
                    name = f"{base_path.stem}-prune{round(100 * fraction)}"
                    record = RESULTS_DIR / args.phase / "runs" / f"{name}.json"
                    if not record.exists():
                        jobs.append((name, dict(base_path=str(base_path), fraction=fraction, record_path=str(record),
                                                out_path=str(base_path.with_name(f"{name}.pt")))))
    run_pool(prune_job, jobs, args.workers, threads=1)


if __name__ == "__main__":
    main()
