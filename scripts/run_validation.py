"""Validation phase on session 1: every subject, fold and condition.

--stage full     The 22-channel EEGNet for the maximum budget, scoring the held-out runs
                 after every epoch and keeping a checkpoint every 25 epochs. These runs set
                 the epoch budget (choose_budget.py) and supply the weight-norm ranking.
--stage subsets  Every channel-selection condition, trained for the frozen epoch budget.
                 The 22-channel model at that budget is taken from its checkpoint.

Finished runs are skipped, so an interrupted stage can simply be restarted.
"""

import argparse
import json

import torch

from mibci import data, selection
from mibci.artifacts import completed
from mibci.experiment import MODELS_DIR, RESULTS_DIR, device_for, run_pool, session1, train_run, write_json
from mibci.model import build_eegnet, load_bundle, save_bundle
from mibci.train import ChannelStandardizer, TrainConfig, predict, scores

VAL_DIR = RESULTS_DIR / "validation"
VAL_MODELS = MODELS_DIR / "validation"
SPLITS = ["fold-0", "fold-1", "fold-2"]
MAX_EPOCHS = 500
CHECKPOINT_EVERY = 25


def full_job(subject: int, split: str) -> None:
    name = f"sub-{subject:02d}_{split}"
    record = train_run(
        subject, split, selection.FULL, list(range(len(data.EEG_CHANNELS))), TrainConfig(epochs=MAX_EPOCHS),
        VAL_MODELS / f"{name}_full-22_max.pt", track_val=True,
        checkpoints=range(CHECKPOINT_EVERY, MAX_EPOCHS + 1, CHECKPOINT_EVERY),
        checkpoint_path=VAL_MODELS / f"{name}_full-22_checkpoints.pt",
    )
    write_json(VAL_DIR / "full" / f"{name}.json", record)


def subset_job(subject: int, split: str, condition: str, channels: list[int], epochs: int) -> None:
    name = f"sub-{subject:02d}_{split}_{condition}"
    record = train_run(subject, split, condition, channels, TrainConfig(epochs=epochs), VAL_MODELS / f"{name}.pt")
    write_json(VAL_DIR / "runs" / f"{name}.json", record)


def full_at_budget(subject: int, split: str, epochs: int):
    """The 22-channel model of this fold at the frozen budget, saved and scored like any run."""
    name = f"sub-{subject:02d}_{split}"
    X, y, tr, va = session1(subject, split)
    bundle_path = VAL_MODELS / f"{name}_{selection.FULL}.pt"
    record_path = VAL_DIR / "runs" / f"{name}_{selection.FULL}.json"
    if completed(record_path, [bundle_path], {"epochs": epochs, "subject": subject, "split": split}):
        return load_bundle(bundle_path)[0], X[tr], y[tr]
    scaler = ChannelStandardizer().fit(X[tr])
    model = build_eegnet(X.shape[1])
    model.load_state_dict(torch.load(VAL_MODELS / f"{name}_full-22_checkpoints.pt", weights_only=True)[epochs])
    model.eval()
    channels = list(range(X.shape[1]))
    save_bundle(VAL_MODELS / f"{name}_{selection.FULL}.pt", model, channels, scaler, subject=subject,
                split=split, condition=selection.FULL, seed=0, epochs=epochs, device=device_for(subject))
    full_record = json.loads((VAL_DIR / "full" / f"{name}.json").read_text())
    record = {k: v for k, v in full_record.items() if k not in ("history", "val")}
    record.update(epochs=epochs, train_time_s=full_record["train_time_s"] * epochs / MAX_EPOCHS,
                  val=scores(predict(model, scaler.transform(X[va])), y[va]))
    record["config"]["epochs"] = epochs
    write_json(VAL_DIR / "runs" / f"{name}_{selection.FULL}.json", record)
    return model, X[tr], y[tr]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["full", "subsets"], required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=data.SUBJECTS)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    jobs = []
    if args.stage == "full":
        for subject in args.subjects:
            for split in SPLITS:
                name = f"sub-{subject:02d}_{split}"
                if not completed(VAL_DIR / "full" / f"{name}.json",
                                 [VAL_MODELS / f"{name}_full-22_max.pt",
                                  VAL_MODELS / f"{name}_full-22_checkpoints.pt"],
                                 {"subject": subject, "split": split, "epochs": MAX_EPOCHS}):
                    jobs.append((name, dict(subject=subject, split=split)))
        run_pool(full_job, jobs, args.workers, args.threads)
        return

    epochs = json.loads((VAL_DIR / "budget.json").read_text())["epochs"]
    for subject in args.subjects:
        for split in SPLITS:
            name = f"sub-{subject:02d}_{split}"
            full_model, X_train, y_train = full_at_budget(subject, split, epochs)
            ranking_path = VAL_DIR / "rankings" / f"{name}.json"
            if ranking_path.exists():
                subsets = json.loads(ranking_path.read_text())["subsets"]
            else:
                ranking = selection.rank_channels(X_train, y_train, full_model)
                subsets = selection.condition_subsets(ranking["orders"])
                write_json(ranking_path, {**ranking, "subsets": subsets})
            for condition, channels in subsets.items():
                if condition != selection.FULL and not completed(
                        VAL_DIR / "runs" / f"{name}_{condition}.json",
                        [VAL_MODELS / f"{name}_{condition}.pt"],
                        {"subject": subject, "split": split, "condition": condition, "epochs": epochs}):
                    jobs.append((f"{name}_{condition}", dict(subject=subject, split=split, condition=condition,
                                                              channels=channels, epochs=epochs)))
    run_pool(subset_job, jobs, args.workers, args.threads)


if __name__ == "__main__":
    main()
