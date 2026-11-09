"""Final models: every condition trained on all artifact-free session-1 trials.

Three seeds per subject and condition, for the frozen epoch budget. Channel rankings
are refit on all of session 1; each seed's weight-norm ranking comes from that seed's
22-channel model. Session 2 is not touched here. Requires results/frozen_config.json.

--stage full     the 22-channel model of every subject and seed
--stage subsets  every other condition
"""

import argparse
import json

from mibci import selection
from mibci.experiment import MODELS_DIR, RESULTS_DIR, run_pool, session1, train_run, write_json
from mibci.model import load_bundle
from mibci.train import TrainConfig

FINAL_DIR = RESULTS_DIR / "final"
FINAL_MODELS = MODELS_DIR / "final"


def frozen_config() -> dict:
    path = RESULTS_DIR / "frozen_config.json"
    if not path.exists():
        raise SystemExit("Freeze the analysis plan first: scripts/summarize_validation.py --freeze")
    return json.loads(path.read_text())


def final_job(subject: int, seed: int, condition: str, channels: list[int], epochs: int) -> None:
    name = f"sub-{subject:02d}_seed-{seed}_{condition}"
    record = train_run(subject, "final", condition, channels, TrainConfig(epochs=epochs, seed=seed),
                       FINAL_MODELS / f"{name}.pt")
    write_json(FINAL_DIR / "runs" / f"{name}.json", record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["full", "subsets"], required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    config = frozen_config()

    jobs = []
    for subject in args.subjects:
        if args.stage == "subsets":
            X, y, train_mask, _ = session1(subject, "final")
        for seed in config["seeds"]:
            prefix = f"sub-{subject:02d}_seed-{seed}"
            if args.stage == "full":
                subsets = {selection.FULL: list(range(22))}
            else:
                full_model = load_bundle(FINAL_MODELS / f"{prefix}_{selection.FULL}.pt")[0]
                ranking = selection.rank_channels(X[train_mask], y[train_mask], full_model)
                subsets = selection.condition_subsets(ranking["orders"])
                write_json(FINAL_DIR / "rankings" / f"{prefix}.json", {**ranking, "subsets": subsets})
                subsets.pop(selection.FULL)
            for condition, channels in subsets.items():
                if not (FINAL_DIR / "runs" / f"{prefix}_{condition}.json").exists():
                    jobs.append((f"{prefix}_{condition}", dict(subject=subject, seed=seed, condition=condition,
                                                               channels=channels, epochs=config["epochs"])))
    run_pool(final_job, jobs, args.workers, threads=1)


if __name__ == "__main__":
    main()
