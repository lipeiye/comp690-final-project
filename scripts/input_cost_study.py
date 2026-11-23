"""Post-test study: how much does one extra input cost EEGNet, and does the cost shrink with more data?

scripts/overfit_study.py found that the electrodes dropped from the full montage carry real
information (+4.6 pp on BCI IV 2a against noise at the same positions), and that adding the same
number of inputs that carry nothing costs about as much (-5.0 pp), so the two cancel and 16
electrodes do as well as 22. This study measures that cost as a dose-response curve: relevance-16
plus k inputs of band-limited noise (mibci.overfit.noise_channels; nested, so k = 4 holds the first
four of k = 16), trained on all or part of the training day:

  BCI IV 2a  k in 1, 2, 4, 6, 10, 16; final (6 runs, seeds 0-9), fold-k (4 runs) and rfold-k (2 runs),
             seeds 0-4 on each of the three folds
  Lee2019    k in 4, 8, 17, 32, 46 (46 makes 62 inputs); final (2 runs) and fold-k (1 run), seeds 0-2

k = 0 comes from the earlier studies at the same splits and seeds (2a: scripts/overfit_study.py's
relevance-16; Lee2019: scripts/electrode_data_study.py's relevance-16). Electrodes are the relevance
order of each training partition, as recorded in the electrode-data plan. If the cost comes from the
network fitting its few training trials, it should grow with k and be larger when there are fewer
training trials (PREDICTIONS, fixed for Lee2019 before any of these models was scored; BCI IV 2a
exploratory).

Stages: plan, train, evaluate. Tables: scripts/analyze_input_cost_study.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mibci import data, lee2019, overfit, replication
from mibci.artifacts import require_files, sha256, write_json
from mibci.experiment import device_for, run_pool
from mibci.layout import frozen_config
from mibci.model import build_eegnet
from mibci.paths import RESULTS_DIR
from mibci.study import Study, shard
from mibci.train import ChannelStandardizer, TrainConfig, fit, predict, scores, set_seed

STUDY = Study(
    "input_cost_study",
    "scripts/input_cost_study.py",
    sources=[
        "src/mibci/overfit.py",
        "src/mibci/lee2019.py",
        "src/mibci/replication.py",
        "src/mibci/splits.py",
        "src/mibci/study.py",
    ],
)
ORDERS = RESULTS_DIR / "supplementary" / "electrode_data_study" / "protocol.json"
KEPT = 16
DATASETS = {
    "bnci2a": {
        "added": (1, 2, 4, 6, 10, 16),
        "classes": 4,
        "final_seeds": tuple(range(10)),
        "fold_seeds": tuple(range(5)),
        "splits": ("fold-0", "fold-1", "fold-2", "rfold-0", "rfold-1", "rfold-2"),
    },
    "lee2019": {
        "added": (4, 8, 17, 32, 46),
        "classes": 2,
        "final_seeds": (0, 1, 2),
        "fold_seeds": (0, 1, 2),
        "splits": ("fold-0", "fold-1"),
    },
}
PREDICTIONS = {
    "C1": {
        "claim": "The cost grows with the number of inputs that carry nothing",
        "contrast": "Lee2019, final models: accuracy with 8 added inputs minus accuracy with 46 added",
        "test": "interval lower bound above 0",
    },
    "C2": {
        "claim": "The cost is larger with fewer training trials",
        "contrast": "Lee2019: (17 added minus none) of fold models (one run) minus (17 added minus none) of final "
        "models (both runs); none = relevance-16 of scripts/electrode_data_study.py",
        "test": "interval upper bound below 0",
    },
}


def day1(dataset: str, subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dataset == "bnci2a":
        ses = data.load_session(subject, 1)
        keep = ~ses["artifact"]
        return ses["X"][keep], ses["y"][keep].astype(np.int64), ses["run"][keep]
    ses = lee2019.load_session(subject, 1)
    return ses["X"], ses["y"].astype(np.int64), ses["run"]


def with_noise(X: np.ndarray, job: dict, day: int) -> np.ndarray:
    """The kept electrodes followed by the first `added` of DATASETS' largest number of noise inputs."""
    largest = max(DATASETS[job["dataset"]]["added"])
    noise = overfit.noise_channels(
        np.zeros((len(X), largest, X.shape[2]), np.float32), list(range(largest)), job["dataset"], job["subject"], day
    )
    return np.concatenate([X[:, job["channels"]], noise[:, : job["added"]]], axis=1)


def model_inputs(job: dict) -> list[int]:
    """What each network input is: an electrode index, or -1, -2, ... for the added noise inputs (so the
    saved bundle has one entry per input and mibci.replication.load rebuilds the right network)."""
    return [*job["channels"], *(-(i + 1) for i in range(job["added"]))]


def plan_jobs(orders: dict) -> list[dict]:
    jobs = []
    for key, by_split in orders.items():
        dataset, subject = key.rsplit("_", 1)
        spec = DATASETS[dataset]
        for split in ("final", *spec["splits"]):
            seeds = spec["final_seeds"] if split == "final" else spec["fold_seeds"]
            channels = sorted(by_split[split][:KEPT])
            jobs.extend(
                {
                    "name": f"{dataset}_sub-{int(subject):02d}_seed-{seed}_relevance-16-plus-{k}_{split}",
                    "dataset": dataset,
                    "subject": int(subject),
                    "seed": seed,
                    "split": split,
                    "added": k,
                    "condition": f"relevance-16-plus-{k}",
                    "channels": channels,
                }
                for k in spec["added"]
                for seed in seeds
            )
    return jobs


def create_plan() -> None:
    orders = json.loads(ORDERS.read_text())["orders"]
    plan = STUDY.create(
        plan_jobs(orders),
        purpose="Measure how much each extra input that carries nothing costs EEGNet, as a function of the number of "
        "such inputs and of the amount of training data.",
        claim_boundary="Exploratory for BCI IV 2a, planned after its overfitting-study results were known. For Lee2019 "
        "the predictions were fixed before any of these models was scored on Lee2019 session 2. Two predictions, no "
        "correction for multiplicity. It explains a result; it does not change the frozen 2a configuration.",
        lee2019_new_models_scored_before_plan=False,
        predictions=PREDICTIONS,
        analysis="seeds (and folds) averaged within each participant; 95% participant bootstrap (10,000, seed 690); "
        "BCI IV 2a on artifact-free session-2 trials, Lee2019 on all",
        datasets={
            name: {key: list(v) if isinstance(v, tuple) else v for key, v in spec.items()}
            for name, spec in DATASETS.items()
        },
        kept=KEPT,
        noise="mibci.overfit.noise_channels drawn for the largest k and cut to k, so smaller k are nested",
        baseline="k = 0: relevance-16 of scripts/overfit_study.py (2a) and scripts/electrode_data_study.py (Lee2019)",
        epochs=frozen_config()["epochs"],
        orders_source_sha256=sha256(ORDERS),
    )
    counts: dict[str, int] = {}
    for job in plan["jobs"]:
        key = f"{job['dataset']} {job['split'].split('-')[0]}"
        counts[key] = counts.get(key, 0) + 1
    print(f"{len(plan['jobs'])} jobs: {counts}")


def train_job(job: dict, epochs: int) -> None:
    X, y, runs = day1(job["dataset"], job["subject"])
    train, held = overfit.split_masks(runs, job["split"], job["dataset"])
    inputs = with_noise(X, job, day=1)
    scaler = ChannelStandardizer().fit(inputs[train])
    cfg = TrainConfig(epochs=epochs, seed=job["seed"])
    set_seed(cfg.seed)
    model = build_eegnet(inputs.shape[1], n_classes=DATASETS[job["dataset"]]["classes"])
    device = device_for(job["subject"])
    X_fit = scaler.transform(inputs[train])
    result = fit(model, X_fit, y[train], cfg, device=device)
    path = STUDY.model(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "dataset": job["dataset"],
        "subject": job["subject"],
        "seed": job["seed"],
        "split": job["split"],
        "condition": job["condition"],
        "added": job["added"],
        "epochs": epochs,
        "device": device,
    }
    replication.save(path, model, model_inputs(job), scaler, **meta)
    record = {
        **meta,
        "name": job["name"],
        "n_train": int(train.sum()),
        "train_time_s": result["train_time_s"],
        "train_acc": scores(predict(model, X_fit), y[train])["acc"],
        "model_sha256": sha256(path),
        "protocol_sha256": sha256(STUDY.protocol_path),
    }
    if held is not None:
        logits = predict(model, scaler.transform(inputs[held]))
        record |= {"n_held": int(held.sum()), "within_day": scores(logits, y[held])}
    write_json(STUDY.record("runs", job), record)


def evaluate_job(job: dict) -> None:
    torch.set_num_threads(1)
    path = STUDY.model(job)
    model, scaler, bundle = replication.load(path)
    if bundle["channels"] != model_inputs(job):
        raise RuntimeError(f"Bundle channels differ from the plan: {path}")
    if job["dataset"] == "bnci2a":
        ses = data.load_session(job["subject"], 2, final_evaluation=True)
        clean = ~ses["artifact"]
    else:
        ses = lee2019.load_session(job["subject"], 2, final_evaluation=True)
        clean = np.ones(len(ses["y"]), bool)
    y = ses["y"].astype(np.int64)
    logits = predict(model, scaler.transform(with_noise(ses["X"], job, day=2)))
    write_json(
        STUDY.record("test", job),
        {
            **job,
            "model_sha256": sha256(path),
            "protocol_sha256": sha256(STUDY.protocol_path),
            "primary": scores(logits[clean], y[clean]),
            "secondary": scores(logits, y),
            "pred": logits.argmax(1).tolist(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "train", "evaluate"], required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--shard", help="i/n: run only array task i of n (0-based)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "plan":
        create_plan()
        return
    plan = STUDY.load()
    jobs = [j for j in shard(plan["jobs"], args.shard) if j["dataset"] in args.datasets]
    if args.stage == "train":
        STUDY.log_environment(f"train-{(args.shard or 'all').replace('/', 'of')}")
        pending = STUDY.pending(
            jobs,
            "runs",
            lambda j: [STUDY.model(j)],
            lambda j: {"subject": j["subject"], "seed": j["seed"], "epochs": plan["epochs"]},
        )
        run_pool(train_job, [(j["name"], {"job": j, "epochs": plan["epochs"]}) for j in pending], args.workers, 1)
    else:
        require_files([STUDY.model(j) for j in jobs], "input-cost evaluation inputs (run --stage train first)")
        pending = STUDY.pending(jobs, "test", lambda j: [STUDY.model(j)])
        run_pool(evaluate_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
