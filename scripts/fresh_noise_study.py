"""Post-test study: is the cost of a noise input there because the same noise repeats in every epoch?

scripts/input_cost_study.py appends band-limited noise inputs to the 16 most relevant electrodes and finds that
each costs accuracy; no penalty switched them off (scripts/input_penalty_study.py, scripts/scalefree_penalty_study.py)
but pretraining on other people did (scripts/pretrain_study.py). Two explanations remain: the noise, the same in
all 475 epochs, lets the network remember the training labels (A), or one day of EEG is too little to learn which
inputs are useless (B). This study repeats the input-cost study's relevance-16-plus-k exactly (same electrodes,
seeds, recipe, scored inputs and standardization) except that every training epoch draws new noise for every
training trial (mibci.fresh_noise): the noise never repeats, the EEG and the epochs do not change, so only (A) is
tested.

  BCI IV 2a  fresh-6, fresh-16: final models seeds 0-9, fold-0/1/2 (4 runs) seeds 0-4 (450 models)
  Lee2019    fresh-46: final models and fold-0/1 (one run), seeds 0-2 (486 models)

References (not retrained): fixed noise is relevance-16-plus-k of scripts/input_cost_study.py, no noise is
relevance-16 (2a: scripts/overfit_study.py; Lee2019: scripts/electrode_data_study.py), at the same splits and seeds.
The stage `reference` measures the fixed-noise models as this study measures its own (noise share of the spatial
filters, noise-to-electrode weight ratio, training trials with unseen noise); it reads session 1 only.

Stages: plan, train, reference, evaluate. Lee2019 predictions F1-F3 are fixed in the plan before any of these
models is scored on Lee2019 session 2; BCI IV 2a is exploratory. Tables: scripts/analyze_fresh_noise_study.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mibci import data, fresh_noise, lee2019, noise_mechanism, overfit, replication
from mibci.artifacts import require_files, sha256, write_json
from mibci.experiment import device_for, run_pool
from mibci.layout import frozen_config
from mibci.model import build_eegnet
from mibci.paths import DATA_DIR, RESULTS_DIR
from mibci.study import Study, shard
from mibci.train import ChannelStandardizer, TrainConfig, predict, scores, set_seed

STUDY = Study(
    "fresh_noise_study",
    "scripts/fresh_noise_study.py",
    sources=[
        "src/mibci/fresh_noise.py",
        "src/mibci/noise_mechanism.py",
        "src/mibci/overfit.py",
        "src/mibci/lee2019.py",
        "src/mibci/replication.py",
        "src/mibci/splits.py",
        "src/mibci/study.py",
    ],
)
SUPP = RESULTS_DIR / "supplementary"
COST_PLAN = SUPP / "input_cost_study" / "protocol.json"
COST_MODELS = DATA_DIR / "models" / "supplementary" / "input_cost_study"
DATASETS = {
    "bnci2a": {
        "added": (6, 16),
        "classes": 4,
        "final_seeds": tuple(range(10)),
        "fold_seeds": tuple(range(5)),
        "splits": ("fold-0", "fold-1", "fold-2"),
    },
    "lee2019": {
        "added": (46,),
        "classes": 2,
        "final_seeds": (0, 1, 2),
        "fold_seeds": (0, 1, 2),
        "splits": ("fold-0", "fold-1"),
    },
}
REFERENCES = {
    "fixed": "relevance-16-plus-k of scripts/input_cost_study.py: the same inputs with the noise drawn once (draw 0)",
    "none": "relevance-16 of scripts/overfit_study.py (2a) and scripts/electrode_data_study.py (Lee2019)",
}
PREDICTIONS = {
    "F1": {
        "claim": "Noise that never repeats costs less than noise that repeats in every epoch",
        "contrast": "Lee2019 final models, next day: fresh-46 minus relevance-16-plus-46 (scripts/input_cost_study.py)",
        "test": "interval lower bound above 0",
    },
    "F2": {
        "claim": "Noise that never repeats has no visible cost",
        "contrast": "Lee2019 final models, next day: fresh-46 minus relevance-16",
        "test": "interval lower bound above -2 points (as K1 of scripts/input_kind_study.py)",
    },
    "F3": {
        "claim": "The network turns noise that never repeats down",
        "contrast": "Lee2019 final models: noise-to-electrode weight ratio (RMS of the effective spatial weights, "
        "mibci.noise_mechanism.weight_rms), fresh-46 minus relevance-16-plus-46, per participant",
        "test": "interval upper bound below 0; whether the fresh models' median ratio falls below 1/3 is also reported",
    },
}
READINGS = {
    "F1 and F2 hold": "explanation A: the network would ignore useless inputs; recorded noise is kept because within "
    "one "
    "day it helps to remember the training set, so methods that only see the training set cannot tell it from a "
    "useful electrode; data beyond the training set (choosing electrodes first, pretraining) can",
    "F1 does not hold": "not the remembering of noise: even noise that never repeats is not switched off; the problem "
    "is optimization or architecture (read with the long-training curves)",
    "F1 holds, F2 does not": "both explanations contribute; report the measured shares",
}


def day1(dataset: str, subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training-day trials (2a: artifact-free only, as in every training run), labels and run index."""
    if dataset == "bnci2a":
        ses = data.load_session(subject, 1)
        keep = ~ses["artifact"]
        return ses["X"][keep], ses["y"][keep].astype(np.int64), ses["run"][keep]
    ses = lee2019.load_session(subject, 1)
    return ses["X"], ses["y"].astype(np.int64), ses["run"]


def model_inputs(job: dict) -> list[int]:
    return [*job["channels"], *(-(i + 1) for i in range(job["added"]))]


def plan_jobs() -> list[dict]:
    cost = json.loads(COST_PLAN.read_text())
    jobs = []
    for j in cost["jobs"]:
        spec = DATASETS[j["dataset"]]
        seeds = spec["final_seeds"] if j["split"] == "final" else spec["fold_seeds"]
        if j["added"] not in spec["added"] or j["seed"] not in seeds or j["split"] not in ("final", *spec["splits"]):
            continue
        jobs.append({
            "name": f"{j['dataset']}_sub-{j['subject']:02d}_seed-{j['seed']}_fresh-{j['added']}_{j['split']}",
            "dataset": j["dataset"], "subject": j["subject"], "seed": j["seed"], "split": j["split"],
            "added": j["added"], "largest": max(cost["datasets"][j["dataset"]]["added"]),
            "condition": f"fresh-{j['added']}", "channels": j["channels"], "fixed": j["name"],
        })  # fmt: skip
    return jobs


def create_plan() -> None:
    plan = STUDY.create(
        plan_jobs(),
        purpose="Test whether noise inputs cost accuracy because the same noise repeats in every training epoch (and "
        "so "
        "can help to remember the training labels), by drawing new noise in every epoch.",
        claim_boundary="Exploratory for BCI IV 2a, planned after its test results and after the input-cost, "
        "penalty and pretraining results were known. For Lee2019 the predictions are fixed before any of these models "
        "is scored on Lee2019 session 2; three predictions, no correction for multiplicity. It explains a result; it "
        "does not change the frozen 2a configuration.",
        lee2019_new_models_scored_before_plan=False,
        predictions=PREDICTIONS,
        readings=READINGS,
        references=REFERENCES,
        fresh_noise="every epoch, every training trial: white noise through the preprocessing band-pass after a 2-s "
        "warm-up, divided by its gain (unit variance), numpy seed [694, dataset code, participant, split code, model "
        "seed, epoch]; standardized with the fixed-noise models' statistics (draw 0 of the training trials)",
        scored_inputs="identical to scripts/input_cost_study.py: draw 0 for the held-out runs and session 2",
        unseen_noise=f"mibci.noise_mechanism.appended_noise draw {fresh_noise.UNSEEN_DRAW}, session 1",
        datasets={
            name: {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}
            for name, spec in DATASETS.items()
        },
        analysis="seeds (and folds) averaged within each participant; 95% participant bootstrap (10,000, seed 690); "
        "BCI IV 2a on artifact-free session-2 trials, Lee2019 on all; 2a also with the seed error",
        epochs=frozen_config()["epochs"],
        input_cost_protocol_sha256=sha256(COST_PLAN),
    )
    counts: dict[str, int] = {}
    for job in plan["jobs"]:
        key = f"{job['dataset']} {'final' if job['split'] == 'final' else 'folds'}"
        counts[key] = counts.get(key, 0) + 1
    print(f"{len(plan['jobs'])} jobs: {counts}")


def measures(model, scaler, job: dict, X: np.ndarray, y: np.ndarray, train: np.ndarray) -> dict:
    """Noise share and weight ratio on the draw-0 training inputs; training trials with draw-0 and unseen noise."""
    seen = scaler.transform(
        fresh_noise.draw0(X, job["channels"], job["added"], job["largest"], job["dataset"], job["subject"], 1)[train]
    )
    unseen = np.concatenate(
        [
            X[:, job["channels"]],
            fresh_noise.unseen_noise(len(X), X.shape[2], job["added"], job["largest"], job["dataset"], job["subject"]),
        ],
        axis=1,
    )
    kept = list(range(len(job["channels"])))
    noise = list(range(len(kept), len(kept) + job["added"]))
    return {
        "train_acc_draw0": scores(predict(model, seen), y[train])["acc"],
        "train_acc_unseen": scores(predict(model, scaler.transform(unseen[train])), y[train])["acc"],
        "noise_share": float(np.mean(noise_mechanism.noise_share(model, seen, noise))),
        "weight_rms_kept": noise_mechanism.weight_rms(model, kept),
        "weight_rms_noise": noise_mechanism.weight_rms(model, noise),
    }


def train_job(job: dict, epochs: int) -> None:
    X, y, runs = day1(job["dataset"], job["subject"])
    train, held = overfit.split_masks(runs, job["split"], job["dataset"])
    fixed = fresh_noise.draw0(X, job["channels"], job["added"], job["largest"], job["dataset"], job["subject"], 1)
    scaler = ChannelStandardizer().fit(fixed[train])  # the fixed-noise models' standardization
    electrodes, n_train = X[train][:, job["channels"]], int(train.sum())

    def epoch_inputs(epoch: int) -> np.ndarray:
        noise = fresh_noise.epoch_noise(
            n_train, X.shape[2], job["added"], job["dataset"], job["subject"], job["split"], job["seed"], epoch
        )
        return scaler.transform(np.concatenate([electrodes, noise], axis=1))

    cfg = TrainConfig(epochs=epochs, seed=job["seed"])
    set_seed(cfg.seed)
    model = build_eegnet(fixed.shape[1], n_classes=DATASETS[job["dataset"]]["classes"])
    device = device_for(job["subject"])
    result = fresh_noise.fit_epochwise(model, epoch_inputs, y[train], cfg, device=device)
    path = STUDY.model(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: job[k] for k in ("dataset", "subject", "seed", "split", "condition", "added")}
    meta |= {"epochs": epochs, "device": device}
    replication.save(path, model, model_inputs(job), scaler, **meta)
    record = {
        **meta,
        "name": job["name"],
        "n_train": n_train,
        "train_time_s": result["train_time_s"],
        "final_train_loss": result["history"]["train_loss"][-1],
        "train_loss": [round(v, 5) for v in result["history"]["train_loss"]],
        **measures(model, scaler, job, X, y, train),
        "model_sha256": sha256(path),
        "protocol_sha256": sha256(STUDY.protocol_path),
    }
    if held is not None:
        logits = predict(model, scaler.transform(fixed[held]))
        record |= {"n_held": int(held.sum()), "within_day": scores(logits, y[held])}
    write_json(STUDY.record("runs", job), record)


def reference_job(job: dict) -> None:
    """The fixed-noise model of the same participant, split, seed and k, measured as this study's models."""
    torch.set_num_threads(1)
    path = COST_MODELS / f"{job['fixed']}.pt"
    model, scaler, bundle = replication.load(path)
    if bundle["channels"] != model_inputs(job):
        raise RuntimeError(f"Bundle channels differ from the plan: {path}")
    X, y, runs = day1(job["dataset"], job["subject"])
    train, _ = overfit.split_masks(runs, job["split"], job["dataset"])
    record = {
        "name": job["fixed"],
        "fresh_twin": job["name"],
        **{k: job[k] for k in ("dataset", "subject", "seed", "split", "added")},
        **measures(model, scaler, job, X, y, train),
        "model_sha256": sha256(path),
        "protocol_sha256": sha256(STUDY.protocol_path),
    }
    write_json(STUDY.out / "reference" / f"{job['fixed']}.json", record)


def evaluate_job(job: dict) -> None:
    torch.set_num_threads(1)
    path = STUDY.model(job)
    model, scaler, bundle = replication.load(path)
    if bundle["channels"] != model_inputs(job) or bundle["condition"] != job["condition"]:
        raise RuntimeError(f"Bundle differs from the plan: {path}")
    if job["dataset"] == "bnci2a":
        ses = data.load_session(job["subject"], 2, final_evaluation=True)
        clean = ~ses["artifact"]
    else:
        ses = lee2019.load_session(job["subject"], 2, final_evaluation=True)
        clean = np.ones(len(ses["y"]), bool)
    y = ses["y"].astype(np.int64)
    inputs = fresh_noise.draw0(
        ses["X"], job["channels"], job["added"], job["largest"], job["dataset"], job["subject"], 2
    )
    logits = predict(model, scaler.transform(inputs))
    write_json(
        STUDY.record("test", job),
        {
            **{k: v for k, v in job.items() if k != "channels"},
            "model_sha256": sha256(path),
            "protocol_sha256": sha256(STUDY.protocol_path),
            "primary": scores(logits[clean], y[clean]),
            "secondary": scores(logits, y),
            "pred": logits.argmax(1).tolist(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "train", "reference", "evaluate"], required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--subjects", nargs="+", type=int, help="only these participants (local smoke tests)")
    parser.add_argument("--shard", help="i/n: run only array task i of n (0-based)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "plan":
        create_plan()
        return
    plan = STUDY.load()
    jobs = [
        j
        for j in shard(plan["jobs"], args.shard)
        if j["dataset"] in args.datasets and (not args.subjects or j["subject"] in args.subjects)
    ]
    if args.stage == "train":
        STUDY.log_environment(f"train-{(args.shard or 'all').replace('/', 'of')}")
        pending = STUDY.pending(
            jobs,
            "runs",
            lambda j: [STUDY.model(j)],
            lambda j: {"subject": j["subject"], "seed": j["seed"], "epochs": plan["epochs"]},
        )
        run_pool(train_job, [(j["name"], {"job": j, "epochs": plan["epochs"]}) for j in pending], args.workers, 1)
    elif args.stage == "reference":
        todo = [j for j in jobs if not (STUDY.out / "reference" / f"{j['fixed']}.json").exists()]
        require_files([COST_MODELS / f"{j['fixed']}.pt" for j in todo], "the input-cost study's fixed-noise models")
        run_pool(reference_job, [(j["fixed"], {"job": j}) for j in todo], args.workers, 1)
    else:
        require_files([STUDY.model(j) for j in jobs], "fresh-noise evaluation inputs (run --stage train first)")
        pending = STUDY.pending(jobs, "test", lambda j: [STUDY.model(j)])
        run_pool(evaluate_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
