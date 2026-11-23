"""Post-test study: which extra inputs cost EEGNet accuracy, the number of inputs or their independent fluctuation?

scripts/input_cost_study.py found that every band-limited noise input added to the 16 most relevant
electrodes costs about 0.9 percentage points on BCI IV 2a and 0.3 on Lee2019, and
scripts/noise_mechanism_study.py offered an explanation it did not test: electrodes are highly
correlated, so spatial filters can cancel what they share, while noise inputs are independent and
cannot be cancelled. If so, the cost follows the independent fluctuation an input brings, not the
number of inputs. This study adds k extra inputs of other kinds to the same 16 electrodes
(mibci.input_kind; the kept electrodes are the input-cost plan's, the noise its draw):

  dup       copies of kept electrodes (relevance order, cycled): no new information, no new fluctuation
  mixA      sqrt(1 - a) * a standardized kept electrode + sqrt(a) * the input-cost noise input, a = A / 100
  shared    k copies of one noise waveform: k inputs, one independent fluctuation

  BCI IV 2a  k = 6: dup, mix25, mix50, mix75, shared; final models seeds 0-9, fold-0/1/2 (4 runs) seeds 0-4
  Lee2019    k = 46: dup, mix50, shared; final models and fold-0/1 (one run), seeds 0-2

References (not retrained): k = 0 is relevance-16 (2a: scripts/overfit_study.py; Lee2019:
scripts/electrode_data_study.py) and pure noise (a = 1) is relevance-16-plus-k of scripts/input_cost_study.py,
at the same splits and seeds. Lee2019 predictions (PREDICTIONS) are fixed in the plan before any of these
models is scored on Lee2019 session 2; BCI IV 2a is exploratory.

Stages: plan, train, evaluate, and predictability (session 1 only, no training: how much of each dropped
electrode's variance on held-out runs the kept electrodes cannot predict linearly). Tables:
scripts/analyze_input_kind_study.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mibci import data, input_kind, lee2019, noise_mechanism, overfit, replication
from mibci.artifacts import require_files, sha256, write_json
from mibci.experiment import device_for, run_pool
from mibci.layout import frozen_config
from mibci.model import build_eegnet
from mibci.paths import RESULTS_DIR
from mibci.study import Study, shard
from mibci.train import ChannelStandardizer, TrainConfig, fit, predict, scores, set_seed

STUDY = Study(
    "input_kind_study",
    "scripts/input_kind_study.py",
    sources=[
        "src/mibci/input_kind.py",
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
ORDERS = SUPP / "electrode_data_study" / "protocol.json"
KEPT = 16
DATASETS = {
    "bnci2a": {
        "kinds": ("dup", "mix25", "mix50", "mix75", "shared"),
        "added": 6,
        "largest": 16,  # the input-cost study drew 16 noise inputs per participant and day and cut to k
        "classes": 4,
        "electrodes": 22,
        "final_seeds": tuple(range(10)),
        "fold_seeds": tuple(range(5)),
        "splits": ("fold-0", "fold-1", "fold-2"),
    },
    "lee2019": {
        "kinds": ("dup", "mix50", "shared"),
        "added": 46,
        "largest": 46,
        "classes": 2,
        "electrodes": 62,
        "final_seeds": (0, 1, 2),
        "fold_seeds": (0, 1, 2),
        "splits": ("fold-0", "fold-1"),
    },
}
REFERENCES = {
    "none": "relevance-16 of scripts/overfit_study.py (2a) and scripts/electrode_data_study.py (Lee2019)",
    "noise": "relevance-16-plus-k of scripts/input_cost_study.py (a = 1: the same noise inputs)",
}
PREDICTIONS = {
    "K1": {
        "claim": "Copies cost less than independent noise",
        "contrast": "Lee2019 final models, next day: dup-46 minus relevance-16-plus-46",
        "test": "interval lower bound above 0",
        "also": "dup-46 minus relevance-16 is reported; with its interval lower bound above -2 points, copies "
        "are described as having no visible cost",
    },
    "K2": {
        "claim": "Half noise lies between copies and noise",
        "contrast": "Lee2019 final models, next day: dup-46 minus mix50-46, and mix50-46 minus relevance-16-plus-46",
        "test": "both interval lower bounds above 0",
    },
    "K3": {
        "claim": "One shared noise waveform costs less than independent noise inputs",
        "contrast": "Lee2019 final models, next day: shared-46 minus relevance-16-plus-46",
        "test": "interval lower bound above 0",
    },
}


def day1(dataset: str, subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training-day trials (2a: artifact-free only, as in every training run), labels and run index."""
    if dataset == "bnci2a":
        ses = data.load_session(subject, 1)
        keep = ~ses["artifact"]
        return ses["X"][keep], ses["y"][keep].astype(np.int64), ses["run"][keep]
    ses = lee2019.load_session(subject, 1)
    return ses["X"], ses["y"].astype(np.int64), ses["run"]


def inputs_for(job: dict, X: np.ndarray, day: int, mean: list[float], std: list[float]) -> np.ndarray:
    """Kept electrodes, then the job's extra inputs; noise is the input-cost study's draw for this day."""
    spec = DATASETS[job["dataset"]]
    noise = noise_mechanism.appended_noise(
        len(X), X.shape[2], job["added"], spec["largest"], job["dataset"], job["subject"], day, 0
    )
    return input_kind.model_input(X, job["channels"], job["kind"], job["sources"], noise, mean, std)


def model_inputs(job: dict) -> list[int]:
    """One bundle entry per network input: electrode indices, then -1, -2, ... for the extra inputs."""
    return [*job["channels"], *(-(i + 1) for i in range(job["added"]))]


def plan_jobs() -> list[dict]:
    cost_jobs = json.loads(COST_PLAN.read_text())["jobs"]
    kept = {(j["dataset"], j["subject"], j["split"]): j["channels"] for j in cost_jobs}
    orders = json.loads(ORDERS.read_text())["orders"]
    jobs = []
    for dataset, spec in DATASETS.items():
        subjects = sorted({s for d, s, _ in kept if d == dataset})
        for subject in subjects:
            for split in ("final", *spec["splits"]):
                channels = kept[(dataset, subject, split)]
                order = [int(c) for c in orders[f"{dataset}_{subject}"][split][:KEPT]]
                if sorted(order) != channels:
                    raise ValueError(f"{dataset} {subject} {split}: orders and the input-cost plan disagree")
                sources = input_kind.copied(order, spec["added"])
                seeds = spec["final_seeds"] if split == "final" else spec["fold_seeds"]
                jobs.extend(
                    {
                        "name": f"{dataset}_sub-{subject:02d}_seed-{seed}_{kind}-{spec['added']}_{split}",
                        "dataset": dataset,
                        "subject": subject,
                        "seed": seed,
                        "split": split,
                        "kind": kind,
                        "a": input_kind.mix_fraction(kind),
                        "added": spec["added"],
                        "condition": f"{kind}-{spec['added']}",
                        "channels": channels,
                        "order": order,
                        "sources": sources,
                    }
                    for kind in spec["kinds"]
                    for seed in seeds
                )
    return jobs


def create_plan() -> None:
    plan = STUDY.create(
        plan_jobs(),
        purpose="Test whether the cost of extra inputs comes from their number or from fluctuation unrelated to the "
        "other inputs: copies of kept electrodes, mixes of copies and noise, and one noise waveform repeated, added "
        "to the 16 most relevant electrodes.",
        claim_boundary="Exploratory for BCI IV 2a, planned after its test results and the input-cost and "
        "noise-mechanism results were known. For Lee2019 the predictions are fixed before any of these models is "
        "scored on Lee2019 session 2; three predictions, no correction for multiplicity. It explains a result; it "
        "does not change the frozen 2a configuration.",
        lee2019_new_models_scored_before_plan=False,
        predictions=PREDICTIONS,
        references=REFERENCES,
        kinds={
            "dup": "exact copies of kept electrodes, the kept electrodes in relevance order (most relevant "
            "first), cycled",
            "mixA": "sqrt(1 - a) * the copied electrode standardized with the training partition's mean and standard "
            "deviation + sqrt(a) * the input-cost study's noise input at the same position, a = A / 100",
            "shared": "every extra input is the first noise input of the input-cost draw",
        },
        datasets={
            name: {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}
            for name, spec in DATASETS.items()
        },
        noise="mibci.noise_mechanism.appended_noise draw 0, the input-cost study's (largest k drawn, cut to k)",
        predictability="session 1 only: per fold, a least-squares fit (with intercept) from the 16 kept electrodes "
        "to each dropped electrode on the training runs; 1 - R^2 on the held-out runs (mibci.input_kind)",
        analysis="seeds (and folds) averaged within each participant; 95% participant bootstrap (10,000, seed 690); "
        "BCI IV 2a on artifact-free session-2 trials, Lee2019 on all; 2a also with the seed error",
        epochs=frozen_config()["epochs"],
        input_cost_protocol_sha256=sha256(COST_PLAN),
        orders_source_sha256=sha256(ORDERS),
    )
    counts: dict[str, int] = {}
    for job in plan["jobs"]:
        key = f"{job['dataset']} {'final' if job['split'] == 'final' else 'folds'}"
        counts[key] = counts.get(key, 0) + 1
    print(f"{len(plan['jobs'])} jobs: {counts}")


def train_job(job: dict, epochs: int) -> None:
    X, y, runs = day1(job["dataset"], job["subject"])
    train, held = overfit.split_masks(runs, job["split"], job["dataset"])
    mean, std = input_kind.source_stats(X[train], job["sources"])
    inputs = inputs_for(job, X, 1, mean, std)
    scaler = ChannelStandardizer().fit(inputs[train])
    cfg = TrainConfig(epochs=epochs, seed=job["seed"])
    set_seed(cfg.seed)
    model = build_eegnet(inputs.shape[1], n_classes=DATASETS[job["dataset"]]["classes"])
    device = device_for(job["subject"])
    X_fit = scaler.transform(inputs[train])
    result = fit(model, X_fit, y[train], cfg, device=device)
    path = STUDY.model(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: job[k] for k in ("dataset", "subject", "seed", "split", "condition", "kind", "a", "added")}
    meta |= {"epochs": epochs, "device": device, "source_mean": mean, "source_std": std}
    replication.save(path, model, model_inputs(job), scaler, **meta)
    extra = list(range(len(job["channels"]), len(job["channels"]) + job["added"]))
    record = {
        **meta,
        "name": job["name"],
        "n_train": int(train.sum()),
        "train_time_s": result["train_time_s"],
        "train_acc": scores(predict(model, X_fit), y[train])["acc"],
        "final_train_loss": result["history"]["train_loss"][-1],
        "train_loss": [round(v, 5) for v in result["history"]["train_loss"]],
        "extra_share": noise_mechanism.noise_share(model, X_fit, extra).tolist(),
        "weight_rms_kept": noise_mechanism.weight_rms(model, list(range(len(job["channels"])))),
        "weight_rms_extra": noise_mechanism.weight_rms(model, extra),
        "model_sha256": sha256(path),
        "protocol_sha256": sha256(STUDY.protocol_path),
    }
    if held is not None:
        logits = predict(model, scaler.transform(inputs[held]))
        record |= {
            "n_held": int(held.sum()),
            "within_day": scores(logits, y[held]),
            "within_day_pred": logits.argmax(1).tolist(),
        }
    write_json(STUDY.record("runs", job), record)


def evaluate_job(job: dict) -> None:
    torch.set_num_threads(1)
    path = STUDY.model(job)
    model, scaler, bundle = replication.load(path)
    if bundle["channels"] != model_inputs(job) or bundle["kind"] != job["kind"]:
        raise RuntimeError(f"Bundle differs from the plan: {path}")
    if job["dataset"] == "bnci2a":
        ses = data.load_session(job["subject"], 2, final_evaluation=True)
        clean = ~ses["artifact"]
    else:
        ses = lee2019.load_session(job["subject"], 2, final_evaluation=True)
        clean = np.ones(len(ses["y"]), bool)
    y = ses["y"].astype(np.int64)
    inputs = inputs_for(job, ses["X"], 2, bundle["source_mean"], bundle["source_std"])
    logits = predict(model, scaler.transform(inputs))
    write_json(
        STUDY.record("test", job),
        {
            **{k: v for k, v in job.items() if k not in ("order", "sources")},
            "model_sha256": sha256(path),
            "protocol_sha256": sha256(STUDY.protocol_path),
            "primary": scores(logits[clean], y[clean]),
            "secondary": scores(logits, y),
            "pred": logits.argmax(1).tolist(),
        },
    )


def predictability_job(dataset: str, subject: int, jobs: list[dict]) -> None:
    """Session 1 only: the unexplained share of each dropped electrode's held-out variance, per fold."""
    X, _, runs = day1(dataset, subject)
    spec = DATASETS[dataset]
    out = {"dataset": dataset, "subject": subject, "folds": {}}
    for split in spec["splits"]:
        kept = next(j["channels"] for j in jobs if j["split"] == split)
        dropped = [c for c in range(spec["electrodes"]) if c not in kept]
        train, held = overfit.split_masks(runs, split, dataset)
        with np.errstate(divide="ignore", invalid="ignore"):
            left = input_kind.unexplained_fraction(X, kept, dropped, train, held)
        out["folds"][split] = {
            "kept": kept,
            "dropped": dropped,
            "unexplained": [float(v) if np.isfinite(v) else None for v in left],
        }
    write_json(STUDY.out / "predictability" / f"{dataset}_sub-{subject:02d}.json", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "train", "evaluate", "predictability"], required=True)
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
    elif args.stage == "evaluate":
        require_files([STUDY.model(j) for j in jobs], "input-kind evaluation inputs (run --stage train first)")
        pending = STUDY.pending(jobs, "test", lambda j: [STUDY.model(j)])
        run_pool(evaluate_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)
    else:
        people = sorted({(j["dataset"], j["subject"]) for j in jobs})
        tasks = [
            (
                f"{d} {s}",
                {
                    "dataset": d,
                    "subject": s,
                    "jobs": [j for j in plan["jobs"] if (j["dataset"], j["subject"]) == (d, s)],
                },
            )
            for d, s in people
        ]
        run_pool(predictability_job, tasks, args.workers, 1)


if __name__ == "__main__":
    main()
