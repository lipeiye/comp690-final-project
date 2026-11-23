"""Post-test study: can EEGNet learn to ignore inputs it cannot use, and do all electrodes then beat the best 16?

scripts/noise_mechanism_study.py found the cause of the cost of extra inputs: EEGNet trained as frozen
(Adam, no weight decay) keeps the weights of inputs that carry nothing at about their initial size, so
noise stays in every spatial filter's output and the network learns less from the electrodes. This
study adds the force that is missing, a group-lasso penalty on each input's spatial weights
(mibci.input_penalty), and asks whether the cost then shrinks and whether all electrodes then beat
the 16 most relevant ones.

Conditions (inputs as in the earlier studies; the noise is the input-cost study's draw):
  BCI IV 2a  full-22, relevance-16, relevance-16-plus-6, relevance-16-plus-16
  Lee2019    full-62, relevance-16, relevance-16-plus-46
Penalty strengths lam in 0.001, 0.003, 0.01, 0.03, 0.1; lam = 0 is the earlier studies' models.

Stages:
  plan
  train --part folds  every lam on the training day's folds (2a: 4-run folds, seeds 0-4; Lee2019:
                      one-run folds, seed 0), scored on the held-out runs of the training day
  select              for each dataset and condition, the lam (0 included, from the earlier fold models
                      at the same folds and seeds) with the best mean held-out accuracy; session 1 only
  train --part final  the final models (all of the training day; 2a seeds 0-9, Lee2019 seeds 0-2) at
                      the selected lam; none when lam = 0 is selected
  evaluate            the next day, for those final models

Lee2019 predictions (PREDICTIONS) are fixed before any of these models is scored on Lee2019 session 2;
BCI IV 2a is exploratory. Tables: scripts/analyze_input_penalty_study.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mibci import data, input_penalty, lee2019, noise_mechanism, overfit, replication
from mibci.artifacts import require_files, sha256, write_json
from mibci.experiment import device_for, run_pool
from mibci.layout import frozen_config
from mibci.model import build_eegnet
from mibci.paths import RESULTS_DIR
from mibci.study import Study, shard
from mibci.train import ChannelStandardizer, TrainConfig, predict, scores, set_seed

STUDY = Study(
    "input_penalty_study",
    "scripts/input_penalty_study.py",
    sources=[
        "src/mibci/input_penalty.py",
        "src/mibci/noise_mechanism.py",
        "src/mibci/overfit.py",
        "src/mibci/lee2019.py",
        "src/mibci/replication.py",
        "src/mibci/splits.py",
        "src/mibci/study.py",
    ],
)
SUPP = RESULTS_DIR / "supplementary"
LAMS = (0.001, 0.003, 0.01, 0.03, 0.1)
DATASETS = {
    "bnci2a": {
        "classes": 4,
        "electrodes": 22,
        "largest": 16,
        "conditions": {"full-22": 0, "relevance-16": 0, "relevance-16-plus-6": 6, "relevance-16-plus-16": 16},
        "folds": ("fold-0", "fold-1", "fold-2"),
        "fold_seeds": (0, 1, 2, 3, 4),
        "final_seeds": tuple(range(10)),
    },
    "lee2019": {
        "classes": 2,
        "electrodes": 62,
        "largest": 46,
        "conditions": {"full-62": 0, "relevance-16": 0, "relevance-16-plus-46": 46},
        "folds": ("fold-0", "fold-1"),
        "fold_seeds": (0,),
        "final_seeds": (0, 1, 2),
    },
}
# lam = 0: the earlier studies' models of the same condition, split and seed
UNPENALIZED = {
    ("bnci2a", "full-22"): ("overfit_study", "bnci2a_sub-{s:02d}_seed-{seed}_full-22{suffix}"),
    ("bnci2a", "relevance-16"): ("overfit_study", "bnci2a_sub-{s:02d}_seed-{seed}_relevance-16{suffix}"),
    ("bnci2a", "relevance-16-plus-6"): (
        "input_cost_study",
        "bnci2a_sub-{s:02d}_seed-{seed}_relevance-16-plus-6_{split}",
    ),
    ("bnci2a", "relevance-16-plus-16"): (
        "input_cost_study",
        "bnci2a_sub-{s:02d}_seed-{seed}_relevance-16-plus-16_{split}",
    ),
    ("lee2019", "full-62"): ("overfit_study", "lee2019_sub-{s:02d}_seed-{seed}_full-62{suffix}"),
    ("lee2019", "relevance-16"): ("electrode_data_study", "lee2019_sub-{s:02d}_seed-{seed}_relevance-16_{split}"),
    ("lee2019", "relevance-16-plus-46"): (
        "input_cost_study",
        "lee2019_sub-{s:02d}_seed-{seed}_relevance-16-plus-46_{split}",
    ),
}
PREDICTIONS = {
    "G1": {
        "claim": "With the penalty, inputs that carry nothing cost less",
        "contrast": "Lee2019 final models, next day: (relevance-16-plus-46 minus relevance-16), each at its selected "
        "lam, minus the same without the penalty",
        "test": "interval lower bound above 0",
    },
    "G2": {
        "claim": "With the penalty, all electrodes beat the 16 most relevant",
        "contrast": "Lee2019 final models, next day: full-62 minus relevance-16, each at its selected lam",
        "test": "interval lower bound above 0",
    },
    "G3": {
        "claim": "The penalty switches the noise inputs off",
        "contrast": "Lee2019 relevance-16-plus-46 final models at the selected lam: median input norm of the noise "
        "inputs over that of the electrodes",
        "test": "below 1/3 (only if a lam > 0 is selected for that condition)",
    },
}


def unpenalized_name(dataset: str, condition: str, subject: int, seed: int, split: str) -> tuple[str, str]:
    """(study, job name) of the lam = 0 model; the overfit study names final models without a split suffix."""
    study, pattern = UNPENALIZED[(dataset, condition)]
    suffix = "" if split == "final" else f"_{split}"
    return study, pattern.format(s=subject, seed=seed, split=split, suffix=suffix)


def day1(dataset: str, subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training-day trials (2a: artifact-free only, as in every training run), labels and run index."""
    if dataset == "bnci2a":
        ses = data.load_session(subject, 1)
        keep = ~ses["artifact"]
        return ses["X"][keep], ses["y"][keep].astype(np.int64), ses["run"][keep]
    ses = lee2019.load_session(subject, 1)
    return ses["X"], ses["y"].astype(np.int64), ses["run"]


def inputs_for(job: dict, X: np.ndarray, day: int) -> np.ndarray:
    """The electrodes of the job, then its noise inputs (the input-cost study's draw)."""
    out = X[:, job["channels"]]
    if not job["added"]:
        return out
    spec = DATASETS[job["dataset"]]
    noise = noise_mechanism.appended_noise(
        len(X), X.shape[2], job["added"], spec["largest"], job["dataset"], job["subject"], day, 0
    )
    return np.concatenate([out, noise], axis=1)


def plan_jobs() -> list[dict]:
    ic = json.loads((SUPP / "input_cost_study" / "protocol.json").read_text())["jobs"]
    relevance = {(j["dataset"], j["subject"], j["split"]): j["channels"] for j in ic}
    jobs = []
    for (dataset, subject, split), kept in sorted(relevance.items()):
        spec = DATASETS[dataset]
        if split != "final" and split not in spec["folds"]:
            continue
        seeds = spec["final_seeds"] if split == "final" else spec["fold_seeds"]
        for condition, added in spec["conditions"].items():
            channels = list(range(spec["electrodes"])) if condition.startswith("full") else kept
            jobs.extend(
                {
                    "name": f"{dataset}_sub-{subject:02d}_seed-{seed}_{condition}_lam-{lam}_{split}",
                    "dataset": dataset,
                    "subject": subject,
                    "seed": seed,
                    "split": split,
                    "condition": condition,
                    "added": added,
                    "lam": lam,
                    "channels": channels,
                }
                for lam in LAMS
                for seed in seeds
            )
    return jobs


def create_plan() -> None:
    plan = STUDY.create(
        plan_jobs(),
        purpose="Test the mechanism of the input cost by adding the missing force: a group-lasso penalty that lets "
        "EEGNet switch inputs off. Does the cost of noise inputs shrink, and do all electrodes then beat the best 16?",
        claim_boundary="Exploratory for BCI IV 2a, planned after its test results and the noise-mechanism results "
        "were known. For Lee2019 the predictions are fixed before any of these models is scored on Lee2019 session 2. "
        "lam is chosen on held-out runs of the training day only. Three predictions, no correction for "
        "multiplicity. It does not change the frozen 2a configuration.",
        lee2019_new_models_scored_before_plan=False,
        predictions=PREDICTIONS,
        lams=list(LAMS),
        selection="for each dataset and condition, the lam (0 included) with the highest held-out accuracy of the "
        "training day, averaged over seeds and folds within each participant and then over participants; ties go "
        "to the smaller lam",
        datasets={
            name: {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}
            for name, spec in DATASETS.items()
        },
        unpenalized={f"{d}|{c}": list(v) for (d, c), v in UNPENALIZED.items()},
        analysis="seeds (and folds) averaged within each participant; 95% participant bootstrap (10,000, seed 690); "
        "BCI IV 2a on artifact-free session-2 trials, Lee2019 on all",
        epochs=frozen_config()["epochs"],
        input_cost_protocol_sha256=sha256(SUPP / "input_cost_study" / "protocol.json"),
    )
    counts: dict[str, int] = {}
    for job in plan["jobs"]:
        key = f"{job['dataset']} {'final' if job['split'] == 'final' else 'folds'}"
        counts[key] = counts.get(key, 0) + 1
    print(f"{len(plan['jobs'])} planned jobs (final ones run only at the selected lam): {counts}")


def train_job(job: dict, epochs: int) -> None:
    X, y, runs = day1(job["dataset"], job["subject"])
    train, held = overfit.split_masks(runs, job["split"], job["dataset"])
    inputs = inputs_for(job, X, day=1)
    scaler = ChannelStandardizer().fit(inputs[train])
    cfg = TrainConfig(epochs=epochs, seed=job["seed"])
    set_seed(cfg.seed)
    model = build_eegnet(inputs.shape[1], n_classes=DATASETS[job["dataset"]]["classes"])
    device = device_for(job["subject"])
    X_fit = scaler.transform(inputs[train])
    result = input_penalty.fit_penalized(model, X_fit, y[train], cfg, job["lam"], device=device)
    path = STUDY.model(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: job[k] for k in ("dataset", "subject", "seed", "split", "condition", "added", "lam")}
    meta |= {"epochs": epochs, "device": device}
    channels = [*job["channels"], *(-(i + 1) for i in range(job["added"]))]
    replication.save(path, model, channels, scaler, **meta)
    with torch.no_grad():
        norms = input_penalty.input_norms(model).numpy()
    record = {
        **meta,
        "name": job["name"],
        "n_train": int(train.sum()),
        "train_time_s": result["train_time_s"],
        "train_acc": scores(predict(model, X_fit), y[train])["acc"],
        "final_train_loss": result["history"]["train_loss"][-1],
        "input_norms": [round(float(v), 6) for v in norms],
        "model_sha256": sha256(path),
        "protocol_sha256": sha256(STUDY.protocol_path),
    }
    if held is not None:
        logits = predict(model, scaler.transform(inputs[held]))
        record |= {"n_held": int(held.sum()), "within_day": scores(logits, y[held])}
    write_json(STUDY.record("runs", job), record)


def held_out_accuracy(job: dict) -> float:
    """Held-out accuracy of the training day of a fold job, penalized or (lam = 0) from the earlier study."""
    if job["lam"]:
        record = STUDY.record("runs", job)
    else:
        study, name = unpenalized_name(job["dataset"], job["condition"], job["subject"], job["seed"], job["split"])
        record = SUPP / study / "runs" / f"{name}.json"
    return json.loads(record.read_text())["within_day"]["acc"]


def select(plan: dict) -> None:
    folds = [j for j in plan["jobs"] if j["split"] != "final"]
    missing = [j["name"] for j in folds if not STUDY.record("runs", j).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} fold models are not trained yet, e.g. {missing[0]}")
    rows = [{**j, "acc": held_out_accuracy(j)} for j in folds]
    rows += [{**j, "lam": 0.0, "acc": held_out_accuracy({**j, "lam": 0.0})} for j in folds if j["lam"] == LAMS[0]]
    table: dict = {}
    for dataset, spec in DATASETS.items():
        for condition in spec["conditions"]:
            by_lam = {}
            for lam in (0.0, *LAMS):
                cell = [r for r in rows if (r["dataset"], r["condition"], r["lam"]) == (dataset, condition, lam)]
                per_person: dict[int, list[float]] = {}
                for r in cell:
                    per_person.setdefault(r["subject"], []).append(r["acc"])
                by_lam[lam] = float(np.mean([np.mean(v) for v in per_person.values()]))
            best = max(by_lam.values())
            chosen = min(lam for lam, v in by_lam.items() if v == best)
            table[f"{dataset}|{condition}"] = {"held_out_accuracy": by_lam, "selected_lam": chosen}
    write_json(STUDY.out / "selection.json", {"rule": plan["selection"], "selection": table})
    for key, v in table.items():
        print(key, "selected lam", v["selected_lam"], {k: round(100 * a, 2) for k, a in v["held_out_accuracy"].items()})


def selected_final_jobs(plan: dict) -> list[dict]:
    chosen = json.loads((STUDY.out / "selection.json").read_text())["selection"]
    return [
        j
        for j in plan["jobs"]
        if j["split"] == "final" and j["lam"] == chosen[f"{j['dataset']}|{j['condition']}"]["selected_lam"]
    ]


def evaluate_job(job: dict) -> None:
    torch.set_num_threads(1)
    path = STUDY.model(job)
    model, scaler, bundle = replication.load(path)
    if bundle["channels"] != [*job["channels"], *(-(i + 1) for i in range(job["added"]))]:
        raise RuntimeError(f"Bundle channels differ from the plan: {path}")
    if job["dataset"] == "bnci2a":
        ses = data.load_session(job["subject"], 2, final_evaluation=True)
        clean = ~ses["artifact"]
    else:
        ses = lee2019.load_session(job["subject"], 2, final_evaluation=True)
        clean = np.ones(len(ses["y"]), bool)
    y = ses["y"].astype(np.int64)
    logits = predict(model, scaler.transform(inputs_for(job, ses["X"], day=2)))
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
    parser.add_argument("--stage", choices=["plan", "train", "select", "evaluate"], required=True)
    parser.add_argument("--part", choices=["folds", "final"], help="train: the fold models or the final models")
    parser.add_argument("--shard", help="i/n: run only array task i of n (0-based)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "plan":
        create_plan()
        return
    plan = STUDY.load()
    if args.stage == "select":
        select(plan)
        return
    if args.stage == "train":
        if not args.part:
            raise SystemExit("--stage train needs --part folds or --part final")
        todo = [j for j in plan["jobs"] if j["split"] != "final"] if args.part == "folds" else selected_final_jobs(plan)
        STUDY.log_environment(f"train-{args.part}-{(args.shard or 'all').replace('/', 'of')}")
        pending = STUDY.pending(
            shard(todo, args.shard),
            "runs",
            lambda j: [STUDY.model(j)],
            lambda j: {"subject": j["subject"], "seed": j["seed"], "epochs": plan["epochs"]},
        )
        run_pool(train_job, [(j["name"], {"job": j, "epochs": plan["epochs"]}) for j in pending], args.workers, 1)
        return
    todo = shard(selected_final_jobs(plan), args.shard)
    require_files([STUDY.model(j) for j in todo], "penalized final models (run --stage train --part final first)")
    pending = STUDY.pending(todo, "test", lambda j: [STUDY.model(j)])
    run_pool(evaluate_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
