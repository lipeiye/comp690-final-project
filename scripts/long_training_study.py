"""Post-test study: is the cost of extra inputs a matter of training too briefly?

scripts/noise_mechanism_study.py found that models with noise inputs were still lowering their training
loss at epoch 475, and that models that ended training with a higher loss did worse the next day
(r = -0.62 within participant and condition). The 475-epoch budget was chosen on the validation
curves of all 22 electrodes. This study trains four times as long with the frozen recipe
(mibci.train.fit: Adam, lr 1e-3, batch 64, no early stopping) and keeps the weights at epochs 475,
950 and 1,900:

  BCI IV 2a  relevance-16, relevance-16-plus-6 (the input-cost study's noise), full-22
  Lee2019    relevance-16, relevance-16-plus-46, full-62

Fold models (2a: fold-0/1/2, 4 runs, seeds 0-4; Lee2019: fold-0/1, one run, seed 0) track the held-out
runs of the training day after every epoch. Final models (all of the training day; 2a seeds 0-9,
Lee2019 seeds 0-2) are scored on the next day at all three checkpoints. Each 475-epoch checkpoint is
compared with the earlier study's model of the same condition, split and seed (the same weights are
expected on the same hardware; either way, the contrasts use this study's own 475-epoch checkpoint).
Lee2019 predictions (PREDICTIONS) are fixed before any of these models is scored on Lee2019 session 2;
BCI IV 2a is exploratory.

Stages: plan, train, evaluate. Tables: scripts/analyze_long_training_study.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mibci import data, lee2019, noise_mechanism, overfit, replication
from mibci.artifacts import require_files, sha256, write_json
from mibci.experiment import device_for, run_pool
from mibci.layout import frozen_config
from mibci.model import build_eegnet
from mibci.paths import MODELS_DIR, RESULTS_DIR
from mibci.study import Study, shard
from mibci.train import ChannelStandardizer, TrainConfig, fit, predict, scores, set_seed

STUDY = Study(
    "long_training_study",
    "scripts/long_training_study.py",
    sources=[
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
MULTIPLES = (1, 2, 4)  # checkpoints at these multiples of the frozen budget
DATASETS = {
    "bnci2a": {
        "classes": 4,
        "electrodes": 22,
        "largest": 16,  # the input-cost study drew this many noise inputs and cut to k
        "conditions": {"relevance-16": 0, "relevance-16-plus-6": 6, "full-22": 0},
        "folds": ("fold-0", "fold-1", "fold-2"),
        "fold_seeds": (0, 1, 2, 3, 4),
        "final_seeds": tuple(range(10)),
    },
    "lee2019": {
        "classes": 2,
        "electrodes": 62,
        "largest": 46,
        "conditions": {"relevance-16": 0, "relevance-16-plus-46": 46, "full-62": 0},
        "folds": ("fold-0", "fold-1"),
        "fold_seeds": (0,),
        "final_seeds": (0, 1, 2),
    },
}
# the earlier studies' 475-epoch models of the same condition, split and seed
EARLIER = {
    ("bnci2a", "relevance-16"): ("overfit_study", "bnci2a_sub-{s:02d}_seed-{seed}_relevance-16{suffix}"),
    ("bnci2a", "relevance-16-plus-6"): (
        "input_cost_study",
        "bnci2a_sub-{s:02d}_seed-{seed}_relevance-16-plus-6_{split}",
    ),
    ("bnci2a", "full-22"): ("overfit_study", "bnci2a_sub-{s:02d}_seed-{seed}_full-22{suffix}"),
    ("lee2019", "relevance-16"): ("electrode_data_study", "lee2019_sub-{s:02d}_seed-{seed}_relevance-16_{split}"),
    ("lee2019", "relevance-16-plus-46"): (
        "input_cost_study",
        "lee2019_sub-{s:02d}_seed-{seed}_relevance-16-plus-46_{split}",
    ),
    ("lee2019", "full-62"): ("overfit_study", "lee2019_sub-{s:02d}_seed-{seed}_full-62{suffix}"),
}
PREDICTIONS = {
    "L1": {
        "claim": "Training four times as long shrinks the cost of the 46 noise inputs",
        "contrast": "Lee2019 final models, next day: (relevance-16-plus-46 minus relevance-16) at 1,900 epochs minus "
        "the same difference at 475 epochs (this study's checkpoints)",
        "test": "interval lower bound above 0",
    },
    "L2": {
        "claim": "Trained four times as long, all 62 electrodes beat the 16 most relevant",
        "contrast": "Lee2019 final models, next day, 1,900 epochs: full-62 minus relevance-16",
        "test": "interval lower bound above 0",
    },
    "L3": {
        "claim": "With noise inputs, held-out accuracy of the training day is still rising after epoch 475",
        "contrast": "Lee2019 relevance-16-plus-46 fold models (session 1 only): mean held-out accuracy over epochs "
        "1,800-1,900 minus the mean over epochs 425-525",
        "test": "interval lower bound above 0 (the plan's 'above 0', read as for L1 and L2)",
    },
}


def earlier_model(job: dict):
    study, pattern = EARLIER[(job["dataset"], job["condition"])]
    suffix = "" if job["split"] == "final" else f"_{job['split']}"
    name = pattern.format(s=job["subject"], seed=job["seed"], split=job["split"], suffix=suffix)
    return MODELS_DIR / "supplementary" / study / f"{name}.pt"


def checkpoint(job: dict, epoch: int):
    return STUDY.models / f"{job['name']}_ep-{epoch}.pt"


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


def model_inputs(job: dict) -> list[int]:
    return [*job["channels"], *(-(i + 1) for i in range(job["added"]))]


def plan_jobs(epochs: int) -> list[dict]:
    cost_jobs = json.loads(COST_PLAN.read_text())["jobs"]
    relevance = {(j["dataset"], j["subject"], j["split"]): j["channels"] for j in cost_jobs}
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
                    "name": f"{dataset}_sub-{subject:02d}_seed-{seed}_{condition}_{split}",
                    "dataset": dataset,
                    "subject": subject,
                    "seed": seed,
                    "split": split,
                    "condition": condition,
                    "added": added,
                    "channels": channels,
                    "checkpoints": [m * epochs for m in MULTIPLES],
                }
                for seed in seeds
            )
    return jobs


def create_plan() -> None:
    epochs = frozen_config()["epochs"]
    plan = STUDY.create(
        plan_jobs(epochs),
        purpose="Test whether the cost of extra inputs, and the failure of all electrodes to beat the best 16, come "
        "from training too briefly: the frozen recipe run four times as long, with checkpoints at 1, 2 and 4 times "
        "the frozen budget.",
        claim_boundary="Exploratory for BCI IV 2a, planned after its test results and the input-cost and "
        "noise-mechanism results were known. For Lee2019 the predictions are fixed before any of these models is "
        "scored on Lee2019 session 2; three predictions, no correction for multiplicity. The frozen 2a "
        "configuration (475 epochs) does not change.",
        lee2019_new_models_scored_before_plan=False,
        predictions=PREDICTIONS,
        frozen_epochs=epochs,
        checkpoints=[m * epochs for m in MULTIPLES],
        datasets={
            name: {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}
            for name, spec in DATASETS.items()
        },
        earlier={f"{d}|{c}": list(v) for (d, c), v in EARLIER.items()},
        noise="mibci.noise_mechanism.appended_noise draw 0, the input-cost study's (largest k drawn, cut to k)",
        analysis="seeds (and folds) averaged within each participant; 95% participant bootstrap (10,000, seed 690); "
        "BCI IV 2a on artifact-free session-2 trials, Lee2019 on all; 2a also with the seed error",
        input_cost_protocol_sha256=sha256(COST_PLAN),
    )
    counts: dict[str, int] = {}
    for job in plan["jobs"]:
        key = f"{job['dataset']} {'final' if job['split'] == 'final' else 'folds'}"
        counts[key] = counts.get(key, 0) + 1
    print(f"{len(plan['jobs'])} jobs: {counts}")


def compare_with_earlier(job: dict, state: dict) -> dict:
    """Whether the 475-epoch checkpoint has the earlier model's weights exactly."""
    path = earlier_model(job)
    if not path.exists():
        return {"earlier_model": str(path.name), "found": False}
    earlier = torch.load(path, weights_only=True)["state_dict"]  # noqa: TID251  only the weights are compared
    if set(earlier) != set(state):
        return {"earlier_model": str(path.name), "found": True, "identical": False, "keys_differ": True}
    diff = max(float((state[k].double() - earlier[k].double()).abs().max()) for k in state)
    return {"earlier_model": str(path.name), "found": True, "identical": diff == 0.0, "max_abs_diff": diff}


def train_job(job: dict) -> None:
    X, y, runs = day1(job["dataset"], job["subject"])
    train, held = overfit.split_masks(runs, job["split"], job["dataset"])
    inputs = inputs_for(job, X, day=1)
    scaler = ChannelStandardizer().fit(inputs[train])
    epochs = max(job["checkpoints"])
    cfg = TrainConfig(epochs=epochs, seed=job["seed"])
    set_seed(cfg.seed)
    classes = DATASETS[job["dataset"]]["classes"]
    model = build_eegnet(inputs.shape[1], n_classes=classes)
    device = device_for(job["subject"])
    X_fit = scaler.transform(inputs[train])
    val = None if held is None else (scaler.transform(inputs[held]), y[held])
    result = fit(model, X_fit, y[train], cfg, val=val, checkpoints=tuple(job["checkpoints"]), device=device)
    meta = {k: job[k] for k in ("dataset", "subject", "seed", "split", "condition", "added")}
    meta |= {"device": device}
    STUDY.models.mkdir(parents=True, exist_ok=True)
    at = {}
    for epoch in job["checkpoints"]:
        net = build_eegnet(inputs.shape[1], n_classes=classes)
        net.load_state_dict(result["checkpoints"][epoch])
        path = checkpoint(job, epoch)
        replication.save(path, net, model_inputs(job), scaler, epochs=epoch, **meta)
        at[str(epoch)] = {"train_acc": scores(predict(net, X_fit), y[train])["acc"], "model_sha256": sha256(path)}
        if val is not None:
            logits = predict(net, val[0])
            at[str(epoch)] |= {"within_day": scores(logits, val[1]), "within_day_pred": logits.argmax(1).tolist()}
    history = {k: [round(v, 5) for v in values] for k, values in result["history"].items() if values}
    record = {
        **meta,
        "name": job["name"],
        "epochs": epochs,
        "n_train": int(train.sum()),
        "n_held": int(held.sum()) if held is not None else 0,
        "train_time_s": result["train_time_s"],
        "checkpoints": at,
        "history": history,
        "earlier_475": compare_with_earlier(job, result["checkpoints"][min(job["checkpoints"])]),
        "protocol_sha256": sha256(STUDY.protocol_path),
    }
    write_json(STUDY.record("runs", job), record)


def evaluate_job(job: dict) -> None:
    torch.set_num_threads(1)
    if job["dataset"] == "bnci2a":
        ses = data.load_session(job["subject"], 2, final_evaluation=True)
        clean = ~ses["artifact"]
    else:
        ses = lee2019.load_session(job["subject"], 2, final_evaluation=True)
        clean = np.ones(len(ses["y"]), bool)
    y = ses["y"].astype(np.int64)
    raw = inputs_for(job, ses["X"], day=2)
    at = {}
    for epoch in job["checkpoints"]:
        path = checkpoint(job, epoch)
        model, scaler, bundle = replication.load(path)
        if bundle["channels"] != model_inputs(job) or bundle["epochs"] != epoch:
            raise RuntimeError(f"Bundle differs from the plan: {path}")
        logits = predict(model, scaler.transform(raw))
        at[str(epoch)] = {
            "model_sha256": sha256(path),
            "primary": scores(logits[clean], y[clean]),
            "secondary": scores(logits, y),
            "pred": logits.argmax(1).tolist(),
        }
    write_json(
        STUDY.record("test", job),
        {**{k: v for k, v in job.items() if k != "channels"}, "protocol_sha256": sha256(STUDY.protocol_path), "at": at},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "train", "evaluate"], required=True)
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
            lambda j: [checkpoint(j, e) for e in j["checkpoints"]],
            lambda j: {"subject": j["subject"], "seed": j["seed"], "epochs": max(j["checkpoints"])},
        )
        run_pool(train_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)
        return
    finals = [j for j in jobs if j["split"] == "final"]
    require_files([checkpoint(j, e) for j in finals for e in j["checkpoints"]], "long-training evaluation inputs")
    pending = STUDY.pending(finals, "test", lambda j: [checkpoint(j, e) for e in j["checkpoints"]])
    run_pool(evaluate_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
