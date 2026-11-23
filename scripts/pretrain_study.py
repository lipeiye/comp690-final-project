"""Post-test study: pretrained on other people's training days, do all electrodes finally beat 16?

Three times more of a person's own data did not shrink the cost of extra inputs
(scripts/input_cost_study.py). This study uses many other people's data instead, the way a product
would: everyone wears the same electrodes, a network is pretrained on other people and then
fine-tuned on the new person (mibci.pretrain). Only training days are used for training, and every
person's inputs are standardized with that person's own training-day mean and standard deviation.

  BCI IV 2a  for each target participant, pretraining on the other 8; seeds 0-4
  Lee2019    six groups of nine by number; for each group, pretraining on the other 45; seeds 0-2

Electrode configurations (one set of inputs for everyone in a pretraining run): all electrodes
(full-22, full-62); group-16, the 16 electrodes with the best mean relevance rank among the
pretraining participants (the target is never looked at); group-16 plus noise inputs (2a 6,
Lee2019 46; each person's own draw, mibci.noise_mechanism.appended_noise draw 0). Three models per
target, configuration and seed, all scored on the target's next day:

  scratch   the target's training day only, the frozen recipe (475 epochs); for all electrodes the
            earlier models of scripts/overfit_study.py, for the other two new models
  pretrain  the pretrained network alone (Adam, lr 1e-3, batch 64), at the epoch the project's budget
            rule picks: the target participants' training-day accuracy after every pretraining epoch,
            all-electrode configuration, averaged over targets and seeds, smoothed with a centred
            moving average; the earliest checkpoint within 0.5 points of the smoothed maximum
  finetune  the pretrained network trained on the target's training day for 100 epochs at lr 1e-4 (the
            project's fine-tuning budget, used for pruning and the matched-budget control)

Lee2019 predictions (PREDICTIONS) are fixed before any of these models is scored on Lee2019 session 2;
BCI IV 2a is exploratory.

Stages: plan; scratch and pretrain (independent); budget (session 1 only); finetune; evaluate.
Tables: scripts/analyze_pretrain_study.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mibci import data, lee2019, noise_mechanism, pretrain, replication
from mibci.artifacts import sha256, write_json
from mibci.experiment import device_for, run_pool
from mibci.layout import frozen_config
from mibci.model import build_eegnet
from mibci.paths import RESULTS_DIR
from mibci.study import Study, shard
from mibci.train import ChannelStandardizer, TrainConfig, fit, predict, scores, set_seed

STUDY = Study(
    "pretrain_study",
    "scripts/pretrain_study.py",
    sources=[
        "src/mibci/pretrain.py",
        "src/mibci/noise_mechanism.py",
        "src/mibci/overfit.py",
        "src/mibci/lee2019.py",
        "src/mibci/replication.py",
        "src/mibci/study.py",
    ],
)
SUPP = RESULTS_DIR / "supplementary"
ORDERS = SUPP / "electrode_data_study" / "protocol.json"
KEPT = 16
TOLERANCE = 0.005  # the frozen budget rule's 0.5 points
PRETRAIN = {"lr": 1e-3, "batch_size": 64}
DATASETS = {
    "bnci2a": {
        "classes": 4,
        "electrodes": 22,
        "added": 6,
        "largest": 16,  # the input-cost study drew 16 noise inputs per person and day and cut to k
        "seeds": (0, 1, 2, 3, 4),
        "max_epochs": 300,  # pretraining limit and checkpoint interval, fixed before the plan
        "interval": 5,
        "window": 5,
    },
    "lee2019": {
        "classes": 2,
        "electrodes": 62,
        "added": 46,
        "largest": 46,
        "seeds": (0, 1, 2),
        "max_epochs": 75,
        "interval": 3,
        "window": 3,
    },
}
PREDICTIONS = {
    "T1": {
        "claim": "With all 62 electrodes, pretraining plus fine-tuning beats training from scratch",
        "contrast": "Lee2019, next day: full-62 finetune minus full-62 scratch (scripts/overfit_study.py)",
        "test": "interval lower bound above 0",
    },
    "T2": {
        "claim": "After pretraining and fine-tuning, the 46 noise inputs cost less than when training from scratch",
        "contrast": "Lee2019, next day: (group-16-plus-46 minus group-16) after finetune minus the same from scratch",
        "test": "interval lower bound above 0",
    },
    "T3": {
        "claim": "After pretraining and fine-tuning, all 62 electrodes beat the shared 16",
        "contrast": "Lee2019, next day: full-62 finetune minus group-16 finetune",
        "test": "interval lower bound above 0",
    },
}


def configs(dataset: str) -> dict[str, int]:
    spec = DATASETS[dataset]
    return {f"full-{spec['electrodes']}": 0, f"group-{KEPT}": 0, f"group-{KEPT}-plus-{spec['added']}": spec["added"]}


def subjects(dataset: str) -> list[int]:
    return list(data.SUBJECTS) if dataset == "bnci2a" else list(lee2019.SUBJECTS)


def day(dataset: str, subject: int, which: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One day's trials and labels, and the clean mask (2a training day: artifact-free trials only)."""
    if dataset == "bnci2a":
        ses = data.load_session(subject, which, final_evaluation=which == 2)
        clean = ~ses["artifact"]
        if which == 1:
            return ses["X"][clean], ses["y"][clean].astype(np.int64), np.ones(int(clean.sum()), bool)
        return ses["X"], ses["y"].astype(np.int64), clean
    ses = lee2019.load_session(subject, which, final_evaluation=which == 2)
    return ses["X"], ses["y"].astype(np.int64), np.ones(len(ses["y"]), bool)


def inputs_for(job: dict, subject: int, X: np.ndarray, which: int) -> np.ndarray:
    """The configuration's electrodes, then the person's own noise inputs for that day."""
    out = X[:, job["channels"]]
    if not job["added"]:
        return out
    spec = DATASETS[job["dataset"]]
    noise = noise_mechanism.appended_noise(
        len(X), X.shape[2], job["added"], spec["largest"], job["dataset"], subject, which, 0
    )
    return np.concatenate([out, noise], axis=1)


def model_inputs(job: dict) -> list[int]:
    return [*job["channels"], *(-(i + 1) for i in range(job["added"]))]


def pretraining_groups(dataset: str) -> list[dict]:
    people = subjects(dataset)
    if dataset == "bnci2a":
        return [{"tag": f"loo-{t:02d}", "targets": [t], "pretrain": [s for s in people if s != t]} for t in people]
    groups = sorted({pretrain.lee_group(s) for s in people})
    return [
        {
            "tag": f"group-{g}",
            "targets": [s for s in people if pretrain.lee_group(s) == g],
            "pretrain": [s for s in people if pretrain.lee_group(s) != g],
        }
        for g in groups
    ]


def plan_jobs(orders: dict) -> list[dict]:
    jobs = []
    for dataset, spec in DATASETS.items():
        for group in pretraining_groups(dataset):
            shared = pretrain.group_electrodes(
                [orders[f"{dataset}_{s}"]["final"] for s in group["pretrain"]], spec["electrodes"], KEPT
            )
            for config, added in configs(dataset).items():
                channels = list(range(spec["electrodes"])) if config.startswith("full") else sorted(shared)
                base = {"dataset": dataset, "config": config, "added": added, "channels": channels}
                for seed in spec["seeds"]:
                    pre = f"{dataset}_pre-{group['tag']}_{config}_seed-{seed}"
                    jobs.append(
                        {
                            "name": pre,
                            "kind": "pretrain",
                            **base,
                            "seed": seed,
                            "group": group["tag"],
                            "pretrain_subjects": group["pretrain"],
                            "targets": group["targets"],
                            "group_order": shared,
                            "max_epochs": spec["max_epochs"],
                            "interval": spec["interval"],
                        }
                    )
                    for t in group["targets"]:
                        stem = f"{dataset}_sub-{t:02d}_seed-{seed}_{config}"
                        person = {**base, "seed": seed, "subject": t, "group": group["tag"], "pretrained": pre}
                        jobs.append({"name": f"{stem}_pretrain", "kind": "pretrain-only", **person})
                        jobs.append({"name": f"{stem}_finetune", "kind": "finetune", **person})
                        if not config.startswith("full"):
                            jobs.append({"name": f"{stem}_scratch", "kind": "scratch", **person})
    return jobs


def create_plan() -> None:
    cfg = frozen_config()
    plan = STUDY.create(
        plan_jobs(json.loads(ORDERS.read_text())["orders"]),
        purpose="Test whether more data, from other people, lets all electrodes beat 16: pretraining on other "
        "people's training days, alone and followed by fine-tuning on the target's training day, against training "
        "from scratch, with all electrodes, a shared 16 and the shared 16 plus noise inputs.",
        claim_boundary="Exploratory for BCI IV 2a, planned after its test results and the input-cost results were "
        "known. For Lee2019 the predictions are fixed before any of these models is scored on Lee2019 session 2; "
        "three predictions, no correction for multiplicity. group-16 is one electrode set shared by everyone in a "
        "pretraining run, not each person's own 16. It does not change the frozen 2a configuration.",
        lee2019_new_models_scored_before_plan=False,
        predictions=PREDICTIONS,
        datasets={
            name: {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}
            for name, spec in DATASETS.items()
        },
        configurations={d: configs(d) for d in DATASETS},
        pretraining={
            **PRETRAIN,
            "data": "the pretraining participants' whole training days, each standardized with their own "
            "training-day statistics",
        },
        budget_rule="the target participants' training-day accuracy after every pretraining epoch, all-electrode "
        "configuration, averaged over pretraining runs (2a: targets x seeds; Lee2019: groups x seeds, nine equal "
        "targets pooled), smoothed with a centred moving average of `window` epochs; the earliest saved epoch "
        f"(every `interval`) within {TOLERANCE * 100} points of the smoothed maximum; session 1 only "
        "(mibci.pretrain.budget_epoch, which reproduces the frozen 475 epochs)",
        finetune={
            "epochs": cfg["pruning"]["finetune_epochs"],
            "lr": cfg["pruning"]["finetune_lr"],
            "batch_size": 64,
            "note": "the project's fine-tuning budget (pruning, matched budget)",
        },
        scratch={
            "epochs": cfg["epochs"],
            "full": "scripts/overfit_study.py full-22 / full-62 final models, same seeds",
        },
        analysis="seeds averaged within each participant; 95% participant bootstrap (10,000, seed 690); BCI IV 2a on "
        "artifact-free session-2 trials, Lee2019 on all; 2a also with the seed error",
        epochs=cfg["epochs"],
        orders_source_sha256=sha256(ORDERS),
    )
    counts: dict[str, int] = {}
    for job in plan["jobs"]:
        key = f"{job['dataset']} {job['kind']}"
        counts[key] = counts.get(key, 0) + 1
    print(f"{len(plan['jobs'])} jobs: {counts}")


def checkpoints_path(name: str):
    return STUDY.models / f"{name}_checkpoints.pt"


def pretrain_job(job: dict) -> None:
    spec = DATASETS[job["dataset"]]
    people = []
    for s in job["pretrain_subjects"]:
        X, y, _ = day(job["dataset"], s, 1)
        people.append((inputs_for(job, s, X, 1), y))
    X_pool, y_pool = pretrain.pooled(people)
    del people
    targets = []  # (target, standardized training-day inputs, labels)
    for t in job["targets"]:
        X, y, _ = day(job["dataset"], t, 1)
        X_std, _ = pretrain.standardized(inputs_for(job, t, X, 1))
        targets.append((t, X_std, y))
    X_val = np.concatenate([x for _, x, _ in targets])
    y_val = np.concatenate([y for _, _, y in targets])
    saved = tuple(range(job["interval"], job["max_epochs"] + 1, job["interval"]))
    cfg = TrainConfig(epochs=job["max_epochs"], seed=job["seed"], **PRETRAIN)
    set_seed(cfg.seed)
    model = build_eegnet(X_pool.shape[1], n_classes=spec["classes"])
    device = device_for(job["targets"][0])
    result = fit(model, X_pool, y_pool, cfg, val=(X_val, y_val), checkpoints=saved, device=device)
    path = checkpoints_path(job["name"])
    pretrain.save_checkpoints(path, result["checkpoints"])
    per_target = {}
    net = build_eegnet(X_pool.shape[1], n_classes=spec["classes"])
    for epoch in saved:
        net.load_state_dict(result["checkpoints"][epoch])
        per_target[str(epoch)] = {str(t): scores(predict(net, x), y)["acc"] for t, x, y in targets}
    write_json(
        STUDY.record("pretrain", job),
        {
            **{k: job[k] for k in ("name", "dataset", "group", "config", "seed", "added", "max_epochs", "interval")},
            "device": device,
            "n_train": len(y_pool),
            "n_val": len(y_val),
            "train_time_s": result["train_time_s"],
            "history": {k: [round(v, 5) for v in vals] for k, vals in result["history"].items()},
            "per_target_acc": per_target,
            "checkpoints_sha256": sha256(path),
            "protocol_sha256": sha256(STUDY.protocol_path),
        },
    )


def train_person(job: dict, start: dict | None, cfg: TrainConfig) -> None:
    """Scratch (start None) or fine-tuning (start: pretrained weights) on the target's whole training day."""
    X, y, _ = day(job["dataset"], job["subject"], 1)
    inputs = inputs_for(job, job["subject"], X, 1)
    X_fit, scaler = pretrain.standardized(inputs)
    model = build_eegnet(inputs.shape[1], n_classes=DATASETS[job["dataset"]]["classes"])
    if start is not None:
        model.load_state_dict(start)
    before = scores(predict(model, X_fit), y)["acc"] if start is not None else None
    set_seed(cfg.seed)
    device = device_for(job["subject"])
    result = fit(model, X_fit, y, cfg, device=device)
    path = STUDY.model(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: job[k] for k in ("dataset", "subject", "seed", "config", "kind", "added")}
    meta |= {"epochs": cfg.epochs, "lr": cfg.lr, "device": device}
    replication.save(path, model, model_inputs(job), scaler, **meta)
    write_json(
        STUDY.record("runs", job),
        {
            **meta,
            "name": job["name"],
            "n_train": len(y),
            "train_time_s": result["train_time_s"],
            "train_acc_before": before,
            "train_acc": scores(predict(model, X_fit), y)["acc"],
            "final_train_loss": result["history"]["train_loss"][-1],
            "model_sha256": sha256(path),
            "protocol_sha256": sha256(STUDY.protocol_path),
        },
    )


def scratch_job(job: dict, epochs: int) -> None:
    train_person(job, None, TrainConfig(epochs=epochs, seed=job["seed"]))


def budget_epochs() -> dict:
    return json.loads((STUDY.out / "budget.json").read_text())["budget"]


def finetune_job(job: dict, epochs: int, lr: float) -> None:
    start = pretrain.load_checkpoint(checkpoints_path(job["pretrained"]), budget_epochs()[job["dataset"]]["epochs"])
    train_person(job, start, TrainConfig(epochs=epochs, lr=lr, seed=job["seed"]))


def budget(plan: dict) -> None:
    out = {}
    for dataset, spec in DATASETS.items():
        full = [
            j
            for j in plan["jobs"]
            if j["kind"] == "pretrain" and j["dataset"] == dataset and j["config"].startswith("full")
        ]
        missing = [j["name"] for j in full if not STUDY.record("pretrain", j).exists()]
        if missing:
            raise SystemExit(f"{len(missing)} pretraining runs are not finished, e.g. {missing[0]}")
        curves = np.array([json.loads(STUDY.record("pretrain", j).read_text())["history"]["val_acc"] for j in full])
        saved = range(spec["interval"], spec["max_epochs"] + 1, spec["interval"])
        out[dataset] = {"runs": len(full), **pretrain.budget_epoch(curves, saved, spec["window"], TOLERANCE)}
        print(dataset, {k: v for k, v in out[dataset].items() if k != "smoothed"})
    write_json(STUDY.out / "budget.json", {"rule": plan["budget_rule"], "budget": out})


def evaluate_job(job: dict) -> None:
    torch.set_num_threads(1)
    X2, y2, clean = day(job["dataset"], job["subject"], 2)
    if job["kind"] == "pretrain-only":
        epoch = budget_epochs()[job["dataset"]]["epochs"]
        X1, _, _ = day(job["dataset"], job["subject"], 1)
        scaler = ChannelStandardizer().fit(inputs_for(job, job["subject"], X1, 1))
        model = build_eegnet(len(model_inputs(job)), n_classes=DATASETS[job["dataset"]]["classes"])
        model.load_state_dict(pretrain.load_checkpoint(checkpoints_path(job["pretrained"]), epoch))
        model.eval()
        source = {"pretrain_epoch": epoch, "checkpoints_sha256": sha256(checkpoints_path(job["pretrained"]))}
    else:
        model, scaler, bundle = replication.load(STUDY.model(job))
        if bundle["channels"] != model_inputs(job):
            raise RuntimeError(f"Bundle channels differ from the plan: {STUDY.model(job)}")
        source = {"model_sha256": sha256(STUDY.model(job))}
    logits = predict(model, scaler.transform(inputs_for(job, job["subject"], X2, 2)))
    write_json(
        STUDY.record("test", job),
        {
            **{k: v for k, v in job.items() if k != "channels"},
            **source,
            "protocol_sha256": sha256(STUDY.protocol_path),
            "primary": scores(logits[clean], y2[clean]),
            "secondary": scores(logits, y2),
            "pred": logits.argmax(1).tolist(),
        },
    )


def ready(job: dict) -> bool:
    """Whether a job's model exists, so that evaluate can run before every model is trained."""
    if job["kind"] == "pretrain-only":
        return checkpoints_path(job["pretrained"]).exists() and (STUDY.out / "budget.json").exists()
    return STUDY.model(job).exists() and STUDY.record("runs", job).exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage", choices=["plan", "pretrain", "scratch", "budget", "finetune", "evaluate"], required=True
    )
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--subjects", nargs="+", type=int, help="only these targets (local smoke tests)")
    parser.add_argument("--shard", help="i/n: run only array task i of n (0-based)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1, help="torch threads per pretraining run")
    args = parser.parse_args()
    if args.stage == "plan":
        create_plan()
        return
    plan = STUDY.load()
    if args.stage == "budget":
        budget(plan)
        return

    def mine(kind: str) -> list[dict]:
        jobs = [j for j in plan["jobs"] if j["kind"] == kind and j["dataset"] in args.datasets]
        if args.subjects:
            jobs = [j for j in jobs if set(j.get("targets", [j.get("subject")])) & set(args.subjects)]
        return shard(jobs, args.shard)

    STUDY.log_environment(f"{args.stage}-{(args.shard or 'all').replace('/', 'of')}")
    if args.stage == "pretrain":
        pending = [j for j in mine("pretrain") if not STUDY.record("pretrain", j).exists()]
        run_pool(pretrain_job, [(j["name"], {"job": j}) for j in pending], args.workers, args.threads)
    elif args.stage == "scratch":
        pending = STUDY.pending(mine("scratch"), "runs", lambda j: [STUDY.model(j)])
        run_pool(scratch_job, [(j["name"], {"job": j, "epochs": plan["epochs"]}) for j in pending], args.workers, 1)
    elif args.stage == "finetune":
        ft = plan["finetune"]
        todo = [j for j in mine("finetune") if checkpoints_path(j["pretrained"]).exists()]
        pending = STUDY.pending(todo, "runs", lambda j: [STUDY.model(j)])
        kwargs = {"epochs": ft["epochs"], "lr": ft["lr"]}
        run_pool(finetune_job, [(j["name"], {"job": j, **kwargs}) for j in pending], args.workers, 1)
    else:
        todo = [j for kind in ("pretrain-only", "finetune", "scratch") for j in mine(kind)]
        waiting = [j for j in todo if not ready(j)]
        if waiting:
            print(f"{len(waiting)} jobs have no model yet (e.g. {waiting[0]['name']}); run evaluate again later")
        pending = [j for j in todo if ready(j) and not STUDY.record("test", j).exists()]
        run_pool(evaluate_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
