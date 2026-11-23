"""Post-test study: a direct test of the overfitting explanation for "16 electrodes beat 22".

On BCI IV 2a, relevance-16 decodes the next day 1.3 points better than full-22 (ten seeds). The
explanation offered so far: the six dropped electrodes carry little information, and with a few
hundred training trials the network fits patterns in them that do not carry over. That was
consistent with the data but never tested. This study makes the explanation's predictions
checkable, on BCI IV 2a (9 people, seeds 0-9) and Lee 2019 (54 people, seeds 0-2), where the
reduced montage is relevance-45 of 62 (the share 16 of 22 keeps, as in the replication study):

  full, reduced            the two montages, retrained here with the frozen recipe on Longleaf
  reduced-plus-noise       full montage, the dropped electrodes replaced by band-limited Gaussian
                           noise (mibci.overfit.noise_channels): inputs that carry nothing
  reduced-plus-shuffled    full montage, the dropped electrodes' own signals taken from another
                           trial of the same day: real EEG, unrelated to the trial
  full-chdrop, reduced-chdrop   trained with channel dropout p = 0.2 (regularization)
  fold and rfold splits    full and reduced trained on whole runs of the training day (2a: 4 or 2
                           of 6 runs; Lee2019: 1 of 2), electrodes ranked on those runs only;
                           each model is scored on the held-out runs of the training day and on
                           the next day, which separates fitting the trials from fitting the day

Predictions of the explanation (PREDICTIONS) and their tests are fixed in the plan. For Lee2019
they are prospective: no model of the new conditions had been scored on its session 2 when the
plan was written. For BCI IV 2a the same contrasts are exploratory, because the 16-versus-22
result motivated them.

Stages: plan (session-1 rankings for every split, jobs), train, evaluate.
Tables: scripts/analyze_overfit_study.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mibci import data, lee2019, overfit, replication
from mibci.artifacts import require_files, sha256, write_json
from mibci.experiment import device_for, run_pool
from mibci.layout import FINAL_DIR, VAL_DIR, frozen_config
from mibci.model import build_eegnet
from mibci.paths import RESULTS_DIR
from mibci.study import Study, shard
from mibci.train import ChannelStandardizer, TrainConfig, fit, predict, scores, set_seed

STUDY = Study(
    "overfit_study",
    "scripts/overfit_study.py",
    sources=["src/mibci/overfit.py", "src/mibci/lee2019.py", "src/mibci/replication.py", "src/mibci/selection.py",
             "src/mibci/splits.py", "src/mibci/study.py"],
)  # fmt: skip
REPLICATION_PLAN = RESULTS_DIR / "supplementary" / "replication_study" / "protocol.json"
DATASETS = {
    "bnci2a": {"seeds": tuple(range(10)), "full": 22, "reduced": 16, "classes": 4,
               "splits": ("fold-0", "fold-1", "fold-2", "rfold-0", "rfold-1", "rfold-2")},
    "lee2019": {"seeds": (0, 1, 2), "full": 62, "reduced": 45, "classes": 2, "splits": ("fold-0", "fold-1")},
}  # fmt: skip
H = ("seeds averaged within each participant; 95% participant bootstrap (10,000, seed 690); BCI IV 2a scored on "
     "artifact-free session-2 trials (the frozen primary analysis), Lee2019 on all session-2 trials")  # fmt: skip
PREDICTIONS = {
    "P1": {"claim": "Less training data widens the full montage's deficit",
           "contrast": "(reduced minus full) of models trained on part of the training day minus (reduced minus full) "
           "of models trained on all of it, both scored on the next day. Lee2019: fold models (one run, 100 trials) "
           "against final models (both runs). BCI IV 2a: fold (4 runs) and rfold (2 runs) against final (6 runs)",
           "test": "Lee2019: interval lower bound above 0"},
    "P2": {"claim": "The deficit also appears on held-out trials of the training day, not only on the next day",
           "contrast": "reduced minus full on the held-out run(s) of the training day, fold models",
           "test": "Lee2019: interval lower bound above 0"},
    "P3": {"claim": "Inputs that carry nothing hurt: noise in place of the dropped electrodes lowers accuracy",
           "contrast": "reduced-plus-noise minus reduced, next day", "test": "Lee2019: interval upper bound below 0"},
    "P4": {"claim": "Channel dropout helps the full montage more than the reduced one",
           "contrast": "(full-chdrop minus full) minus (reduced-chdrop minus reduced), next day",
           "test": "Lee2019: interval lower bound above 0"},
}  # fmt: skip
DESCRIPTIVE = [
    "reduced-plus-shuffled minus reduced, and minus full (real EEG that is unrelated to the trial)",
    "full-chdrop minus reduced (does regularization make dropping electrodes unnecessary?)",
    "(reduced minus full) on the next day minus (reduced minus full) within the training day, same fold models",
    "training-set accuracy, held-out accuracy within the day and next-day accuracy of full and reduced",
    "every contrast for BCI IV 2a, with the seed error of each mean (2a has ten seeds)",
]


def day1(dataset: str, subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Training-day trials (2a: artifact-free only, as in every training run), labels and run index."""
    if dataset == "bnci2a":
        ses = data.load_session(subject, 1)
        keep = ~ses["artifact"]
        return ses["X"][keep], ses["y"][keep].astype(np.int64), ses["run"][keep]
    ses = lee2019.load_session(subject, 1)
    return ses["X"], ses["y"].astype(np.int64), ses["run"]


def reduced_lists(dataset: str, subject: int) -> dict[str, list[int]]:
    """The reduced montage for each split: archived for the final models, ranked anew for the folds.

    Each fold's electrodes are ranked on that fold's training runs only, with the frozen relevance
    rule; the archived rankings are reproduced as a check (2a validation folds and final models,
    Lee2019 replication plan)."""
    spec = DATASETS[dataset]
    X, y, runs = day1(dataset, subject)
    final_order = overfit.relevance_order(X, y)
    if dataset == "bnci2a":
        archived = json.loads((FINAL_DIR / "rankings" / f"sub-{subject:02d}_seed-0.json").read_text())
        lists = {"final": archived["subsets"]["relevance-16"]}
    else:
        archived = json.loads(REPLICATION_PLAN.read_text())["rankings"][str(subject)]["order"]
        lists = {"final": sorted(archived[: spec["reduced"]])}
    if sorted(final_order[: spec["reduced"]]) != lists["final"]:
        raise RuntimeError(f"{dataset} participant {subject}: the relevance rule no longer reproduces the archive")
    for split in spec["splits"]:
        train, _ = overfit.split_masks(runs, split, dataset)
        lists[split] = sorted(overfit.relevance_order(X[train], y[train])[: spec["reduced"]])
        if dataset == "bnci2a" and split.startswith("fold"):
            stored = json.loads((VAL_DIR / "rankings" / f"sub-{subject:02d}_{split}.json").read_text())
            if stored["subsets"]["relevance-16"] != lists[split]:
                raise RuntimeError(f"participant {subject} {split}: differs from the validation ranking")
    return lists


def plan_jobs(lists: dict[tuple[str, int], dict[str, list[int]]]) -> list[dict]:
    jobs = []
    for (dataset, subject), reduced in lists.items():
        spec = DATASETS[dataset]
        full, k = list(range(spec["full"])), spec["reduced"]
        dropped = sorted(set(full) - set(reduced["final"]))
        f, r = f"full-{spec['full']}", f"relevance-{k}"
        # condition: (split, channels, replaced, replacement, channel dropout)
        specs = {
            f: ("final", full, [], None, 0.0),
            r: ("final", reduced["final"], [], None, 0.0),
            f"{r}-plus-noise": ("final", full, dropped, "noise", 0.0),
            f"{r}-plus-shuffled": ("final", full, dropped, "shuffled", 0.0),
            f"{f}-chdrop": ("final", full, [], None, overfit.DROPOUT_P),
            f"{r}-chdrop": ("final", reduced["final"], [], None, overfit.DROPOUT_P),
        }
        for split in spec["splits"]:
            specs[f"{f}_{split}"] = (split, full, [], None, 0.0)
            specs[f"{r}_{split}"] = (split, reduced[split], [], None, 0.0)
        for key, (split, channels, replaced, replacement, dropout) in specs.items():
            condition = key.split("_")[0]
            jobs.extend(
                {"name": f"{dataset}_sub-{subject:02d}_seed-{seed}_{key}", "dataset": dataset, "subject": subject,
                 "seed": seed, "split": split, "condition": condition, "channels": channels, "replaced": replaced,
                 "replacement": replacement, "dropout": dropout}
                for seed in spec["seeds"]
            )  # fmt: skip
    return jobs


def create_plan() -> None:
    lists = {("bnci2a", s): reduced_lists("bnci2a", s) for s in data.SUBJECTS}
    lists |= {("lee2019", s): reduced_lists("lee2019", s) for s in lee2019.SUBJECTS}
    plan = STUDY.create(
        plan_jobs(lists),
        purpose="Test the explanation that the full montage decodes the next day worse because, with a few hundred "
        "training trials, its extra low-information electrodes let the network fit patterns that do not carry over: "
        "less data, noise or shuffled signals in place of the dropped electrodes, channel dropout, and held-out runs "
        "of the training day against the next day.",
        claim_boundary="Exploratory for BCI IV 2a: planned after its 16-versus-22 test result was known, which "
        "motivated it. For Lee2019 the predictions and tests were fixed in this plan before any model of the new "
        "conditions was scored on Lee2019 session 2, but Lee2019's own full-62 and relevance-45 test results were "
        "already known from the replication study. Four predictions, no correction for multiplicity. It explains a "
        "result; it does not change the frozen 2a configuration.",
        lee2019_new_conditions_scored_before_plan=False,
        predictions=PREDICTIONS,
        descriptive=DESCRIPTIVE,
        analysis=H,
        datasets={
            name: {**spec, "seeds": list(spec["seeds"]), "splits": list(spec["splits"])}
            for name, spec in DATASETS.items()
        },
        epochs=frozen_config()["epochs"],
        dropout_p=overfit.DROPOUT_P,
        noise="white Gaussian noise through the preprocessing band-pass (causal 4th-order Butterworth 4-38 Hz at "
        "250 Hz, 2-s warm-up), unit variance, drawn once per dataset, participant and day (mibci.overfit)",
        shuffled="the dropped electrodes of each trial copied from another trial of the same day, a fixed derangement "
        "per dataset, participant and day (mibci.overfit)",
        reduced_montage="BCI IV 2a relevance-16 (archived final rankings); Lee2019 relevance-45 (replication plan); "
        "fold models: relevance order of the fold's own training runs",
        reduced_lists={f"{d}_{s}": v for (d, s), v in lists.items()},
    )
    counts: dict[str, int] = {}
    for job in plan["jobs"]:
        key = f"{job['dataset']} {job['condition']} {job['split'].split('-')[0]}"
        counts[key] = counts.get(key, 0) + 1
    print(f"{len(plan['jobs'])} jobs: {counts}")


def train_job(job: dict, epochs: int) -> None:
    X, y, runs = day1(job["dataset"], job["subject"])
    train, held = overfit.split_masks(runs, job["split"], job["dataset"])
    X_train = overfit.replace(X[train], job, day=1)[:, job["channels"]]
    scaler = ChannelStandardizer().fit(X_train)
    cfg = TrainConfig(epochs=epochs, seed=job["seed"])
    set_seed(cfg.seed)
    model = build_eegnet(len(job["channels"]), n_classes=DATASETS[job["dataset"]]["classes"])
    device = device_for(job["subject"])
    X_fit = scaler.transform(X_train)
    if job["dropout"]:
        result = overfit.fit_channel_dropout(model, X_fit, y[train], cfg, p=job["dropout"], device=device)
    else:
        result = fit(model, X_fit, y[train], cfg, device=device)
    path = STUDY.model(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"dataset": job["dataset"], "subject": job["subject"], "seed": job["seed"], "split": job["split"],
            "condition": job["condition"], "epochs": epochs, "device": device}  # fmt: skip
    replication.save(path, model, job["channels"], scaler, **meta)
    record = {**meta, "name": job["name"], "n_train": int(train.sum()), "train_time_s": result["train_time_s"],
              "train_acc": scores(predict(model, X_fit), y[train])["acc"], "model_sha256": sha256(path),
              "protocol_sha256": sha256(STUDY.protocol_path)}  # fmt: skip
    if held is not None:
        logits = predict(model, scaler.transform(X[held][:, job["channels"]]))
        record |= {"n_held": int(held.sum()), "within_day": scores(logits, y[held]),
                   "within_day_pred": logits.argmax(1).tolist()}  # fmt: skip
    write_json(STUDY.record("runs", job), record)


def evaluate_job(job: dict) -> None:
    torch.set_num_threads(1)
    path = STUDY.model(job)
    model, scaler, bundle = replication.load(path)
    if bundle["channels"] != job["channels"]:
        raise RuntimeError(f"Bundle channels differ from the plan: {path}")
    if job["dataset"] == "bnci2a":
        ses = data.load_session(job["subject"], 2, final_evaluation=True)
        clean, extra = ~ses["artifact"], {"artifact": ses["artifact"].astype(int).tolist()}
    else:
        ses = lee2019.load_session(job["subject"], 2, final_evaluation=True)
        clean, extra = np.ones(len(ses["y"]), bool), {"run": ses["run"].tolist()}
    y = ses["y"].astype(np.int64)
    logits = predict(model, scaler.transform(overfit.replace(ses["X"], job, day=2)[:, job["channels"]]))
    write_json(STUDY.record("test", job), {
        **job, "model_sha256": sha256(path), "protocol_sha256": sha256(STUDY.protocol_path),
        "primary": scores(logits[clean], y[clean]), "secondary": scores(logits, y),
        "pred": logits.argmax(1).tolist(), "y": y.tolist(), **extra,
    })  # fmt: skip


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
        require_files([STUDY.model(j) for j in jobs], "overfit-study evaluation inputs (run --stage train first)")
        pending = STUDY.pending(jobs, "test", lambda j: [STUDY.model(j)])
        run_pool(evaluate_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
