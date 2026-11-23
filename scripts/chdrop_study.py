"""Post-test study: why does channel dropout cost EEGNet so much, and does it buy tolerance of a failed electrode?

scripts/overfit_study.py trained with channel dropout (p = 0.2: each training example loses each
electrode with that probability) and lost 11 points on BCI IV 2a and 2.7 on Lee2019 the next day. The
explanation tested here (mibci.chdrop): with electrodes missing, a spatial filter cannot cancel what the
electrodes share, so its output varies more in training than at test; the batch norm after it stores
the training variance and divides by it at test, which shrinks the signal. If so, re-estimating the batch
norms on the training day with every electrode present (mibci.noise_mechanism.recalibrate) recovers much
of the loss.

  A  (no training) the archived final models of scripts/overfit_study.py with channel dropout (2a
     full-22-chdrop and relevance-16-chdrop, seeds 0-9; Lee2019 full-62-chdrop and relevance-45-chdrop, seeds
     0-2) and without (full-22, relevance-16; full-62, relevance-45), scored the next day as trained and with
     the batch norms re-estimated on the whole training day; per batch norm, stored over actual variance.
     Scoring as trained must reproduce the archived predictions.
  B  (training) channel dropout p = 0.05 and 0.1 for all electrodes (2a full-22: final seeds 0-9, fold-0/1/2
     seeds 0-4; Lee2019 full-62: final and fold-0/1, seeds 0-2), and p = 0.2 fold models, so that every p has
     held-out runs of the training day. Every model is scored as trained and with re-estimated batch norms.
     The fold models without dropout (scripts/overfit_study.py) are scored both ways too.
  C  (scoring) a failed electrode: every all-electrode final model on the next day with one electrode at its
     training mean, each in turn, and with three at once (five fixed sets), as trained and re-estimated.

Lee2019 predictions (PREDICTIONS) are fixed before any of these models is scored on Lee2019 session 2;
BCI IV 2a is exploratory. Stages: plan, train, reference (fold models without dropout; session 1),
evaluate. Tables: scripts/analyze_chdrop_study.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mibci import chdrop, data, lee2019, noise_mechanism, overfit, replication
from mibci.artifacts import sha256, write_json
from mibci.experiment import device_for, run_pool
from mibci.layout import frozen_config
from mibci.model import build_eegnet
from mibci.paths import MODELS_DIR, RESULTS_DIR
from mibci.study import Study, shard
from mibci.train import ChannelStandardizer, TrainConfig, predict, scores, set_seed

STUDY = Study(
    "chdrop_study",
    "scripts/chdrop_study.py",
    sources=[
        "src/mibci/chdrop.py",
        "src/mibci/noise_mechanism.py",
        "src/mibci/overfit.py",
        "src/mibci/lee2019.py",
        "src/mibci/replication.py",
        "src/mibci/splits.py",
        "src/mibci/study.py",
    ],
)
SUPP = RESULTS_DIR / "supplementary"
OVERFIT = SUPP / "overfit_study"
NEW_P = (0.05, 0.1)
ARCHIVED_P = 0.2
FAILURE = {"single": "every electrode in turn", "sets": 5, "size": 3, "seed": 694}
DATASETS = {
    "bnci2a": {
        "classes": 4,
        "electrodes": 22,
        "archived": ("full-22-chdrop", "relevance-16-chdrop", "full-22", "relevance-16"),
        "final_seeds": tuple(range(10)),
        "fold_seeds": (0, 1, 2, 3, 4),
        "folds": ("fold-0", "fold-1", "fold-2"),
    },
    "lee2019": {
        "classes": 2,
        "electrodes": 62,
        "archived": ("full-62-chdrop", "relevance-45-chdrop", "full-62", "relevance-45"),
        "final_seeds": (0, 1, 2),
        "fold_seeds": (0, 1, 2),
        "folds": ("fold-0", "fold-1"),
    },
}
PREDICTIONS = {
    "B1": {
        "claim": "Re-estimating the batch norms recovers next-day accuracy of the p = 0.2 models",
        "contrast": "Lee2019 full-62-chdrop final models (scripts/overfit_study.py), next day: re-estimated minus as "
        "trained; also the share of the loss recovered, (re-estimated minus as trained) over (full-62 without "
        "dropout minus as trained)",
        "test": "interval lower bound above 0",
    },
    "B2": {
        "claim": "p = 0.05 costs less than p = 0.2",
        "contrast": "Lee2019 all-electrode final models, next day, as trained: p = 0.05 minus p = 0.2 (also reported "
        "with re-estimated batch norms)",
        "test": "interval lower bound above 0",
    },
    "B3": {
        "claim": "A model trained with channel dropout loses less when one electrode fails",
        "contrast": "Lee2019 all-electrode final models, next day, one electrode at its training mean, averaged over "
        "the 62: (intact minus failed) of full-62 without dropout as trained, minus the same of the dropout model "
        "with re-estimated batch norms, at the p of 0.05, 0.1, 0.2 with the best re-estimated held-out accuracy of "
        "the training day (fold models)",
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


def day2(dataset: str, subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dataset == "bnci2a":
        ses = data.load_session(subject, 2, final_evaluation=True)
        return ses["X"], ses["y"].astype(np.int64), ~ses["artifact"]
    ses = lee2019.load_session(subject, 2, final_evaluation=True)
    return ses["X"], ses["y"].astype(np.int64), np.ones(len(ses["y"]), bool)


def overfit_name(dataset: str, subject: int, seed: int, condition: str, split: str) -> str:
    return f"{dataset}_sub-{subject:02d}_seed-{seed}_{condition}" + ("" if split == "final" else f"_{split}")


def plan_jobs() -> list[dict]:
    earlier = {j["name"]: j for j in json.loads((OVERFIT / "protocol.json").read_text())["jobs"]}
    jobs = []
    for dataset, spec in DATASETS.items():
        full = list(range(spec["electrodes"]))
        people = sorted({j["subject"] for j in earlier.values() if j["dataset"] == dataset})
        for s in people:
            for split in ("final", *spec["folds"]):
                seeds = spec["final_seeds"] if split == "final" else spec["fold_seeds"]
                ps = NEW_P if split == "final" else (*NEW_P, ARCHIVED_P)
                jobs.extend(
                    {
                        "name": f"{dataset}_sub-{s:02d}_seed-{seed}_full-{spec['electrodes']}-p{p}_{split}",
                        "kind": "train",
                        "dataset": dataset,
                        "subject": s,
                        "seed": seed,
                        "split": split,
                        "condition": f"full-{spec['electrodes']}-p{p}",
                        "p": p,
                        "channels": full,
                    }
                    for p in ps
                    for seed in seeds
                )
                for seed in seeds if split != "final" else ():
                    name = overfit_name(dataset, s, seed, f"full-{spec['electrodes']}", split)
                    jobs.append(
                        {
                            "name": f"{name}_reference",
                            "kind": "reference",
                            "earlier": name,
                            "dataset": dataset,
                            "subject": s,
                            "seed": seed,
                            "split": split,
                            "condition": f"full-{spec['electrodes']}",
                            "p": 0.0,
                            "channels": earlier[name]["channels"],
                        }
                    )
            for condition in spec["archived"]:
                for seed in spec["final_seeds"]:
                    name = overfit_name(dataset, s, seed, condition, "final")
                    jobs.append(
                        {
                            "name": f"{name}_archived",
                            "kind": "archived",
                            "earlier": name,
                            "dataset": dataset,
                            "subject": s,
                            "seed": seed,
                            "split": "final",
                            "condition": condition,
                            "p": earlier[name]["dropout"],
                            "channels": earlier[name]["channels"],
                        }
                    )
    return jobs


def create_plan() -> None:
    plan = STUDY.create(
        plan_jobs(),
        purpose="Test whether channel dropout costs EEGNet accuracy because the batch norms after the spatial filters "
        "store a training variance the test never sees, and whether dropout buys tolerance of a failed electrode.",
        claim_boundary="Exploratory for BCI IV 2a, planned after its test results and the overfitting-study results "
        "were known. For Lee2019 the predictions are fixed before any of the new models, and before any re-estimated "
        "or failed-electrode version of the archived models, is scored on Lee2019 session 2; the archived models' "
        "as-trained next-day scores were known. Three predictions, no correction for multiplicity.",
        lee2019_new_models_scored_before_plan=False,
        predictions=PREDICTIONS,
        new_p=list(NEW_P),
        archived_p=ARCHIVED_P,
        failure=FAILURE,
        recalibration="mibci.noise_mechanism.recalibrate: the batch norms after the spatial filters (bnorm_1, bnorm_2) "
        "re-estimated in one cumulative pass over the training trials with every electrode present, dropout off",
        datasets={
            name: {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}
            for name, spec in DATASETS.items()
        },
        analysis="seeds (and folds) averaged within each participant; 95% participant bootstrap (10,000, seed 690); "
        "BCI IV 2a on artifact-free session-2 trials, Lee2019 on all; 2a also with the seed error",
        epochs=frozen_config()["epochs"],
        overfit_protocol_sha256=sha256(OVERFIT / "protocol.json"),
    )
    counts: dict[str, int] = {}
    for job in plan["jobs"]:
        key = f"{job['dataset']} {job['kind']} {'final' if job['split'] == 'final' else 'folds'}"
        counts[key] = counts.get(key, 0) + 1
    print(f"{len(plan['jobs'])} jobs: {counts}")


def model_path(job: dict):
    if job["kind"] == "train":
        return STUDY.model(job)
    return MODELS_DIR / "supplementary" / "overfit_study" / f"{job['earlier']}.pt"


def both_ways(model, X_train: np.ndarray) -> dict:
    """The model as trained and with its batch norms re-estimated on the (standardized) training trials."""
    return {"as_trained": model.eval(), "recalibrated": noise_mechanism.recalibrate(model, X_train)}


def held_out_scores(job: dict, model, scaler, X, y, runs) -> dict:
    """Held-out runs of the training day, both ways, with the variance ratios before and after."""
    train, held = overfit.split_masks(runs, job["split"], job["dataset"])
    X_fit = scaler.transform(X[train][:, job["channels"]])
    X_held = scaler.transform(X[held][:, job["channels"]])
    versions = both_ways(model, X_fit)
    out = {"n_held": int(held.sum())}
    for name, m in versions.items():
        logits = predict(m, X_held)
        out[name] = {
            "within_day": scores(logits, y[held]),
            "pred": logits.argmax(1).tolist(),
            "variance_ratios": chdrop.variance_ratios(m, X_fit),
        }
    return out


def train_job(job: dict, epochs: int) -> None:
    X, y, runs = day1(job["dataset"], job["subject"])
    train, _ = overfit.split_masks(runs, job["split"], job["dataset"])
    inputs = X[:, job["channels"]]
    scaler = ChannelStandardizer().fit(inputs[train])
    cfg = TrainConfig(epochs=epochs, seed=job["seed"])
    set_seed(cfg.seed)
    model = build_eegnet(inputs.shape[1], n_classes=DATASETS[job["dataset"]]["classes"])
    device = device_for(job["subject"])
    X_fit = scaler.transform(inputs[train])
    result = overfit.fit_channel_dropout(model, X_fit, y[train], cfg, p=job["p"], device=device)
    path = STUDY.model(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: job[k] for k in ("dataset", "subject", "seed", "split", "condition", "p")}
    meta |= {"epochs": epochs, "device": device}
    replication.save(path, model, job["channels"], scaler, **meta)
    record = {
        **meta,
        "name": job["name"],
        "n_train": int(train.sum()),
        "train_time_s": result["train_time_s"],
        "train_acc": scores(predict(model, X_fit), y[train])["acc"],
        "final_train_loss": result["history"]["train_loss"][-1],
        "model_sha256": sha256(path),
        "protocol_sha256": sha256(STUDY.protocol_path),
    }
    if job["split"] != "final":
        record |= held_out_scores(job, model, scaler, X, y, runs)
    write_json(STUDY.record("runs", job), record)


def reference_job(job: dict) -> None:
    """A fold model without dropout (overfit study): held-out runs both ways; as trained must match its record."""
    model, scaler, _ = replication.load(model_path(job))
    X, y, runs = day1(job["dataset"], job["subject"])
    out = held_out_scores(job, model, scaler, X, y, runs)
    earlier = json.loads((OVERFIT / "runs" / f"{job['earlier']}.json").read_text())
    out["matches_earlier_record"] = out["as_trained"]["pred"] == earlier["within_day_pred"]
    keep = ("name", "dataset", "subject", "seed", "split", "condition", "p", "earlier")
    record = {**{k: job[k] for k in keep}, **out, "model_sha256": sha256(model_path(job))}
    write_json(STUDY.record("runs", job), record | {"protocol_sha256": sha256(STUDY.protocol_path)})


def evaluate_job(job: dict) -> None:
    """A final model the next day, as trained and re-estimated; failed electrodes for all-electrode models."""
    torch.set_num_threads(1)
    path = model_path(job)
    model, scaler, bundle = replication.load(path)
    if bundle["channels"] != job["channels"]:
        raise RuntimeError(f"Bundle channels differ from the plan: {path}")
    X1, _, _ = day1(job["dataset"], job["subject"])
    X_fit = scaler.transform(X1[:, job["channels"]])
    X2, y2, clean = day2(job["dataset"], job["subject"])
    X_test = scaler.transform(X2[:, job["channels"]])
    out = {}
    singles = [[e] for e in range(len(job["channels"]))]
    sets = chdrop.failure_sets(len(job["channels"]), FAILURE["size"], FAILURE["sets"], FAILURE["seed"])
    for name, m in both_ways(model, X_fit).items():
        logits = predict(m, X_test)
        out[name] = {
            "primary": scores(logits[clean], y2[clean]),
            "secondary": scores(logits, y2),
            "pred": logits.argmax(1).tolist(),
            "variance_ratios": chdrop.variance_ratios(m, X_fit),
        }
        if job["condition"].startswith("full"):
            fail = {}
            for label, groups in (("single", singles), ("three", sets)):
                fail[label] = [
                    float((predict(m, chdrop.with_broken(X_test, g))[clean].argmax(1) == y2[clean]).mean())
                    for g in groups
                ]
            out[name]["failed"] = fail
    if job["kind"] == "archived":
        earlier = json.loads((OVERFIT / "test" / f"{job['earlier']}.json").read_text())
        out["matches_earlier_record"] = out["as_trained"]["pred"] == earlier["pred"]
    record = {**{k: v for k, v in job.items() if k != "channels"}, **out, "failure_sets": sets}
    record |= {"model_sha256": sha256(path), "protocol_sha256": sha256(STUDY.protocol_path)}
    write_json(STUDY.record("test", job), record)


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
    STUDY.log_environment(f"{args.stage}-{(args.shard or 'all').replace('/', 'of')}")
    if args.stage == "train":
        pending = STUDY.pending(
            [j for j in jobs if j["kind"] == "train"],
            "runs",
            lambda j: [STUDY.model(j)],
            lambda j: {"subject": j["subject"], "seed": j["seed"], "epochs": plan["epochs"]},
        )
        run_pool(train_job, [(j["name"], {"job": j, "epochs": plan["epochs"]}) for j in pending], args.workers, 1)
    elif args.stage == "reference":
        pending = [j for j in jobs if j["kind"] == "reference" and not STUDY.record("runs", j).exists()]
        run_pool(reference_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)
    else:
        finals = [j for j in jobs if j["split"] == "final"]
        waiting = [j for j in finals if not model_path(j).exists()]
        if waiting:
            print(f"{len(waiting)} final models are not trained yet (e.g. {waiting[0]['name']}); run evaluate again")
        pending = [j for j in finals if model_path(j).exists() and not STUDY.record("test", j).exists()]
        run_pool(evaluate_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
