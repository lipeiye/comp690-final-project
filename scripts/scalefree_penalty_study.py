"""Post-test study: with a penalty that no rescaling can lower, does EEGNet switch off inputs it cannot use?

scripts/input_penalty_study.py found why a group lasso on each input's spatial weights does not switch
inputs off: a batch norm follows the spatial filters, so the overall size of a filter's weights does not
matter to the network, and the penalty is lowered most cheaply by shrinking every weight together. The
penalty here (mibci.scalefree_penalty) scales each filter's weight vector to unit length before taking
each input's norm over the filters, so it can be lowered only by concentrating the filters on fewer
inputs. If the batch norm's scale invariance was the reason, this penalty should switch the noise inputs
off, and the cost of those inputs should shrink.

Conditions, folds, selection and final models follow scripts/input_penalty_study.py:
  BCI IV 2a  full-22, relevance-16, relevance-16-plus-6, relevance-16-plus-16
  Lee2019    full-62, relevance-16, relevance-16-plus-46
lam = 0 is the earlier studies' models of the same condition, split and seed.

Stages:
  pilot               session 1 only, before the plan: BCI IV 2a participants 1 and 2, fold-0, seed 0,
                      relevance-16-plus-6 and -plus-16 at PILOT_LAMS, to choose the five strengths in LAMS
                      (from "the noise-to-electrode norm ratio hardly moves" to "the ratio is below 0.1")
  plan
  reference           lam = 0: the earlier studies' models measured as this study's (norms from their weights,
                      accuracies from their records), so that the tables read records only
  train --part folds  every lam on the training day's folds (2a: 4-run folds, seeds 0-4; Lee2019: one-run
                      folds, seed 0), scored on the held-out runs of the training day
  select              for each dataset and condition, the lam (0 included) with the best mean held-out
                      accuracy; when that is 0, also the largest lam within 1 point of the best (a second
                      reading); session 1 only
  train --part final  the final models (2a seeds 0-9, Lee2019 seeds 0-2) at the selected lam and at the
                      second-reading lam; none for lam = 0
  evaluate            the next day, for those final models

Lee2019 predictions (PREDICTIONS) are fixed before any of these models is scored on Lee2019 session 2;
BCI IV 2a is exploratory. Tables: scripts/analyze_scalefree_penalty_study.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mibci import data, lee2019, noise_mechanism, overfit, replication, scalefree_penalty
from mibci.artifacts import require_files, sha256, write_json
from mibci.experiment import device_for, run_pool
from mibci.layout import frozen_config
from mibci.model import build_eegnet
from mibci.paths import MODELS_DIR, RESULTS_DIR
from mibci.study import Study, shard
from mibci.train import ChannelStandardizer, TrainConfig, predict, scores, set_seed

STUDY = Study(
    "scalefree_penalty_study",
    "scripts/scalefree_penalty_study.py",
    sources=[
        "src/mibci/scalefree_penalty.py",
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
PILOT_LAMS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
PILOT = {
    "subjects": (1, 2),
    "split": "fold-0",
    "seed": 0,
    "conditions": ("relevance-16-plus-6", "relevance-16-plus-16"),
}
# From the pilot (session 1 only): at 0.001 the noise-to-electrode ratio of g moved 6-17% and held-out accuracy
# did not; at 0.1 it fell to 0.064 for participant 1 with 16 noise inputs; at 0.3 and above accuracy fell to
# near chance and the ratio went back to about 1. The five strengths span 0.001-0.1 (the input-penalty grid).
LAMS = (0.001, 0.003, 0.01, 0.03, 0.1)
SECOND_READING_POINTS = 1.0  # the largest lam within this many points of the best held-out accuracy
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
    "N1": {
        "claim": "With the scale-free penalty, inputs that carry nothing cost less",
        "contrast": "Lee2019 final models, next day: (relevance-16-plus-46 minus relevance-16), each at its selected "
        "lam, minus the same without the penalty",
        "test": "interval lower bound above 0",
    },
    "N2": {
        "claim": "With the scale-free penalty, all electrodes beat the 16 most relevant",
        "contrast": "Lee2019 final models, next day: full-62 minus relevance-16, each at its selected lam",
        "test": "interval lower bound above 0",
    },
    "N3": {
        "claim": "Some strength switches the noise inputs off without costing held-out accuracy",
        "contrast": "Lee2019 relevance-16-plus-46 fold models (session 1 only), per lam: mean held-out accuracy minus "
        "that at lam = 0, and the median over models of (median normalized norm g of the noise inputs over that of "
        "the electrodes)",
        "test": "some lam > 0 with the accuracy difference at least -1 point and the norm ratio below 1/3",
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


def model_inputs(job: dict) -> list[int]:
    return [*job["channels"], *(-(i + 1) for i in range(job["added"]))]


def relevance_table() -> dict:
    ic = json.loads((SUPP / "input_cost_study" / "protocol.json").read_text())["jobs"]
    return {(j["dataset"], j["subject"], j["split"]): j["channels"] for j in ic}


def plan_jobs(lams) -> list[dict]:
    jobs = []
    for (dataset, subject, split), kept in sorted(relevance_table().items()):
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
                    "relevance_16": kept,
                }
                for lam in lams
                for seed in seeds
            )
    return jobs


def pilot_jobs() -> list[dict]:
    table = relevance_table()
    return [
        {
            "name": f"bnci2a_sub-{s:02d}_seed-{PILOT['seed']}_{condition}_lam-{lam}_{PILOT['split']}",
            "dataset": "bnci2a",
            "subject": s,
            "seed": PILOT["seed"],
            "split": PILOT["split"],
            "condition": condition,
            "added": DATASETS["bnci2a"]["conditions"][condition],
            "lam": lam,
            "channels": table[("bnci2a", s, PILOT["split"])],
            "relevance_16": table[("bnci2a", s, PILOT["split"])],
        }
        for s in PILOT["subjects"]
        for condition in PILOT["conditions"]
        for lam in PILOT_LAMS
    ]


def norm_summary(model, job: dict) -> dict:
    """Normalized norms g (this study's penalty) and plain norms, and the noise-to-electrode ratios of medians."""
    with torch.no_grad():
        g = scalefree_penalty.input_norms(model).numpy()
        plain = model.conv_spatial.weight[:, 0, :, 0].norm(dim=0).numpy()
    n = len(job["channels"])
    out = {"g": [round(float(v), 6) for v in g], "plain_norms": [round(float(v), 6) for v in plain]}
    if job["added"]:
        out["g_ratio"] = float(np.median(g[n:]) / np.median(g[:n]))
        out["plain_ratio"] = float(np.median(plain[n:]) / np.median(plain[:n]))
    if job["condition"].startswith("full"):
        top = sorted(int(i) for i in np.argsort(-g)[:16])
        out["top16_by_g"] = top
        out["top16_overlap_with_relevance"] = len(set(top) & set(job["relevance_16"]))
    return out


def train_job(job: dict, epochs: int, kind: str = "runs") -> None:
    X, y, runs = day1(job["dataset"], job["subject"])
    train, held = overfit.split_masks(runs, job["split"], job["dataset"])
    inputs = inputs_for(job, X, day=1)
    scaler = ChannelStandardizer().fit(inputs[train])
    cfg = TrainConfig(epochs=epochs, seed=job["seed"])
    set_seed(cfg.seed)
    model = build_eegnet(inputs.shape[1], n_classes=DATASETS[job["dataset"]]["classes"])
    device = device_for(job["subject"])
    X_fit = scaler.transform(inputs[train])
    result = scalefree_penalty.fit_scalefree(model, X_fit, y[train], cfg, job["lam"], device=device)
    meta = {k: job[k] for k in ("dataset", "subject", "seed", "split", "condition", "added", "lam")}
    meta |= {"epochs": epochs, "device": device}
    record = {
        **meta,
        "name": job["name"],
        "n_train": int(train.sum()),
        "train_time_s": result["train_time_s"],
        "train_acc": scores(predict(model, X_fit), y[train])["acc"],
        "final_train_loss": result["history"]["train_loss"][-1],
        "final_penalty": result["history"]["penalty"][-1],
        **norm_summary(model, job),
        "protocol_sha256": sha256(STUDY.protocol_path) if kind == "runs" else None,
    }
    if kind == "runs":
        path = STUDY.model(job)
        path.parent.mkdir(parents=True, exist_ok=True)
        replication.save(path, model, model_inputs(job), scaler, **meta)
        record["model_sha256"] = sha256(path)
    if held is not None:
        logits = predict(model, scaler.transform(inputs[held]))
        record |= {"n_held": int(held.sum()), "within_day": scores(logits, y[held])}
    write_json(STUDY.record(kind, job), record)


def unpenalized_record(job: dict) -> dict:
    study, name = unpenalized_name(job["dataset"], job["condition"], job["subject"], job["seed"], job["split"])
    return json.loads((SUPP / study / "runs" / f"{name}.json").read_text())


def held_out_accuracy(job: dict) -> float:
    """Held-out accuracy of the training day of a fold job, penalized or (lam = 0) from the earlier study."""
    record = json.loads(STUDY.record("runs", job).read_text()) if job["lam"] else unpenalized_record(job)
    return record["within_day"]["acc"]


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
            second = None
            if chosen == 0.0:
                near = [lam for lam, v in by_lam.items() if lam > 0 and v >= best - SECOND_READING_POINTS / 100]
                second = max(near) if near else None
            table[f"{dataset}|{condition}"] = {
                "held_out_accuracy": by_lam,
                "selected_lam": chosen,
                "second_reading_lam": second,
            }
    write_json(STUDY.out / "selection.json", {"rule": plan["selection"], "selection": table})
    for key, v in table.items():
        print(key, "selected", v["selected_lam"], "second", v["second_reading_lam"],
              {k: round(100 * a, 2) for k, a in v["held_out_accuracy"].items()})  # fmt: skip


def selected_final_jobs(plan: dict) -> list[dict]:
    chosen = json.loads((STUDY.out / "selection.json").read_text())["selection"]

    def wanted(job: dict) -> bool:
        cell = chosen[f"{job['dataset']}|{job['condition']}"]
        return job["lam"] in (cell["selected_lam"], cell["second_reading_lam"])

    return [j for j in plan["jobs"] if j["split"] == "final" and wanted(j)]


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
    logits = predict(model, scaler.transform(inputs_for(job, ses["X"], day=2)))
    write_json(
        STUDY.record("test", job),
        {
            **{k: v for k, v in job.items() if k != "relevance_16"},
            "model_sha256": sha256(path),
            "protocol_sha256": sha256(STUDY.protocol_path),
            "primary": scores(logits[clean], y[clean]),
            "secondary": scores(logits, y),
            "pred": logits.argmax(1).tolist(),
        },
    )


def pilot_summary() -> list[dict]:
    """Per pilot strength: mean held-out accuracy and noise-to-electrode ratio of g, with lam = 0 alongside."""
    rows = []
    for job in pilot_jobs():
        path = STUDY.record("pilot", job)
        if path.exists():
            r = json.loads(path.read_text())
            rows.append({k: r[k] for k in ("subject", "condition", "lam", "g_ratio", "plain_ratio")} | {
                "held_out_acc": r["within_day"]["acc"]})  # fmt: skip
    for job in pilot_jobs():
        if job["lam"] != PILOT_LAMS[0]:
            continue
        model, _, _ = replication.load(model_path_unpenalized(job))
        ratios = norm_summary(model, job)
        rows.append({"subject": job["subject"], "condition": job["condition"], "lam": 0.0,
                     "g_ratio": ratios["g_ratio"], "plain_ratio": ratios["plain_ratio"],
                     "held_out_acc": unpenalized_record(job)["within_day"]["acc"]})  # fmt: skip
    return sorted(rows, key=lambda r: (r["condition"], r["lam"], r["subject"]))


def model_path_unpenalized(job: dict):
    study, name = unpenalized_name(job["dataset"], job["condition"], job["subject"], job["seed"], job["split"])
    return MODELS_DIR / "supplementary" / study / f"{name}.pt"


def reference_job(job: dict) -> None:
    """lam = 0: the earlier study's model of this condition, split and seed, measured as this study's models are
    (norms from its weights; its held-out accuracy, or next-day accuracy for a final model, from its records)."""
    study, name = unpenalized_name(job["dataset"], job["condition"], job["subject"], job["seed"], job["split"])
    model, _, _ = replication.load(model_path_unpenalized(job))
    record = {
        **{k: job[k] for k in ("dataset", "subject", "seed", "split", "condition", "added")},
        "lam": 0.0,
        "name": job["name"].replace(f"_lam-{job['lam']}_", "_lam-0.0_"),
        "earlier": [study, name],
        **norm_summary(model, job),
        "protocol_sha256": sha256(STUDY.protocol_path),
    }
    earlier = unpenalized_record(job)
    if job["split"] == "final":
        test = json.loads((SUPP / study / "test" / f"{name}.json").read_text())
        record["next_day_acc"] = test["primary"]["acc"]
    else:
        record["within_day"] = earlier["within_day"]
    record["train_acc"] = earlier.get("train_acc")
    write_json(STUDY.out / "reference" / f"{record['name']}.json", record)


def create_plan() -> None:
    if not LAMS:
        raise SystemExit("Set LAMS from the pilot (--stage pilot) before making the plan")
    pilot = pilot_summary()
    if len(pilot) != len(pilot_jobs()) + len(PILOT["subjects"]) * len(PILOT["conditions"]):
        raise SystemExit("The pilot is incomplete; run --stage pilot first")
    plan = STUDY.create(
        plan_jobs(LAMS),
        purpose="Test whether the batch norm's scale invariance is why a penalty did not switch off useless inputs: a "
        "penalty that no rescaling of a spatial filter can lower. Does it switch the noise inputs off, does their cost "
        "shrink, and do all electrodes then beat the best 16?",
        claim_boundary="Exploratory for BCI IV 2a, planned after its test results and the input-penalty results were "
        "known. For Lee2019 the predictions are fixed before any of these models is scored on Lee2019 session 2. lam "
        "is chosen on held-out runs of the training day only, and the five strengths were chosen on a session-1 pilot "
        "of two 2a participants. Three predictions, no correction for multiplicity. It does not change the frozen 2a "
        "configuration.",
        lee2019_new_models_scored_before_plan=False,
        predictions=PREDICTIONS,
        lams=list(LAMS),
        pilot={
            "design": "session 1 only: BCI IV 2a participants 1 and 2, fold-0 (4 runs), seed 0, relevance-16-plus-6 "
            "and relevance-16-plus-16, lam in PILOT_LAMS; lam = 0 from the input-cost study's models",
            "pilot_lams": list(PILOT_LAMS),
            "aim": "five strengths from 'the noise-to-electrode ratio of g hardly moves' to 'the ratio is below 0.1'",
            "chosen": "0.001 (ratio 6-17% lower, held-out accuracy unchanged) to 0.1 (ratio 0.064 for participant 1 "
            "with 16 noise inputs); 0.3 and above put held-out accuracy near chance with the ratio back near 1",
            "script": "the pilot ran from this script before the reference stage and LAMS were added; the training "
            "code was the same",
            "rows": pilot,
        },
        selection="for each dataset and condition, the lam (0 included) with the highest held-out accuracy of the "
        "training day, averaged over seeds and folds within each participant and then over participants; ties go "
        "to the smaller lam. When lam = 0 is selected, a second reading also trains the final models at the largest "
        f"lam within {SECOND_READING_POINTS} point of the best",
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
    print(f"{len(plan['jobs'])} planned jobs (final ones run only at the selected lams): {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["pilot", "plan", "reference", "train", "select", "evaluate"], required=True)
    parser.add_argument("--part", choices=["folds", "final"], help="train: the fold models or the final models")
    parser.add_argument("--subjects", nargs="+", type=int, help="only these participants (local smoke tests)")
    parser.add_argument("--shard", help="i/n: run only array task i of n (0-based)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    epochs = frozen_config()["epochs"]
    if args.stage == "pilot":
        todo = [j for j in pilot_jobs() if not STUDY.record("pilot", j).exists()]
        run_pool(train_job, [(j["name"], {"job": j, "epochs": epochs, "kind": "pilot"}) for j in todo], args.workers, 1)
        for row in pilot_summary():
            print(row)
        return
    if args.stage == "plan":
        create_plan()
        return
    plan = STUDY.load()
    if args.stage == "select":
        select(plan)
        return
    if args.stage == "reference":
        first = [j for j in plan["jobs"] if j["lam"] == LAMS[0]]
        run_pool(reference_job, [(j["name"], {"job": j}) for j in shard(first, args.shard)], args.workers, 1)
        return
    keep = (lambda j: j["subject"] in args.subjects) if args.subjects else (lambda j: True)
    if args.stage == "train":
        if not args.part:
            raise SystemExit("--stage train needs --part folds or --part final")
        todo = [j for j in plan["jobs"] if j["split"] != "final"] if args.part == "folds" else selected_final_jobs(plan)
        STUDY.log_environment(f"train-{args.part}-{(args.shard or 'all').replace('/', 'of')}")
        pending = STUDY.pending(
            [j for j in shard(todo, args.shard) if keep(j)],
            "runs",
            lambda j: [STUDY.model(j)],
            lambda j: {"subject": j["subject"], "seed": j["seed"], "epochs": plan["epochs"]},
        )
        run_pool(train_job, [(j["name"], {"job": j, "epochs": plan["epochs"]}) for j in pending], args.workers, 1)
        return
    todo = [j for j in shard(selected_final_jobs(plan), args.shard) if keep(j)]
    require_files([STUDY.model(j) for j in todo], "penalized final models (run --stage train --part final first)")
    pending = STUDY.pending(todo, "test", lambda j: [STUDY.model(j)])
    run_pool(evaluate_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
