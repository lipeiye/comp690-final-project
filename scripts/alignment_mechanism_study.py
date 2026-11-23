"""Post-test study: why does day-to-day alignment hurt on Lee2019 and help on BCI IV 2a?

Euclidean alignment (mibci.alignment) whitens each recording day with its own mean spatial covariance.
On BCI IV 2a it helped on average (scripts/alignment_study.py); with Lee2019's 62 electrodes it cost 5.5
points (scripts/replication_study.py). Two explanations were mixed together: the covariance is estimated
badly from 100-200 trials, or it is estimated well but whitening blows up directions of tiny variance,
mostly noise, to the size of the useful ones (the same mechanism as many unit-variance unrelated inputs,
scripts/input_kind_study.py). mibci.alignment_variants separates them.

  A  (no training, session 1 only) per participant and montage (2a: all 22, relevance-16; Lee2019: all 62,
     relevance-16): eigenvalues of the alignment covariance, condition number, directions below 1% of the
     largest, the Ledoit-Wolf coefficient (trials as units), and how far apart the whitening matrices of two
     interleaved halves of the trials are.
  B  (training, exploratory on both datasets: plain alignment's next-day results were known) the variants
     ea-shrink10, ea-lw, ea-top and ea-floor (the last added after part A's spectra; see design_note).
     Final models: all 22 (2a) and all 62 (Lee2019) electrodes, seeds 0-2,
     the training day aligned with its own covariance, the next day with all of its unlabeled trials, as in
     the earlier studies. Fold models (session 1 only; 2a fold-0/1/2, Lee2019 fold-0/1, seeds 0-2): the
     training runs aligned with their own covariance, the held-out runs with theirs; plain alignment and the
     three variants. References: unaligned models of scripts/overfit_study.py, plain-aligned final models of
     scripts/alignment_study.py (2a) and scripts/replication_study.py (Lee2019).

The three expectations (EXPECTATIONS) are written in the plan before any model is trained; they are
exploratory statements, not prospective tests. Stages: plan, spectra, train, evaluate. Tables:
scripts/analyze_alignment_mechanism_study.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mibci import alignment_variants, data, lee2019, overfit, replication
from mibci.artifacts import sha256, write_json
from mibci.experiment import device_for, run_pool
from mibci.layout import frozen_config
from mibci.model import build_eegnet
from mibci.paths import RESULTS_DIR
from mibci.study import Study, shard
from mibci.train import ChannelStandardizer, TrainConfig, fit, predict, scores, set_seed

STUDY = Study(
    "alignment_mechanism_study",
    "scripts/alignment_mechanism_study.py",
    sources=[
        "src/mibci/alignment_variants.py",
        "src/mibci/alignment.py",
        "src/mibci/overfit.py",
        "src/mibci/lee2019.py",
        "src/mibci/replication.py",
        "src/mibci/splits.py",
        "src/mibci/study.py",
    ],
)
SUPP = RESULTS_DIR / "supplementary"
COST_PLAN = SUPP / "input_cost_study" / "protocol.json"
FINAL_VARIANTS = ("ea-shrink10", "ea-lw", "ea-top", "ea-floor")
FOLD_VARIANTS = alignment_variants.VARIANTS
SEEDS = (0, 1, 2)
DATASETS = {
    "bnci2a": {"classes": 4, "electrodes": 22, "folds": ("fold-0", "fold-1", "fold-2")},
    "lee2019": {"classes": 2, "electrodes": 62, "folds": ("fold-0", "fold-1")},
}
REFERENCES = {
    "unaligned": "scripts/overfit_study.py full-22 / full-62 (final and fold models), seeds 0-2",
    "plain alignment, final": "scripts/alignment_study.py full-22-ea (2a) and scripts/replication_study.py full-62-ea "
    "(Lee2019), seeds 0-2, next day aligned with all of its trials",
}
EXPECTATIONS = {
    "M1": {
        "statement": "With Lee2019's 62 electrodes, plain alignment lowers accuracy on held-out runs of the "
        "training day too, so the harm does not come from the change between days",
        "contrast": "Lee2019 fold models: ea minus unaligned, held-out runs",
        "reading": "interval upper bound below 0",
    },
    "M2": {
        "statement": "ea-top and ea-shrink10 each remove at least half of the harm on Lee2019",
        "contrast": "Lee2019 final models, next day: (variant minus ea) over (unaligned minus ea)",
        "reading": "at least 0.5 for both",
    },
    "M3": {
        "statement": "The Ledoit-Wolf coefficient is small, so the covariance is estimated well and the harm comes "
        "from blowing up small directions",
        "contrast": "Lee2019, all 62 electrodes, training day: mean coefficient over participants",
        "reading": "below 0.01",
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


def plan_jobs() -> list[dict]:
    cost = json.loads(COST_PLAN.read_text())["jobs"]
    relevance = {(j["dataset"], j["subject"]): j["channels"] for j in cost if j["split"] == "final"}
    jobs = []
    for (dataset, subject), kept in sorted(relevance.items()):
        spec = DATASETS[dataset]
        full = list(range(spec["electrodes"]))
        for montage, channels in ((f"full-{spec['electrodes']}", full), ("relevance-16", kept)):
            jobs.append(
                {
                    "name": f"{dataset}_sub-{subject:02d}_{montage}_spectrum",
                    "kind": "spectrum",
                    "dataset": dataset,
                    "subject": subject,
                    "montage": montage,
                    "channels": channels,
                }
            )
        for split in ("final", *spec["folds"]):
            variants = FINAL_VARIANTS if split == "final" else FOLD_VARIANTS
            jobs.extend(
                {
                    "name": f"{dataset}_sub-{subject:02d}_seed-{seed}_full-{spec['electrodes']}-{variant}_{split}",
                    "kind": "train",
                    "dataset": dataset,
                    "subject": subject,
                    "seed": seed,
                    "split": split,
                    "variant": variant,
                    "channels": full,
                }
                for variant in variants
                for seed in SEEDS
            )
    return jobs


def create_plan() -> None:
    plan = STUDY.create(
        plan_jobs(),
        purpose="Separate two reasons Euclidean alignment hurts with Lee2019's 62 electrodes: a badly estimated "
        "covariance, or tiny-variance directions blown up by whitening; and ask whether it hurts within one day.",
        claim_boundary="Exploratory on both datasets: plain alignment's next-day results on both were known when this "
        "was planned. The expectations are written before any of these models is trained, but they are not "
        "prospective tests. Alignment uses unlabeled next-day trials, which the frozen protocol does not.",
        expectations=EXPECTATIONS,
        references=REFERENCES,
        variants={
            "ea": "mibci.alignment exactly",
            "ea-shrink10": f"covariance shrunk {alignment_variants.SHRINK} toward the identity at its mean eigenvalue",
            "ea-lw": "the same with the Ledoit-Wolf coefficient, trials as units (mibci.alignment_variants)",
            "ea-top": f"only directions with eigenvalue >= {alignment_variants.TOP_FLOOR} of the largest whitened; "
            "the others set to zero",
            "ea-floor": f"eigenvalues below {alignment_variants.TOP_FLOOR} of the largest raised to it before "
            "whitening (kept, not blown up); added to the planned design after the smoke test's spectra (see "
            "design_note)",
        },
        design_note="docs/FINAL_EXECUTION_PLAN.md planned ea-top with a 1% floor. Before this plan, the 2a smoke "
        "test's spectra (training day) put 16-18 of 22 directions below 1% of the largest, and five Lee2019 "
        "participants 45-52 of 62, so ea-top keeps 4-6 (2a) or 10-17 (Lee2019) directions: if it hurts, that "
        "shows the small directions carry information, not that blowing them up is harmless. ea-top is run as "
        "planned; ea-floor, the same 1% boundary with the small directions kept at that floor instead of zeroed, "
        "is added so that the two readings can be told apart. M2 is judged as written (ea-top and ea-shrink10); "
        "ea-floor is reported beside it.",
        seeds=list(SEEDS),
        datasets={
            name: {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()}
            for name, spec in DATASETS.items()
        },
        next_day_alignment="all of the next day's unlabeled trials (as the earlier studies' ea-all); also its first "
        "run (2a, scored on runs 2-6) or its offline run (Lee2019, scored on the online run)",
        analysis="seeds (and folds) averaged within each participant; 95% participant bootstrap (10,000, seed 690); "
        "BCI IV 2a on artifact-free session-2 trials, Lee2019 on all",
        epochs=frozen_config()["epochs"],
        input_cost_protocol_sha256=sha256(COST_PLAN),
    )
    counts: dict[str, int] = {}
    for job in plan["jobs"]:
        key = f"{job['dataset']} {job['kind']} {job.get('split', '')}".strip()
        counts[key] = counts.get(key, 0) + 1
    print(f"{len(plan['jobs'])} jobs: {counts}")


def spectrum_job(job: dict) -> None:
    X, _, _ = day1(job["dataset"], job["subject"])
    out = alignment_variants.spectrum(X[:, job["channels"]])
    write_json(STUDY.record("spectra", job), {**job, **out, "protocol_sha256": sha256(STUDY.protocol_path)})


def train_job(job: dict, epochs: int) -> None:
    X, y, runs = day1(job["dataset"], job["subject"])
    X = X[:, job["channels"]]
    train, held = overfit.split_masks(runs, job["split"], job["dataset"])
    X_train = alignment_variants.aligned(X[train], job["variant"])
    scaler = ChannelStandardizer().fit(X_train)
    cfg = TrainConfig(epochs=epochs, seed=job["seed"])
    set_seed(cfg.seed)
    model = build_eegnet(X.shape[1], n_classes=DATASETS[job["dataset"]]["classes"])
    device = device_for(job["subject"])
    X_fit = scaler.transform(X_train)
    result = fit(model, X_fit, y[train], cfg, device=device)
    path = STUDY.model(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: job[k] for k in ("dataset", "subject", "seed", "split", "variant")}
    meta |= {"epochs": epochs, "device": device}
    replication.save(path, model, job["channels"], scaler, **meta)
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
        X_held = scaler.transform(alignment_variants.aligned(X[held], job["variant"]))
        logits = predict(model, X_held)
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
    if bundle["channels"] != job["channels"]:
        raise RuntimeError(f"Bundle channels differ from the plan: {path}")
    if job["dataset"] == "bnci2a":
        ses = data.load_session(job["subject"], 2, final_evaluation=True)
        clean, first = ~ses["artifact"], ses["run"] == 0
        X2 = ses["X"][:, job["channels"]]
        inputs = {"all": (X2, clean), "first-run": (X2[first], clean & ~first)}
    else:
        ses = lee2019.load_session(job["subject"], 2, final_evaluation=True)
        offline = ses["run"] == 0
        X2 = ses["X"][:, job["channels"]]
        inputs = {"all": (X2, np.ones(len(offline), bool)), "offline-run": (X2[offline], ~offline)}
    y = ses["y"].astype(np.int64)
    out = {}
    for name, (reference, scored) in inputs.items():
        logits = predict(model, scaler.transform(alignment_variants.aligned(X2, job["variant"], reference=reference)))
        out[name] = {
            "scored_trials": int(scored.sum()),
            "acc": float((logits.argmax(1)[scored] == y[scored]).mean()),
            "pred": logits.argmax(1).tolist(),
        }
        if name == "all":
            out[name]["primary"] = scores(logits[scored], y[scored])
    write_json(
        STUDY.record("test", job),
        {
            **{k: v for k, v in job.items() if k != "channels"},
            "aligned_with": out,
            "model_sha256": sha256(path),
            "protocol_sha256": sha256(STUDY.protocol_path),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "spectra", "train", "evaluate"], required=True)
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
    if args.stage == "spectra":
        pending = [j for j in jobs if j["kind"] == "spectrum" and not STUDY.record("spectra", j).exists()]
        run_pool(spectrum_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)
    elif args.stage == "train":
        pending = STUDY.pending(
            [j for j in jobs if j["kind"] == "train"],
            "runs",
            lambda j: [STUDY.model(j)],
            lambda j: {"subject": j["subject"], "seed": j["seed"], "epochs": plan["epochs"]},
        )
        run_pool(train_job, [(j["name"], {"job": j, "epochs": plan["epochs"]}) for j in pending], args.workers, 1)
    else:
        finals = [j for j in jobs if j["kind"] == "train" and j["split"] == "final"]
        waiting = [j for j in finals if not STUDY.model(j).exists()]
        if waiting:
            print(f"{len(waiting)} final models are not trained yet (e.g. {waiting[0]['name']}); run evaluate again")
        pending = [j for j in finals if STUDY.model(j).exists() and not STUDY.record("test", j).exists()]
        run_pool(evaluate_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
