"""Post-test study: why does an input that carries nothing cost EEGNet accuracy, and does the noise draw matter?

scripts/input_cost_study.py measured the cost of k band-limited noise inputs added to 16 electrodes:
about 0.9 percentage points per input on BCI IV 2a and 0.3 on Lee2019, as large on held-out runs of
the training day as on the next day, and the same at every amount of training data. Two questions
remained.

  Mechanism (stage heldout; session 1 only). Every fold model of that study and its k = 0 baseline
  (BCI IV 2a: 4-run and 2-run folds, seeds 0-4, relevance-16 from scripts/overfit_study.py; Lee2019:
  one-run folds, seeds 0-2, relevance-16 from scripts/electrode_data_study.py), scored on the held-out
  runs of the training day as trained, with the spatial weights of the noise inputs set to zero, and
  both again with the batch norms after the spatial filters re-estimated on the training trials
  (mibci.noise_mechanism). Also the share of each spatial filter's output that is noise, after
  training and at initialization (rebuilt from the seed), and the weights' sizes.

  Noise draws (stages train, evaluate; BCI IV 2a final models, seeds 0-9). Each earlier study drew one
  noise sample per participant and day, shared by every seed, and the same condition (16 electrodes
  plus 6 noise inputs) cost 7.29 points with the input-cost draw and 4.96 with the overfit-study draw.
  EEGNet's spatial filters span every input, so the two placements (appended, or in the six dropped
  places) cannot matter except through which initial weight meets which input. Trained arms: four
  new draws appended; the overfit-study draw appended; the input-cost draw in the dropped places. The
  existing arms (input-cost relevance-16-plus-6, overfit relevance-16-plus-noise) and relevance-16
  are scored again as references. Training records the loss of every epoch; scoring repeats the
  mechanism measurements on the next day.

Exploratory (the test results and the input-cost results were known). Tables:
scripts/analyze_noise_mechanism_study.py.
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
    "noise_mechanism_study",
    "scripts/noise_mechanism_study.py",
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
SOURCES = ("input_cost_study", "overfit_study", "electrode_data_study")
CLASSES = {"bnci2a": 4, "lee2019": 2}
LARGEST = {"bnci2a": 16, "lee2019": 46}  # the input-cost study drew this many noise inputs and cut to k
NEW_DRAWS = (1, 2, 3, 4)
TRAINED_ARMS = (*(f"draw-{d}" for d in NEW_DRAWS), "overfit-draw-appended", "input-cost-draw-in-place")
QUESTIONS = {
    "M1": {
        "question": "How much of the held-out cost comes back when the noise inputs' spatial weights are set to zero?",
        "contrast": "per participant: (noise removed, batch norms re-estimated) minus (as trained, batch norms "
        "re-estimated), over (as trained) minus k = 0 (as trained); BCI IV 2a 4-run folds, k = 6 and k = 16",
        "reading": "above 0.5: mostly noise passing through the trained filters; below 0.5: mostly a network that "
        "learned less from the electrodes",
    },
    "M2": {
        "question": "Do models whose spatial filters carry more noise do worse?",
        "contrast": "within participant and condition, correlation of the mean noise share after training with "
        "held-out accuracy, pooled; BCI IV 2a folds and Lee2019 folds separately",
        "reading": "interval upper bound below 0: yes",
    },
    "M3": {
        "question": "Does training lower the noise share below its value at initialization?",
        "contrast": "mean noise share after training over the same share at initialization, per k",
        "reading": "below 1: training learns to suppress the noise inputs, by that factor",
    },
    "B1": {
        "question": "Does the cost of six noise inputs depend on the noise drawn?",
        "contrast": "BCI IV 2a final models: the six appended draws (input-cost, overfit-study, four new); the "
        "spread of their costs beyond what seed noise explains",
        "reading": "a draw standard deviation whose interval excludes 0",
    },
    "B2": {
        "question": "Does the earlier 2.3-point difference follow the draw or the placement?",
        "contrast": "input-cost draw in the dropped places minus the overfit-study draw appended, against the "
        "original difference (input-cost appended minus overfit-study in place)",
        "reading": "same sign and size: the draw; near 0: the placement",
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


def inputs_for(job: dict, X: np.ndarray, day: int) -> tuple[np.ndarray, list[int]]:
    """The network input of a job on one day's trials, and the positions of its noise inputs."""
    arm, ch, n_times = job["arm"], job["channels"], X.shape[2]

    def drawn(added: int, draw: int) -> np.ndarray:
        return noise_mechanism.appended_noise(
            len(X), n_times, added, LARGEST[job["dataset"]], job["dataset"], job["subject"], day, draw
        )

    appended = list(range(len(ch), len(ch) + job["added"]))
    if job["added"] == 0:
        return X[:, ch], []
    if arm == "input-cost":
        return np.concatenate([X[:, ch], drawn(job["added"], 0)], axis=1), appended
    if arm.startswith("draw-"):
        return np.concatenate([X[:, ch], drawn(job["added"], int(arm.split("-")[1]))], axis=1), appended
    replaced = job["replaced"]
    if arm == "overfit-draw-appended":
        noise = overfit.noise_channels(X, replaced, job["dataset"], job["subject"], day)[:, replaced]
        return np.concatenate([X[:, ch], noise], axis=1), appended
    if arm == "overfit-draw-in-place":
        return overfit.noise_channels(X, replaced, job["dataset"], job["subject"], day), replaced
    if arm == "input-cost-draw-in-place":
        out = X.copy()
        out[:, replaced] = drawn(len(replaced), 0)
        return out, replaced
    raise ValueError(f"unknown arm {arm}")


def model_path(job: dict):
    if job["source"] == STUDY.name:
        return STUDY.model(job)
    return MODELS_DIR / "supplementary" / job["source"] / f"{job['source_name']}.pt"


def plan_jobs() -> list[dict]:
    plans = {s: json.loads((SUPP / s / "protocol.json").read_text())["jobs"] for s in SOURCES}
    ic_seeds = {"bnci2a": set(range(5)), "lee2019": set(range(3))}
    jobs = []

    def mechanism(source: str, j: dict, arm: str, added: int) -> dict:
        return {
            "name": f"heldout_{j['name']}",
            "kind": "heldout",
            "source": source,
            "source_name": j["name"],
            "dataset": j["dataset"],
            "subject": j["subject"],
            "seed": j["seed"],
            "split": j["split"],
            "arm": arm,
            "added": added,
            "channels": j["channels"],
        }

    jobs += [
        mechanism("input_cost_study", j, "input-cost", j["added"])
        for j in plans["input_cost_study"]
        if j["split"] != "final"
    ]
    for source, dataset in (("overfit_study", "bnci2a"), ("electrode_data_study", "lee2019")):
        for j in plans[source]:
            fold = j["split"] != "final" and j["dataset"] == dataset and j["condition"] == "relevance-16"
            if fold and j["seed"] in ic_seeds[dataset]:
                jobs.append(mechanism(source, j, "relevance-16", 0))

    final16 = {
        j["subject"]: j["channels"]
        for j in plans["input_cost_study"]
        if j["dataset"] == "bnci2a" and j["split"] == "final"
    }
    in_place = {
        (j["subject"], j["seed"]): j
        for j in plans["overfit_study"]
        if j["dataset"] == "bnci2a" and j["condition"] == "relevance-16-plus-noise"
    }
    base = {"kind": "draw", "dataset": "bnci2a", "split": "final"}
    for (subject, seed), of in sorted(in_place.items()):
        common = {**base, "subject": subject, "seed": seed, "channels": final16[subject], "replaced": of["replaced"]}
        for arm in TRAINED_ARMS:
            name = f"bnci2a_sub-{subject:02d}_seed-{seed}_{arm}"
            jobs.append({"name": name, **common, "source": STUDY.name, "source_name": name, "arm": arm, "added": 6})
        ic = f"bnci2a_sub-{subject:02d}_seed-{seed}_relevance-16-plus-6_final"
        references = [
            ("input_cost_study", ic, "input-cost", 6),
            ("overfit_study", of["name"], "overfit-draw-in-place", 6),
            ("overfit_study", f"bnci2a_sub-{subject:02d}_seed-{seed}_relevance-16", "relevance-16", 0),
        ]
        for source, source_name, arm, added in references:
            jobs.append(
                {
                    "name": f"reference_{source_name}",
                    **common,
                    "kind": "reference",
                    "source": source,
                    "source_name": source_name,
                    "arm": arm,
                    "added": added,
                }
            )
    return jobs


def create_plan() -> None:
    plan = STUDY.create(
        plan_jobs(),
        purpose="Find out why an input that carries nothing costs EEGNet accuracy (does the trained network pass "
        "the noise through, or did it learn less from the electrodes?) and how much the cost depends on the noise "
        "sample drawn.",
        claim_boundary="Exploratory, planned after the test results and the input-cost results were known; it "
        "explains a result and changes no earlier configuration. The mechanism part uses held-out runs of the "
        "training day only. The questions are read as stated in the plan, without correction for multiplicity.",
        questions=QUESTIONS,
        arms={
            "trained": list(TRAINED_ARMS),
            "references": ["input-cost", "overfit-draw-in-place", "relevance-16"],
            "new_draw_seed": noise_mechanism.DRAW_SEED,
        },
        sources={s: sha256(SUPP / s / "protocol.json") for s in SOURCES},
        analysis="seeds (and folds) averaged within each participant; 95% participant bootstrap (10,000, seed 690); "
        "BCI IV 2a on artifact-free trials",
        epochs=frozen_config()["epochs"],
    )
    counts: dict[str, int] = {}
    for job in plan["jobs"]:
        counts[job["kind"]] = counts.get(job["kind"], 0) + 1
    print(f"{len(plan['jobs'])} jobs: {counts}")


def accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    return scores(predict(model, X), y)["acc"]


def mechanism(model, X_fit: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray, noise: list[int], job: dict) -> dict:
    """Accuracy as trained and with the noise path removed, each with re-estimated batch norms; noise
    shares and weight sizes after training and at initialization (fit inputs)."""
    out = {
        "as_trained": accuracy(model, X_eval, y_eval),
        "recalibrated": accuracy(noise_mechanism.recalibrate(model, X_fit), X_eval, y_eval),
    }
    if not noise:
        return out
    removed = noise_mechanism.without_noise_path(model, noise)
    set_seed(job["seed"])
    init = build_eegnet(X_fit.shape[1], n_classes=CLASSES[job["dataset"]]).eval()
    electrodes = [c for c in range(X_fit.shape[1]) if c not in noise]
    return out | {
        "noise_removed": accuracy(removed, X_eval, y_eval),
        "noise_removed_recalibrated": accuracy(noise_mechanism.recalibrate(removed, X_fit), X_eval, y_eval),
        "noise_share_trained": noise_mechanism.noise_share(model, X_fit, noise).round(5).tolist(),
        "noise_share_init": noise_mechanism.noise_share(init, X_fit, noise).round(5).tolist(),
        "weight_rms": {
            "noise": noise_mechanism.weight_rms(model, noise),
            "electrodes": noise_mechanism.weight_rms(model, electrodes),
            "noise_init": noise_mechanism.weight_rms(init, noise),
            "electrodes_init": noise_mechanism.weight_rms(init, electrodes),
        },
    }


def heldout_job(job: dict) -> None:
    torch.set_num_threads(1)
    X, y, runs = day1(job["dataset"], job["subject"])
    train, held = overfit.split_masks(runs, job["split"], job["dataset"])
    inputs, noise = inputs_for(job, X, day=1)
    path = model_path(job)
    model, scaler, bundle = replication.load(path)
    if len(bundle["channels"]) != inputs.shape[1]:
        raise RuntimeError(f"{path}: the bundle has {len(bundle['channels'])} inputs, the job {inputs.shape[1]}")
    Xs = scaler.transform(inputs)
    recorded = json.loads((SUPP / job["source"] / "runs" / f"{job['source_name']}.json").read_text())
    write_json(
        STUDY.record("heldout", job),
        {
            **job,
            "n_train": int(train.sum()),
            "n_held": int(held.sum()),
            "recorded_within_day_acc": recorded["within_day"]["acc"],
            **mechanism(model, Xs[train], Xs[held], y[held], noise, job),
            "model_sha256": sha256(path),
            "protocol_sha256": sha256(STUDY.protocol_path),
        },
    )


def train_job(job: dict, epochs: int) -> None:
    X, y, _ = day1(job["dataset"], job["subject"])
    inputs, noise = inputs_for(job, X, day=1)
    scaler = ChannelStandardizer().fit(inputs)
    cfg = TrainConfig(epochs=epochs, seed=job["seed"])
    set_seed(cfg.seed)
    model = build_eegnet(inputs.shape[1], n_classes=CLASSES[job["dataset"]])
    device = device_for(job["subject"])
    X_fit = scaler.transform(inputs)
    result = fit(model, X_fit, y, cfg, device=device)
    path = STUDY.model(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    if job["arm"] == "input-cost-draw-in-place":  # montage order, noise inputs numbered -1, -2, ...
        channels = [-(noise.index(c) + 1) if c in noise else c for c in range(inputs.shape[1])]
    else:
        channels = [*job["channels"], *(-(i + 1) for i in range(len(noise)))]
    meta = {k: job[k] for k in ("dataset", "subject", "seed", "split", "arm", "added")} | {"epochs": epochs}
    replication.save(path, model, channels, scaler, noise_positions=noise, device=device, **meta)
    write_json(
        STUDY.record("runs", job),
        {
            **meta,
            "name": job["name"],
            "noise_positions": noise,
            "n_train": len(y),
            "train_time_s": result["train_time_s"],
            "train_acc": accuracy(model, X_fit, y),
            "train_loss": [round(v, 5) for v in result["history"]["train_loss"]],
            "model_sha256": sha256(path),
            "protocol_sha256": sha256(STUDY.protocol_path),
        },
    )


def evaluate_job(job: dict) -> None:
    torch.set_num_threads(1)
    X1, _, _ = day1(job["dataset"], job["subject"])
    inputs1, noise = inputs_for(job, X1, day=1)
    ses = data.load_session(job["subject"], 2, final_evaluation=True)
    clean = ~ses["artifact"]
    inputs2, _ = inputs_for(job, ses["X"], day=2)
    path = model_path(job)
    model, scaler, bundle = replication.load(path)
    if len(bundle["channels"]) != inputs2.shape[1]:
        raise RuntimeError(f"{path}: the bundle has {len(bundle['channels'])} inputs, the job {inputs2.shape[1]}")
    y = ses["y"].astype(np.int64)
    X_fit, X_test = scaler.transform(inputs1), scaler.transform(inputs2)
    logits = predict(model, X_test)
    record = {
        **job,
        "primary": scores(logits[clean], y[clean]),
        "pred": logits.argmax(1).tolist(),
        **mechanism(model, X_fit, X_test[clean], y[clean], noise, job),
        "model_sha256": sha256(path),
        "protocol_sha256": sha256(STUDY.protocol_path),
    }
    if job["kind"] == "reference":
        record["recorded_next_day_acc"] = json.loads(
            (SUPP / job["source"] / "test" / f"{job['source_name']}.json").read_text()
        )["primary"]["acc"]
    write_json(STUDY.record("test", job), record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "heldout", "train", "evaluate"], required=True)
    parser.add_argument("--shard", help="i/n: run only array task i of n (0-based)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "plan":
        create_plan()
        return
    plan = STUDY.load()
    jobs = shard(plan["jobs"], args.shard)
    if args.stage == "heldout":
        todo = [j for j in jobs if j["kind"] == "heldout"]
        require_files([model_path(j) for j in todo], "the input-cost and baseline fold models")
        pending = STUDY.pending(todo, "heldout")
        run_pool(heldout_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)
    elif args.stage == "train":
        STUDY.log_environment(f"train-{(args.shard or 'all').replace('/', 'of')}")
        todo = [j for j in jobs if j["kind"] == "draw"]
        pending = STUDY.pending(
            todo,
            "runs",
            lambda j: [STUDY.model(j)],
            lambda j: {"subject": j["subject"], "seed": j["seed"], "epochs": plan["epochs"]},
        )
        run_pool(train_job, [(j["name"], {"job": j, "epochs": plan["epochs"]}) for j in pending], args.workers, 1)
    else:
        todo = [j for j in jobs if j["kind"] in ("draw", "reference")]
        require_files([model_path(j) for j in todo], "noise-draw evaluation inputs (run --stage train first)")
        pending = STUDY.pending(todo, "test", lambda j: [model_path(j)])
        run_pool(evaluate_job, [(j["name"], {"job": j}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
