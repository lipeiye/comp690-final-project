"""Prospective replication on a second two-day dataset: Lee et al. (2019) OpenBMI, 54 people.

The BCI IV 2a work (nine people) found that dropping the least relevant electrodes did not
hurt, that relevance beat random picks, that the people decoded worst seemed to gain most from
fewer electrodes, that 8-bit integer arithmetic lost almost nothing, which tensors need 8 bits,
and that day-to-day alignment helped some people a lot and others not at all. This study fixes
hypotheses that carry those findings over (HYPOTHESES) and the exact tests, before any Lee2019
session-2 trial is scored, and tests them on 54 new people: left- versus right-hand imagery,
62 electrodes, two days, 100 offline and 100 online trials per day (mibci.lee2019). Session 1
trains, ranks and calibrates; session 2 only scores.

Conditions, each x 54 participants x seeds 0, 1, 2, trained with the frozen recipe (EEGNet-8,2,
475 epochs, Adam 1e-3, batch 64, per-channel standardization fit on session 1):

  full-62         every electrode
  relevance-K     the K electrodes with the highest session-1 relevance (mibci.selection: Fisher
                  score of mu and beta log power), K in 8, 16, 20, 32, 45 (45 of 62 is the share
                  16 of 22 that the frozen 2a configuration kept)
  motor-20        the 20 sensorimotor electrodes of the dataset paper (FC, C and CP rows): no data
  random-K        three random lists fixed by seed 690, K in 16 and 45; list i trains with seed i-1
  <cond>-ea       full-62, relevance-45, relevance-16 after Euclidean alignment of session 1

Scored on session 2: FP32 (PyTorch); simulated float, 8/8, 4/8 and the precision map read from
BCI IV 2a session 1 (mibci.fakequant via mibci.replication, calibrated on the model's own
session-1 trials). Aligned models: session 2 aligned with its offline run (an online system
could do this after a warm-up; scored on the online run), with all its trials, and not aligned.

Stages: plan (session-1 rankings, jobs, hypotheses), train, evaluate.
Tables and the hypothesis tests: scripts/analyze_replication_study.py.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mibci import alignment, fakequant, lee2019, replication, selection
from mibci.artifacts import require_files, sha256, write_json
from mibci.experiment import device_for, run_pool
from mibci.layout import frozen_config
from mibci.model import build_eegnet
from mibci.paths import RESULTS_DIR
from mibci.study import Study, shard
from mibci.train import ChannelStandardizer, TrainConfig, fit, predict, scores, set_seed

STUDY = Study(
    "replication_study",
    "scripts/replication_study.py",
    sources=["src/mibci/lee2019.py", "src/mibci/replication.py", "src/mibci/selection.py", "src/mibci/alignment.py",
             "src/mibci/fakequant.py", "src/mibci/study.py", "src/mibci/layout.py"],
)  # fmt: skip
SEEDS = (0, 1, 2)
RELEVANCE_COUNTS = (8, 16, 20, 32, 45)
RANDOM_COUNTS = (16, 45)
N_RANDOM, RANDOM_SEED = 3, 690
ALIGNED = ("full-62", "relevance-45", "relevance-16")
MOTOR = ["FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
         "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"]  # fmt: skip
MAP_SOURCE = RESULTS_DIR / "supplementary" / "quant_rule_study" / "protocol.json"
MAP_NAME = "pooled-0.5"
H = "all session-2 trials; seeds averaged within each participant; 95% participant bootstrap (10,000, seed 690)"
HYPOTHESES = {
    "H1": {"primary": True,
           "claim": "Dropping the least relevant electrodes does not hurt accuracy",
           "contrast": "relevance-45 minus full-62, FP32", "test": "interval lower bound above -2 pp"},
    "H1b": {"claim": "The same at 16 electrodes, the count deployed for BCI IV 2a",
            "contrast": "relevance-16 minus full-62, FP32", "test": "interval lower bound above -2 pp"},
    "H2": {"claim": "Relevance beats random electrodes of the same count",
           "contrast": "relevance-16 minus random-16 (mean of its three lists), FP32",
           "test": "interval lower bound above 0"},
    "H2b": {"claim": "The same at 45 electrodes", "contrast": "relevance-45 minus random-45, FP32",
            "test": "interval lower bound above 0"},
    "H3": {"claim": "People decoded worse with all electrodes gain more from dropping the least relevant ones",
           "contrast": "Spearman rho between full-62 accuracy on the session-2 offline run and relevance-45 minus "
           "full-62 on the online run (separate trials, so measurement noise cannot make the correlation)",
           "test": "rho below 0, one-sided permutation p < 0.05 (10,000 permutations, seed 690)"},
    "H3b": {"claim": "The same with 16 electrodes", "contrast": "as H3 with relevance-16",
            "test": "rho below 0, one-sided permutation p < 0.05"},
    "H4": {"claim": "8-bit integer arithmetic loses almost nothing",
           "contrast": "simulated 8/8 minus simulated float, for full-62, relevance-45 and relevance-16",
           "test": "in each: interval lower bound above -0.5 pp and no participant losing more than 2 pp"},
    "H4b": {"claim": "The precision map read from BCI IV 2a session 1 carries over to another dataset",
            "contrast": f"the 2a map ({MAP_NAME}) minus 8/8, for full-62 and relevance-16",
            "test": "in both: interval lower bound above -2 pp"},
    "H5": {"claim": "Unsupervised day-to-day alignment helps",
           "contrast": "full-62-ea with session 2 aligned on its offline run, minus full-62 unaligned, "
           "both scored on the online run", "test": "interval lower bound above 0"},
    "H5b": {"claim": "The same with 45 and 16 electrodes, and with all session-2 trials aligned",
            "contrast": "relevance-45-ea and relevance-16-ea as H5; each -ea condition aligned with all of session 2 "
            "against its unaligned model on all trials", "test": "interval lower bound above 0, each"},
}  # fmt: skip


def channel_names() -> list[str]:
    names = None
    for subject in lee2019.SUBJECTS:
        found = [str(c) for c in lee2019.load_session(subject, 1)["channels"]]
        if names is not None and found != names:
            raise RuntimeError(f"participant {subject} has a different electrode order")
        names = found
    return names


def relevance_order(subject: int) -> dict:
    """Session-1 relevance (mibci.selection) and its order, as the frozen 2a rule computes it.

    Two participants (50: F9, 51: FTT10h) have an electrode that is exactly zero in some
    session-1 trials, so its log power is -inf and one NaN would turn every min-max-scaled
    score into NaN. Such an electrode (zero power in any session-1 trial) is ranked last; the
    others are scored and ordered as usual (found and decided on session 1 only)."""
    ses = lee2019.load_session(subject, 1)
    with np.errstate(divide="ignore"):
        logpow = selection.band_log_power(ses["X"])
    usable = np.flatnonzero(np.isfinite(logpow).all(axis=(0, 2)))
    dropouts = [int(c) for c in np.setdiff1d(np.arange(logpow.shape[1]), usable)]
    rel = selection.relevance(logpow[:, usable], ses["y"])
    ranked = selection.greedy_ranking(rel, np.zeros((len(rel), len(rel))), 0.0)
    scores = [None] * logpow.shape[1]
    for position, channel in enumerate(usable):
        scores[channel] = float(rel[position])
    return {"relevance_scores": scores, "order": [int(usable[i]) for i in ranked] + dropouts, "dropouts": dropouts}


def plan_jobs(names: list[str], rankings: dict[int, dict]) -> list[dict]:
    rng = np.random.default_rng(RANDOM_SEED)
    randoms = {k: [sorted(rng.choice(len(names), size=k, replace=False).tolist()) for _ in range(N_RANDOM)]
               for k in RANDOM_COUNTS}  # fmt: skip
    motor = sorted(names.index(n) for n in MOTOR)
    jobs = []
    for subject in lee2019.SUBJECTS:
        order = rankings[subject]["order"]
        subsets = {"full-62": list(range(len(names)))} | {f"relevance-{k}": sorted(order[:k]) for k in RELEVANCE_COUNTS}
        subsets["motor-20"] = motor
        # (condition, seed, channels, aligned): every subset with every seed; random list i with seed i - 1
        specs = [(c, seed, channels, False) for c, channels in subsets.items() for seed in SEEDS]
        specs += [(f"random{i + 1}-{k}", i, channels, False) for k, lists in randoms.items()
                  for i, channels in enumerate(lists)]  # fmt: skip
        specs += [(f"{c}-ea", seed, subsets[c], True) for c in ALIGNED for seed in SEEDS]
        jobs.extend(
            {"name": f"sub-{subject:02d}_seed-{seed}_{condition}", "subject": subject, "seed": seed,
             "condition": condition, "channels": channels, "alignment": aligned}
            for condition, seed, channels, aligned in specs
        )  # fmt: skip
    return jobs


def create_plan() -> None:
    names = channel_names()
    rankings = {s: relevance_order(s) for s in lee2019.SUBJECTS}
    source = json.loads(MAP_SOURCE.read_text())
    plan = STUDY.create(
        plan_jobs(names, rankings),
        purpose="Test, on 54 new people recorded on two days, hypotheses carried over from the BCI IV 2a findings: "
        "electrode reduction by relevance, relevance against random, who gains from fewer electrodes, 8-bit "
        "arithmetic, the session-1 precision map, and day-to-day alignment.",
        claim_boundary="Prospective for Lee2019: conditions, hypotheses and tests were fixed in this plan before any "
        "Lee2019 session-2 trial was scored, but they were chosen knowing the BCI IV 2a test results they carry "
        "over. It concerns this dataset's 54 people and its two-class task, not the frozen 2a configuration, which "
        "it does not change. Bit widths below 8 are simulated; nothing is timed.",
        status="prospective_replication",
        lee2019_session2_scored_before_plan=False,
        hypotheses=HYPOTHESES,
        analysis=H,
        dataset="Lee et al. 2019 (OpenBMI) motor imagery via MOABB Lee2019_MI; mibci.lee2019 preprocessing",
        electrodes=names,
        seeds=list(SEEDS),
        epochs=frozen_config()["epochs"],
        relevance_counts=list(RELEVANCE_COUNTS),
        random_counts=list(RANDOM_COUNTS),
        motor=MOTOR,
        rankings={str(s): r for s, r in rankings.items()},
        precision_map={"name": MAP_NAME, **source["pooled_maps"][MAP_NAME], "source_sha256": sha256(MAP_SOURCE)},
        alignment="Euclidean alignment per recording day (He and Wu 2019), as scripts/alignment_study.py",
    )
    counts = {}
    for job in plan["jobs"]:
        counts[job["condition"]] = counts.get(job["condition"], 0) + 1
    print(f"{len(plan['jobs'])} jobs: {counts}")


def train_job(job: dict, epochs: int) -> None:
    ses = lee2019.load_session(job["subject"], 1)
    X, y = ses["X"][:, job["channels"]], ses["y"].astype(np.int64)
    if job["alignment"]:
        X = alignment.aligned(X)
    scaler = ChannelStandardizer().fit(X)
    cfg = TrainConfig(epochs=epochs, seed=job["seed"])
    set_seed(cfg.seed)
    model = build_eegnet(len(job["channels"]), n_classes=len(lee2019.CLASSES))
    device = device_for(job["subject"])
    result = fit(model, scaler.transform(X), y, cfg, device=device)
    path = STUDY.model(job)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"subject": job["subject"], "seed": job["seed"], "condition": job["condition"], "epochs": epochs,
            "device": device, "alignment": job["alignment"]}  # fmt: skip
    replication.save(path, model, job["channels"], scaler, **meta)
    write_json(STUDY.record("runs", job), {
        **meta, "name": job["name"], "n_train": len(y), "train_time_s": result["train_time_s"],
        "train_acc": scores(predict(model, scaler.transform(X)), y)["acc"], "model_sha256": sha256(path),
        "protocol_sha256": sha256(STUDY.protocol_path),
    })  # fmt: skip


def scored(pred: np.ndarray, y: np.ndarray, run: np.ndarray) -> dict:
    return {
        "acc_all": float((pred == y).mean()),
        "acc_offline": float((pred[run == 0] == y[run == 0]).mean()),
        "acc_online": float((pred[run == 1] == y[run == 1]).mean()),
        "pred": pred.tolist(),
    }


def evaluate_job(job: dict, precision_map: dict) -> None:
    torch.set_num_threads(1)
    path = STUDY.model(job)
    model, scaler, bundle = replication.load(path)
    if bundle["channels"] != job["channels"]:
        raise RuntimeError(f"Bundle channels differ from the plan: {path}")
    X1 = lee2019.load_session(job["subject"], 1)["X"][:, job["channels"]]
    ses = lee2019.load_session(job["subject"], 2, final_evaluation=True)
    X2, y, run = ses["X"][:, job["channels"]], ses["y"], ses["run"]
    out = {}
    if job["alignment"]:
        inputs = {"ea-offline": alignment.aligned(X2, reference=X2[run == 0]), "ea-all": alignment.aligned(X2),
                  "none": X2}  # fmt: skip
        for name, x in inputs.items():
            out[name] = scored(predict(model, scaler.transform(x)).argmax(1), y, run)
    else:
        X_test, calibration = scaler.transform(X2), scaler.transform(X1)
        logits = predict(model, X_test)
        out["fp32"] = scored(logits.argmax(1), y, run)
        folded = replication.folded(model)
        ranges = fakequant.calibrate(folded, calibration)
        settings = {"float": (None, None), "8/8": (8, 8), "4/8": (4, 8),
                    "map-2a": (precision_map["weights"], precision_map["activations"])}  # fmt: skip
        for name, (wb, ab) in settings.items():
            sim = fakequant.logits(fakequant.quantized(folded, ranges, wb, ab), X_test)
            out[name] = scored(sim.argmax(1), y, run)
            if name == "float":
                out[name]["max_abs_logit_difference_from_fp32"] = float(np.abs(sim - logits).max())
    write_json(STUDY.record("test", job), {
        **job, "scores": out, "y": y.tolist(), "run": run.tolist(), "model_sha256": sha256(path),
        "protocol_sha256": sha256(STUDY.protocol_path),
    })  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["plan", "train", "evaluate"], required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=lee2019.SUBJECTS)
    parser.add_argument("--shard", help="i/n: run only array task i of n (0-based)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "plan":
        create_plan()
        return
    plan = STUDY.load()
    jobs = [j for j in shard(plan["jobs"], args.shard) if j["subject"] in args.subjects]
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
        require_files([STUDY.model(j) for j in jobs], "replication evaluation inputs (run --stage train first)")
        pending = STUDY.pending(jobs, "test", lambda j: [STUDY.model(j)])
        kwargs = {"precision_map": plan["precision_map"]}
        run_pool(evaluate_job, [(j["name"], {"job": j, **kwargs}) for j in pending], args.workers, 1)


if __name__ == "__main__":
    main()
