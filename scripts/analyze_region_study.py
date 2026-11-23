"""Tables and figures for the region study (scripts/region_study.py).

Question: which functional scalp regions carry the motor-imagery signal, and how few electrodes
does a montage made of whole regions need? The 22 electrodes form nine fine regions
(mibci.regions.FINE), and each of the 511 non-empty region sets was trained for every
participant. Bit i of a record's `mask` is the i-th region in FINE order.

Reads, under MIBCI_RESULTS_DIR (default results/):
  supplementary/region_study/protocol.json      the plan: every job and its region set
  supplementary/region_study/runs/<name>.json   cv jobs: scores on held-out session-1 runs (folds 0-2)
  supplementary/region_study/test/<name>.json   final jobs: session-2 scores (--stage evaluate)
  test/sub-XX_seed-0_{relevance-16,relevance-8,full-22}.json            archived main-study runs
  supplementary/electrode_count/test/sweep/sub-XX_seed-0_relevance-K.json  K = 20, 18, 14, 12, 10

Computes, per participant: the value of every region set (accuracy, and the recall of each
class; the three folds pooled with weights equal to their held-out trials); exact Shapley values
of the nine regions, i.e. the accuracy a region adds averaged over every set it can join, with
the empty set scoring chance (25 %); the pairwise Shapley interaction index; the same values for
the four coarse regions (mibci.regions.COARSE) from the 15 sets made of whole coarse regions; and,
for every budget of 2 to 22 electrodes, the region set that each of three rules picks: the
participant's own best cross-validation accuracy, the best mean cross-validation accuracy of the
other eight participants, and the best session-2 accuracy (the oracle). Means over participants
carry the participant bootstrap interval of the frozen plan (mibci.stats).

Boundary: cross-validation numbers use session 1 only and are test-free. Every session-2 number
is exploratory, because the study was planned after the main study's test results were known.
The oracle chooses with the test day, so it is an upper bound, not a method. Session-2 EEG is
never loaded here; session-2 numbers come only from the score records listed above.

Writes to supplementary/region_study/: coalitions.csv, shapley.csv, shapley_coarse.csv,
interactions.csv, frontier.csv, frontier_choices.csv, relevance_comparison.csv,
fig_region_shapley.png, fig_class_shapley.png, fig_budget_frontier.png, fig_scalp_shapley.png,
summary.md and analysis.json. Refuses to run until every planned record exists: exact Shapley
values need all 511 sets.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from math import factorial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from mibci import data, regions
from mibci.artifacts import ArtifactError, require_files, write_json
from mibci.paths import RESULTS_DIR, display
from mibci.stats import FROZEN_BOOTSTRAP, bootstrap_ci

OUT = RESULTS_DIR / "supplementary" / "region_study"
SWEEP = RESULTS_DIR / "supplementary" / "electrode_count" / "test" / "sweep"
FINE = list(regions.FINE)  # the players, in the bit order of `mask`
COARSE = list(regions.COARSE)
CLASSES = data.CLASSES
CHANCE = 1 / len(CLASSES)  # the value of the empty set: guessing among four classes
SEED = 0  # the region study trains seed 0 only, so it is compared with the seed-0 relevance runs
BUDGETS = list(range(2, len(data.EEG_CHANNELS) + 1))
RELEVANCE_COUNTS = (20, 18, 16, 14, 12, 10, 8)
ARCHIVED = (16, 8)  # electrode counts kept in results/test; the others come from the electrode-count sweep
TIE = 1e-9  # equal accuracies can differ in the last bits after pooling; unequal ones (count ratios) by far more
METHODS = {
    "own_cv": "own cross-validation (test-free)",
    "population": "other eight participants' cross-validation (test-free)",
    "oracle": "oracle: best on session 2 (chosen with the test day)",
    "all_22": "all 22 electrodes",
}
COLORS = {"own_cv": "#2a78d6", "population": "#eb6834", "oracle": "#1baf7a"}
INK, SECONDARY, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list("shapley", ["#9b1c1f", "#e34948", "#f0efec", "#3987e5", "#104281"])


@dataclass
class Landscape:
    """Value of every region set for every participant; index 0 of the last axis is the empty set (chance)."""

    subjects: list[int]
    cv: np.ndarray  # (participants, 1 + classes, sets): accuracy, then each class's recall; held-out session-1 runs
    test: np.ndarray  # the same on the artifact-free session-2 trials
    test_all: np.ndarray  # (participants, sets): accuracy on all session-2 trials


# ---------------------------------------------------------------- region sets


def members(mask: int, names: list[str]) -> list[str]:
    return [name for i, name in enumerate(names) if (mask >> i) & 1]


def set_name(mask: int, names: list[str]) -> str:
    return regions.coalition_name(tuple(members(mask, names)))


def set_sizes(n: int) -> np.ndarray:
    """Number of players in each of the sets 0 .. 2**n - 1."""
    return np.array([bin(m).count("1") for m in range(2**n)])


def electrodes_per_set() -> np.ndarray:
    per_region = np.array([len(regions.FINE[name]) for name in FINE])
    bits = (np.arange(2 ** len(FINE))[:, None] >> np.arange(len(FINE))) & 1
    return bits @ per_region


def coarse_sets() -> np.ndarray:
    """Fine mask of each coarse region set (index: coarse mask); each coarse region must be a union of fine ones."""
    fine_masks = []
    for name in COARSE:
        bits = [i for i, fine in enumerate(FINE) if set(regions.FINE[fine]) <= set(regions.COARSE[name])]
        if sorted(e for i in bits for e in regions.FINE[FINE[i]]) != sorted(regions.COARSE[name]):
            raise ValueError(f"coarse region {name} is not a union of fine regions")
        fine_masks.append(sum(1 << i for i in bits))
    return np.array([sum(f for c, f in enumerate(fine_masks) if (m >> c) & 1) for m in range(2 ** len(COARSE))])


# ---------------------------------------------------------------- records


def require_records(paths: list[Path], context: str) -> None:
    """require_files for thousands of records: say how many are missing and show a few, not all of them."""
    missing = [p for p in paths if not p.is_file() or p.stat().st_size == 0]
    if missing:
        shown = "\n  ".join(display(p) for p in missing[:5])
        raise ArtifactError(f"{context}: {len(missing)} of {len(paths)} records missing or empty, e.g.\n  {shown}")


def read_record(path: Path, job: dict) -> dict:
    """A record, after checking that it belongs to the job the plan lists under its name."""
    record = json.loads(path.read_text())
    for key in ("subject", "mask", "regions"):
        if record[key] != job[key]:
            raise ArtifactError(f"{display(path)}: {key} is {record[key]!r}, but the plan says {job[key]!r}")
    return record


def load_plan() -> list[dict]:
    path = OUT / "protocol.json"
    require_files([path], "region-study analysis (run scripts/region_study.py --stage plan first)")
    jobs = json.loads(path.read_text())["jobs"]
    for job in jobs:
        if job["regions"] != members(job["mask"], FINE):
            raise ArtifactError(f"{job['name']}: mask {job['mask']} does not name the regions {job['regions']}")
    return jobs


def with_chance(values: np.ndarray, what: str) -> np.ndarray:
    """Every non-empty set needs a score; the empty set (index 0) scores chance."""
    if np.isnan(values[..., 1:]).any():
        raise ArtifactError(f"{what}: a participant lacks a score for some region set; the plan is incomplete")
    values[..., 0] = CHANCE
    return values


def cv_values(jobs: list[dict], subjects: list[int]) -> np.ndarray:
    """Accuracy and per-class recall on the held-out session-1 runs, from the three folds' confusion
    matrices added together (a class's recall pools its trials over folds, whose class counts differ)."""
    cv = [j for j in jobs if j["part"] == "cv"]
    paths = [OUT / "runs" / f"{j['name']}.json" for j in cv]
    require_records(paths, "region-study analysis (run scripts/region_study.py --stage train --part cv first)")
    pooled = np.zeros((len(subjects), 2 ** len(FINE), len(CLASSES), len(CLASSES)))
    for job, path in zip(cv, paths, strict=True):
        record = read_record(path, job)
        if record["split"] != job["split"] or record["part"] != "cv":
            raise ArtifactError(f"{display(path)}: not the {job['split']} cross-validation run the plan lists")
        pooled[subjects.index(job["subject"]), job["mask"]] += np.array(record["val"]["confusion"])
    with np.errstate(invalid="ignore", divide="ignore"):
        accuracy = np.trace(pooled, axis1=2, axis2=3) / pooled.sum(axis=(2, 3))
        recall = np.diagonal(pooled, axis1=2, axis2=3) / pooled.sum(axis=3)
        values = np.concatenate([accuracy[:, None, :], recall.transpose(0, 2, 1)], axis=1)
    return with_chance(values, "cross-validation")


def test_values(jobs: list[dict], subjects: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Session-2 accuracy and per-class recall (artifact-free trials), and accuracy on all trials."""
    final = [j for j in jobs if j["part"] == "final"]
    paths = [OUT / "test" / f"{j['name']}.json" for j in final]
    require_records(paths, "region-study analysis (run scripts/region_study.py --stage evaluate first)")
    values = np.full((len(subjects), 2 + len(CLASSES), 2 ** len(FINE)), np.nan)
    for job, path in zip(final, paths, strict=True):
        record = read_record(path, job)
        primary = record["primary"]
        scores = [primary["acc"], *primary["recall"], record["secondary"]["acc"]]
        values[subjects.index(job["subject"]), :, job["mask"]] = scores
    values = with_chance(values, "session 2")
    return values[:, :-1], values[:, -1]


def coalition_table(land: Landscape, electrodes: np.ndarray) -> pd.DataFrame:
    grid = np.meshgrid(np.arange(len(land.subjects)), np.arange(1, 2 ** len(FINE)), indexing="ij")
    s, mask = (a.ravel() for a in grid)
    table = {
        "subject": np.asarray(land.subjects)[s],
        "mask": mask,
        "regions": [set_name(m, FINE) for m in mask],
        "n_regions": set_sizes(len(FINE))[mask],
        "n_electrodes": electrodes[mask],
    }
    for basis, values in (("cv", land.cv), ("test", land.test)):
        table[f"{basis}_acc_pct"] = 100 * values[s, 0, mask]
        for i, name in enumerate(CLASSES):
            table[f"{basis}_recall_{name}_pct"] = 100 * values[s, 1 + i, mask]
    table["test_acc_all_trials_pct"] = 100 * land.test_all[s, mask]
    return pd.DataFrame(table)


# ---------------------------------------------------------------- Shapley values


def players(v: np.ndarray) -> int:
    n = v.shape[-1].bit_length() - 1
    if v.shape[-1] != 2**n:
        raise ValueError("the last axis must hold the value of all 2**n sets")
    return n


def shapley(v: np.ndarray) -> np.ndarray:
    """Exact Shapley value of each player; v[..., mask] is the value of the set `mask`.

    phi_i = sum over sets S without i of |S|! (n - |S| - 1)! / n! * (v(S + i) - v(S))."""
    n = players(v)
    sets, size = np.arange(2**n), set_sizes(n)
    weight = np.array([factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)])
    without = np.array([sets[((sets >> i) & 1) == 0] for i in range(n)])  # (n, 2**(n-1))
    joined = without | (1 << np.arange(n))[:, None]
    return ((v[..., joined] - v[..., without]) * weight[size[without]]).sum(-1)


def interactions(v: np.ndarray) -> np.ndarray:
    """Shapley interaction index of every pair of players, in itertools.combinations order.

    I_ij = sum over sets S without i and j of |S|! (n - |S| - 2)! / (n - 1)! * (v(S+i+j) - v(S+i) - v(S+j) + v(S))."""
    n = players(v)
    sets, size = np.arange(2**n), set_sizes(n)
    pairs = list(combinations(range(n), 2))
    weight = np.array([factorial(k) * factorial(n - k - 2) / factorial(n - 1) for k in range(n - 1)])
    rest = np.array([sets[(((sets >> i) | (sets >> j)) & 1) == 0] for i, j in pairs])  # (pairs, 2**(n-2))
    bit_i = np.array([1 << i for i, _ in pairs])[:, None]
    bit_j = np.array([1 << j for _, j in pairs])[:, None]
    delta = v[..., rest | bit_i | bit_j] - v[..., rest | bit_i] - v[..., rest | bit_j] + v[..., rest]
    return (delta * weight[size[rest]]).sum(-1)


def games(land: Landscape) -> dict[tuple[str, str], np.ndarray]:
    """(partition, basis) -> value of every set in percentage points, shape (participants, 1 + classes, sets)."""
    coarse = coarse_sets()
    out = {}
    for basis, values in (("cv", land.cv), ("test", land.test)):
        out["fine", basis] = 100 * values
        out["coarse", basis] = 100 * values[..., coarse]
    return out


def efficiency_error(phi: np.ndarray, v: np.ndarray) -> float:
    """Largest gap between a participant's summed Shapley values and v(all) - v(empty); zero up to rounding."""
    return float(np.abs(phi.sum(-1) - (v[..., -1] - v[..., 0])).max())


def interval(values: np.ndarray) -> dict:
    """Mean over participants with the participant bootstrap interval of the frozen plan."""
    lo, hi = bootstrap_ci(values)
    return {"mean": float(np.mean(values)), "ci_low": lo, "ci_high": hi}


def shapley_table(phi: dict, partition: str, subjects: list[int]) -> pd.DataFrame:
    names = regions.PARTITIONS[partition]
    rows = []
    for basis in ("cv", "test"):
        values = phi[partition, basis]
        rows.extend(
            {
                "subject": subject,
                "basis": basis,
                "region": name,
                "n_electrodes": len(names[name]),
                "value_pp": values[s, 0, i],
                **{f"value_{c}_pp": values[s, 1 + k, i] for k, c in enumerate(CLASSES)},
            }
            for s, subject in enumerate(subjects)
            for i, name in enumerate(names)
        )
    return pd.DataFrame(rows)


def shapley_summary(phi: dict, partition: str) -> dict:
    """{basis: {region: {"accuracy": interval, "recall": {class: interval}}}}, in percentage points."""
    return {
        basis: {
            name: {
                "accuracy": interval(phi[partition, basis][:, 0, i]),
                "recall": {c: interval(phi[partition, basis][:, 1 + k, i]) for k, c in enumerate(CLASSES)},
            }
            for i, name in enumerate(regions.PARTITIONS[partition])
        }
        for basis in ("cv", "test")
    }


def interaction_table(pair_values: np.ndarray, subjects: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"subject": subject, "region_a": a, "region_b": b, "value_pp": pair_values[s, p]}
            for s, subject in enumerate(subjects)
            for p, (a, b) in enumerate(combinations(FINE, 2))
        ]
    )


def pair_summary(pair_values: np.ndarray) -> list[dict]:
    return [{"a": a, "b": b, **interval(pair_values[:, p])} for p, (a, b) in enumerate(combinations(FINE, 2))]


def top_regions(phi: np.ndarray, subjects: list[int]) -> dict[str, str]:
    """Each participant's region with the largest cross-validation Shapley value."""
    return {f"P{subject}": FINE[int(np.argmax(phi[s, 0]))] for s, subject in enumerate(subjects)}


# ---------------------------------------------------------------- electrode budgets


def pick(values: np.ndarray, allowed: np.ndarray, electrodes: np.ndarray) -> int:
    """The best allowed set: the highest value, then fewer electrodes, then the lower mask."""
    candidates = np.flatnonzero(allowed)
    best = values[candidates].max()
    tied = candidates[values[candidates] >= best - TIE]
    return int(tied[np.lexsort((tied, electrodes[tied]))[0]])


def choice_rows(land: Landscape, electrodes: np.ndarray, budget: int) -> list[dict]:
    """The set each rule picks for each participant with at most `budget` electrodes, and how that set scores."""
    cv, test = land.cv[:, 0], land.test[:, 0]
    full = len(electrodes) - 1
    allowed = (electrodes <= budget) & (np.arange(len(electrodes)) > 0)
    rows = []
    for s, subject in enumerate(land.subjects):
        picked = {
            "own_cv": pick(cv[s], allowed, electrodes),
            "population": pick(np.delete(cv, s, axis=0).mean(0), allowed, electrodes),  # leaves participant s out
            "oracle": pick(test[s], allowed, electrodes),
            "all_22": full,
        }
        rows.extend(
            {
                "budget": budget,
                "method": method,
                "subject": subject,
                "mask": mask,
                "regions": set_name(mask, FINE),
                "n_electrodes": int(electrodes[mask]),
                "cv_acc_pct": 100 * cv[s, mask],
                "test_acc_pct": 100 * test[s, mask],
                "all22_test_acc_pct": 100 * test[s, full],
                "minus_all22_pp": 100 * (test[s, mask] - test[s, full]),
            }
            for method, mask in picked.items()
        )
    return rows


def frontier_table(chosen: pd.DataFrame) -> pd.DataFrame:
    """Per budget and rule: mean session-2 accuracy and the paired difference from all 22 electrodes."""
    rows = []
    for (budget, method), g in chosen.groupby(["budget", "method"], sort=False):
        d = g["minus_all22_pp"].to_numpy()
        lo, hi = bootstrap_ci(d)
        common, times = Counter(g["regions"]).most_common(1)[0]
        rows.append(
            {
                "budget": budget,
                "method": method,
                "test_acc_pct": g["test_acc_pct"].mean(),
                "cv_acc_pct": g["cv_acc_pct"].mean(),
                "minus_all22_pp": d.mean(),
                "ci_low_pp": lo,
                "ci_high_pp": hi,
                "n_better": int((d > 0).sum()),
                "n_worse": int((d < 0).sum()),
                "mean_electrodes": g["n_electrodes"].mean(),
                "distinct_sets": g["regions"].nunique(),
                "most_common_set": common,
                "most_common_count": times,
            }
        )
    return pd.DataFrame(rows)


def relevance_runs(subjects: list[int]) -> pd.DataFrame:
    """Seed-0 session-2 accuracy (artifact-free trials, FP32) of the relevance-ranked montages."""
    wanted = [
        (k, s, (RESULTS_DIR / "test" if k in ARCHIVED else SWEEP) / f"sub-{s:02d}_seed-{SEED}_relevance-{k}.json")
        for k in RELEVANCE_COUNTS
        for s in subjects
    ]
    require_files([w[2] for w in wanted], "relevance comparison (archived results/test and the electrode-count sweep)")
    rows = []
    for k, s, path in wanted:
        record = json.loads(path.read_text())
        if (record["subject"], record["seed"], record["condition"]) != (s, SEED, f"relevance-{k}"):
            raise ArtifactError(f"{display(path)}: not the seed-{SEED} relevance-{k} run of participant {s}")
        rows.append({"electrodes": k, "subject": s, "acc_pct": 100 * record["primary"]["fp32"]["acc"]})
    return pd.DataFrame(rows)


def relevance_table(chosen: pd.DataFrame, relevance: pd.DataFrame) -> pd.DataFrame:
    """Region set chosen with at most k electrodes minus relevance-k, paired by participant."""
    rows = []
    for k in RELEVANCE_COUNTS:
        rel = relevance[relevance["electrodes"] == k].set_index("subject")["acc_pct"]
        for method in ("own_cv", "population"):
            region = chosen[(chosen["budget"] == k) & (chosen["method"] == method)].set_index("subject")
            d = (region["test_acc_pct"] - rel.reindex(region.index)).to_numpy()
            lo, hi = bootstrap_ci(d)
            rows.append(
                {
                    "electrodes": k,
                    "method": method,
                    "region_acc_pct": region["test_acc_pct"].mean(),
                    "region_mean_electrodes": region["n_electrodes"].mean(),
                    "relevance_acc_pct": rel.mean(),
                    "minus_relevance_pp": d.mean(),
                    "ci_low_pp": lo,
                    "ci_high_pp": hi,
                    "n_region_better": int((d > 0).sum()),
                    "n_relevance_better": int((d < 0).sum()),
                    "per_participant_pp": [round(float(x), 2) for x in d],
                }
            )
    return pd.DataFrame(rows)


def full_montage_check(jobs: list[dict], subjects: list[int]) -> list[dict]:
    """The all-22 region set repeats the main study's full-22 recipe with seed 0: compare it with the archived run."""
    final = {j["subject"]: j for j in jobs if j["part"] == "final" and j["mask"] == 2 ** len(FINE) - 1}
    paths = [RESULTS_DIR / "test" / f"sub-{s:02d}_seed-{SEED}_full-22.json" for s in subjects]
    require_files(paths, "pipeline check (archived full-22 runs in results/test)")
    rows = []
    for s, path in zip(subjects, paths, strict=True):
        archived = json.loads(path.read_text())
        ours = json.loads((OUT / "test" / f"{final[s]['name']}.json").read_text())
        if ours["y"] != archived["y"]:
            raise ArtifactError(f"{display(path)}: session-2 labels differ from the region study's record")
        rows.append(
            {
                "subject": s,
                "region_study_acc_pct": 100 * ours["primary"]["acc"],
                "archived_acc_pct": 100 * archived["primary"]["fp32"]["acc"],
                "prediction_agreement": float(np.mean(np.array(ours["pred"]) == np.array(archived["pred"]["fp32"]))),
            }
        )
    return rows


# ---------------------------------------------------------------- figures


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK,
            "xtick.color": SECONDARY,
            "ytick.color": SECONDARY,
        }
    )


def figure_shapley(fine: dict, all22_pp: float) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(FINE))[::-1]  # the first region at the top
    cv = [fine["cv"][r]["accuracy"] for r in FINE]
    means = np.array([c["mean"] for c in cv])
    err = np.array([means - [c["ci_low"] for c in cv], [c["ci_high"] for c in cv] - means])
    ax.barh(y, means, height=0.62, color=COLORS["own_cv"], label="cross-validation, session 1 (mean, 95% interval)")
    ax.errorbar(means, y, xerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    test = [fine["test"][r]["accuracy"]["mean"] for r in FINE]
    ax.scatter(
        test, y, s=48, facecolors="white", edgecolors=INK, linewidths=1.3, zorder=3, label="session 2 (exploratory)"
    )
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y, [f"{r}  ({' '.join(regions.FINE[r])})" for r in FINE])
    ax.set_xlabel("Shapley value: accuracy the region adds (percentage points)")
    ax.set_title("What each region adds, averaged over every set of other regions it can join")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="best")
    fig.text(
        0.99,
        0.01,
        f"The nine cross-validation values add up to all-22 accuracy minus chance: {all22_pp:.1f} pp (mean).",
        ha="right",
        fontsize=8,
        color=SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig_region_shapley.png", dpi=150)
    plt.close(fig)


def figure_classes(fine: dict) -> None:
    values = np.array([[fine["cv"][r]["recall"][c]["mean"] for c in CLASSES] for r in FINE])
    limit = float(np.abs(values).max())
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    image = ax.imshow(values, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    for (i, j), v in np.ndenumerate(values):
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=9, color="white" if abs(v) > 0.6 * limit else INK)
    ax.set_xticks(range(len(CLASSES)), [c.replace("_", " ") for c in CLASSES])
    ax.set_yticks(range(len(FINE)), FINE)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, shrink=0.85, label="Shapley value for the recall of the class (pp)")
    ax.set_title("Which region helps which class (cross-validation, session 1)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_class_shapley.png", dpi=150)
    plt.close(fig)


def figure_frontier(frontier: pd.DataFrame, land: Landscape, electrodes: np.ndarray, relevance: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(electrodes) - 1)
    mean_sets = 100 * land.test[:, 0, 1:].mean(0)
    ax.scatter(electrodes[1:] + jitter, mean_sets, s=9, color="#cfcdc6", lw=0, zorder=1, label="every region set")
    for method, marker in (("own_cv", "o"), ("population", "s"), ("oracle", "^")):
        f = frontier[frontier["method"] == method]
        ax.plot(
            f["budget"],
            f["test_acc_pct"],
            marker=marker,
            ms=5,
            lw=2,
            ls="--" if method == "oracle" else "-",
            color=COLORS[method],
            zorder=3,
            label=f"best by {METHODS[method]}",
        )
    full = frontier.loc[frontier["method"] == "all_22", "test_acc_pct"].iloc[0]
    ax.axhline(full, color=MUTED, ls=":", lw=1.2, zorder=2, label=f"all 22 electrodes ({full:.1f}%)")
    rel = relevance.groupby("electrodes")["acc_pct"].mean()
    label = "k most relevant electrodes (per-electrode rule)"
    ax.scatter(rel.index, rel.to_numpy(), marker="D", s=42, color=INK, zorder=4, label=label)
    ax.set(
        xlabel="electrode budget (a region set may use fewer; grey dots: actual electrodes)",
        ylabel="session-2 accuracy (%), mean of nine participants",
        xticks=BUDGETS[::2],
        title="Session-2 accuracy against electrode budget (exploratory; seed 0)",
    )
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.savefig(OUT / "fig_budget_frontier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_scalp(fine: dict) -> None:
    per_electrode = {e: fine["cv"][r]["accuracy"]["mean"] / len(regions.FINE[r]) for r in FINE for e in regions.FINE[r]}
    limit = max(abs(v) for v in per_electrode.values())
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for electrodes in regions.FINE.values():  # a thick grey band joins the electrodes of one region
        for a, b in combinations(electrodes, 2):
            if b in regions.neighbours(a):
                (xa, ya), (xb, yb) = regions.POSITIONS[a], regions.POSITIONS[b]
                ax.plot([xa, xb], [ya, yb], color="#d9d7cf", lw=14, solid_capstyle="round", zorder=1)
    names = list(regions.POSITIONS)
    xy = np.array([regions.POSITIONS[e] for e in names])
    values = np.array([per_electrode[e] for e in names])
    dots = ax.scatter(
        xy[:, 0], xy[:, 1], c=values, cmap=DIVERGING, vmin=-limit, vmax=limit, s=950, edgecolors=INK, lw=0.6, zorder=2
    )
    for e, (x, y), v in zip(names, xy, values, strict=True):
        ink = "white" if abs(v) > 0.6 * limit else INK
        ax.text(x, y + 0.12, e, ha="center", va="center", fontsize=8.5, weight="bold", color=ink, zorder=3)
        ax.text(x, y - 0.17, f"{v:+.1f}", ha="center", va="center", fontsize=7.5, color=ink, zorder=3)
    ax.plot([-0.35, 0, 0.35], [2.55, 2.95, 2.55], color=MUTED, lw=1.2)
    ax.text(0, 3.1, "nose", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.text(-3.6, 0, "left", ha="right", va="center", fontsize=9, color=MUTED)
    ax.text(3.6, 0, "right", ha="left", va="center", fontsize=9, color=MUTED)
    ax.set(xlim=(-4.4, 4.4), ylim=(-3.6, 3.5), aspect="equal")
    ax.axis("off")
    fig.colorbar(dots, ax=ax, shrink=0.72, label="pp per electrode (region's Shapley value / its electrodes)")
    ax.set_title("Cross-validation Shapley value per electrode; grey bands join a region")
    fig.savefig(OUT / "fig_scalp_shapley.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- summary.md, one function per section


def ci_text(x: dict, digits: int = 2) -> str:
    return f"{x['mean']:+.{digits}f} [{x['ci_low']:+.{digits}f}, {x['ci_high']:+.{digits}f}]"


def intro_lines(subjects: list[int], counts: dict, all22: dict) -> list[str]:
    return [
        "# Region study (supplementary, post-test; generated by scripts/analyze_region_study.py)",
        "",
        "The 22 electrodes form nine scalp regions. Every one of the 511 non-empty sets of regions was trained "
        f"for each of the {len(subjects)} participants: {counts['cv']:,} cross-validation runs on session 1 (three "
        f"folds per set) and {counts['test']:,} final models scored on session 2. With all 22 electrodes the mean "
        f"accuracy is {all22['cv']:.2f}% in cross-validation and {all22['test']:.2f}% on session 2 "
        "(artifact-free trials). Chance is 25%.",
        "",
        "**What can be claimed.** Cross-validation (CV) numbers use session 1 only, so they are test-free. Every "
        "session-2 number is exploratory: the study was planned after the main study's test results were known. "
        "The oracle picks the best set on session 2 itself, so it is an upper bound, not a usable method.",
        "",
        f"Intervals are {FROZEN_BOOTSTRAP['level']:.0%} percentile bootstrap intervals over the participants "
        f"({FROZEN_BOOTSTRAP['resamples']:,} resamples, seed {FROZEN_BOOTSTRAP['seed']}), as in the frozen plan.",
    ]


def shapley_lines(fine: dict, efficiency: dict, top: dict) -> list[str]:
    lines = [
        "",
        "## What each region adds (Shapley value, percentage points of accuracy)",
        "",
        "A region's Shapley value is the accuracy it adds when it joins a set of other regions, averaged over all "
        "such sets so that every set size counts equally. The empty set scores chance (25%), so a participant's "
        "nine values add up to that participant's all-22 accuracy minus 25 points.",
        "",
        "| region | electrodes | CV [95% interval] | CV per electrode | session 2, exploratory [95% interval] |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {r} | {' '.join(regions.FINE[r])} | {ci_text(fine['cv'][r]['accuracy'])} | "
        f"{fine['cv'][r]['accuracy']['mean'] / len(regions.FINE[r]):+.2f} | {ci_text(fine['test'][r]['accuracy'])} |"
        for r in FINE
    )
    lines += [
        "",
        f"Check: summed values equal all-22 accuracy minus chance to within {max(efficiency.values()):.1e} points, "
        "for every participant, basis and class.",
        "",
        "Region with the largest CV value, by participant: " + ", ".join(f"{p} {r}" for p, r in top.items()) + ".",
    ]
    return lines


def class_lines(fine: dict) -> list[str]:
    lines = [
        "",
        "## Which region helps which class (CV Shapley value for each class's recall, pp)",
        "",
        "Recall is the share of a class's trials labelled correctly; each column adds up to the all-22 recall of "
        "that class minus 25 points. Means over participants; intervals are in analysis.json.",
        "",
        "| region | " + " | ".join(c.replace("_", " ") for c in CLASSES) + " |",
        "| --- |" + " ---: |" * len(CLASSES),
    ]
    lines.extend(
        f"| {r} | " + " | ".join(f"{fine['cv'][r]['recall'][c]['mean']:+.2f}" for c in CLASSES) + " |" for r in FINE
    )
    return lines


def interaction_lines(pairs: list[dict]) -> list[str]:
    ordered = sorted(pairs, key=lambda p: p["mean"], reverse=True)
    lines = [
        "",
        "## Pairs of regions (Shapley interaction index, CV, pp)",
        "",
        "Positive: two regions add more together than separately (they complement each other). Negative: they "
        "carry overlapping information, so the second adds less once the first is in. The three largest and the "
        "three smallest of the 36 pairs (all pairs: interactions.csv and analysis.json):",
        "",
        "| regions | interaction [95% interval] |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {p['a']} + {p['b']} | {ci_text(p)} |" for p in [*ordered[:3], *ordered[-3:]])
    return lines


def coarse_lines(coarse: dict) -> list[str]:
    lines = [
        "",
        "## Four coarse regions (Shapley value from the 15 sets made of whole coarse regions, pp)",
        "",
        "| coarse region | electrodes | CV [95% interval] | session 2, exploratory [95% interval] |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {r} | {len(regions.COARSE[r])} | {ci_text(coarse['cv'][r]['accuracy'])} | "
        f"{ci_text(coarse['test'][r]['accuracy'])} |"
        for r in COARSE
    )
    return lines


def frontier_cell(r: pd.Series) -> str:
    return (
        f"{r['test_acc_pct']:.1f}; {r['minus_all22_pp']:+.1f} [{r['ci_low_pp']:+.1f}, {r['ci_high_pp']:+.1f}]; "
        f"{r['n_better']}/{r['n_worse']}"
    )


def frontier_lines(frontier: pd.DataFrame) -> list[str]:
    table = frontier.set_index(["budget", "method"])
    lines = [
        "",
        "## How few electrodes: the best region set within a budget (session-2 accuracy, exploratory)",
        "",
        "Three rules pick a region set with at most the budgeted number of electrodes:",
        "",
        "- **own CV**: the participant's own best CV accuracy. Test-free.",
        "- **population**: the best mean CV accuracy of the other eight participants, so the choice never sees "
        "the participant it is used for. Test-free.",
        "- **oracle**: the best session-2 accuracy. It chooses with the test day: an upper bound only.",
        "",
        "Ties go to fewer electrodes, then to the lower set number (`mask`). Each cell: mean session-2 accuracy "
        "(%); difference from all 22 electrodes (pp) [95% interval]; participants better/worse than with all 22.",
        "",
        "| budget | own CV | population | oracle |",
        "| ---: | --- | --- | --- |",
    ]
    lines.extend(
        f"| {k} | " + " | ".join(frontier_cell(table.loc[(k, m)]) for m in ("own_cv", "population", "oracle")) + " |"
        for k in BUDGETS
    )
    lines += [
        "",
        "Population rule: the set chosen most often across the nine leave-one-out choices.",
        "",
        "| budget | most common set | times chosen (of 9) | different sets chosen |",
        "| ---: | --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {k} | {r['most_common_set']} | {r['most_common_count']} | {r['distinct_sets']} |"
        for k in BUDGETS
        for r in [table.loc[(k, "population")]]
    )
    return lines


def relevance_lines(versus: pd.DataFrame) -> list[str]:
    lines = [
        "",
        "## Region sets against the per-electrode relevance rule (session 2, exploratory, seed 0)",
        "",
        "The region set is each participant's own-CV choice with at most k electrodes, so it may use fewer. "
        "relevance-k keeps the k electrodes with the highest training-day relevance (archived runs and the "
        "electrode-count sweep, seed 0, FP32). The last column repeats the comparison for the population rule.",
        "",
        "| k | region set % (electrodes used) | relevance-k % | region minus relevance (pp) [95% interval] "
        "| region better/worse | population rule minus relevance (pp) [95% interval] |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by = versus.set_index(["electrodes", "method"])
    for k in RELEVANCE_COUNTS:
        own, pop = by.loc[(k, "own_cv")], by.loc[(k, "population")]
        lines.append(
            f"| {k} | {own['region_acc_pct']:.1f} ({own['region_mean_electrodes']:.1f}) | "
            f"{own['relevance_acc_pct']:.1f} | {own['minus_relevance_pp']:+.2f} "
            f"[{own['ci_low_pp']:+.2f}, {own['ci_high_pp']:+.2f}] | "
            f"{own['n_region_better']}/{own['n_relevance_better']} | {pop['minus_relevance_pp']:+.2f} "
            f"[{pop['ci_low_pp']:+.2f}, {pop['ci_high_pp']:+.2f}] |"
        )
    return lines


def check_lines(rows: list[dict]) -> list[str]:
    agree = [100 * r["prediction_agreement"] for r in rows]
    diff = [r["region_study_acc_pct"] - r["archived_acc_pct"] for r in rows]
    return [
        "",
        "## Pipeline check: the all-22 set against the archived full-22 run (seed 0)",
        "",
        "Both follow the same recipe with the same seed. Session-2 predictions agree on "
        f"{min(agree):.1f}-{max(agree):.1f}% of trials, depending on the participant; accuracy differs by "
        f"{min(diff):+.2f} to {max(diff):+.2f} pp. Training on other hardware can change the numbers slightly; "
        "large differences would point to a change in the pipeline.",
    ]


def files_lines() -> list[str]:
    return [
        "",
        "## Files",
        "",
        "- coalitions.csv: every participant and region set: electrodes, CV and session-2 accuracy and per-class "
        "recall, and session-2 accuracy on all trials",
        "- shapley.csv, shapley_coarse.csv: Shapley values per participant, basis (cv or test) and region",
        "- interactions.csv: pairwise interaction index per participant (CV)",
        "- frontier.csv: each rule at each budget; frontier_choices.csv: the set picked for each participant",
        "- relevance_comparison.csv: region sets against relevance-k",
        "- figures: fig_region_shapley.png, fig_class_shapley.png, fig_budget_frontier.png, fig_scalp_shapley.png",
    ]


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    jobs = load_plan()
    subjects = sorted({j["subject"] for j in jobs})
    cv = cv_values(jobs, subjects)
    land = Landscape(subjects, cv, *test_values(jobs, subjects))
    electrodes = electrodes_per_set()
    coalition_table(land, electrodes).to_csv(OUT / "coalitions.csv", index=False)

    values = games(land)
    phi = {key: shapley(v) for key, v in values.items()}
    efficiency = {f"{p}/{b}": efficiency_error(phi[p, b], v) for (p, b), v in values.items()}
    if max(efficiency.values()) > 1e-9:
        raise RuntimeError(f"Shapley values do not add up to v(all) - v(empty): {efficiency}")
    shapley_table(phi, "fine", subjects).to_csv(OUT / "shapley.csv", index=False)
    shapley_table(phi, "coarse", subjects).to_csv(OUT / "shapley_coarse.csv", index=False)
    pair_values = interactions(values["fine", "cv"][:, 0])
    interaction_table(pair_values, subjects).to_csv(OUT / "interactions.csv", index=False)
    summary = {partition: shapley_summary(phi, partition) for partition in ("fine", "coarse")}
    pairs = pair_summary(pair_values)
    top = top_regions(phi["fine", "cv"], subjects)

    chosen = pd.DataFrame([row for k in BUDGETS for row in choice_rows(land, electrodes, k)])
    chosen.to_csv(OUT / "frontier_choices.csv", index=False)
    frontier = frontier_table(chosen)
    frontier.to_csv(OUT / "frontier.csv", index=False)
    relevance = relevance_runs(subjects)
    versus = relevance_table(chosen, relevance)
    versus.to_csv(OUT / "relevance_comparison.csv", index=False)
    check = full_montage_check(jobs, subjects)

    all22 = {basis: float(100 * getattr(land, basis)[:, 0, -1].mean()) for basis in ("cv", "test")}
    style()
    figure_shapley(summary["fine"], all22["cv"] - 100 * CHANCE)
    figure_classes(summary["fine"])
    figure_frontier(frontier, land, electrodes, relevance)
    figure_scalp(summary["fine"])

    counts = {part: sum(j["part"] == part for j in jobs) for part in ("cv", "final")}
    counts["test"] = counts.pop("final")
    lines = [
        *intro_lines(subjects, counts, all22),
        *shapley_lines(summary["fine"], efficiency, top),
        *class_lines(summary["fine"]),
        *interaction_lines(pairs),
        *coarse_lines(summary["coarse"]),
        *frontier_lines(frontier),
        *relevance_lines(versus),
        *check_lines(check),
        *files_lines(),
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    write_json(
        OUT / "analysis.json",
        {
            "status": "supplementary_post_test",
            "boundary": "CV values use session 1 only and are test-free. Session-2 values are exploratory: the "
            "study was planned after the main study's test results were known. The oracle chooses with session 2 "
            "and is an upper bound, not a method.",
            "bootstrap": FROZEN_BOOTSTRAP,
            "participants": subjects,
            "records": counts,
            "regions": {"fine": regions.FINE, "coarse": regions.COARSE, "mask_bit_order": FINE},
            "chance": CHANCE,
            "all22_acc_pct": all22,
            "shapley_efficiency_max_error_pp": efficiency,
            "shapley_pp": summary["fine"],
            "shapley_coarse_pp": summary["coarse"],
            "interactions_cv_pp": pairs,
            "top_region_cv": top,
            "frontier": frontier.to_dict("records"),
            "relevance_comparison": versus.to_dict("records"),
            "pipeline_check_full22": check,
        },
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
