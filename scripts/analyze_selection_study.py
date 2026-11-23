"""Tables and figures for the electrode-selection study (scripts/selection_study.py).

The study asks whether subsets chosen by judging electrodes jointly (Riemannian distance between
class covariances, CSP patterns, a wrapper around a fast decoder, a learned concrete selector), by a
fixed physiology prior, or pooled over the other eight participants (pop-*) decode better than the
frozen per-electrode relevance rule, at 4 to 16 electrodes. Every subset trained EEGNet with the
frozen recipe on session 1 and was scored once on session 2.

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only, never EEG:
  supplementary/selection_study/protocol.json         the plan: one job per method, count, participant, seed
  supplementary/selection_study/test/<job>.json       the session-2 scores of every planned job
  supplementary/selection_study/rankings/sub-XX.json  session-1 orders and the wrapper's cross-validated curve
  test/sub-XX_seed-S_{full-22,relevance-8,relevance-16}.json                     archived runs, seeds 0-2
  supplementary/electrode_count/test/sweep/sub-XX_seed-S_relevance-{10,12}.json  electrode-count sweep, seeds 0-2
  frozen_config.json                                  the frozen bootstrap plan (key "uncertainty")

Writes to RESULTS_DIR/supplementary/selection_study/: accuracy.csv (method x electrode count),
electrode_frequency.csv (how many participants' 8-electrode subsets contain each electrode),
fig_accuracy_vs_k.png, fig_electrode_frequency_k8.png, summary.md and analysis.json.

Accuracy is on the artifact-free session-2 trials. Seeds are averaged within each participant
first, then the nine participants are summarized; paired differences per participant get the
frozen participant bootstrap interval (mibci.stats). Exploratory: the rankings use session 1 only,
but the study was planned after the main study's test results were known. When an input is
missing, the script lists what is missing and exits with status 1 before writing anything.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle

from mibci import data
from mibci.artifacts import sha256, write_json
from mibci.layout import FROZEN_CONFIG
from mibci.paths import RESULTS_DIR, display
from mibci.regions import POSITIONS
from mibci.stats import bootstrap_ci

OUT = RESULTS_DIR / "supplementary/selection_study"
PLAN = OUT / "protocol.json"
ARCHIVE = RESULTS_DIR / "test"
SWEEP = RESULTS_DIR / "supplementary/electrode_count/test/sweep"
# Relevance subsets of these sizes were trained before this study (seeds 0-2), so its plan does not repeat them.
ARCHIVED_RELEVANCE = {8: ARCHIVE, 10: SWEEP, 12: SWEEP, 16: ARCHIVE}
PLANNED_SEEDS = (0, 1, 2)  # only to list missing earlier runs before the plan exists; otherwise the plan's seeds
# The earlier runs were trained on the M4, on its GPU (MPS) for these participants and on its CPU for the others;
# scripts/selection_control_study.py retrains them on Longleaf CPUs, the hardware of this study's runs.
CONTROL = RESULTS_DIR / "supplementary/selection_control_study"
MPS_PARTICIPANTS = [1, 2, 3, 4, 5, 6]
FULL = "full-22"
N_ALL = len(data.EEG_CHANNELS)
MAP_K = 8
MAJORITY = len(data.SUBJECTS) // 2 + 1
# Table order, label, colour and line style; a population montage takes the colour of the ranking it pools.
METHODS = {
    "relevance": ("relevance (frozen rule)", "#2a78d6", "-"),
    "riemann": ("Riemannian distance", "#eb6834", "-"),
    "csp": ("CSP patterns", "#1baf7a", "-"),
    "wrapper": ("wrapper (surrogate decoder)", "#eda100", "-"),
    "learned": ("learned (concrete selector)", "#e87ba4", "-"),
    "prior": ("physiology prior", "#008300", "-"),
    "pop-relevance": ("population relevance", "#2a78d6", "--"),
    "pop-riemann": ("population Riemannian", "#eb6834", "--"),
    "pop-wrapper": ("population wrapper", "#eda100", "--"),
}
OWN_DATA = ("relevance", "riemann", "csp", "wrapper", "learned")
NO_OWN_DATA = ("prior", "pop-relevance", "pop-riemann", "pop-wrapper")
STATS = ("diff_pp", "ci_low_pp", "ci_high_pp", "n_better", "n_worse", "n_tied")
TIE = 1e-9  # percentage points; seed means closer than this are the same accuracy
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e6"
CLAIM_BOUNDARY = (
    "Exploratory, planned after the main study's test results were known. Every ranking and subset was chosen "
    "from session 1 only; a participant's population (pop-*) montage comes from the other eight participants' "
    "session-1 rankings and never uses that participant's data. Session-2 accuracies are not a prospective test "
    "and do not replace the frozen final configuration."
)


# ---------------------------------------------------------------- inputs


def present(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def input_groups(plan: dict | None) -> dict[str, list[Path]]:
    """Every input file, grouped the way the missing-input report lists them."""
    seeds = plan["seeds"] if plan else PLANNED_SEEDS
    stems = [f"sub-{s:02d}_seed-{seed}" for s in data.SUBJECTS for seed in seeds]
    groups = {
        "study plan (scripts/selection_study.py --stage plan)": [PLAN],
        "session-1 rankings (scripts/selection_study.py --stage rank)": [
            OUT / "rankings" / f"sub-{s:02d}.json" for s in data.SUBJECTS
        ],
        "earlier runs (results/test: full-22, relevance-8, relevance-16; electrode-count sweep: relevance-10, -12)": [
            *(ARCHIVE / f"{stem}_{FULL}.json" for stem in stems),
            *(folder / f"{stem}_relevance-{k}.json" for k, folder in ARCHIVED_RELEVANCE.items() for stem in stems),
        ],
        "frozen analysis plan": [FROZEN_CONFIG],
    }
    if plan:
        groups["selection-study test records (--stage train, then --stage evaluate)"] = [
            OUT / "test" / f"{job['name']}.json" for job in plan["jobs"]
        ]
    return groups


def missing_inputs(groups: dict[str, list[Path]]) -> list[str]:
    lines = []
    for label, paths in groups.items():
        missing = [p for p in paths if not present(p)]
        if missing:
            more = f" and {len(missing) - 3} more" if len(missing) > 3 else ""
            shown = ", ".join(display(p) for p in missing[:3])
            lines.append(f"  {label}: {len(missing)} of {len(paths)} missing: {shown}{more}")
    return lines


def load_rankings(plan: dict) -> dict[int, dict]:
    """Session-1 rankings, after checking that they are the files the plan was built from."""
    rankings = {}
    for subject in data.SUBJECTS:
        path = OUT / "rankings" / f"sub-{subject:02d}.json"
        if plan["ranking_sha256"].get(path.name) != sha256(path):
            raise SystemExit(f"{display(path)} differs from the ranking the plan was built from (ranking_sha256)")
        rankings[subject] = json.loads(path.read_text())
    return rankings


def run_row(method: str, k: int, record: dict, acc: float, source: str) -> dict:
    # records store montage indices (this study, the sweep) or electrode names (archived runs)
    channels = frozenset(data.EEG_CHANNELS[c] if isinstance(c, int) else c for c in record["channels"])
    if len(channels) != k:
        raise SystemExit(f"{record['name']}: {len(channels)} electrodes recorded, {k} expected")
    return {
        "method": method,
        "k": k,
        "subject": record["subject"],
        "seed": record["seed"],
        "acc": 100 * acc,
        "channels": channels,
        "source": source,
    }


def load_runs(plan: dict) -> pd.DataFrame:
    """One row per trained model: this study's planned jobs, then the earlier relevance and all-22 runs."""
    digest = sha256(PLAN)
    rows = []
    for job in plan["jobs"]:
        path = OUT / "test" / f"{job['name']}.json"
        record = json.loads(path.read_text())
        if record.get("protocol_sha256") != digest or any(record.get(key) != value for key, value in job.items()):
            raise SystemExit(f"{display(path)} does not match its job in {display(PLAN)}; it belongs to another plan")
        rows.append(run_row(job["method"], job["k"], record, record["primary"]["acc"], "this study"))
    for subject in data.SUBJECTS:
        for seed in plan["seeds"]:
            stem = f"sub-{subject:02d}_seed-{seed}"
            record = json.loads((ARCHIVE / f"{stem}_{FULL}.json").read_text())
            rows.append(run_row(FULL, N_ALL, record, record["primary"]["fp32"]["acc"], "earlier runs"))
            for k, folder in ARCHIVED_RELEVANCE.items():
                record = json.loads((folder / f"{stem}_relevance-{k}.json").read_text())
                rows.append(run_row("relevance", k, record, record["primary"]["fp32"]["acc"], "earlier runs"))
    return pd.DataFrame(rows)


def check_complete(runs: pd.DataFrame, methods: list[str], counts: list[int], seeds: list[int]) -> None:
    """Every method and count needs every participant with every seed, exactly once."""
    sizes = runs.groupby(["method", "k"]).size()
    expected = [(FULL, N_ALL), *((m, k) for m in methods for k in counts)]
    short = [f"{m}-{k}" for m, k in expected if sizes.get((m, k), 0) != len(data.SUBJECTS) * len(seeds)]
    if short or runs.duplicated(["method", "k", "subject", "seed"]).any():
        raise SystemExit(f"Runs missing or repeated for: {', '.join(short) or 'duplicated jobs'}")


def subsets(runs: pd.DataFrame) -> dict[tuple, frozenset[str]]:
    """The electrodes of each method, count and participant (the same for every seed)."""
    chosen = {}
    for key, group in runs.groupby(["method", "k", "subject"]):
        unique = set(group["channels"])
        if len(unique) != 1:
            raise SystemExit(f"{key}: the seeds of one participant were trained on different electrodes")
        chosen[key] = unique.pop()
    return chosen


# ---------------------------------------------------------------- statistics


def paired(a: pd.Series, b: pd.Series, plan: dict) -> dict:
    """a minus b per participant (seed means, %), with the participant bootstrap interval."""
    if not a.index.equals(b.index):
        raise ValueError("paired differences need the same participants on both sides")
    d = (a - b).to_numpy()
    low, high = bootstrap_ci(d, plan)
    return {
        "diff_pp": float(d.mean()),
        "ci_low_pp": low,
        "ci_high_pp": high,
        "n_better": int((d > TIE).sum()),
        "n_worse": int((d < -TIE).sum()),
        "n_tied": int((np.abs(d) <= TIE).sum()),
    }


def accuracy_table(runs: pd.DataFrame, chosen: dict, methods: list[str], counts: list[int], plan: dict):
    """accuracy.csv: one row per method and count, with differences from relevance and from all 22."""
    acc = runs.groupby(["method", "k", "subject"])["acc"].mean()  # seeds averaged within participant
    source = runs.groupby(["method", "k"])["source"].first()
    full = acc.loc[(FULL, N_ALL)]
    rows = [
        {
            "method": FULL,
            "label": "all 22 electrodes",
            "k": N_ALL,
            "source": source[(FULL, N_ALL)],
            "acc_mean_pct": full.mean(),
            "acc_sd_participants_pp": full.std(),
        }
    ]
    for method in methods:
        for k in counts:
            a = acc.loc[(method, k)]
            shared = [len(chosen[(method, k, s)] & chosen[("relevance", k, s)]) for s in data.SUBJECTS]
            row = {
                "method": method,
                "label": METHODS[method][0],
                "k": k,
                "source": source[(method, k)],
                "acc_mean_pct": a.mean(),
                "acc_sd_participants_pp": a.std(),
                "shared_with_relevance": float(np.mean(shared)),
            }
            if method != "relevance":
                row |= {f"vs_relevance_{key}": v for key, v in paired(a, acc.loc[("relevance", k)], plan).items()}
            row |= {f"vs_full22_{key}": v for key, v in paired(a, full, plan).items()}
            rows.append(row)
    columns = ["method", "label", "k", "source", "acc_mean_pct", "acc_sd_participants_pp", "shared_with_relevance"]
    columns += [f"{side}_{s}" for side in ("vs_relevance", "vs_full22") for s in STATS]
    table = pd.DataFrame(rows).reindex(columns=columns)
    for column in columns:
        if column.endswith(("n_better", "n_worse", "n_tied")):
            table[column] = table[column].astype("Int64")
    return table, acc


def load_control() -> pd.Series | None:
    """Seed-mean accuracy (%) of the references retrained on Longleaf, by (condition, subject); None until complete."""
    plan_path = CONTROL / "protocol.json"
    if not plan_path.is_file():
        return None
    plan, digest = json.loads(plan_path.read_text()), sha256(plan_path)
    rows = []
    for job in plan["jobs"]:
        path = CONTROL / "test" / f"{job['name']}.json"
        if not path.is_file():
            return None
        record = json.loads(path.read_text())
        if record.get("protocol_sha256") != digest:
            raise SystemExit(f"{display(path)} does not belong to {display(plan_path)}")
        rows.append({"condition": job["condition"], "subject": job["subject"], "acc": 100 * record["primary"]["acc"]})
    return pd.DataFrame(rows).groupby(["condition", "subject"])["acc"].mean()


def hardware_control(acc: pd.Series, methods: list[str], counts: list[int], plan: dict):
    """The earlier references retrained on this study's hardware: the hardware effect, and every method against them."""
    ctrl = load_control()
    if ctrl is None:
        return None, ["", "## Same-hardware check", "", "Pending: scripts/selection_control_study.py retrains the "
                      "earlier reference runs on the Longleaf CPUs that trained this study's runs."]  # fmt: skip
    ks = [k for k in counts if f"relevance-{k}" in ctrl.index.get_level_values(0)]
    rows = []
    for k in [*ks, N_ALL]:
        condition = FULL if k == N_ALL else f"relevance-{k}"
        archived = acc.loc[(FULL, N_ALL)] if k == N_ALL else acc.loc[("relevance", k)]
        retrained = ctrl.loc[condition]
        diff = retrained - archived
        rows.append({"kind": "retrained minus earlier", "method": condition, "k": k,
                     **paired(retrained, archived, plan),
                     "mps_participants_pp": float(diff.loc[MPS_PARTICIPANTS].mean()),
                     "cpu_participants_pp": float(diff.drop(MPS_PARTICIPANTS).mean())})  # fmt: skip
    rows.extend(
        {"kind": "minus retrained relevance", "method": method, "k": k,
         **paired(acc.loc[(method, k)], ctrl.loc[f"relevance-{k}"], plan)}
        for method in methods
        if method != "relevance"
        for k in ks
    )  # fmt: skip
    table = pd.DataFrame(rows)
    hw = table[table.kind == "retrained minus earlier"]
    lines = ["", "## Same-hardware check", "",
             "The earlier reference runs were trained on the M4 (its GPU for participants 1-6, its CPU for 7-9); this "
             "study's runs on Longleaf CPUs. scripts/selection_control_study.py retrained the references there with "
             "the same electrodes, recipe and seeds. Retrained minus earlier, by condition (pp):", "",
             "| condition | retrained minus earlier [95% interval] better/worse | participants 1-6 (GPU before) "
             "| participants 7-9 (CPU before) |", "| --- | --- | ---: | ---: |"]  # fmt: skip
    for _, r in hw.iterrows():
        lines.append(f"| {r.method} | {plain_diff(r)} | {r.mps_participants_pp:+.2f} | {r.cpu_participants_pp:+.2f} |")
    lines += ["", "Every method minus relevance retrained on the same hardware (pp):", "",
              "| method | " + " | ".join(f"{k} electrodes" for k in ks) + " |",
              "| --- |" + " --- |" * len(ks)]  # fmt: skip
    vs = table[table.kind == "minus retrained relevance"].set_index(["method", "k"])
    for method in dict.fromkeys(vs.index.get_level_values(0)):
        lines.append(f"| {METHODS[method][0]} | " + " | ".join(plain_diff(vs.loc[(method, k)]) for k in ks) + " |")
    return table, lines


def frequency_table(chosen: dict, methods: list[str]) -> pd.DataFrame:
    """electrode_frequency.csv: for each electrode, how many participants' 8-electrode subsets contain it."""
    rows = []
    for name in data.EEG_CHANNELS:
        x, y = POSITIONS[name]
        counts = {m: sum(name in chosen[(m, MAP_K, s)] for s in data.SUBJECTS) for m in methods}
        rows.append({"electrode": name, "x": x, "y": y} | counts)
    return pd.DataFrame(rows)


def wrapper_estimate(rankings: dict[int, dict], counts: list[int]) -> pd.Series:
    """Mean over participants of the wrapper's own session-1 cross-validated accuracy (%) at each count."""
    curves = np.array([rankings[s]["wrapper_curve"] for s in data.SUBJECTS])  # 22 electrodes first, then one fewer
    if curves.shape != (len(data.SUBJECTS), N_ALL - 1):
        raise SystemExit(f"wrapper_curve should run from {N_ALL} electrodes down to 2; found shape {curves.shape}")
    return pd.Series({k: 100 * float(curves[:, N_ALL - k].mean()) for k in (N_ALL, *sorted(counts, reverse=True))})


# ---------------------------------------------------------------- figures


def figure_accuracy(table: pd.DataFrame, methods: list[str], counts: list[int]) -> None:
    full = table.loc[table.method == FULL, "acc_mean_pct"].iloc[0]
    panels = (
        ("Chosen from the participant's own session-1 data", [m for m in OWN_DATA if m in methods]),
        ("Chosen without the participant's data", [m for m in NO_OWN_DATA if m in methods]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)
    for ax, (title, shown) in zip(axes, panels, strict=True):
        ax.axhline(full, color=MUTED, lw=1)
        ax.annotate(
            f"all 22 electrodes: {full:.1f}%",
            (counts[0], full),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=8,
            color=MUTED,
        )
        if "relevance" not in shown:  # the frozen rule, faint, as the yardstick in this panel too
            label, color, _ = METHODS["relevance"]
            rows = table[table.method == "relevance"].sort_values("k")
            ax.plot(
                rows.k,
                rows.acc_mean_pct,
                color=color,
                lw=1.2,
                alpha=0.45,
                marker="o",
                ms=4,
                label=f"{label}, for comparison",
            )
        for method in shown:
            label, color, style = METHODS[method]
            rows = table[table.method == method].sort_values("k")
            fill = "white" if style == "--" else color
            ax.plot(rows.k, rows.acc_mean_pct, color=color, ls=style, lw=2, marker="o", ms=5, mfc=fill, label=label)
        ax.set(title=title, xlabel="electrodes kept", xticks=counts)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_ylabel("held-out-day accuracy (%), mean of 9 participants")
    fig.text(
        0.995,
        0.01,
        "seeds averaged within participant; differences with intervals in accuracy.csv",
        ha="right",
        fontsize=7,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig_accuracy_vs_k.png", dpi=200)
    plt.close(fig)


def scalp_map(ax, freq: pd.DataFrame, method: str, cmap, norm) -> None:
    ax.add_patch(Circle((0, -0.5), 3.7, fill=False, ec="#8c8c8c", lw=0.8))
    ax.plot([-0.35, 0, 0.35], [3.15, 3.6, 3.15], color="#8c8c8c", lw=0.8)  # the nose marks the front
    counts = freq[method].to_numpy()
    ax.scatter(freq.x, freq.y, c=counts, cmap=cmap, norm=norm, s=360, edgecolors="#8c8c8c", linewidths=0.5)
    for name, x, y, count in zip(freq.electrode, freq.x, freq.y, counts, strict=True):
        ink = "white" if count >= MAJORITY else INK
        ax.text(x, y + 0.19, name, ha="center", va="center", fontsize=5, color=ink)
        ax.text(x, y - 0.17, str(count), ha="center", va="center", fontsize=7, fontweight="bold", color=ink)
    ax.set(xlim=(-4, 4), ylim=(-4.4, 3.9), aspect="equal")
    ax.set_title(METHODS[method][0], fontsize=9)


def figure_frequency(freq: pd.DataFrame, methods: list[str]) -> None:
    cmap = LinearSegmentedColormap.from_list("participants", ["#f6f9fe", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
    norm = Normalize(0, len(data.SUBJECTS))
    columns = 3
    rows = -(-len(methods) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.1 * columns + 0.9, 3.1 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, method in enumerate(methods):
        scalp_map(axes.flat[i], freq, method, cmap, norm)
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(), shrink=0.45, pad=0.02)
    bar.set_label(f"participants (of {len(data.SUBJECTS)}) whose {MAP_K} electrodes include it", fontsize=8)
    fig.suptitle(f"How often each electrode is among the {MAP_K} chosen (front of the head at the top)", fontsize=10)
    fig.savefig(OUT / f"fig_electrode_frequency_k{MAP_K}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- summary.md


def signed(value: float) -> str:
    return f"{round(value, 1) + 0.0:+.1f}"  # + 0.0 turns a rounded -0.0 into 0.0


def fmt_diff(row: pd.Series, prefix: str) -> str:
    """'+1.2 [-0.8, +3.1] 6/3': mean difference (pp), 95% interval, participants better/worse."""
    if pd.isna(row[f"{prefix}_diff_pp"]):
        return ""
    low, high = signed(row[f"{prefix}_ci_low_pp"]), signed(row[f"{prefix}_ci_high_pp"])
    return f"{signed(row[f'{prefix}_diff_pp'])} [{low}, {high}] {row[f'{prefix}_n_better']}/{row[f'{prefix}_n_worse']}"


def plain_diff(row) -> str:
    """fmt_diff for a row of paired() fields without a prefix."""
    low, high = signed(row["ci_low_pp"]), signed(row["ci_high_pp"])
    return f"{signed(row['diff_pp'])} [{low}, {high}] {int(row['n_better'])}/{int(row['n_worse'])}"


def fmt_acc(row: pd.Series) -> str:
    return f"{row['acc_mean_pct']:.1f}"


def wide(table: pd.DataFrame, methods, counts: list[int], cell) -> list[str]:
    """Methods as rows, electrode counts as columns."""
    lines = ["| method | " + " | ".join(f"{k} electrodes" for k in counts) + " |", "| --- |" + " ---: |" * len(counts)]
    for method in methods:
        rows = table[table.method == method].set_index("k")
        lines.append(f"| {METHODS[method][0]} | " + " | ".join(cell(rows.loc[k]) for k in counts) + " |")
    return lines


def clear_lines(table: pd.DataFrame, prefix: str) -> list[str]:
    """Where the interval lies entirely above or entirely below zero, by method."""
    rows = table.dropna(subset=[f"{prefix}_diff_pp"])
    lines = []
    for side, mask in (("above", rows[f"{prefix}_ci_low_pp"] > 0), ("below", rows[f"{prefix}_ci_high_pp"] < 0)):
        items = [
            f"{label} at {', '.join(map(str, group.k))} electrodes"
            for label, group in rows[mask].groupby("label", sort=False)
        ]
        lines.append(f"- Interval entirely {side} zero: " + ("; ".join(items) if items else "nowhere") + ".")
    return lines


def intro_lines(n_new: int, n_earlier: int, counts: list[int]) -> list[str]:
    return [
        "# Electrode-selection study (supplementary, post-test; generated by scripts/analyze_selection_study.py)",
        "",
        "Question: can another rule pick better electrode subsets than the frozen relevance rule? Each method chose "
        f"{', '.join(map(str, counts))} electrodes, and EEGNet was trained on each subset with the frozen recipe on "
        "all of session 1 (the training day), then scored once on session 2 (the held-out day). "
        f"{n_new} new runs; {n_earlier} earlier runs supply relevance at 8, 10, 12 and 16 electrodes and all 22 "
        "electrodes (results/test and the electrode-count sweep, same recipe and seeds).",
        "",
        "The methods, in plain words:",
        "",
        "- relevance (frozen rule): scores each electrode on its own, by how well its band power separates the "
        "classes.",
        "- Riemannian distance: keeps the subset whose class-average covariance matrices (how electrodes vary "
        "together) stay farthest apart.",
        "- CSP patterns: ranks electrodes by their weight in the common spatial patterns that best separate "
        "each class from the rest.",
        "- wrapper: drops electrodes one at a time, each time the one whose loss a fast surrogate decoder "
        "(tangent space + logistic regression) notices least, judged by cross-validation over session-1 runs.",
        "- learned: a concrete (Gumbel-softmax) selector trained end to end in front of EEGNet, one per count.",
        "- physiology prior: a fixed order around C3, C4 and Cz, the classic hand and foot sites; uses no data.",
        "- population (pop-*): one montage for a new user, from the average rank of each electrode over the "
        "other eight participants' relevance, Riemannian or wrapper rankings.",
    ]


def boundary_lines(seeds: list[int], plan: dict) -> list[str]:
    return [
        "",
        "## What this can and cannot show",
        "",
        "- Every ranking and subset was chosen from session 1 only. The population montages of a participant use "
        "only the other eight participants' session-1 rankings, never that participant's own data.",
        "- The session-2 numbers are exploratory: this study was planned after the main study's test results were "
        "known, so it is not a test of a hypothesis fixed in advance, and it does not replace the frozen final "
        "configuration (relevance-16-prune50, INT8).",
        "- Nine participants. An interval that contains zero does not show that two methods are equal, and none of "
        "this shows how new people would fare.",
        "",
        "## How to read the numbers",
        "",
        "Accuracy is the share of artifact-free session-2 trials classified correctly (four classes, chance 25%). "
        f"Each participant's accuracy is the mean of seeds {', '.join(map(str, seeds))}; tables give the mean of "
        "the nine participants. Differences are in percentage points (pp), written as mean [95% interval] "
        "better/worse: the interval resamples the nine participants "
        f"({plan['resamples']:,} bootstrap resamples, seed {plan['seed']}), and better/worse counts the participants "
        "whose accuracy is higher/lower (the rest are tied).",
    ]


def leader_lines(table: pd.DataFrame, counts: list[int]) -> list[str]:
    lines = [
        "",
        "## Highest mean at each count",
        "",
        "| electrodes | highest mean | accuracy % | minus relevance (pp) | minus all 22 (pp) |",
        "| ---: | --- | ---: | --- | --- |",
    ]
    for k in counts:
        rows = table[table.k == k]
        best = rows.loc[rows.acc_mean_pct.idxmax()]
        versus = "(this is the relevance rule)" if best["method"] == "relevance" else fmt_diff(best, "vs_relevance")
        lines.append(
            f"| {k} | {best['label']} | {best['acc_mean_pct']:.1f} | {versus} | {fmt_diff(best, 'vs_full22')} |"
        )
    n = int(table["vs_relevance_diff_pp"].notna().sum())
    lines += [
        "",
        f"Picking the highest of {table.method.nunique() - 1} means after the fact favours a lucky method. There are "
        f"{n} comparisons with relevance; if no method truly differed, roughly one in twenty intervals would still "
        "exclude zero by chance, so one such interval on its own is weak evidence.",
    ]
    return lines


def electrode_lines(freq: pd.DataFrame, table: pd.DataFrame, methods: list[str]) -> list[str]:
    shared = table[table.k == MAP_K].set_index("method")["shared_with_relevance"]
    lines = [
        "",
        f"## Which electrodes the {MAP_K}-electrode subsets use",
        "",
        f"Counts are participants (of {len(data.SUBJECTS)}); fig_electrode_frequency_k{MAP_K}.png shows them on "
        "the scalp grid, electrode_frequency.csv lists every electrode.",
        "",
        f"| method | electrodes shared with the relevance subset (mean, of {MAP_K}) "
        f"| chosen for at least {MAJORITY} participants |",
        "| --- | ---: | --- |",
    ]
    for method in methods:
        common = freq[freq[method] >= MAJORITY].sort_values(method, ascending=False, kind="stable")
        listed = ", ".join(f"{e} ({c})" for e, c in zip(common.electrode, common[method], strict=True))
        lines.append(f"| {METHODS[method][0]} | {shared[method]:.1f} | {listed or 'none'} |")
    return lines


def wrapper_lines(estimate: pd.Series, table: pd.DataFrame) -> list[str]:
    session2 = table[table.method == "wrapper"].set_index("k")["acc_mean_pct"]
    session2[N_ALL] = table.loc[table.method == FULL, "acc_mean_pct"].iloc[0]
    lines = [
        "",
        "## The wrapper's own estimate on the training day",
        "",
        "The wrapper chose its subsets by the cross-validated accuracy of a fast surrogate decoder on session 1. "
        "Next to it, EEGNet trained on the same subset and scored on session 2. The two use different decoders and "
        "days, so compare how each changes with the count, not the levels.",
        "",
        "| electrodes | surrogate decoder, session-1 cross-validation % | EEGNet, session 2 % |",
        "| ---: | ---: | ---: |",
    ]
    lines += [f"| {k} | {estimate[k]:.1f} | {session2[k]:.1f} |" for k in estimate.index]
    return lines


def summary_lines(table, freq, estimate, methods, counts, seeds, plan, n_new) -> list[str]:
    others = [m for m in methods if m != "relevance"]
    n_earlier = len(data.SUBJECTS) * len(seeds) * (len(ARCHIVED_RELEVANCE) + 1)
    return [
        *intro_lines(n_new, n_earlier, counts),
        *boundary_lines(seeds, plan),
        "",
        "## Mean accuracy (%)",
        "",
        *wide(table, methods, counts, fmt_acc),
        "",
        f"All 22 electrodes: {table.loc[table.method == FULL, 'acc_mean_pct'].iloc[0]:.1f}%.",
        "",
        "## Difference from the relevance rule at the same count (pp)",
        "",
        *wide(table, others, counts, partial(fmt_diff, prefix="vs_relevance")),
        "",
        *clear_lines(table, "vs_relevance"),
        "",
        "## Difference from all 22 electrodes (pp)",
        "",
        *wide(table, methods, counts, partial(fmt_diff, prefix="vs_full22")),
        "",
        *clear_lines(table, "vs_full22"),
        *leader_lines(table, counts),
        *electrode_lines(freq, table, methods),
        *wrapper_lines(estimate, table),
    ]


def records(frame: pd.DataFrame) -> list[dict]:
    """Rows as plain JSON values (missing values become null)."""
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    plan = json.loads(PLAN.read_text()) if present(PLAN) else None
    missing = missing_inputs(input_groups(plan))
    if missing:
        if plan is None:
            missing.append("  (the study's own test records are listed in the plan; they are checked once it exists)")
        raise SystemExit("analyze_selection_study.py: inputs are missing; nothing was written.\n" + "\n".join(missing))
    boot = json.loads(FROZEN_CONFIG.read_text())["uncertainty"]
    planned = {job["method"] for job in plan["jobs"]} | {"relevance"}
    unknown = planned - set(METHODS)
    if unknown:
        raise SystemExit(f"Methods without a label in this script: {', '.join(sorted(unknown))}")
    methods = [m for m in METHODS if m in planned]
    counts, seeds = sorted(plan["counts"]), plan["seeds"]
    rankings = load_rankings(plan)
    runs = load_runs(plan)
    check_complete(runs, methods, counts, seeds)
    chosen = subsets(runs)
    table, acc = accuracy_table(runs, chosen, methods, counts, boot)
    freq = frequency_table(chosen, methods)
    estimate = wrapper_estimate(rankings, counts)
    lines = summary_lines(table, freq, estimate, methods, counts, seeds, boot, len(plan["jobs"]))
    control, control_lines = hardware_control(acc, methods, counts, boot)
    lines += control_lines

    figure_accuracy(table, methods, counts)
    figure_frequency(freq, methods)
    table.to_csv(OUT / "accuracy.csv", index=False)
    freq.to_csv(OUT / "electrode_frequency.csv", index=False)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    write_json(
        OUT / "analysis.json",
        {
            "status": "supplementary_post_test",
            "historical_test_results_already_known": True,
            "claim_boundary": CLAIM_BOUNDARY,
            "bootstrap": boot,
            "new_runs": len(plan["jobs"]),
            "methods": methods,
            "counts": counts,
            "seeds": seeds,
            "accuracy": records(table),
            "per_participant_acc_pct": {
                m if m == FULL else f"{m}-{k}": [round(float(v), 4) for v in acc.loc[(m, k)]]
                for m, k in acc.index.droplevel(2).unique()
            },
            f"electrode_frequency_k{MAP_K}": records(freq),
            "wrapper_session1_estimate_pct": {str(k): v for k, v in estimate.items()},
            "hardware_control": None if control is None else records(control),
        },
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
