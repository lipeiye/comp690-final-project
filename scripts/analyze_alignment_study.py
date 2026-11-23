"""Tables and a figure for the day-to-day alignment study (scripts/alignment_study.py).

Euclidean alignment whitens each recording day with that day's own average spatial covariance;
it needs no labels. The study trained full-22, relevance-16 and relevance-8 models on aligned
session 1 and scored them on session 2 three ways: aligned with all of its trials (ea-all,
offline), aligned with its first run only and scored on runs 2-6 (ea-first-run, which an online
system could do after a short warm-up), and not aligned at all (none). Each record also holds the
archived unaligned model of the same participant, seed and electrodes, scored on the same trials:
the baseline every variant is paired with.

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only, never EEG:
  supplementary/alignment_study/protocol.json    the plan: one job per condition, participant and seed
  supplementary/alignment_study/test/<job>.json  aligned accuracies and the unaligned baseline of every job
  frozen_config.json                             the frozen bootstrap plan (key "uncertainty")

Writes to RESULTS_DIR/supplementary/alignment_study/: alignment.csv (condition x variant against the
matched baseline), per_participant.csv, subsets_vs_full22.csv (relevance-16 and relevance-8 minus
full-22 without and with alignment, and the change), fig_alignment_by_participant.png, summary.md
and analysis.json.

Seeds are averaged within each participant first, then the nine participants are summarized;
paired differences get the frozen participant bootstrap interval (mibci.stats). Exploratory and
outside the frozen protocol: alignment uses unlabeled session-2 trials, and the study was planned
after the main study's test results were known. When an input is missing, the script lists what
is missing and exits with status 1 before writing anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mibci import data
from mibci.artifacts import sha256, write_json
from mibci.layout import FROZEN_CONFIG
from mibci.paths import RESULTS_DIR, display
from mibci.stats import bootstrap_ci

OUT = RESULTS_DIR / "supplementary/alignment_study"
PLAN = OUT / "protocol.json"
FULL = "full-22"
# accuracy column -> where the record keeps it: the aligned model three ways, the unaligned baseline on two trial sets
COLUMNS = {
    "ea_all": ("acc", "ea-all"),
    "ea_first_run": ("acc", "ea-first-run"),
    "none": ("acc", "none"),
    "baseline_all_clean": ("baseline_acc", "all_clean"),
    "baseline_runs_2_to_6": ("baseline_acc", "clean_runs_2_to_6"),
}
# variant -> (aligned model's column, matched baseline column, trials both are scored on, what session 2 gets)
VARIANTS = {
    "ea-all": ("ea_all", "baseline_all_clean", "all artifact-free trials", "aligned with all of its trials"),
    "ea-first-run": (
        "ea_first_run",
        "baseline_runs_2_to_6",
        "artifact-free trials of runs 2-6",
        "aligned with run 1 only",
    ),
    "none": ("none", "baseline_all_clean", "all artifact-free trials", "not aligned"),
}
HIGHLIGHT = 2  # day-2 signals about 1.5 times larger than day 1 (electrode-count study, amplitude what-if)
TIE = 1e-9  # percentage points; seed means closer than this are the same accuracy
MUTED, GRID, SHADE = "#52514e", "#e6e6e6", "#f0efec"
PAIR_COLOURS = {"ea-all": "#2a78d6", "ea-first-run": "#eb6834"}
CLAIM_BOUNDARY = (
    "Exploratory and outside the frozen protocol. Alignment uses unlabeled session-2 trials: all of them for "
    "ea-all (offline only), the first run for ea-first-run (which an online system could collect during a short "
    "warm-up). Planned after the main study's test results were known; it does not replace the frozen final "
    "configuration."
)


# ---------------------------------------------------------------- inputs


def present(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def missing_inputs(plan: dict | None) -> list[str]:
    groups = {
        "study plan (scripts/alignment_study.py --stage plan)": [PLAN],
        "frozen analysis plan": [FROZEN_CONFIG],
    }
    if plan:
        groups["alignment-study test records (--stage train, then --stage evaluate)"] = [
            OUT / "test" / f"{job['name']}.json" for job in plan["jobs"]
        ]
    lines = []
    for label, paths in groups.items():
        missing = [p for p in paths if not present(p)]
        if missing:
            more = f" and {len(missing) - 3} more" if len(missing) > 3 else ""
            shown = ", ".join(display(p) for p in missing[:3])
            lines.append(f"  {label}: {len(missing)} of {len(paths)} missing: {shown}{more}")
    if lines and plan is None:
        lines.append("  (the study's own test records are listed in the plan; they are checked once it exists)")
    return lines


def load_runs(plan: dict) -> pd.DataFrame:
    """One row per planned job: accuracies in % of the scored trials."""
    digest = sha256(PLAN)
    rows = []
    for job in plan["jobs"]:
        path = OUT / "test" / f"{job['name']}.json"
        record = json.loads(path.read_text())
        if record.get("protocol_sha256") != digest or any(record.get(key) != value for key, value in job.items()):
            raise SystemExit(f"{display(path)} does not match its job in {display(PLAN)}; it belongs to another plan")
        scores = {column: 100 * record[group][name] for column, (group, name) in COLUMNS.items()}
        rows.append({"condition": job["condition"], "subject": job["subject"], "seed": job["seed"], **scores})
    runs = pd.DataFrame(rows)
    seeds = runs.groupby(["condition", "subject"])["seed"].nunique()
    expected = len(plan["conditions"]) * len(data.SUBJECTS)
    if len(seeds) != expected or (seeds != len(plan["seeds"])).any() or len(runs) != expected * len(plan["seeds"]):
        raise SystemExit("The plan does not cover every condition, participant and seed exactly once")
    return runs


# ---------------------------------------------------------------- statistics


def describe(d: pd.Series, plan: dict) -> dict:
    """Paired differences (one per participant, pp): mean, bootstrap interval, counts, and participant 2."""
    values = d.to_numpy()
    low, high = bootstrap_ci(values, plan)
    return {
        "diff_pp": float(values.mean()),
        "ci_low_pp": low,
        "ci_high_pp": high,
        "n_better": int((values > TIE).sum()),
        "n_worse": int((values < -TIE).sum()),
        "n_tied": int((np.abs(values) <= TIE).sum()),
        f"participant_{HIGHLIGHT}_pp": float(d.loc[HIGHLIGHT]),
        f"rank_of_participant_{HIGHLIGHT}": int((values > d.loc[HIGHLIGHT] + TIE).sum() + 1),
        f"mean_without_participant_{HIGHLIGHT}_pp": float(d.drop(HIGHLIGHT).mean()),
    }


def alignment_table(pp: pd.DataFrame, conditions: list[str], plan: dict) -> pd.DataFrame:
    """alignment.csv: every condition and variant against the unaligned model on the same trials."""
    rows = []
    for condition in conditions:
        t = pp.loc[condition]
        for variant, (aligned, baseline, trials, _) in VARIANTS.items():
            rows.append(
                {
                    "condition": condition,
                    "variant": variant,
                    "trials_scored": trials,
                    "baseline": f"archived unaligned model, baseline_acc.{COLUMNS[baseline][1]}",
                    "acc_mean_pct": t[aligned].mean(),
                    "acc_sd_participants_pp": t[aligned].std(),
                    "baseline_mean_pct": t[baseline].mean(),
                    "baseline_sd_participants_pp": t[baseline].std(),
                    **describe(t[aligned] - t[baseline], plan),
                }
            )
    return pd.DataFrame(rows)


def subset_table(pp: pd.DataFrame, conditions: list[str], plan: dict) -> pd.DataFrame:
    """subsets_vs_full22.csv: fewer electrodes minus all 22, unaligned and aligned on the same trials."""
    full = pp.loc[FULL]
    rows = []
    for condition in (c for c in conditions if c != FULL):
        t = pp.loc[condition]
        for variant in ("ea-all", "ea-first-run"):
            aligned, baseline, trials, _ = VARIANTS[variant]
            unaligned_diff = t[baseline] - full[baseline]
            aligned_diff = t[aligned] - full[aligned]
            for models, d in (
                ("unaligned (archived models)", unaligned_diff),
                (f"aligned ({variant})", aligned_diff),
                ("change from alignment", aligned_diff - unaligned_diff),
            ):
                rows.append({"a": condition, "b": FULL, "trials_scored": trials, "models": models, **describe(d, plan)})
    return pd.DataFrame(rows)


def per_participant(pp: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """per_participant.csv: seed means (%) and each variant's gain over its matched baseline (pp)."""
    per = pp.copy()
    for aligned, baseline, _, _ in VARIANTS.values():
        per[f"{aligned}_gain_pp"] = per[aligned] - per[baseline]
    per["seeds"] = runs.groupby(["condition", "subject"])["seed"].nunique()
    return per.rename(columns={c: f"{c}_pct" for c in COLUMNS}).reset_index()


# ---------------------------------------------------------------- figure


def figure(pp: pd.DataFrame, conditions: list[str]) -> None:
    fig, axes = plt.subplots(1, len(conditions), figsize=(4.3 * len(conditions), 4.4), sharey=True, squeeze=False)
    x = np.arange(len(data.SUBJECTS))
    where = data.SUBJECTS.index(HIGHLIGHT)
    for ax, condition in zip(axes[0], conditions, strict=True):
        t = pp.loc[condition]
        ax.axvspan(where - 0.45, where + 0.45, color=SHADE, zorder=0)
        for i, (offset, variant) in enumerate(zip((-0.18, 0.18), PAIR_COLOURS, strict=True)):
            aligned, baseline, trials, treatment = VARIANTS[variant]
            colour, xs = PAIR_COLOURS[variant], x + offset
            ax.vlines(xs, t[baseline], t[aligned], color=colour, lw=1.4, zorder=2)
            unaligned = "archived unaligned model, same trials" if i == 0 else None  # one legend entry for both
            ax.scatter(
                xs, t[baseline], s=28, facecolors="white", edgecolors=MUTED, linewidths=1, zorder=3, label=unaligned
            )
            ax.scatter(xs, t[aligned], s=28, color=colour, zorder=3, label=f"session 2 {treatment}; {trials}")
        ax.set(title=condition, xticks=x, xticklabels=[f"P{s}" for s in data.SUBJECTS], xlabel="participant")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("held-out-day accuracy (%), mean of seeds")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.text(
        0.995,
        0.955,
        f"shaded: participant {HIGHLIGHT}, whose day-2 signals are about 1.5 times larger",
        ha="right",
        fontsize=7,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(OUT / "fig_alignment_by_participant.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- summary.md


def signed(value: float, digits: int = 2) -> str:
    return f"{round(value, digits) + 0.0:+.{digits}f}"  # + 0.0 turns a rounded -0.0 into 0.0


def fmt_diff(row: pd.Series) -> str:
    """'+1.17 [-1.27, +3.75] 6/3': mean difference (pp), 95% interval, participants better/worse."""
    interval = f"[{signed(row['ci_low_pp'])}, {signed(row['ci_high_pp'])}]"
    return f"{signed(row['diff_pp'])} {interval} {row['n_better']}/{row['n_worse']}"


def intro_lines(n_runs: int, conditions: list[str], seeds: list[int], plan: dict) -> list[str]:
    return [
        "# Day-to-day alignment study (supplementary, post-test; generated by scripts/analyze_alignment_study.py)",
        "",
        f"What was done: {n_runs} EEGNet models ({', '.join(conditions)}; 9 participants; seeds "
        f"{', '.join(map(str, seeds))}) were trained with the frozen recipe on session 1 after Euclidean alignment. "
        "Alignment multiplies each recording day by the inverse square root of that day's average spatial "
        "covariance, so that on average the electrodes vary together in the same way on both days. It needs no "
        "labels. The aligned models were scored on session 2 three ways:",
        "",
        "- ea-all: session 2 aligned with all of its trials. Offline only: it needs the whole day in advance.",
        "- ea-first-run: session 2 aligned with its first run (48 trials) and scored on runs 2-6. An online system "
        "could do this after a short warm-up.",
        "- none: the aligned model fed session-2 trials that were not aligned; shows how much it relies on alignment.",
        "",
        "Every number is paired with the archived unaligned model of the same participant, seed and electrodes, "
        "scored on the same trials.",
        "",
        "## What this can and cannot show",
        "",
        "- Alignment uses session-2 trials without their labels: all of them for ea-all, only the first run for "
        "ea-first-run. The frozen protocol uses no session-2 data before scoring, so this is outside it.",
        "- Exploratory: planned after the main study's test results were known. It does not replace the frozen "
        "final configuration (relevance-16-prune50, INT8).",
        "- Nine participants. An interval that contains zero does not show that alignment changes nothing, and none "
        "of this shows how new people would fare.",
        "",
        "## How to read the numbers",
        "",
        "Accuracy is the share of artifact-free session-2 trials classified correctly (four classes, chance 25%), "
        "averaged over seeds within each participant; tables give the mean of the nine participants. Differences "
        "are in percentage points (pp), written as mean [95% interval] better/worse: the interval resamples the "
        f"nine participants ({plan['resamples']:,} bootstrap resamples, seed {plan['seed']}), and better/worse "
        "counts participants whose accuracy is higher/lower (the rest are tied).",
    ]


def alignment_lines(table: pd.DataFrame) -> list[str]:
    lines = [
        "",
        "## Aligned models against the archived unaligned models (same trials)",
        "",
        "| electrodes | session 2 | trials scored | model trained on aligned session 1, % "
        f"| archived unaligned model, % | difference (pp) | participant {HIGHLIGHT} (pp) |",
        "| --- | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['condition']} | {VARIANTS[r['variant']][3]} ({r['variant']}) | {r['trials_scored']} "
            f"| {r['acc_mean_pct']:.2f} | {r['baseline_mean_pct']:.2f} | {fmt_diff(r)} "
            f"| {signed(r[f'participant_{HIGHLIGHT}_pp'], 1)} |"
        )
    return lines


def subset_lines(subsets: pd.DataFrame) -> list[str]:
    lines = [
        "",
        "## Does alignment change the 16-versus-22 comparison?",
        "",
        "Fewer electrodes minus all 22, paired by participant, with the same scoring on both sides. "
        '"Change" is the aligned difference minus the unaligned difference, per participant.',
        "",
        "| comparison | trials scored | unaligned (archived models) | aligned | change from alignment |",
        "| --- | --- | --- | --- | --- |",
    ]
    for (a, trials), group in subsets.groupby(["a", "trials_scored"], sort=False):
        cells = [fmt_diff(row) for _, row in group.iterrows()]
        variant = group["models"].iloc[1].removeprefix("aligned ")
        lines.append(f"| {a} minus {FULL} | {trials} {variant} | " + " | ".join(cells) + " |")
    return lines


def highlight_lines(table: pd.DataFrame, subsets: pd.DataFrame, pp: pd.DataFrame, conditions: list[str]) -> list[str]:
    p = HIGHLIGHT
    lines = [
        "",
        f"## Participant {p}",
        "",
        f"Participant {p}'s day-2 signals are about 1.5 times larger than on day 1 (measured in the electrode-count "
        "study), the clearest case of the shift that alignment is meant to remove.",
        "",
    ]
    for condition in conditions:
        t = pp.loc[(condition, p)]
        rows = table[table.condition == condition].set_index("variant")
        lines.append(
            f"- {condition}: all trials {t['baseline_all_clean']:.1f}% unaligned, {t['ea_all']:.1f}% aligned "
            f"({signed(rows.loc['ea-all', f'participant_{p}_pp'], 1)} pp, rank "
            f"{rows.loc['ea-all', f'rank_of_participant_{p}']} of {len(data.SUBJECTS)}); runs 2-6 "
            f"{t['baseline_runs_2_to_6']:.1f}% unaligned, {t['ea_first_run']:.1f}% aligned with run 1 "
            f"({signed(rows.loc['ea-first-run', f'participant_{p}_pp'], 1)} pp). Mean gain from ea-all without "
            f"participant {p}: {signed(rows.loc['ea-all', f'mean_without_participant_{p}_pp'])} pp, against "
            f"{signed(rows.loc['ea-all', 'diff_pp'])} pp with all nine."
        )
    for _, r in subsets[subsets.trials_scored == VARIANTS["ea-all"][2]].iterrows():
        if r["models"] != "change from alignment":
            lines.append(
                f"- {r['a']} minus {FULL}, {r['models']}, all trials: participant {p} "
                f"{signed(r[f'participant_{p}_pp'], 1)} pp (nine-participant mean {signed(r['diff_pp'])} pp)."
            )
    return lines


def per_participant_lines(pp: pd.DataFrame, conditions: list[str]) -> list[str]:
    lines = [
        "",
        "## Per participant: unaligned, then aligned with all session-2 trials (%, all artifact-free trials)",
        "",
        "| participant | " + " | ".join(conditions) + " |",
        "| ---: |" + " --- |" * len(conditions),
    ]
    for subject in data.SUBJECTS:
        cells = []
        for condition in conditions:
            t = pp.loc[(condition, subject)]
            gain = t["ea_all"] - t["baseline_all_clean"]
            cells.append(f"{t['baseline_all_clean']:.1f} to {t['ea_all']:.1f} ({signed(gain, 1)})")
        name = f"**{subject}**" if subject == HIGHLIGHT else str(subject)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def records(frame: pd.DataFrame) -> list[dict]:
    """Rows as plain JSON values."""
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    plan = json.loads(PLAN.read_text()) if present(PLAN) else None
    missing = missing_inputs(plan)
    if missing:
        raise SystemExit("analyze_alignment_study.py: inputs are missing; nothing was written.\n" + "\n".join(missing))
    boot = json.loads(FROZEN_CONFIG.read_text())["uncertainty"]
    conditions, seeds = plan["conditions"], plan["seeds"]
    if FULL not in conditions:
        raise SystemExit(f"The plan has no {FULL} condition to compare against")
    runs = load_runs(plan)
    pp = runs.groupby(["condition", "subject"])[list(COLUMNS)].mean()  # seeds averaged within participant
    table = alignment_table(pp, conditions, boot)
    subsets = subset_table(pp, conditions, boot)
    per = per_participant(pp, runs)
    lines = [
        *intro_lines(len(runs), conditions, seeds, boot),
        *alignment_lines(table),
        *subset_lines(subsets),
        *highlight_lines(table, subsets, pp, conditions),
        *per_participant_lines(pp, conditions),
    ]

    figure(pp, conditions)
    table.to_csv(OUT / "alignment.csv", index=False)
    per.to_csv(OUT / "per_participant.csv", index=False)
    subsets.to_csv(OUT / "subsets_vs_full22.csv", index=False)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    write_json(
        OUT / "analysis.json",
        {
            "status": "supplementary_post_test",
            "historical_test_results_already_known": True,
            "claim_boundary": CLAIM_BOUNDARY,
            "bootstrap": boot,
            "runs": len(runs),
            "conditions": conditions,
            "seeds": seeds,
            "highlighted_participant": HIGHLIGHT,
            "alignment": records(table),
            "subsets_vs_full22": records(subsets),
            "per_participant": records(per),
        },
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
