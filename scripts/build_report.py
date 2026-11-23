"""Build a portable Markdown research report from retained experiment evidence.

Does not train, choose models or overwrite historical tables/figures. New charts
and derived tables are written only under results/maintenance and docs/assets.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
RESULTS = ROOT / "results"
# Two trailing spaces end a Markdown line without starting a new paragraph.
LINE_BREAK = "  "


def read_csv(relative):
    with (RESULTS / relative).open(newline="") as stream:
        return list(csv.DictReader(stream))


def table(headers, rows):
    return (
        "| "
        + " | ".join(headers)
        + " |\n| "
        + " | ".join(["---"] * len(headers))
        + " |\n"
        + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)
    )


def span(r):
    """A mean difference with its 95% participant interval, in points: '+1.23 [-0.45, +2.80]'."""
    return f"{r['mean_diff_pp']:+.2f} [{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]"


def pm(r, g):
    """Group g's difference from 22 electrodes with its seed standard error, such as '+1.2 ± 0.4'."""
    return f"{r[g + '_minus_full_pp']:+.1f} ± {r[g + '_seed_se_pp']:.1f}"


def people(subjects):
    """Participant labels such as 'P1, P4, P9'."""
    return ", ".join(f"P{s}" for s in subjects)


def word(n):
    """Counts from two to seven as words, other numbers as digits."""
    return {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}.get(n, str(n))


def series(counts, value):
    """value(k) for each electrode count k, signed and comma-separated."""
    return ", ".join(f"{value(k):+.1f}" for k in counts)


def both(items):
    """Items joined as 'a, b and c'."""
    return ", ".join(items[:-1]) + (" and " if len(items) > 1 else "") + items[-1]


def electrode_selection_section(a):
    """Section 6.1: how the rule chooses electrodes and which ones it removes."""
    rows_by = a["selection"]["by_scalp_row"]
    frontal = sum(v["dropped_at_16"] for k, v in rows_by.items() if k.startswith("front"))
    total = sum(v["dropped_at_16"] for v in rows_by.values())
    kept = a["selection"]["kept_at_16_count_per_electrode"]
    always = [e for e, n in kept.items() if n == 9]
    sel = read_csv("supplementary/electrode_count/selection_by_participant.csv")

    def mean(key):
        return statistics.mean(float(r[key]) for r in sel)

    return (
        "### 6.1 How electrodes are chosen, and what the rule removes\n\n"
        "Selection is neither random nor based on an anatomical atlas. For each participant, electrodes are ranked "
        "from that participant's training-day band power by the rule in Section 2, and the network is retrained on "
        "the kept electrodes. The frozen configuration uses the relevance-only variant: the 16 electrodes with the "
        "highest Fisher score. Random subsets exist only as a control. Although the rule receives no anatomical "
        f"information, its choices are anatomically structured. Across participants, {frontal} of the {total} "
        "removed electrode slots at 16 electrodes fall on Fz and the fronto-central row, which hold 6 of the 22 "
        "electrodes.\n\n"
        + table(
            ["Scalp row", "Electrodes", "Times removed at 16", "Share of all removals"],
            [
                [
                    k,
                    v["electrodes"],
                    f"{v['dropped_at_16']}/{v['possible']}",
                    f"{100 * v['share_of_all_drops']:.0f}%",
                ]
                for k, v in rows_by.items()
            ],
        )
        + f"\n\nKept for all nine participants: {', '.join(always)}. The removed electrodes carry less class "
        f"information on both days: their mean Fisher score is {mean('fisher_day1_dropped'):.3f} against "
        f"{mean('fisher_day1_kept'):.3f} for kept electrodes on the training day, and "
        f"{mean('fisher_day2_dropped'):.3f} against {mean('fisher_day2_kept'):.3f} on the held-out day. Six "
        f"electrodes are 27% of the montage but hold {100 * mean('dropped_share_of_fisher_day1'):.0f}% of the "
        f"summed Fisher score on the training day and {100 * mean('dropped_share_of_fisher_day2'):.0f}% on the "
        "held-out day. Their day-to-day change in band power is no larger than that of kept electrodes "
        f"({mean('day_shift_dropped'):.2f} against {mean('day_shift_kept'):.2f} within-day standard deviations), "
        "so removing unstable electrodes is not the explanation for the result below. The held-out-day figures are "
        "a post-test description and fed no selection."
    )


def electrode_count_curve(comp, cond, group_rows):
    """The electrode-count part of Section 6.2: the sweep table, its figure and how the curve reads."""
    ten_everywhere = all((f"relevance-{k}", "fp32", "0-9") in cond for k in (20, 18, 14, 12, 10))
    sweep = []
    for k in (22, 20, 18, 16, 14, 12, 10, 8):
        name = "full-22" if k == 22 else f"relevance-{k}"
        seeds = "0-9" if (name, "fp32", "0-9") in cond else "0-2"
        row = group_rows[name]
        d = comp.get((name, "full-22", "fp32", seeds))
        sweep.append(
            [
                k,
                10 if seeds == "0-9" else 3,
                f"{float(cond[(name, 'fp32', seeds)]['acc_mean']):.2f}",
                "0" if d is None else span(d),
                "" if d is None else f"{d['seed_se_pp']:.2f}",
                "0" if k == 22 else pm(row, "below"),
                "0" if k == 22 else pm(row, "above"),
            ]
        )
    text = (
        "**Accuracy against electrode count.** Subsets are the first k electrodes of the same relevance order"
        + (", and every count has ten seeds" if ten_everywhere else "; counts other than 22, 16 and 8 have three seeds")
        + ".\n\n"
        + table(
            [
                "Electrodes",
                "Seeds",
                "Accuracy (%)",
                "Minus 22 electrodes (pp) [95% interval]",
                "Seed standard error",
                "Lower group, minus 22 (pp)",
                "Upper group, minus 22 (pp)",
            ],
            sweep,
        )
        + "\n\nThe two group columns compare against the ten-seed 22-electrode accuracy of the same participants."
        "\n\n![Accuracy against electrode count, for all participants and by "
        "group](../results/supplementary/electrode_count/fig_electrode_count.png)\n\n"
    )
    if ten_everywhere:

        def vs_full(k):
            return comp[(f"relevance-{k}", "full-22", "fp32", "0-9")]

        def lower(k):
            return group_rows[f"relevance-{k}"]["below_minus_full_pp"]

        def upper(k):
            return group_rows[f"relevance-{k}"]["above_minus_full_pp"]

        apart = [
            str(k) for k in (20, 18, 16, 14, 12, 10, 8) if vs_full(k)["ci_low_pp"] > 0 or vs_full(k)["ci_high_pp"] < 0
        ]
        text += (
            "With ten seeds at every count, mean accuracy rises slightly while the six least relevant electrodes are "
            f"removed ({series((20, 18, 16), lambda k: vs_full(k)['mean_diff_pp'])} points at 20, 18 and 16 "
            "electrodes) and falls once more are removed "
            f"({series((14, 12, 10, 8), lambda k: vs_full(k)['mean_diff_pp'])} at 14, 12, 10 and 8). Sixteen is the "
            "top of a gentle hump, not an isolated point; the three-seed curve looked more jagged because each of its "
            "points carried about one point of seed noise. The two groups explain the shape. The upper group stays "
            f"level down to 16 electrodes ({series((20, 18, 16), upper)}) and then loses accuracy at every step "
            f"({series((14, 12, 10, 8), upper)}): its first six removals cost nothing and the later ones do. The lower "
            f"group gains most at 18 and 16 electrodes ({series((18, 16), lower)}) and stays within about a point of "
            f"its 22-electrode accuracy at the other counts ({series((20, 14, 12, 10, 8), lower)}). Sixteen is "
            "therefore not a special number. It is the count at which removals have not yet cost the participants "
            "whose models work, while those whose models barely work are at their best. "
            + (
                f"The interval that resamples participants excludes zero only at {both(apart)} electrodes.\n\n"
                if apart
                else "At no count does the interval that resamples participants exclude zero.\n\n"
            )
        )
    else:
        text += (
            "Counts with three seeds carry about one point of seed noise each, so the shape between 20 and 10 "
            "electrodes is not resolved here.\n\n"
        )
    return text


def electrode_count_section(a, comp):
    """Section 6.2 and its headline sentence: is the 16-electrode gain real, and where does it come from?"""
    r3, r7, r10 = (comp[("relevance-16", "full-22", "fp32", s)] for s in ("0-2", "3-9", "0-9"))
    luck = a["seed_luck"]["relevance-16 minus full-22"]
    conditions = read_csv("supplementary/electrode_count/conditions.csv")
    cond = {(r["condition"], r["precision"], r["seeds"]): r for r in conditions}
    run_sd = float(cond[("full-22", "fp32", "0-9")]["run_sd_within_participant"])
    repro = min(v["prediction_agreement"]["fp32"] for v in a["repro"].values())
    baseline = {
        int(r["subject"]): float(r["full-22_mean"])
        for r in read_csv("supplementary/electrode_count/per_participant.csv")
    }
    text = (
        "### 6.2 Sixteen against 22 electrodes: is the difference real, and where does it come from?\n\n"
        "In the frozen analysis, 16 selected electrodes scored 1.2 to 1.8 points above all 22, with intervals that "
        f"include zero. One training run moves by about {run_sd:.1f} points with the seed alone (standard "
        "deviation of single runs within a participant), and each model had three seeds. Seven more seeds were "
        "therefore trained for the 22-, 16- and 8-electrode models, with the same recipe and device allocation. "
        f"Two archived runs trained again gave the archived predictions on {100 * repro:.0f}% of held-out-day "
        "trials, so the pipeline itself has not drifted.\n\n"
        + table(
            [
                "Comparison (FP32)",
                "Seeds 0-2 (archived)",
                "Seeds 3-9 (new)",
                "All ten seeds",
                "Seed standard error",
                "Participants better/worse",
            ],
            [
                [
                    f"{x} minus {y}",
                    span(comp[(x, y, "fp32", "0-2")]),
                    span(comp[(x, y, "fp32", "3-9")]),
                    span(comp[(x, y, "fp32", "0-9")]),
                    f"{comp[(x, y, 'fp32', '0-9')]['seed_se_pp']:.2f}",
                    f"{comp[(x, y, 'fp32', '0-9')]['n_a_better']}/{comp[(x, y, 'fp32', '0-9')]['n_b_better']}",
                ]
                for x, y in [
                    ("relevance-16", "full-22"),
                    ("proposed-16", "full-22"),
                    ("relevance-8", "full-22"),
                    ("proposed-16", "relevance-16"),
                    ("proposed-8", "relevance-8"),
                ]
            ],
        )
        + "\n\nDifferences are percentage points. Brackets are 95% intervals that resample the nine participants: "
        "they say how far the mean could move with nine other people. The seed standard error holds the "
        "participants fixed and says how far the ten-seed mean could move if the same models were trained again. "
        "The new seeds are an independent draw of training randomness, not new participants.\n\n"
        f"**The difference is not seed luck.** The seven new seeds reproduce it ({r7['mean_diff_pp']:+.2f} points, "
        f"against {r3['mean_diff_pp']:+.2f} for the archived three), and the ten-seed value of "
        f"{r10['mean_diff_pp']:+.2f} is {r10['mean_diff_pp'] / r10['seed_se_pp']:.1f} seed standard errors from "
        f"zero. A three-seed estimate of this difference carries about ±{1.96 * luck['sd_pp']:.1f} points of seed "
        "noise, which is why the original run could not settle it."
    )
    if a.get("seed_ensemble"):
        e22, e16 = a["seed_ensemble"]["full-22"], a["seed_ensemble"]["relevance-16"]
        text += (
            " Averaging the class probabilities of the ten seeds of each model into one prediction removes most "
            f"run-to-run variance, and the difference does not shrink: {e22['ensemble']:.1f}% with 22 electrodes "
            f"and {e16['ensemble']:.1f}% with 16 ({e16['ensemble'] - e22['ensemble']:+.1f} points)."
        )
    text += (
        f"\n\n**It is not established for other people.** The interval that resamples participants runs from "
        f"{r10['ci_low_pp']:+.2f} to {r10['ci_high_pp']:+.2f} points"
        + (", so it still reaches zero." if r10["ci_low_pp"] <= 0 else ", which excludes zero only narrowly.")
        + " Nine participants cannot show that the gain would appear in new users.\n\n"
    )
    g = a["groups"]
    group_rows = {r["condition"]: r for r in a["by_baseline_group"]}
    g16, grand, gleast = (
        group_rows["relevance-16"],
        group_rows["random-16 (three fixed subsets)"],
        group_rows["leastrelevant-16"],
    )
    text += (
        f"**It is uneven across participants.** The {word(len(g['below']))} participants who score below "
        f"{g['threshold_pct']:.0f}% with all 22 electrodes ({people(g['below'])}; "
        f"{min(baseline[s] for s in g['below']):.0f}-{max(baseline[s] for s in g['below']):.0f}%) gain "
        f"{pm(g16, 'below')} points on average from the 16-electrode subset. The {word(len(g['above']))} who score "
        f"above it ({people(g['above'])}; "
        f"{min(baseline[s] for s in g['above']):.0f}-{max(baseline[s] for s in g['above']):.0f}%) change by "
        f"{pm(g16, 'above')}. Here ± is the standard error from training seeds. The rank correlation between a "
        f"participant's 22-electrode accuracy and the gain is {g['spearman_baseline_vs_gain']:+.2f} (n = 9, p = "
        f"{g['spearman_p']:.2f}). The grouping was made after seeing the results, so it describes these "
        "participants and is not a tested hypothesis.\n\n"
        "![Ten seeds per model for 22 and 16 electrodes; participants ordered by 22-electrode "
        "accuracy](../results/supplementary/electrode_count/fig_16_vs_22_seeds.png)\n\n"
    )
    text += electrode_count_curve(comp, cond, group_rows)
    worst = comp[("leastrelevant-16", "full-22", "fp32", "0-2")]
    text += (
        "**Which electrodes are removed matters.** Keeping the 16 least relevant electrodes instead of the 16 most "
        f"relevant gives {span(worst)} points against all 22. Three fixed random 16-electrode subsets change the "
        f"upper group by {pm(grand, 'above')} points and the lower group by {pm(grand, 'below')}; the least "
        f"relevant subset changes them by {pm(gleast, 'above')} and {pm(gleast, 'below')}. For participants whose "
        "models work, only the relevance-selected subset keeps the accuracy of the full montage.\n\n"
    )
    fit = a.get("train_fit_by_participant") or []
    if fit:

        def avg(key, subjects=None):
            return statistics.mean(r[key] for r in fit if subjects is None or r["subject"] in subjects)

        def gap(c, subjects=None):
            return avg(c + "_train", subjects) - avg(c + "_test", subjects)

        text += (
            "**Fit to the training day.** Models with 22 electrodes fit the training day slightly better and the "
            f"held-out day worse: {avg('full-22_train'):.1f}% and {avg('full-22_test'):.1f}% with 22 electrodes, "
            f"{avg('relevance-16_train'):.1f}% and {avg('relevance-16_test'):.1f}% with 16. The gap between the "
            f"two days is {gap('full-22'):.1f} points with 22 electrodes and {gap('relevance-16'):.1f} with 16. In "
            f"the lower group it falls from {gap('full-22', g['below']):.1f} to "
            f"{gap('relevance-16', g['below']):.1f} points; in the upper group it is "
            f"{gap('full-22', g['above']):.1f} and {gap('relevance-16', g['above']):.1f}.\n\n"
        )
    what_if = {(r["scaling"], r["a"]): r for r in a.get("amplitude_what_if", [])}
    if what_if:
        p, gain, e = (what_if[(s, "relevance-16")] for s in ("protocol", "one_gain", "per_electrode"))
        text += (
            "**One of the gains is an amplitude effect.** Participant 2's held-out-day signals are about 1.5 times "
            "larger than on the training day on every electrode; the other eight participants are within about "
            "10%. As a diagnostic outside the frozen protocol, the same models were scored again after rescaling "
            "the held-out-day input to the training amplitude. This uses session-2 statistics but no labels.\n\n"
            + table(
                [
                    "Input scaling",
                    "22 electrodes (%)",
                    "16 by relevance (%)",
                    "16 minus 22 (pp) [95% interval]",
                    "Participant 2: 22 electrodes (%)",
                    "Participant 2: 16 electrodes (%)",
                ],
                [
                    [
                        n,
                        f"{r['acc_b']:.2f}",
                        f"{r['acc_a']:.2f}",
                        span(r),
                        f"{r['per_participant_acc_b'][1]:.1f}",
                        f"{r['per_participant_acc_a'][1]:.1f}",
                    ]
                    for n, r in [
                        ("As evaluated", p),
                        ("One gain for the recording", gain),
                        ("Each electrode re-standardized", e),
                    ]
                ],
            )
            + f"\n\nWith matched amplitude, participant 2 scores {gain['per_participant_acc_b'][1]:.1f}% with 22 "
            f"electrodes and {gain['per_participant_acc_a'][1]:.1f}% with 16, so that participant's gain of "
            f"{p['per_participant_pp'][1]:+.1f} points exists only when the input lies outside the range the "
            f"models were trained on. The nine-participant difference falls from {p['mean_diff_pp']:+.2f} to "
            f"{gain['mean_diff_pp']:+.2f} points. The rescaling is not a proposed change to the pipeline: it "
            "lowers participant 2's accuracy and raises participant 7's, and it needs statistics of the test "
            "recording.\n\n"
        )
    low = [r for r in fit if r["subject"] in g["below"]]
    regime = (
        (
            "their networks fit the training day at "
            f"{min(r['full-22_train'] for r in low):.0f}-{max(r['full-22_train'] for r in low):.0f}% and reach "
            f"{min(r['full-22_test'] for r in low):.0f}-{max(r['full-22_test'] for r in low):.0f}% on the other "
            "day, so most of what they learn does not transfer"
        )
        if low
        else "most of what their networks learn on the training day does not transfer"
    )
    text += (
        "**Reading.** The removed electrodes are mostly frontal and carry about half the class information of the "
        "kept ones (Section 6.1), so participants whose models work lose nothing. Participants whose models barely "
        f"work are in a different regime: {regime}. An earlier version of this report read the gain as fewer inputs "
        "leaving less room to fit the training trials, the motive reviews of EEG channel selection give [7, 8]. "
        "The later studies in Section 6.12 do not support that reading: the dropped electrodes do carry "
        "information, every added input costs about as much, and the cost does not shrink with three times more "
        "training data; on 54 new people, the participants decoded worst did not gain most (Section 6.11). Two "
        "claims are supported. Sixteen relevance-selected electrodes are not worse than all 22 on average, and the "
        "gain of about one point is reproducible for these nine participants."
    )
    headline = (
        f"With ten training seeds per model, 16 relevance-selected electrodes score **{r10['mean_diff_pp']:+.2f} "
        f"points** against all 22 (95% participant interval [{r10['ci_low_pp']:+.2f}, {r10['ci_high_pp']:+.2f}]). "
        f"Seven new seeds reproduce the archived difference ({r7['mean_diff_pp']:+.2f} against "
        f"{r3['mean_diff_pp']:+.2f}), so it is not seed luck, but it comes from the {word(len(g['below']))} "
        f"participants who score below {g['threshold_pct']:.0f}% with 22 electrodes ({pm(g16, 'below')} points; "
        f"the other {word(len(g['above']))}: {pm(g16, 'above')})."
    )
    return text, headline


def electrode_sections(base):
    """Sections 6.1 and 6.2 with their headline sentence, or a note that the runs are pending."""
    path = base / "electrode_count/analysis.json"
    if not path.exists():
        return [], [
            "### 6.1-6.2 Electrode selection and electrode count\n\nThe ten-seed and electrode-count runs "
            "(`scripts/run_electrode_study.py`) are **pending**; no outcome is claimed here."
        ]
    a = json.loads(path.read_text())
    comp = {(r["a"], r["b"], r["precision"], r["seeds"]): r for r in a["comparisons"]}
    selection = electrode_selection_section(a)
    text, headline = electrode_count_section(a, comp)
    return [headline], [selection, text]


def redundancy_section(base):
    """Section 6.3: whether removing the scalp-wide band-power component would change the selection."""
    parts = []
    path = base / "redundancy_check/summary.json"
    if path.exists():
        o = json.loads(path.read_text())["overall"]
        label = {
            "original": "Spearman on log band power (as used)",
            "common_mode_removed": "Same, after removing the scalp-average log power per trial",
        }
        rows = []
        for key, v in o.items():
            name, penalty = key.rsplit("_penalty_", 1)
            rows.append(
                [
                    label[name],
                    f"{v['mean_abs_correlation']:.2f}",
                    penalty,
                    f"{v['participants_changed_at_16']}/9",
                    max(v["swapped_at_16"]),
                    f"{v['participants_changed_at_8']}/9",
                    max(v["swapped_at_8"]),
                    f"{v['relevance_to_penalty_spread_ratio']:.1f}",
                ]
            )
        parts.append(
            "### 6.3 Would a different redundancy measure change the selection?\n\n"
            "Band power rises and falls together over the whole scalp, so every electrode looks similarly redundant. "
            "The planned follow-up was to remove that shared component before measuring redundancy and retrain. A "
            "check on training-day data alone, with no training and without reading session 2, shows that this would "
            "change little:\n\n"
            + table(
                [
                    "Redundancy measure",
                    "Mean abs. correlation",
                    "Penalty",
                    "Participants whose 16 change",
                    "Most electrodes swapped",
                    "Participants whose 8 change",
                    "Most electrodes swapped",
                    "Spread of relevance / spread of penalty",
                ],
                rows,
            )
            + "\n\nThe last column compares, across the remaining candidates at each greedy step, the standard "
            "deviation of the relevance score with that of the penalty term. Relevance varies several times more than "
            "the penalty, so the order is decided almost entirely by relevance, whichever redundancy measure is used. "
            "The negative result in Section 3.1 is therefore a property of this relevance-minus-penalty rule, not only "
            "of the correlation measure, and no retraining was spent on the alternative measure."
        )
    return [], parts


def int8_section(base):
    """Section 6.4 and, once the timing run exists, its headline sentence; a pending note before the study."""
    headline, parts = [], []
    path = base / "int8_study/analysis.json"
    if path.exists():
        a = json.loads(path.read_text())
        acc = a["all_conditions_diff_vs_fp32_pp"]
        text = (
            "### 6.4 Why INT8 was slower on the M4, and what avoids it\n\n"
            "The slowdown is a property of one runtime kernel, not of the processor. The M4 reports the 8-bit "
            "dot-product and matrix-multiply instruction sets (FEAT_DotProd and FEAT_I8MM), and within the same INT8 "
            "model the temporal convolution is faster than in FP32. Only the spatial convolution is slow. ONNX Runtime "
            "1.31.1 has fast INT8 paths for ordinary convolutions and for depthwise convolutions with exactly one "
            "filter per input channel; EEGNet's spatial layer has eight groups with two filters each, so it takes the "
            "generic path that gathers each group's input and calls a narrow matrix product [6].\n\n"
        )
        if "latency_full22_ms" in a:
            lat = a["latency_full22_ms"]
            bench = json.loads((base / "int8_study/benchmark.json").read_text())
            latency = read_csv("supplementary/int8_study/latency.csv")
            row = {(r["condition"], r["variant"]): r for r in latency}
            ref = lat["fp32"]
            names = [
                ("fp32", "FP32, as exported"),
                ("int8", "INT8, as exported"),
                ("int8-u8act", "INT8, unsigned activations"),
                ("int8-pertensor", "INT8, per-tensor weights"),
                ("int8-qoperator", "INT8, QOperator format"),
                ("int16", "16-bit activations, 8-bit weights"),
                ("int16-w16", "16-bit activations and weights"),
                ("int8-mixed", "INT8, spatial convolution left in FP32"),
                ("int8-dense", "INT8, grouped convolution rewritten as dense"),
                ("fp32-spatial-first", "FP32, first two layers swapped"),
                ("int8-spatial-first", "INT8, first two layers swapped"),
            ]
            env = bench["environment"]
            text += (
                table(
                    [
                        "Variant (22 electrodes)",
                        "Median time (ms)",
                        "Relative to FP32",
                        "Accuracy minus FP32, all conditions (pp)",
                        "Parameter bytes",
                    ],
                    [
                        [
                            label,
                            f"{lat[k]:.3f}",
                            f"{lat[k] / ref:.2f}x",
                            "0"
                            if k == "fp32"
                            else (
                                "0.00, same predictions"
                                if k == "fp32-spatial-first"
                                else (
                                    (f"{acc[k]['mean']:+.2f} [{acc[k]['ci_low']:+.2f}, {acc[k]['ci_high']:+.2f}]")
                                    .replace("-0.00", "0.00")
                                    .replace("+0.00", "0.00")
                                    if k in acc
                                    else "not scored"
                                )
                            ),
                            f"{int(row[('full-22', k)]['parameter_bytes']):,}",
                        ]
                        for k, label in names
                    ],
                )
                + f"\n\nTimes are batch-1, single-thread medians on the original {env['cpu']} Mac mini "
                f"({env['hw_model']}) with ONNX Runtime {env['onnxruntime']}, for the archived participant-1, seed-0 "
                f"model. All variants of one architecture were timed in {int(row[('full-22', 'fp32')]['rounds'])} "
                f"interleaved rounds of {int(row[('full-22', 'fp32')]['runs_per_round'])} calls; the load average was "
                f"{env['load_average_before'][0]:.0f} before and {env['load_average_after'][0]:.0f} after, with this "
                "project's training jobs suspended for the measurement. Accuracy is the mean over all 17 conditions "
                f"and {a['n_models']} archived models, seeds averaged within participant.\n\n"
                "![Latency of the deployment variants](../results/supplementary/int8_study/fig_latency.png)\n\n"
            )

            def sf(c, v):
                return float(row[(c, v)]["median_ms"])

            text += (
                "Changing quantization settings does not help, because the same kernel runs. Sixteen-bit quantization "
                "does not help either: the CPU provider has no 16-bit convolution kernel, so the graph executes the "
                "four float convolutions plus extra quantize and dequantize steps. Three changes do help, none of "
                "which needs retraining. Leaving the one slow layer in FP32 (mixed precision) removes the penalty. "
                "Rewriting the grouped convolution as an ordinary convolution with zeros outside each group's block "
                "gives logits **identical** to the archived INT8 graph on every session-2 trial of all "
                f"{a['n_models']} models, while the layer runs about five times faster. Swapping the first two layers "
                "goes further.\n\n"
                "EEGNet's temporal and spatial convolutions are both linear and only BatchNorm lies between them, so "
                "they commute: applying the 16 spatial filters first and the temporal filters second computes the same "
                "function. MI-BMInet builds this order into its architecture to save memory and computation [3]; here "
                "it is applied to the already trained EEGNet as a graph rewrite. In FP32 the swapped graph reproduces "
                f"the archived predictions for all {a['n_models']} models (largest logit difference "
                f"{a['fp32_spatial_first_max_abs_logit_diff_train']:.0e}; lowest prediction agreement "
                f"{100 * a['fp32_spatial_first_prediction_agreement_min']:.0f}%).\n\n"
            )
            costs = read_csv("supplementary/int8_study/costs.csv")
            cost = {r["condition"]: r for r in costs}
            macs_saved_swapped = 100 * (
                1 - int(cost["proposed-8"]["spatial-first_macs"]) / int(cost["full-22"]["spatial-first_macs"])
            )
            macs_saved_exported = 100 * (
                1 - int(cost["proposed-8"]["original_macs"]) / int(cost["full-22"]["original_macs"])
            )
            text += (
                table(
                    [
                        "Electrodes",
                        "MACs, as exported",
                        "MACs, swapped",
                        "Peak activation values, as exported",
                        "Peak activation values, swapped",
                        "FP32 as exported (ms)",
                        "INT8 swapped (ms)",
                    ],
                    [
                        [
                            k,
                            f"{int(cost[c]['original_macs']):,}",
                            f"{int(cost[c]['spatial-first_macs']):,}",
                            f"{int(cost[c]['original_peak_activation_elems']):,}",
                            f"{int(cost[c]['spatial-first_peak_activation_elems']):,}",
                            f"{sf(c, 'fp32'):.3f}",
                            f"{sf(c, 'int8-spatial-first'):.3f}",
                        ]
                        for k, c in [(22, "full-22"), (16, "relevance-16"), (8, "proposed-8")]
                    ],
                )
                + "\n\nThe swap stores the temporal filters twice, which adds 504 values. It also changes one argument "
                "of this report. With the exported layer order, the temporal convolution runs on every electrode, so "
                "fewer electrodes mean proportionally less computation. After the swap, the electrode count only "
                "affects the small spatial layer: going from 22 to 8 electrodes removes "
                f"{macs_saved_swapped:.0f}% "
                "of the multiply-accumulates instead of "
                f"{macs_saved_exported:.0f}%. "
                "The case for fewer electrodes then rests on hardware and set-up cost, not on computation.\n\n"
                f"For the frozen configuration's architecture, INT8 takes {sf('relevance-16-prune50', 'int8'):.3f} ms "
                f"as exported and {sf('relevance-16-prune50', 'int8-spatial-first'):.3f} ms after the swap. The frozen "
                "configuration itself is unchanged; these are post-test engineering measurements on a desktop CPU and "
                "say nothing about microcontroller latency."
            )
            probe = base / "int8_study/coreml_probe.json"
            if probe.exists():
                c = json.loads(probe.read_text())
                t = {(r["graph"], r["backend"]): r["median_ms"] for r in c["rows"]}
                taken = [
                    r["nodes_by_provider"].get("CoreMLExecutionProvider", 0)
                    for r in c["rows"]
                    if r["graph"].startswith("int8") and r["backend"] != "cpu"
                ]
                text += (
                    "\n\nOne more option is specific to this machine: Apple's CoreML backend, which can also use the "
                    "GPU and the Neural Engine. It was timed on the same model with the same interleaved rounds (load "
                    f"average {c['environment']['load_average_before'][0]:.0f}). Restricted to the CPU, CoreML runs "
                    f"the exported FP32 graph in {t[('fp32', 'coreml-cpu-only')]:.3f} ms against "
                    f"{t[('fp32', 'cpu')]:.3f} ms for the CPU provider, but CoreML manages its own threads, so this is "
                    "not a single-thread number. With the GPU and the Neural Engine allowed it takes "
                    f"{t[('fp32', 'coreml-any-unit')]:.3f} ms. CoreML accepts only {min(taken)} to {max(taken)} nodes "
                    "of the INT8 graphs and leaves the rest to the CPU provider, which makes them slower "
                    f"({t[('int8', 'coreml-cpu-only')]:.3f} ms as exported, "
                    f"{t[('int8-spatial-first', 'coreml-cpu-only')]:.3f} ms swapped). No CoreML configuration beats "
                    f"the swapped graphs on the plain CPU provider ({t[('fp32-spatial-first', 'cpu')]:.3f} ms in FP32, "
                    f"{t[('int8-spatial-first', 'cpu')]:.3f} ms in INT8), so the layer swap remains the improvement "
                    "that suits this machine."
                )
            headline.append(
                "The INT8 slowdown comes from ONNX Runtime's kernel for the grouped spatial convolution, not from the "
                "M4: swapping EEGNet's first two layers, an exact rewrite that needs no retraining, runs in "
                f"**{lat['fp32-spatial-first']:.3f} ms** in FP32 and **{lat['int8-spatial-first']:.3f} ms** in INT8 "
                f"against {ref:.3f} ms for the exported FP32 graph, with accuracy unchanged "
                f"({acc['int8-spatial-first']['mean']:+.2f} points over all conditions). Sixteen-bit quantization was "
                f"slower than FP32 ({lat['int16'] / ref:.2f}x)."
            )
        else:
            text += "Accuracy of the variants has been scored; the timing run is **pending**."
        parts.append(text)
    else:
        parts.append("### 6.4 INT8 on the M4\n\nThe deployment study (`scripts/int8_study.py`) is **pending**.")
    return headline, parts


def frozen_vs_later_section(base):
    """Section 6.5: the frozen configuration next to the variants that the later studies make available."""
    parts = []
    if all((base / "int8_study" / name).exists() for name in ("accuracy.csv", "latency.csv", "costs.csv")):
        acc = {
            (r["condition"], r["variant"]): float(r["acc_mean"])
            for r in read_csv("supplementary/int8_study/accuracy.csv")
        }
        lat = {(r["condition"], r["variant"]): r for r in read_csv("supplementary/int8_study/latency.csv")}
        cost = {r["condition"]: r for r in read_csv("supplementary/int8_study/costs.csv")}
        ref = acc[("full-22", "fp32")]

        def row(label, c, v):
            graph = "spatial-first" if v.endswith("spatial-first") else "original"
            return [
                label,
                f"{acc[(c, v)]:.2f}",
                "0" if (c, v) == ("full-22", "fp32") else f"{acc[(c, v)] - ref:+.2f}",
                f"{float(lat[(c, v)]['median_ms']):.3f}",
                f"{int(lat[(c, v)]['parameter_bytes']):,}",
                f"{int(cost[c][graph + '_macs']):,}",
                f"{int(cost[c][graph + '_peak_activation_elems']):,}",
            ]

        def ms(c, v):
            return float(lat[(c, v)]["median_ms"])

        saved = int(lat[("relevance-16", "int8-spatial-first")]["parameter_bytes"]) - int(
            lat[("relevance-16-prune50", "int8-spatial-first")]["parameter_bytes"]
        )
        text = (
            "### 6.5 The frozen configuration next to what the later studies make available\n\n"
            "The frozen configuration is unchanged and remains the final result of the planned study. The table places "
            "it next to variants built from the same archived models (seeds 0-2), so the trade-offs can be read in one "
            "place.\n\n"
            + table(
                [
                    "Model",
                    "Accuracy (%)",
                    "Minus full-22 FP32 (pp)",
                    "Median time (ms)",
                    "Parameter bytes",
                    "MACs",
                    "Peak activation values",
                ],
                [
                    row("22 electrodes, FP32, as exported (baseline)", "full-22", "fp32"),
                    row("Frozen final: 16 electrodes, 50% pruned, INT8, as exported", "relevance-16-prune50", "int8"),
                    row(
                        "Frozen final with the first two layers swapped, INT8",
                        "relevance-16-prune50",
                        "int8-spatial-first",
                    ),
                    row("16 electrodes, not pruned, layers swapped, INT8", "relevance-16", "int8-spatial-first"),
                    row("22 electrodes, not pruned, layers swapped, INT8", "full-22", "int8-spatial-first"),
                    row(
                        "8 electrodes (redundancy-aware), not pruned, layers swapped, INT8",
                        "proposed-8",
                        "int8-spatial-first",
                    ),
                ],
            )
            + "\n\nOnce the layers are swapped, pruning the last layer no longer buys time "
            f"({ms('relevance-16-prune50', 'int8-spatial-first'):.3f} ms pruned, "
            f"{ms('relevance-16', 'int8-spatial-first'):.3f} ms unpruned) and saves {saved:,} parameter bytes"
        )
        matched = base / "matched_budget/analysis.json"
        if matched.exists():
            c = next(
                r
                for r in json.loads(matched.read_text())["comparisons"]
                if r["base"] == "relevance-16" and r["precision"] == "int8" and r["a"] == "50" and r["b"] == "0"
            )
            text += (
                f", while the matched-budget comparison in Section 5 attributes {abs(c['difference_pp']):.2f} points "
                "of accuracy to it"
            )
        text += (
            ". Going from 22 to 16 electrodes then saves "
            f"{1000 * (ms('full-22', 'int8-spatial-first') - ms('relevance-16', 'int8-spatial-first')):.0f} "
            "microseconds per decision, so its value is in hardware and set-up, not in computation. If a deployment "
            "were chosen today, 16 unpruned electrodes with swapped layers in INT8 would be the natural candidate. "
            "That is an observation made after the test results were known. It is not a new final configuration, and "
            "its accuracy carries the same participant-level uncertainty as every other row."
        )
        parts.append(text)
    return [], parts


def interval(diff, low, high):
    """'+1.23 [-0.45, +2.80]' from a difference and its 95% participant interval, in points."""
    return f"{diff:+.2f} [{low:+.2f}, {high:+.2f}]"


def pending(title, script):
    return [], [f"### {title}\n\nThis study (`{script}`) is **pending**; no outcome is claimed here."]


def bit_width_section(base):
    """Section 6.6: how many bits the decoder needs, from simulated quantization of every archived model."""
    path, rule_path = base / "quant_study/analysis.json", base / "quant_rule_study/analysis.json"
    title = "6.6 How many bits the decoder needs, on any processor"
    if not path.exists() or not rule_path.exists():
        return pending(title, "scripts/quant_study.py, scripts/quant_rule_study.py")
    a = json.loads(path.read_text())
    ptq = {r["setting"]: r for r in a["ptq_pooled"] if r["condition"] == "all conditions"}
    qat = {r["variant"]: r for r in a["qat"] if r["condition"] == "all QAT conditions"}
    sens = {r["tensor"]: r["diff_pp"] for r in a["sensitivity_pooled"] if r["session"] == "session1"}
    rule = {(r["condition"], r["setting"]): r for r in json.loads(rule_path.read_text())["settings"]}
    rows = []
    for setting, label in (
        ("16/16", "16-bit weights and activations (INT16)"),
        ("8/8", "8-bit weights and activations (INT8)"),
        ("4/8", "4-bit weights, 8-bit activations"),
        ("4/4", "4-bit weights and activations"),
    ):
        r = ptq[setting]
        rows.append(["Rounding only", label, f"{r['acc_mean']:.2f}",
                     interval(r["vs_float_diff_pp"], r["vs_float_ci_low_pp"], r["vs_float_ci_high_pp"]),
                     f"{r['vs_float_n_losing_over_2pp']}/9"])  # fmt: skip
    for setting, label in (
        ("pooled-0.5", "Widths per tensor from training-day sensitivity, pooled over participants"),
        ("own-0.5", "Widths per tensor from each model's own training-day sensitivity"),
    ):
        r = rule[("all conditions", setting)]
        rows.append(["Rounding only", label, f"{r['accuracy']:.2f}",
                     interval(r["vs_float_diff_pp"], r["vs_float_low_pp"], r["vs_float_high_pp"]),
                     f"{r['vs_float_n_lose_more_than_2']}/9"])  # fmt: skip
    for variant, label in (
        ("w8a8", "8-bit weights and activations"),
        ("w4a8", "4-bit weights, 8-bit activations"),
        ("w4a4-early8", "4-bit, but 8-bit input and temporal-filter output"),
        ("w4a4", "4-bit weights and activations"),
    ):
        r = qat[variant]
        rows.append(["Fine-tuned with the rounding (4 architectures)", label, f"{r['acc_after']:.2f}",
                     interval(r["vs_fp32ft_diff_pp"], r["vs_fp32ft_ci_low_pp"], r["vs_fp32ft_ci_high_pp"]),
                     f"{r['vs_fp32ft_n_losing_over_2pp']}/9"])  # fmt: skip
    early = ("activations:input@4", "activations:conv_t@4")
    others = [v for k, v in sens.items() if k.startswith("activations:") and k not in early]
    pooled = rule[("all conditions", "pooled-0.5")]
    eight = rule[("all conditions", "8/8")]
    final = rule[("relevance-16-prune50", "pooled-0.5")]
    final8 = rule[("relevance-16-prune50", "8/8")]
    rho = a["precision_rule"]["spearman_session1_vs_session2"]["rho"]
    agreement = a["simulator_check"]["agreement_with_archived_int8_pct_mean"]
    text = (
        f"### {title}\n\n"
        "The review asked whether 16-bit integers would do better than INT8. Speed belongs to a runtime and a "
        "processor; the number of bits a decoder needs does not, so it was measured without either. Every archived "
        "final model (459: 17 conditions, 9 participants, 3 seeds) was rounded to integer grids of each width and "
        "evaluated in floating point (`scripts/quant_study.py`, `mibci.fakequant`). The simulated 8-bit network "
        f"agrees with the archived ONNX Runtime INT8 predictions on {agreement:.2f}% of session-2 trials, so it "
        "stands in for the deployed INT8.\n\n"
        + table(["Method", "Bit widths", "Accuracy (%)", "Minus reference (pp) [95%]", "Lose > 2 pp"], rows)
        + "\n\nAccuracy is on artifact-free session-2 trials, pooled over conditions. The reference is the same "
        "models in floating point; for the fine-tuned rows it is the same 100 epochs of fine-tuning without rounding, "
        "because extra training alone changes accuracy.\n\n"
        f"Sixteen bits buys nothing over eight: INT8 loses {-ptq['8/8']['vs_float_diff_pp']:.2f} points and no "
        "participant loses more than 2. Below eight bits, two activations break first: rounding the input alone to 4 "
        f"bits costs {-sens['activations:input@4']:.1f} points of training-day accuracy and the temporal filters' "
        f"output {-sens['activations:conv_t@4']:.1f}, against at most {-min(others):.1f} for any other activation; "
        f"among the weights, the spatial filters' ({-sens['weights:conv_s@4']:.1f}). The tensors break in nearly the "
        f"same order on both days (Spearman {rho:.2f} over 20 single-tensor tests), so widths can be chosen from the "
        "training day. Pooled over participants with a threshold of 0.5 points, that choice keeps 8 bits for the "
        "spatial filters' weights and for seven of the ten activation tensors and rounds the rest to 4. Without "
        f"retraining it loses {-pooled['vs_float_diff_pp']:.2f} points, no participant loses more than 2, and the "
        f"weights take {pooled['weight_bytes']:,.0f} bytes on average instead of {eight['weight_bytes']:,.0f} at INT8 "
        f"(for the frozen final architecture {final['weight_bytes']:,.0f} instead of {final8['weight_bytes']:,.0f}, "
        f"losing {-final['vs_float_diff_pp']:.2f} points). Widths read from each model's own training day lose more "
        f"({-rule[('all conditions', 'own-0.5')]['vs_float_diff_pp']:.2f} points): one participant's training trials "
        "are too few to choose bit widths. Fine-tuning with the rounding in the loop brings 4-bit weights with 8-bit "
        f"activations to within {-qat['w4a8']['vs_fp32ft_diff_pp']:.2f} points of its control.\n\n"
        "Bit widths below 8 are simulated: nothing below 8 bits ran on a low-bit kernel or was timed. The 0.5-point "
        "threshold was picked after the bit-width study's session-2 results were known, so the per-tensor widths are "
        "an exploratory result (`results/supplementary/quant_rule_study/summary.md`)."
    )
    saving = 100 * (1 - pooled["weight_bytes"] / eight["weight_bytes"])
    headline = (
        f"INT8 loses {abs(ptq['8/8']['vs_float_diff_pp']):.1f} points against floating point and INT16 adds nothing; "
        "widths chosen per tensor from training-day data keep accuracy within "
        f"{abs(pooled['vs_float_diff_pp']):.1f} points with {saving:.0f}% fewer weight bytes than INT8 (simulated)."
    )
    return [headline], [text]


def portability_section(base):
    """Section 6.7: the same ONNX graphs timed and scored on other processors."""
    lat_path, port_path = base / "hw_latency/latency.csv", base / "int8_portability/machines.csv"
    title = "6.7 The same INT8 files on other processors"
    if not lat_path.exists() or not port_path.exists():
        return pending(title, "scripts/hw_latency.py, scripts/int8_portability.py")
    lat = read_csv("supplementary/hw_latency/latency.csv")
    by = {(r["machine"], r["condition"], r["variant"]): float(r["median_us"]) for r in lat}
    cpus = {r["machine"]: r["cpu"] for r in lat}
    condition = "relevance-16-prune50"
    rows = []
    for machine in sorted(cpus, key=lambda m: (cpus[m] != "Apple M4", cpus[m], m)):
        variants = ("fp32", "int8", "int8-spatial-first", "fp32-spatial-first", "int16")
        t = {v: by[(machine, condition, v)] for v in variants}
        rows.append([machine, cpus[machine].replace("(R)", ""), f"{t['fp32']:.0f}",
                     f"{t['int8']:.0f} ({t['int8'] / t['fp32']:.2f}x)",
                     f"{t['int16']:.0f} ({t['int16'] / t['fp32']:.2f}x)",
                     f"{t['int8-spatial-first']:.0f}", f"{t['fp32-spatial-first']:.0f}"])  # fmt: skip
    machines = read_csv("supplementary/int8_portability/machines.csv")
    seen, port_rows = set(), []
    for m in machines:
        if m["cpu"] in seen:
            continue
        seen.add(m["cpu"])
        rr = next((float(x["rr_acc_change_vs_fp32_pp"]) for x in machines
                   if x["cpu"] == m["cpu"] and x["rr_acc_change_vs_fp32_pp"]), None)  # fmt: skip
        port_rows.append([m["cpu"].replace("(R)", ""), m["integer_extensions"], f"{m['fp32_models_changed']}/459",
                          f"{m['int8_models_changed']}/459", f"{float(m['int8_acc_change_pp']):+.2f}",
                          "" if rr is None else f"{rr:+.2f}"])  # fmt: skip
    changed = [m for m in machines if int(m["int8_models_changed"]) > 0]
    worst = min(float(m["int8_acc_change_pp"]) for m in changed) if changed else 0.0
    text = (
        f"### {title}\n\n"
        "**Speed.** The frozen final graph (16 electrodes, 50% pruned) was timed on Longleaf nodes with "
        "`scripts/hw_latency.py`, next to the M4 numbers of Section 6.4 (microseconds per decision, median; ratios "
        "to the same machine's FP32):\n\n"
        + table(
            [
                "Machine",
                "CPU",
                "FP32",
                "INT8 as exported",
                "INT16 activations",
                "INT8, layers swapped",
                "FP32, layers swapped",
            ],
            rows,
        )
        + "\n\nINT8 as exported is slower than FP32 on every processor measured, including the AMD EPYC 9654, "
        "whose AVX-512 VNNI multiplies 8-bit integers in hardware, so the M4 has no special problem: the slow path is "
        "the runtime's grouped spatial convolution. Swapping the first two layers speeds INT8 up everywhere; on the "
        "M4 it is then faster than FP32, but on the x86 servers FP32 with the same swap is faster still. INT16 "
        "activations are slower than FP32 on every machine. Times on shared servers are noisy; ratios within one "
        "machine are more reliable than times across machines.\n\n"
        "**Accuracy.** The archived FP32 and INT8 files of all 459 final models were scored again on session 2 on "
        "each processor and compared with the predictions archived on the M4 (`scripts/int8_portability.py`):\n\n"
        + table(
            [
                "CPU",
                "8-bit related extensions",
                "FP32 models changed",
                "INT8 models changed",
                "INT8 accuracy change (pp)",
                "7-bit weights: accuracy vs FP32 (pp)",
            ],
            port_rows,
        )
        + "\n\nFP32 gives the same predictions everywhere. INT8 reproduces every prediction on the M4 and the AMD "
        f"EPYC 9654, but on every processor without VNNI it changes predictions in nearly every model and costs "
        f"{abs(worst):.2f} points on average. This is what the documented behaviour of ONNX Runtime's x86 8-bit "
        "kernels predicts: without VNNI they add pairs of 8-bit products in 16 bits (VPMADDUBSW), which can "
        "saturate, while Arm dot products and VNNI accumulate in 32 bits. ONNX Runtime documents 7-bit weights "
        "(reduce_range) as the "
        "remedy; with them every one of the 459 models gives the same accuracy and agreement on all four processors "
        "tested, at a small cost against FP32. So whether INT8 keeps its accuracy depends on the processor, not only "
        "how fast it runs. These checks ran after the test results were known and concern deployment, not the frozen "
        "analysis."
    )
    headline = (
        "The same INT8 files lose accuracy on x86 processors without VNNI "
        f"({worst:+.1f} points); 7-bit weights make the results identical on every processor tested."
    )
    return [headline], [text]


def alignment_section(base):
    """Section 6.8: unsupervised Euclidean alignment of each recording day."""
    path = base / "alignment_study/alignment.csv"
    title = "6.8 Aligning the two recording days without labels"
    if not path.exists():
        return pending(title, "scripts/alignment_study.py")
    rows = read_csv("supplementary/alignment_study/alignment.csv")
    shown = []
    label = {"ea-all": "aligned with all of session 2", "ea-first-run": "aligned with its first run (48 trials)"}
    for r in rows:
        if r["variant"] not in label:
            continue
        shown.append([r["condition"], label[r["variant"]], r["trials_scored"], f"{float(r['acc_mean_pct']):.2f}",
                      f"{float(r['baseline_mean_pct']):.2f}",
                      interval(float(r["diff_pp"]), float(r["ci_low_pp"]), float(r["ci_high_pp"])),
                      f"{r['n_better']}/{r['n_worse']}"])  # fmt: skip
    per = read_csv("supplementary/alignment_study/per_participant.csv")
    full = [float(r["ea_all_gain_pp"]) for r in per if r["condition"] == "full-22"]
    spread = f" (with all 22 electrodes, from {min(full):+.1f} to {max(full):+.1f} points)" if full else ""
    text = (
        f"### {title}\n\n"
        "Euclidean alignment (He and Wu 2019) multiplies each recording day by the inverse square root of that day's "
        "average spatial covariance, so that the electrodes co-vary the same way on both days; it needs no labels. "
        "Models were retrained with the frozen recipe on aligned session-1 trials and scored on aligned session-2 "
        "trials, paired with the archived unaligned model of the same participant, seed and electrodes "
        "(`scripts/alignment_study.py`):\n\n"
        + table(
            [
                "Electrodes",
                "Session 2",
                "Trials scored",
                "Aligned (%)",
                "Unaligned (%)",
                "Difference (pp) [95%]",
                "Better/worse",
            ],
            shown,
        )
        + f"\n\nAlignment helps on average, but it moves individual participants in both directions{spread}. "
        "The interval for all 22 electrodes includes losses of several points, and with nine participants the "
        "study cannot say whether alignment helps a new user. It uses unlabeled session-2 trials, which the "
        "frozen protocol does not, and it was planned after the test results were known. The second dataset tests "
        "it prospectively (Section 6.11)."
    )
    return [], [text]


SELECTION_LABELS = {
    "relevance": "Relevance, one electrode at a time (frozen rule)",
    "riemann": "Riemannian distance between class covariances",
    "csp": "Common spatial patterns",
    "wrapper": "Wrapper: backward elimination with a fast decoder",
    "learned": "Learned selector in front of EEGNet",
    "prior": "Fixed order around C3, Cz, C4 (no data)",
    "pop-relevance": "Population: other eight participants' relevance",
    "pop-riemann": "Population: other eight participants' Riemannian",
    "pop-wrapper": "Population: other eight participants' wrapper",
}


def selection_section(base):
    """Section 6.10: joint, prior and population electrode selection against the frozen per-electrode rule."""
    path = base / "selection_study/analysis.json"
    title = "6.10 Choosing electrodes together instead of one at a time"
    if not path.exists():
        return pending(title, "scripts/selection_study.py")
    a = json.loads(path.read_text())
    acc = {(r["method"], int(r["k"])): r for r in a["accuracy"]}
    counts = a["counts"]
    rows = [[SELECTION_LABELS[m], *(f"{acc[(m, k)]['acc_mean_pct']:.1f}" for k in counts)] for m in a["methods"]]
    full = acc[("full-22", 22)]["acc_mean_pct"]
    r8 = acc[("riemann", 8)]

    def vs(r, ref):
        return interval(r[f"vs_{ref}_diff_pp"], r[f"vs_{ref}_ci_low_pp"], r[f"vs_{ref}_ci_high_pp"])

    short = {"riemann": "the Riemannian rule", "csp": "common spatial patterns", "wrapper": "the wrapper",
             "learned": "the learned selector", "prior": "the fixed order", "pop-relevance": "population relevance",
             "pop-riemann": "population Riemannian", "pop-wrapper": "population wrapper"}  # fmt: skip
    above = []
    for m in a["methods"][1:]:
        ks = [str(k) for k in counts if acc[(m, k)]["vs_relevance_ci_low_pp"] > 0]
        if ks:
            above.append(f"{short[m]} at {both(ks)}")
    mean_over_counts = {m: statistics.mean(acc[(m, k)]["acc_mean_pct"] for k in counts) for m in a["methods"]}
    lowest = min(mean_over_counts, key=mean_over_counts.get)
    pairs = [(m, k) for m in ("relevance", "riemann", "wrapper") for k in counts]
    drop = {m: acc[(m, 16)]["acc_mean_pct"] - acc[(m, 8)]["acc_mean_pct"] for m in a["methods"]}
    gains = [acc[("riemann", k)]["vs_relevance_n_better"] for k in counts if k <= 10]
    own_better = sum(acc[(m, k)]["acc_mean_pct"] > acc[(f"pop-{m}", k)]["acc_mean_pct"] for m, k in pairs)
    control = {(r["kind"], r["method"], r["k"]): r for r in a.get("hardware_control") or []}
    hw = [r for (kind, _, _), r in control.items() if kind == "retrained minus earlier"]
    same = control.get(("minus retrained relevance", "riemann", 8))
    rr_path = base / "replication_riemann_study/analysis.json"
    rr = {r["hypothesis"]: r for r in json.loads(rr_path.read_text())["hypotheses"]} if rr_path.exists() else {}
    text = (
        f"### {title}\n\n"
        "The frozen rule scores each electrode on its own. A motor-imagery decoder works through spatial filters, "
        "which combine electrodes, so `scripts/selection_study.py` compared rules that judge electrodes jointly, a "
        "fixed physiological order, and montages taken from other people. Every ranking used session 1 only; the "
        "population montages of a participant use only the other eight participants' rankings. Each subset was "
        "trained with the frozen recipe (seeds 0-2) and scored once on session 2 (1,350 new runs):\n\n"
        + table(["Method", *(f"{k} electrodes (%)" for k in counts)], rows)
        + f"\n\nAll 22 electrodes: {full:.1f}%. At 16 electrodes the frozen rule is still the best of the nine. Below "
        f"that it loses accuracy fastest (from 16 to 8 electrodes it drops {drop['relevance']:.1f} points, the most "
        "of the nine methods): at 8 electrodes the Riemannian rule (Barachant and Bonnet 2011), which keeps "
        "the subset whose class-average covariance matrices stay farthest apart, reaches "
        f"{r8['acc_mean_pct']:.1f}%, {vs(r8, 'relevance')} points above the frozen rule "
        f"({r8['vs_relevance_n_better']} participants better, {r8['vs_relevance_n_worse']} worse) and "
        f"{vs(r8, 'full22')} points from all 22 electrodes; its 8-electrode subsets share "
        f"{r8['shared_with_relevance']:.1f} of their 8 electrodes with the frozen rule's on average. Intervals "
        f"entirely above the frozen rule: {both(above)} electrodes. Even the fixed order around the hand and foot "
        "areas, which uses no data at all, is among them. Averaged over the six counts, "
        f"{short.get(lowest, lowest)} scores lowest ({mean_over_counts[lowest]:.1f}%, against "
        f"{mean_over_counts['relevance']:.1f}% for the frozen rule), and a montage borrowed from the other eight "
        f"participants scores below the person's own montage from the same method in {own_better} of "
        f"{len(pairs)} cases.\n\n"
        "These are 48 comparisons with the frozen rule on nine people, chosen after the test results were known; "
        "about one interval in twenty would exclude zero by chance alone, and the best method at each count is "
        "picked after the fact. The Riemannian advantage holds at each of 4, 6, 8 and 10 electrodes, with "
        f"{min(gains)} to {max(gains)} of the nine participants gaining at each count, but it is an exploratory "
        "result that would need a prospective test on new people "
        "(`results/supplementary/selection_study/summary.md`)."
    )
    if hw and same:
        low, high = min(r["diff_pp"] for r in hw), max(r["diff_pp"] for r in hw)
        gap = interval(same["diff_pp"], same["ci_low_pp"], same["ci_high_pp"])
        text += (
            "\n\nThe relevance and all-22 references came from the archived runs, which the M4 trained (on its GPU "
            "for participants 1-6), while this study trained on Longleaf CPUs. Section 6.9 found that training "
            "hardware changes individual models like a new seed, so `scripts/selection_control_study.py` retrained "
            "those 135 references on Longleaf with the same electrodes, recipe and seeds. Retrained minus archived "
            f"ranges from {low:+.1f} to {high:+.1f} points across the five conditions, every interval including zero: "
            "hardware changes individual predictions but did not shift accuracy one way. Against the retrained "
            f"references, the Riemannian rule at 8 electrodes is {gap} "
            f"points ({same['n_better']} participants better, {same['n_worse']} worse)."
        )
    if rr:
        text += (
            " The finding was then tested prospectively on 54 new people (Section 6.11): Riemannian selection beat "
            f"the frozen rule at 8 ({interval(rr['R1']['diff_pp'], rr['R1']['low_pp'], rr['R1']['high_pp'])}), 16 "
            f"({interval(rr['R2']['diff_pp'], rr['R2']['low_pp'], rr['R2']['high_pp'])}) and 20 electrodes "
            f"({interval(rr['R3']['diff_pp'], rr['R3']['low_pp'], rr['R3']['high_pp'])})."
        )
    headline = (
        f"Choosing electrodes jointly keeps 8 electrodes at {r8['acc_mean_pct']:.1f}% (Riemannian rule), "
        f"{r8['vs_relevance_diff_pp']:+.1f} points above the frozen rule and {r8['vs_full22_diff_pp']:+.1f} from all "
        "22 (exploratory)."
    )
    return [headline], [text]


def region_section(base):
    """Section 6.9: which functional scalp regions carry the signal (exact Shapley values over all region sets)."""
    path = base / "region_study/analysis.json"
    title = "6.9 Which scalp regions carry the motor-imagery signal"
    if not path.exists():
        return pending(title, "scripts/region_study.py")
    a = json.loads(path.read_text())
    cv, test = a["shapley_pp"]["cv"], a["shapley_pp"]["test"]
    fine = a["regions"]["fine"]

    def ci(v):
        return f"{v['mean']:+.2f} [{v['ci_low']:+.2f}, {v['ci_high']:+.2f}]"

    rows = [[name, " ".join(electrodes), ci(cv[name]["accuracy"]),
             f"{cv[name]['accuracy']['mean'] / len(electrodes):+.2f}", ci(test[name]["accuracy"])]
            for name, electrodes in fine.items()]  # fmt: skip
    classes = ("left_hand", "right_hand", "feet", "tongue")
    recall_rows = [[name, *(f"{cv[name]['recall'][c]['mean']:+.1f}" for c in classes)] for name in fine]
    pairs = sorted(a["interactions_cv_pp"], key=lambda r: r["mean"])[:3]
    frontier = {(r["budget"], r["method"]): r for r in a["frontier"]}
    own16, oracle16 = frontier[(16, "own_cv")], frontier[(16, "oracle")]
    hand = {c: max(fine, key=lambda n: cv[n]["recall"][c]["mean"]) for c in classes}
    low = sorted(fine, key=lambda n: cv[n]["accuracy"]["mean"])[:3]
    check = a["pipeline_check_full22"]
    versus = [r for r in a["relevance_comparison"] if r["method"] == "own_cv"]
    below = sorted(r["electrodes"] for r in versus if r["ci_high_pp"] < 0)
    above = sorted(r["electrodes"] for r in versus if r["ci_low_pp"] > 0)
    spread = [r["minus_relevance_pp"] for r in versus]
    budgets = sorted(r["electrodes"] for r in versus)
    agree = [100 * r["prediction_agreement"] for r in check]
    text = (
        f"### {title}\n\n"
        "The review asked whether electrodes should be chosen by functional brain region. The 22 electrodes were "
        "grouped into nine regions (left and right premotor, motor and somatosensory rows, the midline front and "
        "centre, and parietal; `mibci.regions`), and every one of the 511 non-empty sets of regions was trained for "
        f"every participant: {a['records']['cv']:,} cross-validation runs on session 1 (three folds per set) and "
        f"{a['records']['test']:,} final models scored on session 2 (`scripts/region_study.py`). With all 511 sets "
        "the Shapley value of each region can be computed exactly: the accuracy it adds when it joins the other "
        "regions, averaged over every set it can join. Cross-validation values use session 1 only and are test-free; "
        "session-2 values are exploratory.\n\n"
        + table(
            [
                "Region",
                "Electrodes",
                "Adds, session-1 CV (pp) [95%]",
                "Per electrode (pp)",
                "Adds, session 2 (pp) [95%]",
            ],
            rows,
        )
        + "\n\nThe values of the nine regions add up to each participant's all-22 accuracy minus chance "
        f"(to within {max(a['shapley_efficiency_max_error_pp'].values()):.0e} points). The motor, somatosensory, "
        "midline-centre and parietal regions each add about 5 to 8 points; "
        f"{both(low)} add the least, and per electrode the frontal ones carry least, which is where the frozen "
        "relevance rule removed most electrodes (Section 6.1). Splitting each region's value by class recovers the "
        "motor map from the data alone (session-1 cross-validation, points of class recall):\n\n"
        + table(["Region", "Left hand", "Right hand", "Feet", "Tongue"], recall_rows)
        + f"\n\nThe left hand is decoded mostly from {hand['left_hand']}, the right hand from {hand['right_hand']} and "
        f"the feet from {hand['feet']}: each hand from the opposite hemisphere and the feet from the vertex, as the "
        "motor homunculus predicts. Neighbouring regions overlap: the most negative interactions (two regions adding "
        "less together than separately) are "
        + both([f"{r['a']} with {r['b']} ({r['mean']:+.1f})" for r in pairs])
        + ".\n\nAs a selection rule, whole regions did not beat single electrodes. Chosen by each participant's "
        f"training-day cross-validation, the best region set within a budget of {budgets[0]} to {budgets[-1]} "
        f"electrodes scored {min(spread):+.1f} to {max(spread):+.1f} points against the frozen rule's top electrodes "
        "of the same count (seed 0)"
        + (f", below it at {both([str(k) for k in below])} electrodes" if below else "")
        + (f", above it at {both([str(k) for k in above])}" if above else "")
        + " and with intervals including zero elsewhere. With at most 16 electrodes, the "
        "region set with each participant's best training-day cross-validation scores "
        f"{own16['test_acc_pct']:.1f}% on session 2 ({own16['minus_all22_pp']:+.1f} points from all 22), while the "
        f"set that would have been best on session 2 itself scores {oracle16['test_acc_pct']:.1f}% "
        f"({oracle16['minus_all22_pp']:+.1f}). That ceiling is picked with the test day out of hundreds of noisy "
        "values, so it overstates what any rule could reach. The gap shows that training-day cross-validation ranks "
        "region sets poorly for the next day, through day-to-day change, the noise of ranking 511 sets on a few "
        "hundred trials, or both.\n\n"
        "Training hardware matters for the comparisons with archived runs. The study's all-22 model (seed 0) agrees "
        f"with the archived full-22 seed-0 model on {min(agree):.0f}-{max(agree):.0f}% of session-2 trials: almost "
        "fully for participants 7-9, whose archived models were trained on the M4's CPU like these on Longleaf's, "
        "and much less for participants 1-6, whose archived models were trained on its GPU. Comparisons between new "
        "and archived models therefore carry extra seed-like variation; the control in Section 6.10 retrained the "
        "archived references on the same hardware and found no systematic shift in accuracy."
    )
    headline = (
        "Exact Shapley values over all 511 region sets recover the motor map from the data: each hand is decoded "
        "mainly from the opposite hemisphere and the feet from the midline, while frontal regions add least."
    )
    return [headline], [text]


def replication_section(base):
    """Section 6.11: the prospective replication on Lee et al. (2019), with the Riemannian test."""
    path, rr_path = base / "replication_study/analysis.json", base / "replication_riemann_study/analysis.json"
    title = "6.11 A second two-day dataset: 54 people (Lee et al. 2019)"
    if not path.exists() or not rr_path.exists():
        return pending(title, "scripts/replication_study.py, scripts/replication_riemann_study.py")
    plan = json.loads((base / "replication_study/protocol.json").read_text())
    rr_plan = json.loads((base / "replication_riemann_study/protocol.json").read_text())
    hyp = json.loads(path.read_text())["hypotheses"]
    rr = json.loads(rr_path.read_text())
    conditions = {r["condition"]: r for r in read_csv("supplementary/replication_study/conditions.csv")}

    def what(r):
        """The plan's contrast; for hypotheses tested on several conditions, the one this row tests."""
        if "alignment" in r:
            return f"{r['condition']}: aligned minus unaligned ({r['alignment']})"
        if r["hypothesis"] == "H4":
            return f"{r['condition']}: simulated 8/8 minus simulated float"
        if r["hypothesis"] == "H4b":
            return f"{r['condition']}: the BCI IV 2a precision map minus 8/8"
        return plan["hypotheses"][r["hypothesis"]]["contrast"]

    rows = []
    for r in hyp:
        if "rho" in r:
            estimate, counts = f"rho {r['rho']:+.3f}, one-sided p = {r['p_one_sided']:.2f}", ""
        else:
            estimate = interval(r["estimate_pp"], r["low_pp"], r["high_pp"])
            counts = f"{r['better']}/{r['worse']}"
        rows.append([r["hypothesis"], what(r), estimate, counts, plan["hypotheses"][r["hypothesis"]]["test"],
                     "yes" if r["met"] else "no"])  # fmt: skip
    rows += [[r["hypothesis"], rr_plan["hypotheses"][r["hypothesis"]]["contrast"],
              interval(r["diff_pp"], r["low_pp"], r["high_pp"]), f"{r['better']}/{r['worse']}",
              rr_plan["hypotheses"][r["hypothesis"]]["test"], "yes" if r["met"] else "no"]
             for r in rr["hypotheses"]]  # fmt: skip
    by = {}
    for r in hyp:
        by.setdefault(r["hypothesis"], []).append(r)
    full = float(conditions["full-62"]["accuracy"])
    acc45, acc16 = (float(conditions[f"relevance-{k}"]["accuracy"]) for k in (45, 16))
    best = max(conditions.values(), key=lambda r: float(r["accuracy"]))
    r16 = by["H1b"][0]
    ea = by["H5"][0]
    rr_by = {r["hypothesis"]: r for r in rr["hypotheses"]}
    means = rr["mean_accuracy_pct"]
    text = (
        f"### {title}\n\n"
        "Every BCI IV 2a result above comes from nine people, and every post-test result was seen before it was "
        "explained. To test what carries over, a second two-day dataset was added: Lee et al. (2019, OpenBMI), 54 "
        "people, left- against right-hand imagery (chance 50%), 62 electrodes, 100 offline and 100 online trials "
        "per day, preprocessed like BCI IV 2a (`mibci.lee2019`). Hypotheses, conditions and tests were written into "
        f"a plan at {plan['created_at'][11:16]} on {plan['created_at'][:10]}, before any Lee2019 session-2 trial was "
        "scored (`scripts/replication_study.py`; 1,944 models: 12 conditions, 54 people, 3 seeds, the frozen "
        "recipe). The Riemannian result of Section 6.10 was added as a second plan at "
        f"{rr_plan['created_at'][11:16]}, while the first study's scoring was held so that no session-2 trial had "
        "been scored (`scripts/replication_riemann_study.py`; 486 more models). Each hypothesis is tested exactly "
        "as its plan states; accuracy is on all session-2 trials, seeds averaged within each participant, with the "
        "95% participant bootstrap interval.\n\n"
        + table(["", "Contrast", "Estimate (pp) [95%]", "Better/worse", "Test in the plan", "Met"], rows)
        + "\n\nWhat held:\n\n"
        f"- **Fewer electrodes did not hurt on average.** The 45 most relevant of 62 electrodes, the share the "
        f"frozen BCI IV 2a configuration kept, and even 16 of 62, stayed within the 2-point margin of all 62 "
        f"({acc45:.2f}% and {acc16:.2f}% against "
        f"{full:.2f}%). The average hides a wide spread: at 16 electrodes {r16['lose_more_than_2']} of 54 people lose "
        "more than 2 points and about as many gain.\n"
        "- **Relevance beats random picks at 16 electrodes,** though not clearly at 45, where a random set already "
        "keeps nearly three quarters of the montage.\n"
        "- **8-bit integer arithmetic lost nothing measurable** (simulated, every participant within 2 points), and "
        "the precision map read from BCI IV 2a's training day carried over to a different dataset, montage and "
        "task, at about 0.1-0.2 points below uniform 8 bits.\n"
        "- **Choosing electrodes jointly beat choosing them one at a time** at 8, 16 and 20 electrodes. With 8 "
        f"Riemannian electrodes the decoder reached {means['riemann-8']:.1f}%, above the {full:.1f}% of all 62 (a "
        f"comparison the plan did not test); with 20, {means['riemann-20']:.1f}%.\n\n"
        "What did not hold:\n\n"
        "- **The people decoded worst did not gain most from fewer electrodes.** With baseline and gain measured on "
        "different trials (the offline and online runs of session 2), the correlation is essentially zero. The "
        "BCI IV 2a pattern (rho -0.67 in nine people, both quantities from the same trials) is best read as noise "
        "and regression to the mean, and Section 6.2's account of who benefits should not be generalized.\n"
        f"- **Day-to-day alignment hurt.** With all 62 electrodes it cost {abs(ea['estimate_pp']):.1f} points "
        f"({ea['worse']} of 54 people worse); the loss shrinks with fewer electrodes and is not distinguishable from "
        "zero at 16. BCI IV 2a's average gain (Section 6.8) did not carry over. A later study (Section 6.12) tests "
        "one possible reason: whitening with a 62-electrode covariance estimated from 100 trials amplifies "
        "directions that carry mostly noise.\n\n"
        f"Among the first plan's conditions, {best['condition']} scored highest ({float(best['accuracy']):.1f}%); "
        f"Riemannian-20 from the second plan reached {means['riemann-20']:.1f}%. Both are descriptive. One dataset, "
        "one two-class task and 54 people: the results describe this cohort, and a 'met' non-inferiority test bounds "
        "the average loss without showing that two conditions are equal. Participants 50 and 51 each have one "
        "electrode that is exactly zero in some session-1 trials; the plan ranks it last."
    )
    headline = (
        "On 54 new people (Lee et al. 2019), with hypotheses fixed before scoring: fewer electrodes did not hurt on "
        "average, INT8 and the BCI IV 2a precision map lost almost nothing, and Riemannian selection beat the frozen "
        f"rule at 8, 16 and 20 electrodes (+{rr_by['R1']['diff_pp']:.1f} points at 8); the 'worst decoders gain "
        "most' pattern and day-to-day alignment did not replicate."
    )
    return [headline], [text]


LATER = (  # the final-sprint studies: what each tested and its tests registered on Lee 2019 before scoring
    ("input_kind_study", "the kind of extra input (copies, mixes of copy and noise, one shared noise waveform)",
     ("K1", "K2", "K3")),
    ("long_training_study", "training four times longer", ("L1", "L2", "L3")),
    ("scalefree_penalty_study", "a per-input penalty that the batch norm cannot undo", ("N1", "N2", "N3")),
    ("fresh_noise_study", "the same noise inputs drawn afresh in every training epoch", ("F1", "F2", "F3")),
    ("residual_surrogate_study", "the dropped electrodes with another trial's unpredictable part",
     ("S1", "S2", "S3", "S4")),
    ("pretrain_study", "pretraining on the other people of the same dataset", ("T1", "T2", "T3")),
    ("chdrop_study", "channel dropout and broken electrodes", ("B1", "B2", "B3")),
    ("alignment_mechanism_study", "why day-to-day alignment hurts on Lee 2019", ("M1", "M2", "M3")),
)  # fmt: skip
UNITS = {"F3": "weight ratio", "S4": "Spearman correlation"}  # every other test is in points of accuracy


def verdict(test):
    return "holds" if test["holds"] else "does not hold"


def registered_tests(base, study, keys):
    """'K1 (claim): +1.23 [-0.45, +2.80], holds' for each key, or None while the study has no analysis."""
    path = base / study / "analysis.json"
    if not path.exists():
        return None
    a = json.loads(path.read_text())
    plan = json.loads((base / study / "protocol.json").read_text())
    claims = plan.get("predictions") or plan.get("expectations") or {}
    tests = a.get("tests_lee2019") or a.get("readings") or {}
    out = []
    for key in keys:
        t, c = tests[key], claims.get(key, {})
        claim = c.get("claim") or c.get("statement") or t.get("claim", "")
        unit = f" ({UNITS[key]})" if key in UNITS else " points"
        estimate = t.get("mean", t.get("rho"))  # S4 is a Spearman correlation
        value = f": {interval(estimate, *t['ci'])}{unit}" if estimate is not None and "ci" in t else ""
        out.append(f"{key} ({claim}){value}, {verdict(t)}")
    return out


def measured_terms(base):
    """The dropped electrodes' cost and information as measured by scripts/residual_surrogate_study.py, or ''."""
    path = base / "residual_surrogate_study/analysis.json"
    if not path.exists():
        return ""
    a = json.loads(path.read_text())
    d = {(r["dataset"], r["term"]): r for r in a["decomposition"] if r["level"] == "final"}
    t = a["tests_lee2019"]
    return (
        "Measured directly (`scripts/residual_surrogate_study.py`: the prediction from the kept electrodes plus "
        "another trial's residual in place of the real dropped electrodes), the unexplained part costs "
        f"{interval(d[('bnci2a', 'cost')]['mean'], *d[('bnci2a', 'cost')]['ci'])} points on 2a and "
        f"{interval(d[('lee2019', 'cost')]['mean'], *d[('lee2019', 'cost')]['ci'])} on Lee 2019 (registered "
        f"S2 {verdict(t['S2'])}), and the real residual adds "
        f"{interval(d[('bnci2a', 'information')]['mean'], *d[('bnci2a', 'information')]['ci'])} and "
        f"{interval(d[('lee2019', 'information')]['mean'], *d[('lee2019', 'information')]['ci'])} over it (S1 "
        f"{verdict(t['S1'])}); the prediction alone costs nothing visible (S3 {verdict(t['S3'])}). On Lee 2019 the "
        "two cancel; the linear interpolation on the mixes' curve had estimated a cost of about "
        f"{-a['interpolated_input_kind']['bnci2a']['next_day_cost_at_a']:.1f} and "
        f"{-a['interpolated_input_kind']['lee2019']['next_day_cost_at_a']:.1f} points. "
    )


def kind_paragraph(base, reduced_minus_full_2a):
    """What kind of extra input costs (input_kind_study), and where the real dropped electrodes sit."""
    path = base / "input_kind_study/analysis.json"
    if not path.exists():
        return ""
    a = json.loads(path.read_text())
    rows = {(r["dataset"], r["condition"]): r for r in read_csv("supplementary/input_kind_study/conditions.csv")}
    cost = {k: float(rows[k]["next_day_cost"]) for k in rows if rows[k]["next_day_cost"]}
    rep = json.loads((base / "replication_study/analysis.json").read_text())
    h1b = next(h for h in rep["hypotheses"] if h["hypothesis"] == "H1b")["estimate_pp"]  # relevance-16 minus full-62
    eq, pr = a["a_equivalent"], a["predictability"]
    net = (reduced_minus_full_2a, h1b)
    both = {c: f"{cost[('bnci2a', c)]:+.2f} and {cost[('lee2019', c)]:+.2f}" for c in ("dup", "shared", "noise")}
    return (
        "**What costs is the part of an input the others cannot predict** (`scripts/input_kind_study.py`). With "
        f"6 extra inputs on 2a and 46 on Lee 2019, accuracy changes by {both['dup']} points with copies of kept "
        f"electrodes, {both['shared']} with one noise waveform repeated and {both['noise']} with independent noise; "
        "mixes of a copy and noise lie in between. The real dropped electrodes are mostly predictable: a linear fit "
        f"from the 16 kept ones leaves {100 * pr['bnci2a']['mean']:.0f}% (2a) and {100 * pr['lee2019']['mean']:.0f}% "
        "(Lee 2019) of their variance on held-out runs unexplained. "
        + (
            measured_terms(base)
            or (
                "By linear interpolation mixes like that cost about "
                f"{-eq['bnci2a']['next_day_cost_at_a']:.1f} and {-eq['lee2019']['next_day_cost_at_a']:.1f} points, "
                f"while dropping the real electrodes changes accuracy by only {net[0]:+.2f} and {net[1]:+.2f}; these "
                "are rough estimates (scripts/residual_surrogate_study.py measures both terms directly). "
            )
        )
        + "An earlier draft compared the real electrodes "
        "with noise in their place and called the difference their information; that comparison also credits them "
        "for being predictable, so it overstated both their information and their cost.\n\n"
    )


def later_studies_section(base):
    """Section 6.12: the later account of why fewer electrodes keep accuracy; the course report has it in full."""
    title = "6.12 Later studies: why 16 electrodes are enough"
    needed = ("overfit_study", "input_cost_study", "noise_mechanism_study", "input_penalty_study")
    if not all((base / s / "analysis.json").exists() for s in needed):
        return pending(title, ", ".join(f"scripts/{s}.py" for s in needed))
    over, cost, mech, pen = (json.loads((base / s / "analysis.json").read_text()) for s in needed)
    c = {(x["dataset"], x["contrast"]): x for x in over["contrasts"]}
    noise = c[("bnci2a", "P3: reduced plus noise minus reduced")]
    reduced = c[("bnci2a", "reduced minus full, next day, all of the training day")]
    slopes = {(x["dataset"], x["level"]): x for x in cost["slopes"]}
    share = next(x for x in mech["shares"] if x["dataset"] == "bnci2a" and x["added"] == 6)
    rec = (mech["M1"]["k=6"]["share_recovered"], mech["M1"]["k=16"]["share_recovered"])
    c2, p3 = cost["tests_lee2019"]["C2"], over["tests_lee2019"]["P3"]
    g = pen["tests_lee2019"]
    text = (
        f"### {title}\n\n"
        "The [course report](COURSE_REPORT.pdf) (`docs/COURSE_REPORT.md`, built by `scripts/build_course_report.py`) "
        "is the current account of every study; this section only corrects the reading in Section 6.2.\n\n"
        "**Every extra input costs accuracy.** Independent band-limited noise added to the 16 kept electrodes "
        f"costs {slopes[('bnci2a', 'final')]['slope']:+.2f} points per input on BCI IV 2a and "
        f"{slopes[('lee2019', 'final')]['slope']:+.2f} on Lee 2019, steadily with the number of inputs, and the cost "
        f"does not shrink with three times more training data (2a: {slopes[('bnci2a', 'rfold')]['slope']:+.2f} per "
        f"input with 2 of 6 runs, {slopes[('bnci2a', 'final')]['slope']:+.2f} with all 6; registered C2 on Lee 2019 "
        f"{verdict(c2)}, the effect going the other way). So the cost is not that more inputs leave more room to fit "
        f"the training trials. Noise in place of the six dropped electrodes costs "
        f"{interval(noise['mean'], noise['ci_low'], noise['ci_high'])} points against the 16 alone (Lee 2019, "
        f"registered P3: {p3['mean']:+.2f}, {verdict(p3)}).\n\n"
        + kind_paragraph(base, reduced["mean"])
        + "**EEGNet does not learn to ignore a recorded input it cannot use.** After training, six noise inputs still "
        "make up "
        f"{100 * share['noise_share_trained']:.0f}% of each spatial filter's output variance, against "
        f"{100 * share['noise_share_init']:.0f}% at initialization, and zeroing their weights after training "
        f"recovers only {100 * rec[0]:.0f}-{100 * rec[1]:.0f}% of the loss. A group-lasso penalty on each input's "
        f"weights did not help (registered G1: {verdict(g['G1'])}; G2: {verdict(g['G2'])}) because a batch norm "
        "follows the spatial filters. Recorded noise is the same in every epoch, so the network can use it to "
        "remember the training trials; drawn afresh in every epoch, the same noise is largely switched off "
        "(`scripts/fresh_noise_study.py`, below).\n\n"
        "**Tests of this account,** each with predictions registered on Lee 2019 before its models were scored:\n"
    )
    for study, what, keys in LATER:
        tests = registered_tests(base, study, keys)
        text += f"\n- *{what[0].upper() + what[1:]}* (`scripts/{study}.py`): " + (
            "; ".join(tests) + "." if tests else "still running when this report was built."
        )
    return [], [text]


def post_midterm():
    """Summary paragraph and section text for the studies added after the midterm review.

    All of them were planned after the session-2 results were known. Each part is written
    only when its analysis file exists; otherwise it is reported as pending.
    """
    base = RESULTS / "supplementary"
    headline = []
    parts = []
    sections = (electrode_sections, redundancy_section, int8_section, frozen_vs_later_section,
                bit_width_section, portability_section, alignment_section, region_section,
                selection_section, replication_section, later_studies_section)  # fmt: skip
    for section in sections:
        sentences, texts = section(base)
        headline += sentences
        parts += texts
    summary = (
        (
            "**Added after the midterm review (post-test, exploratory).** "
            + " ".join(headline)
            + " These additions explain the frozen results; they do not change the frozen configuration."
        )
        if headline
        else ""
    )
    body = (
        "All analyses in this section were added after the midterm review of 9 November 2026, when the session-2 "
        "results were already known. They are exploratory, live under `results/supplementary/`, and leave the frozen "
        "configuration and every historical record unchanged.\n\n" + "\n\n".join(parts)
    )
    return summary, body


def write_csv(path, header, rows):
    """Write one header row and the data rows with the csv module."""
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def plot_participant_differences(participant_rows):
    """docs/assets/participant_differences.png: final minus baseline accuracy for each participant."""
    fig, ax = plt.subplots(figsize=(8.6, 4.9), layout="constrained")
    differences = [r[3] for r in participant_rows]
    ax.barh(
        [f"P{r[0]}" for r in participant_rows],
        differences,
        color=["#2e75b6" if v >= 0 else "#b5591a" for v in differences],
    )
    for i, value in enumerate(differences):
        ax.text(
            value + (0.15 if value >= 0 else -0.15),
            i,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
        )
    ax.axvline(0, color="#333333", lw=0.8)
    ax.axvline(-2, color="#555555", ls="--", lw=1, label="2-point loss threshold")
    ax.set(
        xlim=(-6, 9),
        xlabel="Final INT8 minus full-22 FP32 accuracy (percentage points)",
        title="Final configuration: participant-level accuracy differences",
    )
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(ASSETS / "participant_differences.png", dpi=170)
    plt.close(fig)


def plot_class_recall(class_rows):
    """docs/assets/class_recall.png: mean participant recall per class, baseline against final."""
    fig, ax = plt.subplots(figsize=(8.6, 4.3), layout="constrained")
    x = list(range(4))
    ax.bar([v - 0.18 for v in x], [r[1] for r in class_rows], width=0.36, color="#555555", label="Full-22 FP32")
    ax.bar(
        [v + 0.18 for v in x],
        [r[2] for r in class_rows],
        width=0.36,
        color="#2e75b6",
        hatch="//",
        label="Frozen final INT8",
    )
    ax.set(
        xticks=x,
        xticklabels=[r[0] for r in class_rows],
        ylim=(0, 100),
        ylabel="Mean participant recall (%)",
        title="Class recall across nine participants",
    )
    ax.legend(frameon=False)
    fig.savefig(ASSETS / "class_recall.png", dpi=170)
    plt.close(fig)


def memory_section():
    """Section 4.1 from the isolated-process memory runs, or an empty string when there are none."""
    memory_rows = []
    for name, label in [("memory_full22", "Full-22"), ("memory_final_candidate", "16 electrodes, 50% pruned")]:
        path = RESULTS / "supplementary/matched_budget" / (name + ".json")
        if path.exists():
            result = json.loads(path.read_text())
            for precision, row in result["rows"].items():
                memory_rows.append(
                    [
                        label,
                        precision,
                        *[
                            f"{row[key] / 1048576:.2f}"
                            for key in [
                                "baseline_rss_bytes",
                                "after_session_rss_bytes",
                                "after_inference_rss_bytes",
                                "process_lifetime_peak_rss_bytes",
                            ]
                        ],
                    ]
                )
    return (
        (
            "### 4.1 Additional isolated-process measurements on the M5 reproduction\n\n"
            + table(
                [
                    "Model",
                    "Precision",
                    "Baseline RSS (MiB)",
                    "After session (MiB)",
                    "After inference (MiB)",
                    "Lifetime peak (MiB)",
                ],
                memory_rows,
            )
            + "\n\nEach row is one fresh process without PyTorch, using 100 warm-up and 1,000 inference calls. These "
            "are M5 measurements of independently reproduced models, not replacements for the historical M4 timing. "
            "Process RSS includes interpreter, runtime, graph setup, input arrays and allocator overhead; its lifetime "
            "high-water mark is not incremental tensor memory or MCU SRAM. No uncertainty interval or pure "
            "activation-memory compression ratio is inferred from these process measurements."
        )
        if memory_rows
        else ""
    )


def matched_budget_section():
    """Section 5 text and its executive-summary sentence; a pending note before the experiment has run."""
    supplement_path = RESULTS / "supplementary/matched_budget/analysis.json"
    matched_summary = ""
    if supplement_path.exists():
        supplement = json.loads(supplement_path.read_text())
        contrasts = [r for r in supplement["comparisons"] if r["a"] in ["25", "50"] and r["b"] == "0"]
        fig, ax = plt.subplots(figsize=(8.6, 4.9), layout="constrained")
        for i, row in enumerate(contrasts):
            color = "#2e75b6" if row["precision"] == "int8" else "#555555"
            ax.hlines(i, row["ci_low_pp"], row["ci_high_pp"], color=color, lw=1.4)
            ax.scatter(row["difference_pp"], i, marker="s" if row["a"] == "50" else "o", s=38, color=color, zorder=3)
        ax.axvline(0, color="#333333", ls="--", lw=0.8)
        ax.set_yticks(
            range(len(contrasts)), [f"{r['base']} | {r['precision'].upper()} | prune {r['a']}%" for r in contrasts]
        )
        ax.invert_yaxis()
        ax.set(
            title="Pruning versus an equally fine-tuned unpruned model",
            xlabel="Paired accuracy difference (percentage points)\n95% participant bootstrap interval; n = 9",
        )
        fig.savefig(ASSETS / "matched_budget_effects.png", dpi=170)
        plt.close(fig)
        origin = (
            "independent reproduction"
            if supplement["base_origin"] == "independent_reproduction"
            else "matched branching from original bundles"
        )
        supplement_text = (
            f"The additional experiment has been executed as **{origin}**, after the original test results were known. "
            "Both historical pruning bases and all nine participants/three seeds were retained; no finalist was "
            "reselected. Every base was branched into 0%, 25% and 50% pruning, all with the same 100-epoch, 1e-4 Adam "
            "fine-tuning budget.\n\n"
            + table(
                ["Base", "Precision", "Branch A", "Branch B", "A minus B (pp)", "95% paired interval (pp)"],
                [
                    [
                        r["base"],
                        r["precision"],
                        r["a"],
                        r["b"],
                        f"{r['difference_pp']:+.2f}",
                        f"[{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}]",
                    ]
                    for r in supplement["comparisons"]
                ],
            )
            + "\n\n"
            "Here `base` is the 475-epoch model with the recorded origin; `0` is unpruned plus 100 more epochs; `25` "
            "and `50` are equally fine-tuned pruned branches. The `0 - base` contrast estimates the extra-training "
            "effect. The `25/50 - 0` contrasts compare structural pruning at equal fine-tuning budgets. Intervals are "
            "exploratory participant-level bootstrap intervals. See the [supplementary "
            "protocol](../results/supplementary/matched_budget/protocol.json) and [all paired "
            "comparisons](../results/supplementary/matched_budget/paired_comparisons.csv)."
        )
        supplement_text += (
            "\n\n![Matched-budget pruning contrasts with participant intervals](assets/matched_budget_effects.png)"
        )
        selected_contrast = next(
            r
            for r in supplement["comparisons"]
            if r["base"] == "relevance-16" and r["precision"] == "int8" and r["a"] == "50" and r["b"] == "0"
        )
        matched_summary = (
            "The added **M5 matched-budget reproduction** found that 50% pruning of the relevance-16 INT8 model "
            f"reduced mean accuracy by **{abs(selected_contrast['difference_pp']):.2f} percentage points** against an "
            "unpruned model given the same 100 extra epochs (95% exploratory interval "
            f"**[{selected_contrast['ci_low_pp']:+.2f}, {selected_contrast['ci_high_pp']:+.2f}]**). Thus, similar "
            "performance to the original shorter-trained baseline does not imply pruning is accuracy-free. This "
            "supplementary analysis was added after test outcomes were known; it preserves the historical finalist."
        )
        supplement_text += (
            "\n\nFor relevance-16 INT8, the 50%-pruned branch differed from the equally fine-tuned unpruned branch by "
            f"{selected_contrast['difference_pp']:+.2f} points, with a 95% interval "
            f"[{selected_contrast['ci_low_pp']:+.2f}, {selected_contrast['ci_high_pp']:+.2f}]. This isolates the "
            "structural comparison within the recorded fine-tuning procedure more closely than the original baseline "
            "contrast. It remains a post-test, nine-participant reproduction on different hardware and does not "
            "replace the original study."
        )
        agreements = []
        for path in (RESULTS / "supplementary/matched_budget/test").glob("*.json"):
            record = json.loads(path.read_text())
            agreements.extend((record.get("historical_prediction_agreement") or {}).values())
        if agreements:
            supplement_text += (
                f"\n\nAcross {len(agreements)} model/precision pairs comparable with a historical condition, mean "
                f"full-trial prediction agreement was {100 * statistics.mean(agreements):.2f}% (range "
                f"{100 * min(agreements):.2f}–{100 * max(agreements):.2f}%). Agreement is a reproduction diagnostic, "
                "not proof of identical weights; mismatching runs remain included."
            )
    else:
        supplement_text = (
            "A supplementary matched-budget experiment is implemented in `scripts/run_matched_control.py`. Its "
            "analysis is **pending**, and no additional empirical outcome is claimed here. It branches each of the two "
            "historical bases into unpruned, 25%-pruned and 50%-pruned models, all given 100 epochs at 1e-4 with the "
            "same seed and batch schedule. Missing original weights require explicitly labeled independent base "
            "reproduction. This analysis was added after the original test outcomes were known; it cannot be "
            "represented as preregistered or used to reselect the historical finalist."
        )
    return matched_summary, supplement_text


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((RESULTS / "frozen_config.json").read_text())
    chosen = frozen["final_configuration"]
    pp = read_csv("analysis/per_participant_primary.csv")
    summary = read_csv("analysis/conditions_primary.csv")
    efficiencies = read_csv("analysis/efficiency.csv")
    comparisons = read_csv("analysis/comparisons_primary.csv")
    conditions = {(r["condition"], r["precision"]): r for r in summary}
    costs = {(r["condition"], r["precision"]): r for r in efficiencies}
    selected_key = (chosen["condition"], chosen["precision"])
    baseline_key = ("full-22", "fp32")
    b, f = conditions[baseline_key], conditions[selected_key]
    bc, fc = costs[baseline_key], costs[selected_key]
    participant = {(r["condition"], r["precision"], int(r["subject"])): r for r in pp}
    participant_rows = []
    stage_rows = []
    for subject in range(1, 10):
        before = participant[(*baseline_key, subject)]
        after = participant[(*selected_key, subject)]
        participant_rows.append(
            [
                subject,
                float(before["acc"]) * 100,
                float(after["acc"]) * 100,
                (float(after["acc"]) - float(before["acc"])) * 100,
            ]
        )
        stages = [
            float(participant[(c, p, subject)]["acc"])
            for c, p in [baseline_key, ("relevance-16", "fp32"), ("relevance-16-prune50", "fp32"), selected_key]
        ]
        stage_rows.append([subject, *[100 * (stages[i + 1] - stages[i]) for i in range(3)]])
    loss_count = sum(r[3] < -2 for r in participant_rows)
    if loss_count != int(f["n_losing_more_than_2pts"]):
        raise RuntimeError("Participant loss count disagrees with summary")
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"}
    )
    plot_participant_differences(participant_rows)
    class_columns = ["recall_left_hand", "recall_right_hand", "recall_feet", "recall_tongue"]
    class_rows = []
    for column in class_columns:
        a = statistics.mean(float(participant[(*baseline_key, s)][column]) for s in range(1, 10)) * 100
        z = statistics.mean(float(participant[(*selected_key, s)][column]) for s in range(1, 10)) * 100
        class_rows.append([column.removeprefix("recall_").replace("_", " "), a, z, z - a])
    plot_class_recall(class_rows)
    write_csv(
        RESULTS / "maintenance/participant_differences.csv",
        ["subject", "baseline_accuracy_pct", "final_accuracy_pct", "difference_pp"],
        participant_rows,
    )
    write_csv(
        RESULTS / "maintenance/class_recall.csv",
        ["class", "baseline_recall_pct", "final_recall_pct", "difference_pp"],
        class_rows,
    )
    parameter_ratio = float(bc["parameter_bytes"]) / float(fc["parameter_bytes"])
    file_reduction = 100 * (1 - float(fc["file_bytes"]) / float(bc["file_bytes"]))
    modeled_ratio = float(bc["peak_activation_bytes"]) / float(fc["peak_activation_bytes"])
    shown = [
        baseline_key,
        ("full-22", "int8"),
        ("proposed-16", "fp32"),
        ("relevance-16", "fp32"),
        ("weightnorm-16", "fp32"),
        ("proposed-8", "fp32"),
        selected_key,
    ]
    result_table = table(
        [
            "Condition",
            "Precision",
            "Accuracy (%)",
            "Macro-F1 (%)",
            "Difference vs baseline (pp)",
            "95% paired interval (pp)",
        ],
        [
            [
                c,
                p,
                f"{100 * float(conditions[(c, p)]['acc_mean']):.2f}",
                f"{100 * float(conditions[(c, p)]['f1_mean']):.2f}",
                f"{100 * float(conditions[(c, p)]['diff_vs_full_fp32']):+.2f}",
                f"[{100 * float(conditions[(c, p)]['diff_ci_low']):+.2f}, "
                f"{100 * float(conditions[(c, p)]['diff_ci_high']):+.2f}]",
            ]
            for c, p in shown
        ],
    )
    selection_table = table(
        ["Electrodes", "Precision", "Control", "Proposed minus control (pp)", "95% paired interval (pp)"],
        [
            [
                r["electrodes"],
                r["precision"],
                r["family"].removeprefix("proposed vs "),
                f"{100 * float(r['mean_diff']):+.2f}",
                f"[{100 * float(r['ci_low']):+.2f}, {100 * float(r['ci_high']):+.2f}]",
            ]
            for r in comparisons
            if r["family"].startswith("proposed vs")
        ],
    )
    size_table = table(
        ["Quantity", "Full-22 FP32", "Frozen final INT8", "Interpretation"],
        [
            [
                "Initializer/parameter bytes",
                bc["parameter_bytes"],
                fc["parameter_bytes"],
                f"{parameter_ratio:.2f}x ratio; includes initializers such as scales",
            ],
            ["Serialized ONNX bytes", bc["file_bytes"], fc["file_bytes"], f"{file_reduction:.2f}% file reduction"],
            [
                "Modeled peak activation bytes",
                bc["peak_activation_bytes"],
                fc["peak_activation_bytes"],
                f"{modeled_ratio:.2f}x ratio under uniform-precision tensor assumptions",
            ],
            [
                "Actual ONNX float32 input bytes",
                "44000",
                "32000",
                "Both graph interfaces accept float32; quantization is internal",
            ],
            [
                "Median CPU latency (ms)",
                bc["latency_median_ms"],
                fc["latency_median_ms"],
                "Historical batch-1, one-thread measurement",
            ],
            ["MACs", bc["macs"], fc["macs"], "Analytical convolution/dense multiply-accumulate count"],
        ],
    )
    participant_table = table(
        ["Participant", "Full-22 FP32 (%)", "Frozen final INT8 (%)", "Difference (pp)"],
        [[s, f"{a:.2f}", f"{z:.2f}", f"{d:+.2f}"] for s, a, z, d in participant_rows],
    )
    stage_table = table(
        [
            "Participant",
            "22 to 16 electrodes + retraining (pp)",
            "Pruning + 100 extra epochs (pp)",
            "INT8 conversion (pp)",
        ],
        [[s, f"{a:+.2f}", f"{z:+.2f}", f"{d:+.2f}"] for s, a, z, d in stage_rows],
    )
    class_table = table(
        ["Class", "Baseline recall (%)", "Final recall (%)", "Difference (pp)"],
        [[c, f"{a:.2f}", f"{z:.2f}", f"{d:+.2f}"] for c, a, z, d in class_rows],
    )
    memory_text = memory_section()
    matched_summary, supplement_text = matched_budget_section()
    post_summary, post_body = post_midterm()
    report = f"""# Fewer EEG electrodes preserve average decoding accuracy; exported INT8 is slower on the M4 until \
two layers are swapped

Peiye Li · COMP 690-193 · Fall 2026{LINE_BREAK}
Local report assembled from retained experiment evidence; maintenance date: 2026-11-12.

## Executive summary

The frozen 16-electrode, 50%-pruned INT8 configuration reached **{100 * float(f["acc_mean"]):.2f}%** held-out-session \
accuracy, compared with **{100 * float(b["acc_mean"]):.2f}%** for the full 22-electrode FP32 baseline. The mean paired \
difference was **{100 * float(f["diff_vs_full_fp32"]):+.2f} percentage points**, with a 95% participant bootstrap \
interval of **[{100 * float(f["diff_ci_low"]):+.2f}, {100 * float(f["diff_ci_high"]):+.2f}]**. This is close average \
performance, not proof of equivalence: **{loss_count}/9 participants lost more than two points**.

Redundancy-aware electrode selection did not show a clear advantage over relevance-only or spatial-weight selection. \
It outperformed the mean of three fixed random subsets, but most comparisons against the stronger controls had \
intervals crossing zero. The results support a controlled negative finding about this particular redundancy proxy, \
rather than a new superior selection algorithm.

Initializer bytes fell by approximately {parameter_ratio:.1f}x, but the complete ONNX file shrank by only \
{file_reduction:.1f}%. INT8 inference of the full model was about 28% slower than FP32 on the measured Apple M4 CPU. \
The large activation-memory ratio is an analytical tensor estimate; actual ONNX Runtime memory and MCU SRAM were not \
measured in the historical experiment.

{matched_summary}

{post_summary}

## 1. Research question and relation to prior work

The question is whether EEG electrodes selected for both class relevance and low band-power redundancy preserve \
four-class motor-imagery information across recording days, and whether any advantage survives INT8 quantization. \
Electrode count describes physical measurements; pruning removes learned internal features. Their costs and \
information loss should be evaluated separately.

EEGNet uses depthwise and separable convolutions for compact EEG decoding [1]. Q-EEGNet studies an 8-bit, \
hardware-aware embedded EEGNet implementation [2]. MI-BMInet combines spatial-filter channel selection, quantization \
and embedded evaluation [3]. The present project is a controlled software comparison within one fixed EEGNet pipeline. \
The spatial-weight baseline is inspired by MI-BMInet; it is not a reproduction of that architecture or its deployment \
results.

## 2. Data and experimental protocol

BCI Competition IV 2a / BNCI 2014-001 contains nine participants, two recording sessions on different days, 22 EEG \
channels at 250 Hz and four classes: left hand, right hand, feet and tongue. Each session has 288 trials in six runs. \
EOG is excluded. The primary analysis removes expert-flagged artifacts, while the secondary analysis evaluates the \
same models on all trials. Counts and source-file SHA-256 values are retained in \
[data_summary.csv](../results/data_summary.csv).

Each run is processed with a causal fourth-order Butterworth 4–38 Hz band-pass, resetting state at run boundaries. \
Each trial supplies a two-second window from 0.5 to 2.5 seconds after the cue. Normalization uses training-partition \
channel means and standard deviations only.

Session 1 supplies training and three validation folds, each holding out two complete runs. Channel rankings, \
normalization and PTQ calibration are fitted only to the applicable training partition. Session 2 is the final test \
day. The source code requires an explicit final-evaluation flag to load it. The retained [frozen \
configuration](../results/frozen_config.json) fixes the training budget at 475 epochs and seeds at 0, 1 and 2. The \
stored validation-table hash can be checked; this consistency check is not independent proof of the original chronology.

The decoder is Braindecode 2.0.0 EEGNet with F1=8, D=2, F2=16, temporal kernel 64, dropout 0.5 and max-norm \
constraints. Adam uses learning rate 1e-3, batch size 64 and the same fixed epoch budget for the electrode-selection \
conditions. The 22-channel model has 2,420 trainable parameters.

For selection, Welch log power is calculated in 8–13 Hz and 13–30 Hz bands. Relevance is the average multiclass Fisher \
score, scaled to 0–1 across electrodes. Redundancy is the mean absolute Spearman correlation across bands. Greedy \
selection maximizes relevance minus 0.5 times mean redundancy to the already chosen electrodes. Controls use relevance \
alone, spatial-weight L2 norms and three fixed random subsets. Models are retrained with 16 or 8 electrodes. These \
feature correlations do not identify shared neural sources.

Static INT8 PTQ uses ONNX QDQ format, signed activations, per-channel weights and training-only min-max calibration. \
The original bounded pruning comparison drops 25% or 50% of final pointwise filters by L1 norm, slices BatchNorm and \
classifier tensors, and adds 100 fine-tuning epochs at 1e-4. The original unpruned baseline lacks those extra epochs, \
so pruning-versus-baseline accuracy differences confound removal and additional optimization.

The finalist was chosen using validation: fewest electrodes within two points of full-channel FP32 accuracy, then \
fewest parameter bytes, then highest accuracy. Test accuracy never changes this historical choice. Accuracy, macro-F1, \
per-class recall and confusion matrices are retained. Seeds are averaged within participant; participants, not seeds \
or trials, are the nine independent resampling units. Paired intervals use 10,000 percentile bootstrap resamples and \
seed 690. Intervals are exploratory and are not adjusted for all comparisons.

## 3. Held-out-session decoding results

{result_table}

The full tables include all 17 conditions in both precisions: [primary](../results/analysis/conditions_primary.csv) \
and [secondary](../results/analysis/conditions_secondary.csv). These are within-participant transfers to another \
session, not tests on previously unseen people. Differences below are percentage points, not relative percentage \
changes.

### 3.1 Redundancy-aware selection against its controls

{selection_table}

The redundancy term often leaves the selected set unchanged: in the final seed-0 rankings, proposed and relevance-only \
subsets are identical for 6/9 participants at 16 electrodes and 7/9 at 8. The small observed advantages against \
relevance-only cannot establish a reliable improvement, while spatial-weight ranking is competitive. Beating three \
random subsets supports task-aware selection in this setting; it does not establish superiority to stronger \
channel-selection algorithms or robustness over all possible random layouts.

![Paired selection comparisons with participant bootstrap intervals](../results/analysis/slides/fig_forest.png)

### 3.2 Participant variation changes the interpretation of the mean

{participant_table}

![Final configuration minus baseline by participant](assets/participant_differences.png)

The losses are concentrated in participants {", ".join("P" + str(r[0]) for r in participant_rows if r[3] < -2)}. A \
similar group mean can coexist with useful gains for some people and material losses for others. The exploratory \
interval does not establish a two-point noninferiority margin. Electrode lists are fitted separately per person, so \
this experiment does not validate a universal headset layout.

The observed path to the final model can also be decomposed arithmetically:

{stage_table}

These columns sum to each participant's final difference up to displayed rounding. They are an accounting of observed \
conditions, not a causal decomposition: electrode models were retrained, and the original pruning step also adds \
optimization. The matched-budget experiment below addresses the latter confounding explicitly.

### 3.3 Class-specific outcomes

{class_table}

![Mean participant recall by class](assets/class_recall.png)

Class recall is averaged within participant rather than pooled across all trials. These descriptive differences help \
locate errors; they do not establish a causal neural mechanism or a statistically reliable class-specific effect. The \
retained [confusion matrices](../results/analysis/fig_confusion.png) show the complementary pooled error pattern.

## 4. Efficiency: storage, modeled tensors and measured time

{size_table}

The historical measurement used the author's Apple M4 **Mac mini**, ONNX Runtime 1.31.1, batch size one and one \
inference thread, with 100 warm-up calls and 1,000 timed calls. The original hardware record does not establish the \
RAM capacity; the machine type is corrected from the author's clarification. Preprocessing is timed separately. These \
are local CPU measurements, not end-to-end remote-serving latency.

The finalist is about 8% faster than full-22 FP32, but about 25% slower than the unpruned relevance-16 FP32 model \
(0.257 versus 0.206 ms). Its small speed advantage over the full-channel baseline comes from a smaller input, not \
evidence that INT8 accelerated this decoder. The validation decision rule prioritizes electrode count and initializer \
bytes under an accuracy tolerance; it does not select the lowest-latency configuration.

Initializer bytes include weights, biases and quantization metadata. The serialized file also contains the graph and \
other metadata, explaining why a nearly fourfold initializer reduction produces only a small complete-file reduction.

Activation bytes are estimated under a layer-input-plus-output schedule with uniform four-byte FP32 or one-byte INT8 \
tensors. The real QDQ graph includes float operations and conversion buffers; both input interfaces receive float32 \
arrays. The historical `input_bytes` field for INT8 describes the modeled quantized tensor, not the actual interface \
buffer. New benchmark code records those separately. The estimate excludes live mixed-precision tensors, allocator \
behavior, runtime overhead and kernel workspaces. It is neither a measured runtime peak nor verified MCU SRAM usage.

The isolated-process tool `scripts/benchmark_memory.py` can measure process RSS separately. Its lifetime peak includes \
Python, ORT, graph initialization and input arrays; even that measurement must not be labeled tensor memory or MCU SRAM.

The profiler attributes the INT8 slowdown mainly to the grouped spatial convolution: about 185 microseconds versus \
12.8 microseconds in FP32. This observation identifies an expensive quantized operator on this runtime; it does not \
prove an integer kernel is missing. The optimized graph in fact contains QLinearConv operators. Reduced electrode \
count lowers the expensive early temporal-convolution work, while pruning only the late pointwise filters changes MACs \
very little. This explains why parameter count alone predicts latency poorly here. ONNX Runtime documents the \
dependence of quantization performance on hardware, model structure and conversion overhead [4]. Section 6.4 follows \
this observation up after the midterm review: it traces the cost to the runtime's generic path for grouped \
convolutions and measures variants that avoid it. It also shows that the link between electrode count and computation \
described here holds only for the exported layer order.

![Historical per-layer CPU profile](../results/analysis/slides/fig_int8_layers.png)

The decoder needs a two-second observation window and makes its primary decision 2.5 seconds after the cue plus \
processing time. Sub-millisecond compute does not imply sub-millisecond intention-to-action latency.

{memory_text}

## 5. Supplementary matched-budget pruning comparison

{supplement_text}

## 6. Studies added after the midterm review

{post_body}

## 7. Demonstration and claim boundaries

The [recorded replay](../results/demo/replay_participant3.mp4) streams prerecorded EEG at 250 Hz through causal \
preprocessing and displays predictions, cursor movement and compute time. The recorded participant cannot respond to \
the feedback. This validates a replay implementation, not online closed-loop control, asynchronous intent detection, \
power consumption or clinical utility.

Nine participants and one dataset limit uncertainty and generalization. Removing artifacts is an expert-assisted \
offline primary analysis; the all-trials secondary analysis is retained to expose that dependency. There is no MCU \
latency, SRAM or energy measurement, and no claim about implants or new users. A negative result for one fixed \
band-power redundancy penalty does not rule out other redundancy measures, although Section 6.3 shows that one obvious \
alternative would change the selected electrodes very little. The original pruning accuracy comparison needs the \
matched-budget control above.

## 8. Reproducibility and delivery

The repository retains fixed splits, channel lists, seeds, source hashes, configuration, predictions and measurement \
scripts. See [reproduction instructions](REPRODUCIBILITY.md), [work log](WORK_LOG.md), [supplementary \
protocol](SUPPLEMENTARY_PROTOCOL.md) and the [machine-readable audit](../results/maintenance/audit.json). The local \
[slide deck](../presentation/index.html) replaces private blob references with packaged project assets.

Original missing weights must be restored or independently reproduced in separate paths. New artifact checks refuse to \
silently retrain over historical records when model files are missing. Matching predictions from a reproduction do not \
prove identical original weights. Cross-backend bitwise equality is not promised; preserve the recorded software and \
per-participant device allocation [5].

## References and evidence

1. Lawhern et al. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. J. Neural \
Eng. 15, 056013 (2018). [Author paper](https://arxiv.org/abs/1611.08024), DOI 10.1088/1741-2552/aace8c.
2. Schneider et al. Q-EEGNet: an Energy-Efficient 8-bit Quantized Parallel EEGNet Implementation for Edge \
Motor-Imagery Brain–Machine Interfaces (2020). [Author paper](https://arxiv.org/abs/2004.11690).
3. Wang et al. MI-BMInet: An Efficient Convolutional Neural Network for Motor Imagery Brain–Machine Interfaces with \
EEG Channel Selection. [Author paper](https://arxiv.org/abs/2203.14592).
4. ONNX Runtime. [Quantize ONNX \
models](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), including mixed graph \
representation and performance limitations.
5. PyTorch. [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).
6. ONNX Runtime v1.31.1 source, [qlinearconv.cc](https://github.com/microsoft/onnxruntime/blob/v1.31.1/\
onnxruntime/core/providers/cpu/quantization/qlinearconv.cc): the depthwise test requires one input and one output \
channel per group; other grouped convolutions run im2col and a matrix product per group.
7. Alotaiby et al. A review of channel selection algorithms for EEG signal processing. EURASIP J. Adv. Signal Process. \
2015, 66 (2015). DOI 10.1186/s13634-015-0251-9.
8. Abdullah, Faye and Islam. EEG Channel Selection Techniques in Motor Imagery Applications: A Review and New \
Perspectives. Bioengineering 9(12), 726 (2022). DOI 10.3390/bioengineering9120726.

Numerical claims are derived from the linked local CSV/JSON evidence and the reproducible generator \
`scripts/build_report.py`. The audit verifies internal consistency, not independent provenance of the original \
training run. The full original result tables and figures remain available under `results/analysis/`; no historical \
result was replaced by this report.
"""
    (DOCS / "REPORT.md").write_text(report)
    print("Wrote docs/REPORT.md and source-backed charts.")


if __name__ == "__main__":
    main()
