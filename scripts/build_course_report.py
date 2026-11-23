"""Build the COMP 690 course report (docs/COURSE_REPORT.md and its figures) from the project's records.

The course asks for a 10-12 page PDF with fixed sections (Introduction, Background, Related work,
Approach, Experimental setup, Results, Conclusion; then References, Artifacts, Appendices and an Ethics
section of at most one page). Every number here is read from results/ (records and the tables the
analysis scripts wrote from them); nothing is typed in by hand. Studies that have not finished show
as pending. Figures go to docs/course_report/ at the size they are printed, so their text matches the
12-point body. The PDF is made by scripts/export_course_report_pdf.py.

Reads PROJECT_RESULTS_DIR only (results/ of this checkout, whatever MIBCI_RESULTS_DIR says).
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mibci.paths import DOCS_DIR, PROJECT_RESULTS_DIR

RES = PROJECT_RESULTS_DIR
SUPP = RES / "supplementary"
OUT = DOCS_DIR / "COURSE_REPORT.md"
ASSETS = DOCS_DIR / "course_report"
TITLE = (
    "EEG Channel Selection and Model Compression for Motor-Imagery BCI: "
    "Why 16 Electrodes Suffice and What Makes 8-Bit Inference Fast"
)  # the talk's cover title, then the two main findings
AUTHOR = "Peiye Li"
EMAIL = "**[email address: to be filled in by the author]**"
NAVY, BLUE, LIGHT, RED, GREY = "#13294B", "#4B9CD3", "#9ec5e8", "#C8102E", "#6b7280"
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "savefig.dpi": 220,
        "font.family": "DejaVu Sans",
    }
)
WIDTH_IN = 6.5  # the text width at 1-inch margins on US letter
FACT_NAMES = (
    "ms",
    "of",
    "cur",
    "co",
    "me",
    "pe",
    "rep",
    "sel8",
    "reg",
    "qu",
    "q8",
    "q16",
    "q48",
    "q44",
    "mixed",
    "port",
    "nonvnni",
    "vnni",
    "rr",
    "k22",
    "kf",
    "hw",
    "mc",
    "m4_sw",
    "m4_or",
    "kf_fir",
    "figs",
    "kinds",
    "longer",
    "scalefree",
    "pre",
    "chd",
    "align",
    "fresh",
    "surr",
)


# ---------------------------------------------------------------- reading records


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def analysis(study: str):
    return read(SUPP / study / "analysis.json")


def csv(path) -> pd.DataFrame:
    return pd.read_csv(path)


def word(n: int) -> str:
    """Small counts in words, as English prose writes them."""
    return ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")[n] if n < 10 else str(n)


def s(v: float, digits: int = 2) -> str:
    """A signed number with an ASCII minus."""
    return f"{v:+.{digits}f}"


def interval(lo: float, hi: float, digits: int = 2) -> str:
    return f"[{s(lo, digits)}, {s(hi, digits)}]"


def est(d: dict, digits: int = 2, mean: str = "mean", ci: str = "ci") -> str:
    return f"{s(d[mean], digits)} {interval(*d[ci], digits)}"


def verdict(holds) -> str:
    return "holds" if holds else "does not hold"


PENDING = "*(this study is still running; its numbers are added when it finishes)*"


# ---------------------------------------------------------------- facts


def main_study() -> dict:
    cond = csv(RES / "analysis" / "conditions_primary.csv")
    eff = csv(RES / "analysis" / "efficiency.csv")

    def row(df, condition, precision):
        return df[(df.condition == condition) & (df.precision == precision)].iloc[0]

    full, frozen = row(cond, "full-22", "fp32"), row(cond, "relevance-16-prune50", "int8")
    e_full, e_frozen = row(eff, "full-22", "fp32"), row(eff, "relevance-16-prune50", "int8")
    e_full_int8 = row(eff, "full-22", "int8")
    ec = analysis("electrode_count")
    ten = {(c["a"], c["b"], c["precision"]): c for c in ec["comparisons"] if c["seeds"] == "0-9"}
    mb = analysis("matched_budget")
    prune = {(c["base"], c["precision"]): c for c in mb["comparisons"] if c["a"] == "50" and c["b"] == "0"}
    return {
        "full_acc": 100 * full.acc_mean,
        "frozen_acc": 100 * frozen.acc_mean,
        "frozen_diff": 100 * frozen.diff_vs_full_fp32,
        "frozen_ci": (100 * frozen.diff_ci_low, 100 * frozen.diff_ci_high),
        "frozen_losing": int(frozen.n_losing_more_than_2pts),
        "full_params": int(e_full.params),
        "full_macs": int(e_full.macs),
        "full_param_bytes": int(e_full.parameter_bytes),
        "frozen_param_bytes": int(e_frozen.parameter_bytes),
        "full_ms": 1000 * e_full.latency_median_ms,
        "full_int8_ms": 1000 * e_full_int8.latency_median_ms,
        "frozen_ms": 1000 * e_frozen.latency_median_ms,
        "redundancy16": ten[("proposed-16", "relevance-16", "fp32")],
        "relevance8": ten[("relevance-8", "full-22", "fp32")],
        "r16_vs_full": ten[("relevance-16", "full-22", "fp32")],
        "prune16": prune[("relevance-16", "fp32")],
        "prune22": prune[("full-22", "fp32")],
    }


def overfit() -> dict:
    a = analysis("overfit_study")
    c = {(x["dataset"], x["contrast"]): x for x in a["contrasts"]}

    def get(dataset, name):
        x = c[(dataset, name)]
        return {"mean": x["mean"], "ci": [x["ci_low"], x["ci_high"]], "better": x.get("better"), "n": x.get("n")}

    out = {}
    for d in ("bnci2a", "lee2019"):
        out[d] = {
            "reduced_minus_full": get(d, "reduced minus full, next day, all of the training day"),
            "noise_cost": get(d, "P3: reduced plus noise minus reduced"),
            "information": get(d, "information in the dropped electrodes: full minus reduced plus noise"),
            "shuffled_cost": get(d, "reduced plus shuffled minus reduced"),
            "dropout": get(d, "full with channel dropout minus full"),
        }
    out["P3"] = a["tests_lee2019"]["P3"]
    return out


def curves() -> pd.DataFrame:
    return csv(SUPP / "electrode_data_study" / "curves.csv")


def costs() -> dict:
    a = analysis("input_cost_study")
    slopes = {(x["dataset"], x["level"]): x for x in a["slopes"]}
    return {
        "slope_2a": slopes[("bnci2a", "final")],
        "slope_lee": slopes[("lee2019", "final")],
        "slopes": a["slopes"],
        "C1": a["tests_lee2019"]["C1"],
        "C2": a["tests_lee2019"]["C2"],
        "table": csv(SUPP / "input_cost_study" / "costs.csv"),
    }


def mechanism() -> dict:
    a = analysis("noise_mechanism_study")
    shares = {(x["dataset"], x["added"]): x for x in a["shares"]}
    costs = [x["cost"] for x in a["draws"]]  # six distinct draws; two of them also measured in the dropped places
    return {
        "share6": shares[("bnci2a", 6)],
        "share16": shares[("bnci2a", 16)],
        "lee46": shares[("lee2019", 46)],
        "recovered6": a["M1"]["k=6"]["share_recovered"],
        "recovered16": a["M1"]["k=16"]["share_recovered"],
        "draws_mean": float(np.mean(costs)),
        "draws": len(costs),
        "draw_sd": a["B1"]["draw_sd"],
    }


def penalty() -> dict:
    a = analysis("input_penalty_study")
    n = csv(SUPP / "input_penalty_study" / "norms.csv")
    sel = n[(n.dataset == "bnci2a") & (n.condition == "relevance-16-plus-6") & (n["final"].astype(str) == "False")]
    return {
        "G1": a["tests_lee2019"]["G1"],
        "G2": a["tests_lee2019"]["G2"],
        "electrode_norm": (sel.electrode_norm.max(), sel.electrode_norm.min()),
        "noise_norm": (sel.noise_norm.max(), sel.noise_norm.min()),
        "ratio": (
            float((sel.noise_norm / sel.electrode_norm).min()),
            float((sel.noise_norm / sel.electrode_norm).max()),
        ),
    }


def replication() -> dict:
    a = analysis("replication_study")
    h = {x["hypothesis"]: x for x in a["hypotheses"]}  # the first row of a hypothesis tested several ways
    for x in reversed(a["hypotheses"]):
        h[x["hypothesis"]] = x
    rows = {}
    for x in a["hypotheses"]:
        rows.setdefault(x["hypothesis"], []).append(x)
    r = analysis("replication_riemann_study")
    return {
        "h": h,
        "rows": rows,
        "r": {x["hypothesis"]: x for x in r["hypotheses"]},
        "riemann_acc": r["mean_accuracy_pct"],
    }


def selection() -> pd.DataFrame:
    return csv(SUPP / "selection_study" / "accuracy.csv")


def regions() -> dict:
    a = analysis("region_study")
    return {k: v["accuracy"]["mean"] for k, v in a["shapley_pp"]["cv"].items()}


def quant() -> dict:
    a = analysis("quant_study")
    rows = {x["setting"]: x for x in a["ptq_pooled"] if x["condition"] == "all conditions"}
    rule = analysis("quant_rule_study")
    return {
        "rows": rows,
        "agreement": a["simulator_check"]["agreement_with_archived_int8_pct_mean"],
        "rule": rule["settings"],
    }


def portability() -> pd.DataFrame:
    return csv(SUPP / "int8_portability" / "machines.csv")


def latency() -> tuple[pd.DataFrame, pd.DataFrame]:
    return csv(SUPP / "ort_kernel_study" / "latency.csv"), csv(SUPP / "hw_latency" / "latency.csv")


def mcu() -> dict:
    return {
        "library": csv(SUPP / "mcu_study" / "instructions.csv"),
        "study": analysis("mcu_study"),
        "kernel": csv(SUPP / "mcu_kernel_study" / "builds.csv")
        if (SUPP / "mcu_kernel_study" / "builds.csv").exists()
        else None,
        "kernel_checks": (analysis("mcu_kernel_study") or {}).get("checks"),
    }


def compute() -> dict:
    return read(SUPP / "compute_summary.json")


# ---------------------------------------------------------------- figures


def save(fig, name: str) -> str:
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / name, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return f"course_report/{name}"


def fig_pipeline() -> str:
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 1.8))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    boxes = [
        (1, "EEG window\n2 s x 22 or 62\nelectrodes"),
        (17, "electrode\nselection\n(training day)"),
        (33, "EEGNet\ntemporal -> spatial\n-> separable"),
        (49, "pruning and\nINT8 export\n(training day)"),
        (65, "runtime kernels\nlayer swap, custom\nINT8 operators"),
        (81, "target: M4 CPU,\nx86 servers,\nCortex-M0/M4/M55"),
    ]
    for x, text in boxes:
        ax.add_patch(plt.Rectangle((x, 9), 15, 16, fc=LIGHT if x < 60 else "#dbe7f3", ec=NAVY, lw=1))
        ax.text(x + 7.5, 17, text, ha="center", va="center", fontsize=7.6, color=NAVY)
    for x, _ in boxes[:-1]:
        ax.annotate("", xy=(x + 16.6, 17), xytext=(x + 15.2, 17), arrowprops={"arrowstyle": "->", "color": NAVY})
    ax.text(
        25, 4, "chosen on the training day only; the next day is scored once", ha="center", fontsize=7.6, color=GREY
    )
    ax.text(73, 4, "measured: latency, instructions, bytes; checked bit for bit", ha="center", fontsize=7.6, color=GREY)
    return save(fig, "fig1_pipeline.png")


def fig_electrodes(cur: pd.DataFrame, rep: dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.3))
    names = {
        "bnci2a": {"final": "6 runs", "fold": "4 runs", "rfold": "2 runs"},
        "lee2019": {"final": "2 runs", "fold": "1 run"},
    }
    colors = {"final": NAVY, "fold": BLUE, "rfold": LIGHT}
    for ax, dataset in zip(axes, ("bnci2a", "lee2019"), strict=True):
        for level, label in names[dataset].items():
            c = cur[(cur.dataset == dataset) & (cur.level == level)].sort_values("electrodes")
            ax.plot(c.electrodes, c.next_day, "o-", ms=3, color=colors[level], label=f"trained on {label}")
        if dataset == "lee2019":
            for k in (8, 16, 20):
                ax.plot(
                    k,
                    rep["riemann_acc"][f"riemann-{k}"],
                    "D",
                    color=RED,
                    ms=4,
                    label="Riemannian selection" if k == 8 else None,
                )
        ax.set(
            xlabel="electrodes kept",
            ylabel="next-day accuracy (%)",
            title="BCI IV 2a, 9 people, 4 classes" if dataset == "bnci2a" else "Lee 2019, 54 people, 2 classes",
        )
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return save(fig, "fig2_electrodes.png")


def fig_cost(table: pd.DataFrame, kinds, equivalent=None, measured=None) -> str:
    """Left: cost per noise input. Right: cost by kind of input; stars mark the real dropped electrodes' share
    of variance the kept ones cannot predict, at the measured cost of their prediction plus another trial's
    residual (residual_surrogate_study), or before that study at the cost a mix with that share has."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.3))
    ax = axes[0]
    for dataset, color, label in (("bnci2a", NAVY, "BCI IV 2a"), ("lee2019", BLUE, "Lee 2019")):
        c = table[(table.dataset == dataset) & (table.level == "final")].sort_values("added")
        x, y = np.r_[0, c.added], np.r_[0, c.cost]
        err = np.vstack([np.r_[0, c.cost - c.ci_low], np.r_[0, c.ci_high - c.cost]])
        ax.errorbar(x, y, yerr=err, fmt="o-", ms=3, capsize=2, color=color, label=label)
    ax.axhline(0, color="black", lw=0.7)
    ax.set(
        xlabel="noise inputs added to 16 electrodes",
        ylabel="change in next-day accuracy (pp)",
        title="Every useless input costs accuracy",
    )
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax = axes[1]
    if kinds is None:
        ax.text(0.5, 0.5, "input-kind study pending", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        for dataset, color in (("bnci2a", NAVY), ("lee2019", BLUE)):
            t = kinds[(kinds.dataset == dataset) & kinds.a.notna()].sort_values("a")
            ax.errorbar(
                t.a,
                t.next_day_cost,
                yerr=[t.next_day_cost - t.next_day_ci_low, t.next_day_ci_high - t.next_day_cost],
                fmt="o-",
                ms=3,
                capsize=2,
                color=color,
                label="BCI IV 2a, 6 extra" if dataset == "bnci2a" else "Lee 2019, 46 extra",
            )
            sh = kinds[(kinds.dataset == dataset) & (kinds.condition == "shared")]
            if len(sh):
                r = sh.iloc[0]
                x = 1.25 + (-0.03 if dataset == "bnci2a" else 0.03)
                yerr = [[r.next_day_cost - r.next_day_ci_low], [r.next_day_ci_high - r.next_day_cost]]
                ax.errorbar(x, r.next_day_cost, yerr=yerr, fmt="s", ms=4, mfc="white", capsize=2, color=color)
            if measured is not None:
                m = measured[dataset]
                yerr = [[m["mean"] - m["ci"][0]], [m["ci"][1] - m["mean"]]]
                ax.errorbar(equivalent[dataset]["a"], m["mean"], yerr=yerr, fmt="none", capsize=2, color=color)
                ax.plot(equivalent[dataset]["a"], m["mean"], "*", ms=10, color=color, mec="black", mew=0.5, zorder=5)
            elif equivalent is not None:
                e = equivalent[dataset]
                ax.plot(e["a"], e["next_day_cost_at_a"], "*", ms=10, color=color, mec="black", mew=0.5, zorder=5)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0, 1.25], ["copy", ".25", ".5", ".75", "noise", "shared"])
        if equivalent is not None:
            star = "dropped electrodes, measured cost" if measured is not None else "real dropped electrodes"
            ax.plot([], [], "*", ms=10, color="white", mec="black", mew=0.5, label=star)
        ax.set(
            xlabel="share of each extra input that is independent noise",
            ylabel="change (pp)",
            title="What kind of input costs",
            ylim=(-21, 2.5),
        )
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    return save(fig, "fig3_cost.png")


def fig_selection(sel: pd.DataFrame, rep: dict, cur: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.3), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    show = {
        "relevance": ("one at a time (frozen rule)", NAVY, "o-"),
        "riemann": ("Riemannian distance", RED, "D-"),
        "prior": ("fixed order from C3/C4/Cz", GREY, "s--"),
        "csp": ("CSP weights", BLUE, "^:"),
    }
    for method, (label, color, style) in show.items():
        c = sel[sel.method == method].sort_values("k")
        if len(c):
            ax.plot(c.k, c.acc_mean_pct, style, ms=3, color=color, label=label)
    full = sel[sel.method == "full-22"].acc_mean_pct.iloc[0]
    ax.axhline(full, color="black", lw=0.8, ls=":", label="all 22 electrodes")
    ax.set(xlabel="electrodes kept", ylabel="next-day accuracy (%)", title="BCI IV 2a (exploratory)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    ks = (8, 16, 20)
    rel = [rep["riemann_acc"][f"relevance-{k}"] for k in ks]
    rie = [rep["riemann_acc"][f"riemann-{k}"] for k in ks]
    x = np.arange(len(ks))
    ax.bar(x - 0.2, rel, 0.4, color=NAVY, label="frozen rule")
    ax.bar(x + 0.2, rie, 0.4, color=RED, label="Riemannian")
    full62 = cur[(cur.dataset == "lee2019") & (cur.level == "final") & (cur.electrodes == 62)].next_day.iloc[0]
    ax.set_xticks(x, [str(k) for k in ks])
    ax.set_ylim(min(rel) - 4, max(rie) + 2)
    ax.set(xlabel="electrodes kept", ylabel="next-day accuracy (%)", title="Lee 2019 (registered)")
    if full62:
        ax.axhline(full62, color="black", lw=0.8, ls=":")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    return save(fig, "fig4_selection.png")


def fig_latency(kern: pd.DataFrame, hw: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.35), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    order = [
        ("fp32", "FP32"),
        ("int8", "INT8"),
        ("int8-kernel", "INT8 +kernel"),
        ("fp32-spatial-first", "FP32 swap"),
        ("int8-spatial-first", "INT8 swap"),
        ("int8-spatial-first-kernel", "INT8 swap\n+kernel"),
    ]
    t = kern[kern.condition == "full-22"].set_index("variant")
    vals = [t.us[v] for v, _ in order]
    bars = ax.bar(range(len(order)), vals, color=[GREY, RED, NAVY, GREY, RED, NAVY])
    for b, v in zip(bars, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(order)), [label for _, label in order], fontsize=7, rotation=30, ha="right")
    ax.set(ylabel="microseconds per window", title="Apple M4, 22 electrodes, ONNX Runtime", ylim=(0, 1.12 * max(vals)))
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    machines = [
        ("m4-int8-study", "Apple M4"),
        ("c151411", "AMD 9654\n(VNNI)"),
        ("c0309", "Intel 6140"),
        ("intel-no-avx512", "Intel E5 v3"),
    ]
    sub = hw[hw.condition == "relevance-16-prune50"]
    x = np.arange(len(machines))
    series = (
        ("fp32", GREY, "FP32"),
        ("fp32-spatial-first", LIGHT, "FP32 swap"),
        ("int8", RED, "INT8"),
        ("int8-spatial-first", NAVY, "INT8 swap"),
    )
    for i, (variant, color, label) in enumerate(series):
        v = [sub[(sub.machine == m) & (sub.variant == variant)].median_us.iloc[0] for m, _ in machines]
        ax.bar(x + (i - 1.5) * 0.2, v, 0.2, color=color, label=label)
    ax.set_xticks(x, [label for _, label in machines], fontsize=7)
    ax.set(ylabel="microseconds per window", title="frozen model on four CPUs")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return save(fig, "fig5_latency.png")


def fig_mcu(m: dict) -> str:
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.15))
    lib = m["library"]
    cores = [("m0", "Cortex-M0"), ("m4", "Cortex-M4"), ("m55", "Cortex-M55")]
    rows = [
        ("original", "f32", "FP32, original", LIGHT),
        ("original", "int8", "INT8, original", BLUE),
        ("swapped", "f32", "FP32, swapped", "#d1d5db"),
        ("swapped", "int8", "INT8, swapped", NAVY),
    ]
    x = np.arange(len(cores))
    width = 0.15
    for i, (order, precision, label, color) in enumerate(rows):
        r = lib[(lib.condition == "full-22") & (lib.order == order) & (lib.precision == precision)].iloc[0]
        ax.bar(x + (i - 2) * width, [r[c] for c, _ in cores], width, color=color, label=label)
    if m["kernel"] is not None:
        k = m["kernel"][
            (m["kernel"].subject == 1) & (m["kernel"].condition == "full-22") & (m["kernel"].order == "swapped")
        ]
        v = [k[k.core == c].fir_total.iloc[0] / 1e6 for c, _ in cores]
        ax.bar(x + 2 * width, v, width, color=RED, label="INT8, swapped + FIR kernel")
    ax.set_yscale("log")
    ax.set_xticks(x, [label for _, label in cores])
    ax.set(ylabel="million instructions per window", title="22 electrodes on emulated microcontrollers (CMSIS-NN)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()
    return save(fig, "fig6_mcu.png")


# ---------------------------------------------------------------- sections


def front(abstract: str) -> list[str]:
    return [
        f"# {TITLE}",
        "",
        f"**{AUTHOR}** | {EMAIL} | COMP 690-193, Fall 2026 (individual project)",
        "",
        "**Contributions.** **[To be written by the author.]**",
        "",
        "**Overlap with other courses or research.** **[To be written by the author.]**",
        "",
        f"**Abstract.** {abstract}",
    ]


def build() -> str:
    ms, of, cur, co, me, pe = main_study(), overfit(), curves(), costs(), mechanism(), penalty()
    rep, sel, reg, qu, port = replication(), selection(), regions(), quant(), portability()
    kern, hw = latency()
    mc, comp = mcu(), compute()
    kinds = analysis("input_kind_study")
    kinds_table = csv(SUPP / "input_kind_study" / "conditions.csv") if kinds else None
    longer, scalefree, pre = (
        analysis("long_training_study"),
        analysis("scalefree_penalty_study"),
        analysis("pretrain_study"),
    )
    chd, align = analysis("chdrop_study"), analysis("alignment_mechanism_study")
    fresh, surr = analysis("fresh_noise_study"), analysis("residual_surrogate_study")
    measured = (
        {r["dataset"]: r for r in surr["decomposition"] if r["level"] == "final" and r["term"] == "cost"}
        if surr
        else None
    )
    h, r = rep["h"], rep["r"]
    k22 = kern[kern.condition == "full-22"].set_index("variant").us
    kf = kern[kern.condition == "relevance-16-prune50"].set_index("variant").us
    nonvnni = port[port.int8_models_changed > 0]
    vnni = port[(port.int8_models_changed == 0) & port.cpu.str.contains("9654")]
    rr = port[port.rr_acc_change_vs_fp32_pp.notna()]
    q8, q16, q48, q44 = (qu["rows"][k] for k in ("8/8", "16/16", "4/8", "4/4"))
    mixed = next(x for x in qu["rule"] if x["setting"] == "pooled-0.5")
    lib = mc["library"].set_index(["condition", "order", "precision"])
    m4_sw = lib.loc[("full-22", "swapped", "int8"), "m4"]
    m4_or = lib.loc[("full-22", "original", "int8"), "m4"]
    figs = {
        "pipeline": fig_pipeline(),
        "electrodes": fig_electrodes(cur, rep),
        "cost": fig_cost(co["table"], kinds_table, kinds["a_equivalent"] if kinds else None, measured),
        "selection": fig_selection(sel, rep, cur),
        "latency": fig_latency(kern, hw),
        "mcu": fig_mcu(mc),
    }
    sel8 = sel[sel.k == 8].set_index("method")
    kf_fir = None
    if mc["kernel"] is not None:
        kk = mc["kernel"][(mc["kernel"].subject == 1) & (mc["kernel"].condition == "full-22")]
        kf_fir = kk.set_index(["order", "core"])
    ratio_range = (mc["kernel"].total_ratio.min(), mc["kernel"].total_ratio.max()) if mc["kernel"] is not None else None

    abstract = (
        "Wearable brain-computer interfaces must decode motor-imagery EEG on small processors, where every "
        "electrode, bit and multiply costs power. We study a compact decoder (EEGNet) on BCI Competition IV 2a (9 "
        "people, two days) under a protocol frozen before scoring the test day, and retest the main findings on Lee "
        "et al. 2019 (54 people) with predictions registered before scoring. Keeping the "
        "16 most relevant of 22 electrodes does not lower next-day accuracy "
        f"({s(of['bnci2a']['reduced_minus_full']['mean'])} "
        "points); so do 45 and 16 of 62 electrodes on the new people. Every extra input costs "
        "accuracy through the part the other inputs cannot predict, mostly because the network memorizes its "
        "training trials with it; the dropped electrodes add about as much as they cost"
        + (
            f", yet after pretraining on 45 other people all 62 beat a shared 16 by "
            f"{pre['tests_lee2019']['T3']['mean']:.1f} points. "
            if pre
            else ". "
        )
        + "Choosing electrodes jointly (Riemannian distance) beats one-at-a-time relevance by "
        f"{r['R1']['diff_pp']:.1f} points at 8 electrodes on the new people. Eight-bit integers keep accuracy, but "
        "exported INT8 ran slower than float until we reordered two layers and wrote the two missing kernels "
        f"(then {k22['fp32'] / k22['int8-spatial-first-kernel']:.1f}x faster than the original float). On "
        "emulated microcontrollers, our filter-bank kernel cuts instructions "
        + (f"{ratio_range[0]:.1f}-{ratio_range[1]:.1f}x." if ratio_range else "further.")
        + " INT8 accuracy can depend on the CPU."
    )
    L = front(abstract)
    L += [
        "",
        "## 1. Introduction",
        "",
        "A motor-imagery brain-computer interface (BCI) turns imagined movements, such as of the left or the right "
        "hand, into commands. The signal is scalp EEG: microvolt rhythms over the sensorimotor cortex whose power "
        "changes with the imagined movement. A practical, wearable BCI has to decode this on a small processor with "
        "a battery, so the electrodes (each one adds setup time, cost and front-end power), the bits of each number "
        "and the multiply-accumulates per decision are the budget. This project asks how far a compact decoder can "
        "be reduced along those three axes before next-day accuracy suffers, and what actually limits speed on real "
        "hardware.",
        "",
        "Compact networks such as EEGNet [1] made deep decoding of EEG small enough for embedded use, and 8-bit "
        "versions have run on low-power processors [2, 3, 4]. Most evaluations, however, (i) score on data from the "
        "same recording day or tune on the test day, (ii) report parameter counts instead of measured time, (iii) "
        "assume that 8-bit integers make a network faster, and (iv) rest on nine people, the size of the standard "
        "BCI IV 2a benchmark. We address each: every choice is made on the training day only and the next day is "
        "scored once; latency is measured in interleaved rounds and microcontroller cost is counted instruction by "
        "instruction in emulation; integer pipelines are checked bit for bit against reference arithmetic; and the "
        "main findings are tested again on a second two-day dataset of 54 people with the predictions written down "
        "before the new test days were scored.",
        "",
        "The systems principles we use are measurement before belief (a cheaper model is not faster until a timer "
        "says so), making the common case fast (the network's most frequent layer shape gets a dedicated kernel "
        "instead of a general path), co-design of model and kernel (reordering two linear layers changes the cost "
        "by up to 8x without changing a single prediction), matching arithmetic to hardware (the width of the integer "
        "accumulator decides whether INT8 is exact), and scale-out for statistical power (about "
        f"{round(comp['longleaf_models'], -3):,} trainings on a cluster turn single-seed anecdotes into intervals).",
        "",
        "**Contributions.**",
        "",
        "1. A frozen, auditable pipeline on BCI IV 2a (electrode selection, structured pruning, INT8 export) in which "
        "every reported number is regenerated from stored records.",
        "2. Evidence that 16 of 22 electrodes suffice, replicated prospectively on 54 new people, and a mechanism: "
        "an extra input costs what the other inputs cannot predict, mostly because EEGNet memorizes its training "
        "trials with that part; measured for the dropped electrodes, where it cancels what they add, and tested "
        "with copies, mixes, shared noise, noise redrawn every epoch, longer training, two sparsity penalties, and "
        "pretraining on other people, which removes most of the cost.",
        "3. A joint (Riemannian) electrode-selection rule that beats the frozen rule, found on BCI IV 2a and "
        "confirmed prospectively on Lee 2019.",
        "4. A systems account of INT8 on CPUs: why it was slower (a missing grouped-convolution kernel), two custom "
        "ONNX Runtime kernels that are bit-exact on 459 models, a layer reordering, and a cross-CPU accuracy "
        "difference traced to 16-bit accumulation on x86 CPUs without VNNI, with a fix.",
        "5. Microcontroller deployment in emulation (Cortex-M0, M4, M55): CMSIS-NN [5] reproduced bit for bit, and a "
        "dedicated INT8 filter-bank kernel for the temporal layer that the library lacks.",
        "",
        f"**Results in brief.** The frozen configuration (16 electrodes, half of the last layer's filters pruned, "
        f"INT8) scored {ms['frozen_acc']:.2f}% against {ms['full_acc']:.2f}% for all 22 electrodes in float; the mean "
        f"difference is {s(ms['frozen_diff'])} points, but {ms['frozen_losing']} of 9 people lost more than 2 points. "
        f"On 54 new people, 45 and 16 of 62 electrodes were within the registered 2-point margin "
        f"({s(h['H1']['estimate_pp'])} and {s(h['H1b']['estimate_pp'])} points). Eight-bit arithmetic changed accuracy "
        f"by {s(q8['vs_float_diff_pp'])} points; with both kernels and the layer swap the decoder needs "
        f"{k22['int8-spatial-first-kernel']:.0f} us per window on the Apple M4 instead of {k22['fp32']:.0f} us.",
    ]
    L += [
        "",
        "## 2. Background",
        "",
        "**Motor imagery and the data.** Imagining a hand movement lowers the power of the mu (8-12 Hz) and beta "
        "(13-30 Hz) rhythms over the opposite hemisphere's motor cortex; imagined foot movements act near the "
        "midline. BCI Competition IV 2a [6] recorded 9 people on two days with 22 electrodes at 250 Hz: four "
        "classes (left hand, right hand, feet, tongue), 288 trials per day. Lee et al. 2019 [7] recorded 54 people "
        "on two days with 62 electrodes: two classes (left and right hand), 100 offline and 100 online trials per "
        "day. The second day is the test: EEG drifts between sessions (electrode placement, impedance, attention), "
        "so a decoder trained on day 1 and scored on day 2 is the honest measure of what a user would experience.",
        "",
        f"**EEGNet.** EEGNet-8,2 [1] filters each electrode in time with 8 kernels of 64 taps, mixes the electrodes "
        "with 2 spatial filters per temporal filter (a grouped convolution spanning every electrode), and then "
        "applies a separable convolution, pooling and a small dense layer, with batch normalization after each "
        f"convolution stage. With 22 electrodes it has {ms['full_params']:,} parameters and "
        f"{ms['full_macs'] / 1e6:.1f} million multiply-accumulates (MACs) per 2-s window; the temporal filters on "
        "every electrode account for most of them.",
        "",
        "**Integer inference.** INT8 inference stores weights as 8-bit integers with a scale (per output channel), "
        "maps activations to 8 bits with a scale and zero point, accumulates products in 32-bit integers and "
        "rescales the sums with a fixed-point multiplier [8]. ONNX Runtime executes such graphs on CPUs; CMSIS-NN "
        "[5] implements the TensorFlow Lite version of the same arithmetic on Arm Cortex-M microcontrollers. How "
        "fast and how exactly this runs depends on the hardware: Arm dot-product instructions and x86 AVX-512 VNNI "
        "add 8-bit products straight into 32-bit sums, whereas x86 CPUs without VNNI first add pairs of products in "
        "a saturating 16-bit register. On microcontrollers, the Cortex-M0 has no SIMD at all, the M4's SMLAD "
        "performs two 16-bit MACs per instruction, and the M55's Helium vectors perform sixteen 8-bit MACs.",
    ]
    L += [
        "",
        "## 3. Related work",
        "",
        "**Compact and embedded decoders.** EEGNet [1] is the common compact baseline for EEG. Q-EEGNet [2] "
        "quantized it to 8 bits for a parallel ultra-low-power processor, EEG-TCNet [3] ran a temporal convolutional "
        "decoder on an embedded RISC-V MCU, and MI-BMInet [4] combined channel selection with a small network for "
        "edge deployment. Those works report what fits and how fast it runs; ours asks, in addition, why fewer "
        "electrodes do not hurt, and where the runtime, not the hardware, sets the speed.",
        "",
        "**Electrode selection.** Reviews group channel-selection methods into filter, wrapper and embedded "
        "approaches and stress that most are evaluated on few people [9, 10]. Riemannian geometry on spatial "
        "covariance matrices is a strong family of motor-imagery decoders [11]; we use its distance between class "
        "covariances as a selection criterion and compare it with one-at-a-time relevance, CSP weights, wrappers and "
        "a learned selection layer.",
        "",
        "**Compression and alignment.** Pruning and quantization are standard model compression [12, 13]; we prune "
        "whole filters so the dense network shrinks. Euclidean alignment [14] whitens each session's data to reduce "
        "the shift between days without labels; we test it on both datasets and dissect why it fails on one.",
        "",
        "**Runtimes for integer inference.** ONNX Runtime's documentation notes that quantized models are not "
        "always faster [15]; TensorFlow Lite Micro [16] and CMSIS-NN [5] are the usual path on Cortex-M. We locate "
        "the specific operator shapes for which these libraries fall back to general code paths and supply the "
        "missing kernels.",
    ]
    L += approach_and_setup(ms, comp)
    facts = {name: value for name, value in locals().items() if name in FACT_NAMES}
    L += results(facts)
    L += conclusion(ms, of, r, k22, ratio_range)
    L += references()
    L += artifacts(comp)
    L += appendix(facts)
    L += appendix_b(facts)
    L += ethics(comp, analysis("chdrop_study"))
    return "\n".join(L) + "\n"


def approach_and_setup(ms: dict, comp: dict) -> list[str]:
    costs = csv(SUPP / "int8_study" / "costs.csv")
    ratio = costs["original_macs"] / costs["spatial-first_macs"]
    swap = (float(ratio.min()), float(ratio.max()))
    agreement = quant()["agreement"]
    return [
        "",
        "## 4. Approach and system design",
        "",
        "![Figure 1. The decoder pipeline. Every choice on the left is made with the training day only; every "
        "claim on the right is a measurement.](course_report/fig1_pipeline.png)",
        "",
        "**Frozen protocol.** All model choices use session 1 only, with three run-grouped validation folds (two of "
        "the six runs held out each time, so validation sees the drift within a day). The training budget (475 "
        "epochs) was fixed by a rule on the 22-electrode validation curves before any selection result existed; the "
        "configuration to deploy was frozen on 3 November, before session 2 was scored, and session 2 was then "
        "scored once. Code reads session 2 only through a guarded loader, and a lint check fails if any other "
        "file opens it.",
        "",
        "**Electrode selection.** Each electrode's relevance is how well its log band power separates the classes "
        "on the training day; a greedy ranking can subtract a redundancy penalty for electrodes whose power "
        "fluctuates with those already chosen. The frozen rule keeps the 16 most relevant electrodes of each person. "
        "Later studies compare joint criteria: the Riemannian distance between class-mean covariance matrices of the "
        "chosen set, CSP filter weights, a wrapper that drops electrodes one at a time with a fast decoder, a fixed "
        "anatomical order, rankings borrowed from other people, and a learned (concrete) selection layer.",
        "",
        "**Compression.** Structured pruning removes the half of the final pointwise convolution's 16 filters with "
        "the smallest L1 norm and fine-tunes for 100 epochs at a tenth of the learning rate. INT8 export uses ONNX "
        "Runtime static quantization (QDQ format, per-channel weights) calibrated on session 1. The bit-width study "
        "folds batch normalization into the convolutions and simulates any integer grid in float arithmetic; its "
        f"8-bit setting agrees with the real ONNX Runtime INT8 graph on {agreement:.1f}% of next-day trials.",
        "",
        "**Runtime kernels.** Two changes make INT8 fast without changing predictions. First, the temporal and "
        "spatial convolutions are both linear before the first nonlinearity, so their order can be swapped: mixing "
        "electrodes first leaves 16 signals to filter instead of 8 per electrode, "
        f"{swap[0]:.1f}-{swap[1]:.1f}x fewer MACs (8 to 22 electrodes). Second, "
        "where ONNX Runtime had no fast INT8 path, we wrote custom operators in C with NEON intrinsics for the "
        "grouped spatial convolution and for the swapped network's first convolution; they replace only those "
        "nodes in the runtime's own optimized graph and follow its rounding exactly. On microcontrollers we built "
        "the network with CMSIS-NN v8.0.0 and added an INT8 FIR filter-bank kernel for the temporal layer with "
        "plain-C, DSP (SMLAD) and Helium (VMLADAVA) paths, compiled with Arm GNU Toolchain 15.3 and run on QEMU 11 "
        "boards with a plugin that counts instructions per layer.",
        "",
        "**Experiment infrastructure.** Each study after the midterm writes a plan before it runs: the jobs, the "
        "contrasts, the predictions for the new dataset, and the SHA-256 of every source file that decides its "
        "numbers. A study refuses to resume if a pinned file changed, and a lint check rejects edits to pinned "
        f"files. The {comp['totals']['longleaf_models'] if 'totals' in comp else comp['longleaf_models']:,} "
        "trainings on the UNC Longleaf cluster ran as SLURM arrays of these plans; every record was copied back and "
        "every table and figure in this report is rebuilt from the records by one command (Artifacts).",
        "",
        "## 5. Experimental setup",
        "",
        "**Data and preprocessing.** Each run is band-passed causally (4th-order Butterworth, 4-38 Hz) and cut into "
        "one 2-s window per trial, 0.5-2.5 s after the cue. BCI IV 2a trials flagged by the dataset's experts as "
        "artifacts are left out of training and of the primary score; Lee 2019 is downsampled from 1000 to 250 Hz "
        "after the same filter and uses every trial.",
        "",
        "**Training and scoring.** Adam (learning rate 0.001, batch 64), 475 epochs, the paper's dropout of 0.5, "
        "and each electrode standardized with training-day statistics. Each condition is trained with 3-10 seeds per "
        "person. The primary metric is next-day accuracy; held-out runs of the training day are a second, same-day "
        "measure. Seeds are averaged within each person first; the 95% interval is a percentile bootstrap over "
        "people (10,000 resamples), which says what another group of people might show. The seed error, what "
        "retraining the same people would show, is reported separately where it matters.",
        "",
        "**Hardware.** Latency: an Apple M4 Mac mini (ONNX Runtime 1.31.1, one thread, one window at a time, all "
        "versions interleaved in 15 rounds of 500 runs, system load recorded) and Longleaf CPUs: AMD EPYC 9654 "
        "(with AVX-512 VNNI) and, without VNNI, AMD EPYC 75F3 and 7702, Intel Xeon Gold 6140 and Xeon E5-2680 v3. "
        "Microcontrollers: emulated Cortex-M0 (no DSP, no FPU), Cortex-M4F (DSP, FPU) and Cortex-M55 (Helium). "
        "QEMU counts instructions; it does not model cycles or memory wait states.",
        "",
        "**Registered and exploratory results.** BCI IV 2a results after the freeze are exploratory: they were "
        "planned after the test day had been scored. On Lee 2019, each study wrote its predictions and their "
        "decision rule into its plan before any of its models was scored on that dataset's test day; these are "
        "reported as holding or not, with the rule unchanged. Appendix A lists all of them.",
    ]


def results(f: dict) -> list[str]:
    """Section 6; f holds the facts gathered by build (FACT_NAMES)."""
    ms, of, cur, co, me, _, rep, sel8, reg = (f[k] for k in FACT_NAMES[:9])
    qu, q8, q16, q48, q44, mixed, _, nonvnni, rr = (
        f[k] for k in ("qu", "q8", "q16", "q48", "q44", "mixed", "port", "nonvnni", "rr")
    )
    k22, kf, mc, m4_sw, m4_or, kf_fir, figs = (f[k] for k in ("k22", "kf", "mc", "m4_sw", "m4_or", "kf_fir", "figs"))
    kinds, longer, scalefree, pre, chd, align = (f[k] for k in ("kinds", "longer", "scalefree", "pre", "chd", "align"))
    fresh, surr = f["fresh"], f["surr"]
    eight = next(x for x in qu["rule"] if x["setting"] == "8/8")
    kind = kind_facts(kinds, of, rep)
    layers = csv(SUPP / "ort_kernel_study" / "layers.csv")
    costs8 = csv(SUPP / "int8_study" / "costs.csv").set_index("condition")
    pruned = tuple(
        1 - costs8.loc["full-22-prune50", c] / costs8.loc["full-22", c] for c in ("original_macs", "spatial-first_macs")
    )
    l22 = layers[layers.condition == "full-22"].set_index("variant")
    lay = {
        "spatial_slow": l22.spatial["int8"] / l22.spatial["fp32"],
        "temporal_fast": l22.temporal["fp32"] / l22.temporal["int8"],
    }
    paths = SUPP / "int8_paths_study" / "shapes.csv"
    pmap = csv(paths) if paths.exists() else None
    if pmap is not None:
        x86 = pmap[pmap.machine != "m4-mac-mini"]
        m4 = pmap[pmap.machine == "m4-mac-mini"]
        dw = x86[x86.eegnet_layer.fillna("").str.contains("swapped temporal")].int8_over_fp32
        lay |= {
            "shapes": int(m4.name.nunique()),
            "m4_faster": int((m4.int8_over_fp32 < 1).sum()),
            "x86_machines": int(x86.machine.nunique()),
            "x86_faster_max": int(x86.groupby("machine").int8_over_fp32.apply(lambda v: (v < 1).sum()).max()),
            "x86_best": float(x86.int8_over_fp32.min()),
            "x86_dw": (float(dw.min()), float(dw.max())),
        }
    saved = 100 * (1 - mixed["weight_bytes"] / eight["weight_bytes"])
    h, r = rep["h"], rep["r"]
    a = of["bnci2a"]
    c2a = cur[(cur.dataset == "bnci2a") & (cur.level == "final")].set_index("electrodes").next_day
    clee = cur[(cur.dataset == "lee2019") & (cur.level == "final")].set_index("electrodes").next_day
    L = [
        "",
        "## 6. Results",
        "",
        "### 6.1 Fewer electrodes keep accuracy, on 9 and on 54 people",
        "",
        f"Retrained on the same hardware with 10 seeds, the 16 most relevant electrodes scored "
        f"{s(a['reduced_minus_full']['mean'])} points against all 22 {interval(*a['reduced_minus_full']['ci'])} "
        f"(the archived models, partly trained on the M4's GPU, had suggested {s(ms['r16_vs_full']['mean_diff_pp'])}; "
        "a different training device acts like a different seed). On the 54 "
        f"people of Lee 2019, the registered test H1 (45 of 62 electrodes within 2 points of all 62) holds: "
        f"{s(h['H1']['estimate_pp'])} {interval(h['H1']['low_pp'], h['H1']['high_pp'])}; so does H1b with only 16 "
        f"electrodes, {s(h['H1b']['estimate_pp'])} {interval(h['H1b']['low_pp'], h['H1b']['high_pp'])}. The mean hides "
        f"spread: {h['H1b']['lose_more_than_2']} of 54 people lost more than 2 points with 16 electrodes, and about as "
        "many gained. Accuracy rises with the number of electrodes and then flattens (Figure 2): on 2a from "
        f"{c2a[4]:.1f}% with 4 to {c2a[16]:.1f}% with 16 and {c2a[22]:.1f}% with 22; on Lee 2019 from {clee[8]:.1f}% "
        f"with 8 to {clee[20]:.1f}% with 20 and {clee[62]:.1f}% with 62. Halving the training data did not move the "
        "plateau (registered tests D1 and D2 do not hold), so the plateau is not set by the amount of data.",
        "",
        f"![Figure 2. Next-day accuracy against the number of electrodes kept (per-person relevance order), for "
        f"different amounts of training data. Red: Riemannian selection on Lee 2019.]({figs['electrodes']})",
        "",
        "### 6.2 Why 16 are enough: an extra input costs what the others cannot predict",
        "",
        "Every extra input costs accuracy. Independent band-limited noise added to the 16 most relevant electrodes "
        "lowers next-day accuracy steadily with the number of inputs (Figure 3, left): "
        f"{s(co['slope_2a']['slope'])} points per input on 2a and {s(co['slope_lee']['slope'])} on Lee 2019, about 2% "
        "of the accuracy above chance per input on both. The cost does not shrink with three times more training "
        "data (registered C2 does not hold; the effect goes the other way), and it is as large on held-out runs of "
        "the training day as on the next day, so it is a failure to generalize to new trials, not a day-to-day shift.",
        "",
        "What costs is the part of an input that the other inputs cannot predict (Figure 3, right; registered "
        f"{kind['verdicts']}, Appendix A). With 6 extra inputs on 2a and 46 on Lee 2019, accuracy changes by "
        f"{kind['dup']} points with copies of kept electrodes, {kind['shared']} with one noise waveform repeated, "
        f"{kind['mix50']} with copies whose variance is half noise, and {kind['noise']} with independent noise. "
        "The real dropped electrodes are mostly predictable: a linear fit from "
        f"the 16 kept ones leaves only {100 * kind['pr']['bnci2a']['mean']:.0f}% (2a) and "
        f"{100 * kind['pr']['lee2019']['mean']:.0f}% (Lee 2019) of their variance on held-out runs unexplained. "
        + (
            surrogate_sentences(surr)
            if surr
            else f"Read off the mixes' curve by linear interpolation, inputs like that would cost about "
            f"{kind['cost'][0]:.1f} and {kind['cost'][1]:.1f} points and add about {kind['info'][0]:.1f} and "
            f"{kind['info'][1]:.1f}; these numbers are rough. "
        )
        + "Comparing "
        f"the real electrodes with noise in their place ({s(a['information']['mean'])} points; noise there costs "
        f"{s(a['noise_cost']['mean'])}, registered P3 holds) credits them for being predictable and overstates both.",
        "",
        "EEGNet does not learn to ignore these inputs. After training, six noise inputs "
        f"still make up {100 * me['share6']['noise_share_trained']:.0f}% of each spatial filter's output variance, "
        f"against {100 * me['share6']['noise_share_init']:.0f}% at initialization; with 16 noise inputs "
        f"{100 * me['share16']['noise_share_trained']:.0f}% against {100 * me['share16']['noise_share_init']:.0f}%. "
        "Setting their weights to zero after training recovers only "
        f"{100 * me['recovered6']:.0f}-{100 * me['recovered16']:.0f}% of the loss, so the damage is done during "
        "training: the filters learned from the electrodes are worse. A group-lasso penalty on each input's weights "
        "did not help (registered G1, G2 do not hold), and the reason is architectural: a batch norm follows the "
        "spatial filters, so the overall scale of a filter's weights does not matter and the penalty is minimized "
        "by shrinking every weight together (Appendix B).",
        "",
    ]
    L += new_mechanism_results(kinds, longer, scalefree, pre, fresh)
    L += [
        f"![Figure 3. Left: next-day cost of noise inputs added to the 16 most relevant electrodes (95% participant "
        f"intervals). Right: the cost of extra inputs that are copies of kept electrodes, mixes of a copy and noise, "
        f"or one noise waveform repeated; stars: the real dropped electrodes with another trial's unexplained part, "
        f"placed at the share of their variance the kept ones cannot predict.]({figs['cost']})",
        "",
        "### 6.3 How to choose: jointly, not one at a time",
        "",
        f"The frozen rule ranks electrodes one at a time. Its redundancy penalty changed little: with 10 seeds, "
        f"redundancy-aware 16 against relevance-only 16 is {s(ms['redundancy16']['mean_diff_pp'])} "
        f"{interval(ms['redundancy16']['ci_low_pp'], ms['redundancy16']['ci_high_pp'])}, because relevance differences "
        "are several times larger than the penalty. Joint criteria do better with few electrodes (Figure 4). With 8 "
        f"electrodes on 2a, the Riemannian distance criterion reached {sel8.acc_mean_pct['riemann']:.1f}% against "
        f"{sel8.acc_mean_pct['relevance']:.1f}% for the frozen rule (exploratory; it was better at 4, 6, 8 and 10 "
        "electrodes, with 48 comparisons made after the test results were known). Registered on Lee 2019 before "
        f"scoring, it won at 8 electrodes by {s(r['R1']['diff_pp'])} {interval(r['R1']['low_pp'], r['R1']['high_pp'])} "
        f"({r['R1']['better']} of 54 people better), at 16 by {s(r['R2']['diff_pp'])} and at 20 by "
        f"{s(r['R3']['diff_pp'])}; all three hold. With 8 Riemannian electrodes, Lee 2019 accuracy "
        f"({rep['riemann_acc']['riemann-8']:.1f}%) exceeds all 62 electrodes ({clee[62]:.1f}%).",
        "",
        "Training a model on each of the 511 combinations of nine scalp regions (session 1 only for the ranking) "
        "gives exact Shapley values of each region's contribution. The motor and somatosensory strips, the central "
        f"midline and the parietal region contribute 5-6.4 points each (left motor {reg['L-motor']:.1f}, right "
        f"somatosensory {reg['R-somatosensory']:.1f}, central midline {reg['mid-centre']:.1f}, parietal "
        f"{reg['parietal']:.1f}); the frontal regions contribute least (midline front {reg['mid-front']:.1f}). "
        "Split by class, the data recover the textbook layout without being told it: the left hand depends on the "
        "right hemisphere, the right hand on the left, and the feet on the midline. "
        "Choosing whole regions, however, was no better than choosing single electrodes.",
        "",
        "### 6.4 How many bits: eight",
        "",
        f"Simulated over all 459 models, 16-bit weights and activations change accuracy by "
        f"{s(q16['vs_float_diff_pp'])} points and 8-bit by {s(q8['vs_float_diff_pp'])} (no person loses 2 points); "
        f"4-bit weights change it by {s(q48['vs_float_diff_pp'])} and 4 bits everywhere by "
        f"{s(q44['vs_float_diff_pp'])}, near "
        "chance. INT16 therefore buys nothing, and it was slower than FP32 on every CPU we timed. Below 8 bits the "
        "first casualties are the raw input and the temporal layer's output, which still carry the signal's full "
        "amplitude range. A mixed precision map chosen on the training day (8 bits only where a layer is sensitive, "
        f"4 elsewhere) changes accuracy by {s(mixed['vs_float_diff_pp'])} points with {saved:.0f}% "
        "fewer weight bytes than 8 bits; the same map, moved unchanged to Lee 2019, kept its accuracy (registered "
        "H4b holds). These sub-8-bit results are simulated integer grids, not timed on low-bit hardware.",
        "",
        f"![Figure 4. Electrode selection methods. Left: BCI IV 2a, exploratory. Right: Lee 2019, registered "
        f"before scoring; dotted line: all 62 electrodes.]({figs['selection']})",
        "",
        "### 6.5 INT8 on CPUs: missing kernels, layer order, and the accumulator",
        "",
        f"Exported INT8 was slower than FP32 on the M4 ({k22['int8']:.0f} against {k22['fp32']:.0f} us for 22 "
        "electrodes, Figure 5) although every layer has fewer bytes. Profiling on the M4 showed that ONNX "
        "Runtime's INT8 convolution has fast paths only for ordinary and one-filter-per-channel depthwise "
        "convolutions; EEGNet's "
        "spatial layer (8 groups of 2 filters spanning every electrode) falls to a general path that copies each "
        f"group into a buffer and multiplies a 2-column matrix: {lay['spatial_slow']:.1f} times slower than FP32 for "
        f"that layer. The M4 itself executes 8-bit arithmetic well: the temporal layer was "
        f"{lay['temporal_fast']:.2f}x faster in INT8. "
        + (
            f"A map of {lay['shapes']} synthetic convolutions shows these fast paths only on the M4: INT8 beat FP32 "
            f"for {lay['m4_faster']} shapes there and for at most {lay['x86_faster_max']} on "
            f"{word(lay['x86_machines'])} x86 CPUs (Appendix B). "
            if "shapes" in lay
            else ""
        )
        + f"Our custom kernel for "
        f"that layer brings INT8 to {k22['int8-kernel']:.0f} us; swapping the two layers helps every version, and a "
        f"second kernel for the swapped network's first layer gives {k22['int8-spatial-first-kernel']:.0f} us, "
        f"{k22['fp32'] / k22['int8-spatial-first-kernel']:.1f}x faster than the original FP32 and "
        f"{k22['fp32-spatial-first'] / k22['int8-spatial-first-kernel']:.1f}x faster than the fastest FP32. Both "
        "kernels reproduce the runtime's own INT8 logits exactly on all 459 models (459 x 288 = 132,192 "
        f"predictions each). The frozen model takes {kf['int8-spatial-first-kernel']:.0f} us. Pruning saves almost "
        f"nothing in time: halving the last layer removes {pruned[0]:.1%} of the MACs in the original order and "
        f"{pruned[1]:.1%} in the swapped one.",
        "",
        f"Accuracy, too, depends on the processor. The same INT8 files gave identical predictions on the M4 and the "
        f"AMD 9654 (VNNI), but on x86 CPUs without VNNI {int(nonvnni.int8_models_changed.iloc[0])} of 459 models "
        f"changed and accuracy fell by {-nonvnni.int8_acc_change_pp.mean():.2f} points: their 8-bit kernel adds two "
        "8-bit products in a saturating 16-bit register, as ONNX Runtime documents [15]. Restricting weights to 7 "
        "bits (the runtime's reduce_range option, which its documentation recommends for these CPUs) removes the "
        "overflow; the 459 models then agree on every tested CPU and "
        f"lose {abs(rr.rr_acc_change_vs_fp32_pp.mean()):.2f} points against FP32. " + capture_summary(),
        "",
    ]
    L += [
        f"![Figure 5. Left: latency per 2-s window on the Apple M4 (one thread). Right: the frozen model on four "
        f"CPUs; with the first two layers swapped, INT8 beats the exported FP32 graph everywhere, but FP32 with the "
        f"same swap only on the M4.]({figs['latency']})",
        "",
    ]
    L += mcu_results(mc, m4_sw, m4_or, kf_fir, figs)
    L += failures(ms, of, rep, chd, align)
    return L


def capture_summary() -> str:
    """One sentence for Section 6.5; the numbers are in Appendix B."""
    cap = analysis("int8_capture_study")
    if not cap:
        return ""
    layer = next(iter(cap["summaries"][0]["first_differing"])).replace(" conv", " convolution")
    return (
        "Reading the integer tensors each machine actually computes confirms this: on x86 the runtime always uses "
        "unsigned 8-bit activations, VNNI makes that exact, and without VNNI the first difference is in the "
        f"{layer}, where every output that differs is a predicted 16-bit overflow; 7-bit weights are needed in every "
        "layer, and our kernels, rewritten for x86, are exact there too (Appendix B)."
    )


def capture_sentence() -> str:
    """Where the non-VNNI difference starts (int8_capture_study), or nothing before that study has run."""
    cap = analysis("int8_capture_study")
    if not cap:
        return ""
    s = cap["summaries"][0]
    tot, (layer, n) = s["saturation_totals"], next(iter(s["first_differing"].items()))
    lay = csv(SUPP / "int8_capture_study" / "layers.csv")
    logits = lay[lay.label == "logits"]
    every = {v["variant"]: v for v in cap["variants"]}
    single = [v for k, v in every.items() if k.startswith("7bit-") and k != "7bit-all"]
    return (
        "Reading the integer tensors of the graph each machine actually runs (a Gold 6140 without VNNI against the "
        f"AMD 9654, {s['models']} models), the first difference is in the {layer.replace(' conv', ' convolution')} "
        f"in {n} of {s['models']}, and "
        f"every output that differs there ({tot['differ'] / 1e6:.2f} million of {tot['outputs'] / 1e6:.1f} million) "
        "is one whose adjacent pair of unsigned-by-signed 8-bit products leaves the 16-bit range "
        f"({100 * s['share_of_differing_predicted']:.0f}%; the pairing predicts {tot['predicted'] / 1e6:.2f} "
        "million). On x86 the runtime rewrites the graph with unsigned 8-bit activations whether or not the CPU has "
        "VNNI, and VNNI makes that arithmetic exact. The errors grow layer by layer "
        f"({100 * logits.differ.sum() / logits['values'].sum():.0f}% of logits differ); 7-bit weights in any single "
        f"layer leave {min(100 * (1 - v['same_predictions_all'] / v['windows_all']) for v in single):.0f}-"
        f"{max(100 * (1 - v['same_predictions_all'] / v['windows_all']) for v in single):.0f}% of training-day "
        "predictions different, and only all five make the machines agree." + x86_kernel_sentence()
    )


def x86_kernel_sentence() -> str:
    """The two kernels as uint8 C versions on x86 (int8_kernel_x86_study), or nothing before it has run."""
    path = SUPP / "int8_kernel_x86_study" / "machines.csv"
    if not path.exists():
        return ""
    m = csv(path).set_index("machine")
    lat = csv(SUPP / "int8_kernel_x86_study" / "latency.csv")
    vnni, other = m.loc["vnni-amd-9654"], m.loc["novnni-intel-6140"]
    full = lat[lat.architecture == "full-22"].pivot_table(index="machine", columns="variant", values="us")
    return (
        " Rewritten for the unsigned activations x86 uses, our grouped-convolution kernel reproduces the runtime's "
        f"INT8 exactly on the AMD 9654 ({int(vnni.kernel_equals_runtime)}/{int(vnni.models)} models) and, by "
        "computing that one layer exactly, cuts the Gold 6140's loss from "
        f"{abs(other.runtime_acc_minus_m4_pp):.2f} to {abs(other.kernel_acc_minus_m4_pp):.2f} points. As plain C it "
        f"makes x86 INT8 faster ({full.loc['vnni-amd-9654', 'int8']:.0f} to "
        f"{full.loc['vnni-amd-9654', 'int8-kernel']:.0f} us on the AMD 9654) but not faster than FP32 "
        f"({full.loc['vnni-amd-9654', 'fp32']:.0f} us)."
    )


def kind_facts(kinds: dict, of: dict, rep: dict) -> dict:
    """Section 6.2's numbers from input_kind_study: costs by kind of input, and the real electrodes on that axis."""
    t = csv(SUPP / "input_kind_study" / "conditions.csv").set_index(["dataset", "condition"]).next_day_cost
    tests = kinds["tests_lee2019"]
    held = [k for k in ("K1", "K2", "K3") if tests[k]["holds"]]
    eq = kinds["a_equivalent"]
    net = (of["bnci2a"]["reduced_minus_full"]["mean"], rep["h"]["H1b"]["estimate_pp"])  # 16 minus all electrodes
    cost = (-eq["bnci2a"]["next_day_cost_at_a"], -eq["lee2019"]["next_day_cost_at_a"])
    pair = {c: f"{s(t[('bnci2a', c)])} and {s(t[('lee2019', c)])}" for c in ("dup", "shared", "mix50", "noise")}
    return {
        **pair,
        "t": t,
        "pr": kinds["predictability"],
        "verdicts": "K1, K2 and K3 all hold"
        if len(held) == 3
        else "; ".join(f"{k} {verdict(tests[k]['holds'])}" for k in ("K1", "K2", "K3")),
        "net": net,
        "cost": cost,
        "info": (cost[0] - net[0], cost[1] - net[1]),  # what they add = their cost + the net change of keeping them
    }


def new_mechanism_results(kinds, longer, scalefree, pre, fresh=None) -> list[str]:
    n = sum(x is not None for x in (longer, scalefree, pre, fresh))
    L = [
        f"**Testing the explanation.** {word(n).capitalize()} more studies, each with registered predictions on "
        "Lee 2019, test the account above."
    ]
    if longer:
        t = longer["tests_lee2019"]
        L.append(
            f"*Training four times longer* (1,900 epochs) changed neither: the noise cost moved by {est(t['L1'])} "
            f"(L1 {verdict(t['L1']['holds'])}), all 62 electrodes did not beat 16 (L2 {verdict(t['L2']['holds'])}), "
            f"and the noise models' held-out accuracy did not rise after epoch 475 (L3 {verdict(t['L3']['holds'])}; "
            "curves in Appendix B)."
        )
    else:
        L.append(f"*Training four times longer* {PENDING}.")
    if scalefree:
        t = scalefree["tests_lee2019"]
        ratios = [r["g_ratio_median"] for r in scalefree["folds"] if r["g_ratio_median"] is not None and r["lam"] > 0]
        L.append(
            "*A penalty no rescaling can lower* (filters scaled to unit length before the per-input norm) did not "
            "switch the noise off either: at no strength did the noise inputs' median norm fall below a third of the "
            f"electrodes' (ratios {min(ratios):.2f}-{max(ratios):.2f}; N3 {verdict(t['N3']['holds'])}), and the "
            f"noise cost changed by {est(t['N1'])} (N1 {verdict(t['N1']['holds'])}; N2 "
            f"{verdict(t['N2']['holds'])}). So scale invariance is not the whole reason."
        )
    else:
        L.append(f"*A scale-free penalty* {PENDING}.")
    if fresh:
        L.append(fresh_sentence(fresh))
    else:
        L.append(f"*Noise redrawn in every epoch* {PENDING}.")
    if pre:
        t, lee, a2 = pre["tests_lee2019"], pre["lee2019"], pre["bnci2a"]
        acc = {(x["dataset"], x["config"], x["training"]): x["next_day"] for x in pre["accuracy"]}
        L.append(
            "*Pretraining on 45 other people, then fine-tuning on the person's training day* (the same electrodes "
            "for everyone; 'a shared 16' is the best 16 on the pretraining people): with all 62 electrodes, "
            f"next-day accuracy rises from {acc[('lee2019', 'full-62', 'scratch')]:.1f}% to "
            f"{acc[('lee2019', 'full-62', 'finetune')]:.1f}% (T1 {est(t['T1'])}, {verdict(t['T1']['holds'])}); the 46 "
            f"noise inputs cost {s(lee['finetune: noise cost (group-16 plus noise minus group-16)']['mean'])} points "
            f"instead of {s(lee['scratch: noise cost (group-16 plus noise minus group-16)']['mean'])} (T2 "
            f"{verdict(t['T2']['holds'])}); and all 62 electrodes beat a shared 16 by {est(t['T3'])} (T3 "
            f"{verdict(t['T3']['holds'])}), which they did not from scratch "
            f"({s(lee['scratch: all electrodes minus group-16']['mean'])}). With enough data the network does use "
            "the extra electrodes; three times more of the person's own (C2) was not enough. On 2a, eight other "
            "people were too few: pretraining lowered accuracy with all 22 electrodes by "
            f"{abs(a2['full: finetune minus scratch']['mean']):.1f} points."
        )
    else:
        L.append(f"*Pretraining on other people* {PENDING}.")
    return [" ".join(L), ""]


def surrogate_sentences(surr: dict) -> str:
    """Section 6.2: the dropped electrodes' cost and information, measured (residual_surrogate_study)."""
    t = surr["tests_lee2019"]
    d = {(r["dataset"], r["level"], r["term"]): r for r in surr["decomposition"]}
    c = {(r["dataset"], r["condition"]): r for r in surr["conditions"]}
    lee = {k: d[("lee2019", "final", k)] for k in ("cost", "information", "sum", "pred_cost")}
    two = {k: d[("bnci2a", lvl, k)] for lvl, k in (("final", "cost"), ("final", "information"))}
    within = d[("bnci2a", "fold", "information")]
    return (
        "To measure what they add and what they cost, we replaced each trial's unexplained part (the residual of "
        "that fit) with another trial's, which keeps its size and its correlation across electrodes but not its "
        f"relation to the trial. On Lee 2019 these inputs cost {est(lee['cost'])} points against the 16 alone "
        f"(registered S2 {verdict(t['S2']['holds'])}), the real residuals add {est(lee['information'])} over them "
        f"(S1 {verdict(t['S1']['holds'])}), and the prediction alone costs nothing visible "
        f"({s(lee['pred_cost']['mean'])}; S3 {verdict(t['S3']['holds'])}): information and cost cancel "
        f"({s(lee['sum']['mean'])}). On 2a the cost is {s(two['cost']['mean'])} and the information "
        f"{s(two['information']['mean'])} next day ({s(within['mean'])} on held-out runs of the training day). With "
        f"the unexplained part present the Lee 2019 networks fit {c[('lee2019', 'surr')]['train_final']:.0f}% of "
        f"their training trials; with the 16 alone {c[('lee2019', 'none')]['train_final']:.0f}%, and with the "
        f"prediction added {c[('lee2019', 'pred')]['train_final']:.0f}%. Across the 54 people the "
        f"unexplained share did not predict who paid more (S4 {verdict(t['S4']['holds'])}, rho "
        f"{t['S4']['rho']:+.2f}), although a person's cost repeats between halves of the seeds (rho "
        f"{surr['seed_split_cost']['rho']:+.2f}). "
    )


def fresh_sentence(fresh: dict) -> str:
    """Section 6.2: the same noise drawn afresh in every training epoch (fresh_noise_study)."""
    t = fresh["tests_lee2019"]
    con = {(c["dataset"], c["added"], c["level"], c["a"], c["b"]): c for c in fresh["contrasts"]}
    meas = {(m["dataset"], m["added"], m["noise"]): m for m in fresh["measures"]}
    fixed, now = con[("lee2019", 46, "final", "fixed", "none")], con[("lee2019", 46, "final", "fresh", "none")]
    removed = {
        k: con[(d, k, "final", "fresh", "fixed")]["mean"] / -con[(d, k, "final", "fixed", "none")]["mean"]
        for d, k in (("bnci2a", 6), ("bnci2a", 16), ("lee2019", 46))
    }
    rec = meas[("lee2019", 46, "fixed")]
    return (
        "*The same noise drawn afresh in every training epoch* (all else unchanged) "
        f"is switched off: its weights fall to {t['F3']['fresh_median_ratio']:.2f} of the electrodes' (median; "
        f"{meas[('lee2019', 46, 'fresh')]['noise_share']:.0%} of the spatial filters' output, against "
        f"{rec['weight_ratio']:.2f} and {rec['noise_share']:.0%} with recorded noise; F3 "
        f"{verdict(t['F3']['holds'])}), and the cost of 46 noise inputs falls from {s(fixed['mean'])} to "
        f"{s(now['mean'])} points (F1 {verdict(t['F1']['holds'])}), though not to zero (F2 "
        f"{verdict(t['F2']['holds'])}, {interval(*now['ci'])}); on 2a it falls by "
        f"{100 * removed[6]:.0f}% and {100 * removed[16]:.0f}% with 6 and 16. Recorded noise is the same in every "
        "epoch, and the network uses it to remember its training trials: with their own noise the Lee 2019 models "
        f"classify {rec['train_acc_draw0']:.0f}% of them correctly, with noise they have not seen "
        f"{rec['train_acc_unseen']:.1f}%. So most of the cost of noise inputs ({100 * removed[46]:.0f}% on Lee 2019) "
        "is the network memorizing content that repeats; noise that changes, it largely learns to ignore."
    )


def mcu_results(mc, m4_sw, m4_or, kf_fir, figs) -> list[str]:
    st = mc["study"]
    ratios = [v for core in ("m4", "m55") for c in st["fp32_over_int8"][core].values() for v in c.values()]
    fast = (min(ratios), max(ratios))
    m0 = [v for c in st["fp32_over_int8"]["m0"].values() for v in c.values()]
    m0r = (min(m0), max(m0))
    diffs = [x["int8_minus_fp32"] for x in st["accuracy"]]
    acc_span = (min(diffs), max(diffs))
    L = [
        "### 6.6 Microcontrollers: CMSIS-NN and the missing filter-bank kernel",
        "",
        f"On emulated Cortex-M cores, CMSIS-NN's INT8 network reproduced our NumPy integer reference bit for bit in "
        f"{st['int8_bit_exact_builds']} of {st['int8_builds']} builds, and its next-day accuracy differs from FP32 "
        f"by {acc_span[0]:+.2f} to {acc_span[1]:+.2f} points. INT8 needs {m0r[0]:.0f}-{m0r[1]:.0f}x fewer "
        "instructions than FP32 on the M0, which has no floating-point unit, but only "
        f"{fast[0]:.1f}-{fast[1]:.1f}x fewer on the M4 and M55, because both libraries spend most instructions moving "
        "data. Swapping the layers "
        f"cuts the M4's INT8 count from {m4_or:.1f} to {m4_sw:.1f} million instructions per window (Figure 6).",
        "",
    ]
    if kf_fir is not None:
        k = mc["kernel"]
        ch = mc["kernel_checks"]
        m4 = kf_fir.loc[("swapped", "m4")]
        m55 = kf_fir.loc[("swapped", "m55")]
        m0 = kf_fir.loc[("swapped", "m0")]
        L.append(
            "With the layers swapped, the temporal layer (16 FIR filters of 64 taps) still took "
            f"{m4.library_temporal / 1e6:.1f} of the M4's {m4.library_total / 1e6:.1f} million instructions, "
            f"{m4.library_per_mac:.1f} per MAC, since CMSIS-NN treats it as a general depthwise convolution. Our "
            f"filter-bank kernel brings that layer to {m4.fir_per_mac:.2f} instructions per MAC on the M4 (best case "
            f"0.5), {m55.fir_per_mac:.2f} on the M55 (best case 1/16) and {m0.fir_per_mac:.1f} on the M0, and the "
            f"whole network to {m4.fir_total / 1e6:.2f} (M4), {m55.fir_total / 1e6:.2f} (M55) and "
            f"{m0.fir_total / 1e6:.2f} (M0) million instructions, {k.total_ratio.min():.1f}-{k.total_ratio.max():.1f}x "
            f"fewer across all builds. All {ch['bit_exact']} of {ch['builds']} builds give logits identical to the "
            "reference. On the M55 it also removes a 32 KB scratch buffer the library needed. These are instruction "
            "counts, not time or energy."
        )
        L.append("")
    L += [
        f"![Figure 6. Instructions per 2-s window on emulated Cortex-M cores (log scale), CMSIS-NN v8.0.0, 22 "
        f"electrodes.]({figs['mcu']})",
        "",
    ]
    return L


def failures(ms, of, rep, chd, align) -> list[str]:
    h = rep["h"]
    a, lee = of["bnci2a"], of["lee2019"]
    L = [
        "### 6.7 What did not work, and why",
        "",
        f"- *Pruning at a matched budget.* Against an unpruned model fine-tuned for the same 100 epochs, pruning half "
        f"of the last layer changed accuracy by {s(ms['prune16']['difference_pp'])} points "
        f"{interval(ms['prune16']['ci_low_pp'], ms['prune16']['ci_high_pp'])} on the 16-electrode base; it saves "
        "bytes, "
        "not time.",
        f"- *Day-to-day alignment.* Whitening each day helped 2a on average but hurt Lee 2019's 62 electrodes "
        f"(registered H5: {s(h['H5']['estimate_pp'])} {interval(h['H5']['low_pp'], h['H5']['high_pp'])}). "
        + (alignment_reason(align) if align else PENDING),
        f"- *Channel dropout.* Dropping random electrodes in training cost {-a['dropout']['mean']:.2f} points on 2a "
        f"and {-lee['dropout']['mean']:.2f} on Lee 2019. " + (chdrop_reason(chd) if chd else PENDING),
        f"- *Who benefits.* On 2a, people decoded worst with all electrodes seemed to gain most from fewer; measured "
        f"on separate trials in 54 people the correlation is {h['H3']['rho']:+.3f} (registered H3 does not "
        "hold), so it was noise plus regression to the mean.",
        "- *Letting the network choose.* A learned selection layer was the worst selection method, and neither a "
        "group lasso nor a scale-free penalty on the inputs switched the noise inputs off (Section 6.2).",
    ]
    return L


def alignment_reason(a: dict) -> str:
    m = a["readings"]
    spec = csv(SUPP / "alignment_mechanism_study" / "spectra.csv")
    lee = spec[(spec.dataset == "lee2019") & (spec.montage == "full-62")]
    share = {v: 100 * x["share_of_harm_removed"] for v, x in m["M2"]["variants"].items()}
    return (
        f"The harm is not a day-to-day effect: within one day it is {est(m['M1'])} (M1). Nor is the covariance "
        f"badly estimated (Ledoit-Wolf coefficient {m['M3']['mean_coefficient']:.4f}, M3). But "
        f"{lee.below_1pct.mean():.0f} of its {int(lee.electrodes.iloc[0])} directions carry less than 1% of the "
        "largest one's variance, and whitening scales them all to unit variance: many inputs the others cannot "
        "predict, as in Section 6.2. Keeping those directions without amplifying them removes "
        f"{share['ea-floor']:.0f}% of the harm and shrinking the covariance by 10% removes "
        f"{share['ea-shrink10']:.0f}%; "
        f"dropping them removes {share['ea-top']:.0f}%, because it also drops signal (M2, which asked for half, "
        f"{'holds' if m['M2']['holds'] else 'falls just short'})."
    )


def failure_costs(c: dict | None) -> tuple[float, float] | None:
    """2a: points lost to one failed electrode without dropout, and with p = 0.05 (re-estimated)."""
    if not c:
        return None
    fail = {(r["dataset"], r["p"], r["version"]): r["single_mean"] for r in c["failure"] if r["montage"] == "full"}
    return fail[("bnci2a", 0.0, "as_trained")], fail[("bnci2a", 0.05, "recalibrated")]


def chdrop_reason(c: dict) -> str:
    t = c["tests_lee2019"]
    fail = {(r["dataset"], r["p"], r["version"]): r["single_mean"] for r in c["failure"] if r["montage"] == "full"}
    p = t["B3"]["p"]
    return (
        "Re-estimating the batch norms with every electrode present recovers only "
        f"{100 * t['B1']['share_recovered']:.0f}% of the loss on Lee 2019 (B1 {verdict(t['B1']['holds'])}), so the "
        f"damage is in the learned filters, not in mismatched statistics; a smaller p costs less (B2 "
        f"{verdict(t['B2']['holds'])}). Dropout does buy robustness: with one electrode replaced by its training mean "
        f"on the next day, the 2a models lose {fail[('bnci2a', 0.0, 'as_trained')]:.1f} points on average without "
        f"dropout and {fail[('bnci2a', p, 'recalibrated')]:.1f} with p = {p}; on Lee 2019 "
        f"{fail[('lee2019', 0.0, 'as_trained')]:.2f} against {fail[('lee2019', p, 'recalibrated')]:.2f} (B3 "
        f"{verdict(t['B3']['holds'])}, {est(t['B3'])})."
    )


def conclusion(ms, of, r, k22, ratio_range) -> list[str]:
    return [
        "",
        "## 7. Conclusion",
        "",
        "A small EEG decoder can lose a quarter of its electrodes, three quarters of its bits and most of its "
        "latency without losing average next-day accuracy, but none of that comes for free from the model alone. "
        "Fewer electrodes work because the network pays for the part of every input that the others cannot predict, "
        "mostly by memorizing its training trials with it, and the dropped electrodes add about as much as they "
        "cost, until enough data from other people lets the network use them; choosing them jointly "
        "matters most when few are kept, and that result held on 54 new people. INT8 is accurate, but whether it is "
        "fast is decided by the runtime's kernels and the order of two layers, and whether it is exact by the "
        "processor's accumulator. Two lessons carry over beyond EEG: measure each claimed saving on the target, and "
        "when a model's most common layer shape falls to a general path, a short dedicated kernel can matter more "
        "than any further compression. Future work: pretraining across many users so that more electrodes pay off, "
        "electrode selection learned on such data, and timing on physical microcontrollers.",
    ]


def references() -> list[str]:
    refs = [
        "V. J. Lawhern et al. EEGNet: a compact convolutional neural network for EEG-based brain-computer "
        "interfaces. J. Neural Eng. 15(5):056013, 2018.",
        "T. Schneider, X. Wang, M. Hersche, L. Cavigelli, L. Benini. Q-EEGNet: an energy-efficient 8-bit quantized "
        "parallel EEGNet implementation for edge motor-imagery brain-machine interfaces. Proc. IEEE Int. Conf. "
        "Smart Computing (SMARTCOMP), pp. 284-289, 2020.",
        "T. M. Ingolfsson et al. EEG-TCNet: an accurate temporal convolutional network for embedded "
        "motor-imagery brain-machine interfaces. Proc. IEEE Int. Conf. Systems, Man, and Cybernetics (SMC), "
        "pp. 2958-2965, 2020.",
        "X. Wang, M. Hersche, M. Magno, L. Benini. MI-BMInet: an efficient convolutional neural network for motor "
        "imagery brain-machine interfaces with EEG channel selection. IEEE Sensors J. 24(6):8835-8847, 2024.",
        "L. Lai, N. Suda, V. Chandra. CMSIS-NN: efficient neural network kernels for Arm Cortex-M CPUs. "
        "arXiv:1801.06601, 2018.",
        "M. Tangermann et al. Review of the BCI Competition IV. Front. Neurosci. 6:55, 2012.",
        "M.-H. Lee et al. EEG dataset and OpenBMI toolbox for three BCI paradigms: an investigation into BCI "
        "illiteracy. GigaScience 8(5):giz002, 2019.",
        "B. Jacob et al. Quantization and training of neural networks for efficient integer-arithmetic-only "
        "inference. Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), pp. 2704-2713, 2018.",
        "T. Alotaiby, F. E. A. El-Samie, S. A. Alshebeili, I. Ahmad. A review of channel selection algorithms for "
        "EEG signal processing. EURASIP J. Adv. Signal Process. 2015:66, 2015.",
        "Abdullah, I. Faye, M. R. Islam. EEG channel selection techniques in motor imagery applications: a "
        "review and new perspectives. Bioengineering 9(12):726, 2022.",
        "A. Barachant, S. Bonnet, M. Congedo, C. Jutten. Multiclass brain-computer interface classification by "
        "Riemannian geometry. IEEE Trans. Biomed. Eng. 59(4):920-928, 2012.",
        "S. Han, H. Mao, W. J. Dally. Deep compression: compressing deep neural networks with pruning, trained "
        "quantization and Huffman coding. ICLR, 2016.",
        "H. Li, A. Kadav, I. Durdanovic, H. Samet, H. P. Graf. Pruning filters for efficient ConvNets. ICLR, 2017.",
        "H. He, D. Wu. Transfer learning for brain-computer interfaces: a Euclidean space data alignment approach. "
        "IEEE Trans. Biomed. Eng. 67(2):399-410, 2020.",
        "ONNX Runtime. Quantize ONNX models (documentation). https://onnxruntime.ai/docs/performance/"
        "model-optimizations/quantization.html, accessed 2026.",
        "R. David et al. TensorFlow Lite Micro: embedded machine learning for TinyML systems. Proc. Machine "
        "Learning and Systems 3 (MLSys), pp. 800-811, 2021.",
        "O. Ledoit, M. Wolf. A well-conditioned estimator for large-dimensional covariance matrices. J. "
        "Multivariate Anal. 88(2):365-411, 2004.",
    ]
    return ["", "## References", ""] + [f"{i}. {t}" for i, t in enumerate(refs, 1)]


def artifacts(comp: dict) -> list[str]:
    return [
        "",
        "## Artifacts",
        "",
        "**Code and records.** The public repository https://github.com/lipeiye/comp690-final-project "
        "holds the package (`src/mibci/`), every study script and its plan, all result records "
        "(`results/`), the C kernels (`kernels/`, `mcu/`), and this report's generator.",
        "",
        "**Push-button rebuild of every table and figure (minutes, no EEG data needed):**",
        "",
        "`python3 scripts/rebuild_all_tables.py` regenerates every analysis table and figure and this report "
        "from the stored records into a sandbox and compares them byte for byte with the repository "
        "(`python3 tools/golden.py --all` does the same check).",
        "",
        "**Rerunning the experiments.** `scripts/prepare_data.py` and `scripts/prepare_lee2019.py` download and cache "
        "the two datasets (about 65 GB for Lee 2019); `scripts/run_all.sh` reruns the frozen main study (about "
        "8 hours on an M4 Mac mini); each later study runs as `scripts/<name>_study.py --stage plan|train|evaluate` "
        "on a SLURM cluster through `tools/longleaf.py`. The post-midterm studies used "
        f"{comp['longleaf_models'] if 'longleaf_models' in comp else comp['totals']['longleaf_models']:,} trainings "
        "and about "
        f"{comp['longleaf_cpu_hours'] if 'longleaf_cpu_hours' in comp else comp['totals']['longleaf_cpu_hours']:,.0f} "
        "CPU hours. The microcontroller studies need the Arm GNU Toolchain, CMSIS-NN v8.0.0 and QEMU; their exact "
        "versions are recorded in each plan.",
    ]


def appendix_b(f: dict) -> list[str]:
    """Supporting numbers moved out of the main text (not counted toward its length)."""
    pe = f["pe"]
    L = ["", "## Appendix B. Supporting details", ""]
    L.append(
        "**Group lasso (Section 6.2).** With the penalty, median weight norms fell from "
        f"{pe['electrode_norm'][0]:.2f} to {pe['electrode_norm'][1]:.3f} for electrodes and from "
        f"{pe['noise_norm'][0]:.2f} to {pe['noise_norm'][1]:.3f} for noise inputs; the noise-to-electrode ratio stayed "
        f"between {pe['ratio'][0]:.2f} and {pe['ratio'][1]:.2f}, so every weight shrank together."
    )
    curves_csv = SUPP / "long_training_study" / "curves.csv"
    if curves_csv.exists():
        cv = csv(curves_csv).set_index(["dataset", "condition", "epoch"])
        noisy, plain = ("lee2019", "relevance-16-plus-46"), ("lee2019", "relevance-16")
        L += [
            "",
            "**Training curves (Section 6.2).** With 46 noise inputs the Lee 2019 models fitted their training trials "
            f"fastest (training loss {cv.loc[(*noisy, 475), 'train_loss']:.2f} at epoch 475, against "
            f"{cv.loc[(*plain, 475), 'train_loss']:.2f} without noise) yet stayed near chance on held-out runs of the "
            f"same day ({cv.loc[(*noisy, 475), 'held_out']:.1f}% at epoch 475, "
            f"{cv.loc[(*noisy, 1900), 'held_out']:.1f}% at 1,900); held-out accuracy never rose and then fell.",
        ]
    paths = SUPP / "int8_paths_study" / "shapes.csv"
    if paths.exists():
        pmap = csv(paths)
        x86, m4 = pmap[pmap.machine != "m4-mac-mini"], pmap[pmap.machine == "m4-mac-mini"]
        dw = x86[x86.eegnet_layer.fillna("").str.contains("swapped temporal")].int8_over_fp32
        best = x86.int8_over_fp32.min()
        L += [
            "",
            f"**Where INT8 is fast (Section 6.5).** {int(m4.name.nunique())} single convolutions (G groups of M "
            "filters; 22x1 spatial or 1x64 temporal kernels), exported and quantized as the decoder, timed at batch 1 "
            f"on one thread. On the M4, INT8 beat FP32 for {int((m4.int8_over_fp32 < 1).sum())} shapes: the "
            "ungrouped temporal convolutions with at least two filters and the depthwise ones over at least 8 "
            f"channels. On the {word(x86.machine.nunique())} x86 CPUs it beat FP32 for at most "
            f"{int(x86.groupby('machine').int8_over_fp32.apply(lambda v: (v < 1).sum()).max())} shape, by at most "
            f"{100 * (1 - best):.0f}%, and the swapped network's depthwise temporal layer was "
            f"{dw.min():.1f}-{dw.max():.1f}x slower in INT8 than in FP32.",
        ]
    if analysis("int8_capture_study"):
        L += ["", "**Where INT8 differs without VNNI (Section 6.5).** " + capture_sentence().strip()]
    kernel = x86_kernel_sentence()
    if kernel:
        L += ["", "**The kernels on x86 (Section 6.5).** " + kernel.strip()]
    return L


def appendix(f: dict) -> list[str]:
    of, co, rep = f["of"], f["co"], f["rep"]
    kinds, longer, scalefree, pre, chd = (f[k] for k in ("kinds", "longer", "scalefree", "pre", "chd"))
    fresh, surr = f["fresh"], f["surr"]
    h, r = rep["h"], rep["r"]
    rows = [
        ("H1", "45 of 62 electrodes within 2 points of all 62", f"{s(h['H1']['estimate_pp'])}", h["H1"]["met"]),
        ("H1b", "16 of 62 within 2 points", f"{s(h['H1b']['estimate_pp'])}", h["H1b"]["met"]),
        ("H2", "relevance beats random at 16", f"{s(h['H2']['estimate_pp'])}", h["H2"]["met"]),
        ("H3", "worse decoders gain more from fewer electrodes", f"{h['H3']['rho']:+.3f} (rho)", h["H3"]["met"]),
        (
            "H4",
            "8-bit integers keep accuracy (three montages)",
            ", ".join(s(x["estimate_pp"]) for x in rep["rows"]["H4"]),
            all(x["met"] for x in rep["rows"]["H4"]),
        ),
        ("H5", "day-to-day alignment helps", f"{s(h['H5']['estimate_pp'])}", h["H5"]["met"]),
        (
            "R1-R3",
            "Riemannian beats relevance at 8, 16, 20",
            f"{s(r['R1']['diff_pp'])}, {s(r['R2']['diff_pp'])}, {s(r['R3']['diff_pp'])}",
            all(r[k]["met"] for k in ("R1", "R2", "R3")),
        ),
        ("P3", "noise in place of dropped electrodes hurts", s(of["P3"]["mean"]), of["P3"]["holds"]),
        ("C1", "the cost grows with the number of useless inputs", s(co["C1"]["mean"]), co["C1"]["holds"]),
        ("C2", "the cost is larger with fewer training trials", s(co["C2"]["mean"]), co["C2"]["holds"]),
    ]
    for name, study, keys in (
        ("input_kind_study", kinds, ("K1", "K2", "K3")),
        ("residual_surrogate_study", surr, ("S1", "S2", "S3", "S4")),
        ("long_training_study", longer, ("L1", "L2", "L3")),
        ("scalefree_penalty_study", scalefree, ("N1", "N2", "N3")),
        ("fresh_noise_study", fresh, ("F1", "F2", "F3")),
        ("pretrain_study", pre, ("T1", "T2", "T3")),
        ("chdrop_study", chd, ("B1", "B2", "B3")),
    ):
        if not study:
            continue
        claims = read(SUPP / name / "protocol.json")["predictions"]
        for key in keys:
            t = study["tests_lee2019"][key]
            means = (
                [t["mean"]] if "mean" in t else [v["mean"] for v in t.values() if isinstance(v, dict) and "mean" in v]
            )
            means = [t["rho"]] if "rho" in t else means  # a Spearman correlation (S4)
            unit = {"F3": " (weight ratio)", "S4": " (rho)"}.get(key, "")
            rows.append((key, claims[key]["claim"], ", ".join(s(m) for m in means) + unit, t["holds"]))
    L = [
        "",
        "## Appendix A. Predictions registered on Lee 2019 before scoring",
        "",
        "| test | prediction | estimate (points) | result |",
        "|---|---|---:|---|",
    ]
    L += [f"| {k} | {claim} | {e} | {'holds' if ok else 'does not hold'} |" for k, claim, e, ok in rows]
    L += ["", "Studies still running when this report was built add their tests when they finish."]
    return L


def ethics(comp: dict, chd: dict | None = None) -> list[str]:
    cpu = comp["longleaf_cpu_hours"] if "longleaf_cpu_hours" in comp else comp["totals"]["longleaf_cpu_hours"]
    return [
        "",
        "## Ethical, legal, societal and broader implications",
        "",
        "**Data.** Both datasets are public recordings released for research. Lee et al. [7] report approval by "
        "the Korea University Institutional Review Board and written informed consent from all participants; the "
        "published descriptions of BCI Competition IV 2a [6] do not report consent or ethics approval, and we rely "
        "on the organizers' public release for research use. This project collected no new human data and "
        "redistributes no EEG; it stores trained models and derived records only.",
        "",
        "**Privacy.** EEG can identify a person and can reveal health information. A decoder trained on one "
        "person's EEG encodes that person's signals; models trained on real users should be treated as personal "
        "data (stored encrypted, deleted with the account, never pooled for pretraining without consent).",
        "",
        "**Validity and harm.** Average accuracy hides individual losses: the frozen configuration lost more than "
        "2 points for 4 of 9 people, and with 16 of 62 electrodes 24 of 54 people lost that much. A deployed BCI "
        "should therefore be validated per user, with the full montage as a fallback. Results come from "
        "laboratory recordings on two days; they do not establish performance for patients, over "
        "months, or in closed-loop use (our replay of recorded EEG is not online control). INT8 accuracy changed "
        "on some x86 CPUs, so integer models must be checked on the hardware that will run them; the runtime's "
        "documented 7-bit option removed the difference here."
        + (
            " Electrodes also fail in use: on 2a one failed electrode cost a model trained without channel dropout "
            f"{fail[0]:.1f} points on average and one trained with p = 0.05 {fail[1]:.1f} (Section 6.7), so a deployed "
            "decoder should be trained for it or detect failed electrodes and fall back."
            if (fail := failure_costs(chd))
            else ""
        ),
        "",
        f"**Compute.** The studies used about {cpu:,.0f} CPU hours on a shared university cluster and modest local "
        "computing. We reused trained models and records wherever possible instead of retraining.",
    ]


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    text = build()
    OUT.write_text(text)
    words = len(text.split("**Abstract.**")[1].split("\n")[0].split())
    print(f"Wrote {OUT}; abstract {words} words")


if __name__ == "__main__":
    main()
