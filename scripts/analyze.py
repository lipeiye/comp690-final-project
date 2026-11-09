"""Session-2 analysis following results/frozen_config.json, plus the efficiency figures.

Primary analysis: trials without artifact flags. Each participant's metrics are
averaged over the three seeds before the nine participants are summarized, and
paired differences get participant-level bootstrap intervals (exploratory, not
hypothesis tests). Writes CSV tables, summary.md and figures to results/analysis/,
and the same figures with larger type to results/analysis/slides/.
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from mibci import data
from mibci.experiment import RESULTS_DIR
from summarize_validation import parse_condition

OUT = RESULTS_DIR / "analysis"
FULL = "full-22"
MARGIN = 0.02
METHODS = ("proposed", "relevance", "weightnorm", "random")
METHOD_LABEL = {"full": "all 22 electrodes", "proposed": "redundancy-aware (proposed)", "relevance": "relevance only",
                "weightnorm": "spatial-weight norm", "random": "random (mean of 3)"}
METHOD_COLOR = {"full": "#404040", "proposed": "#c55a11", "relevance": "#2e75b6", "weightnorm": "#548235",
                "random": "#8c8c8c"}
CLASS_LABELS = ["left hand", "right hand", "feet", "tongue"]
SLIDE_RC = {"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 14,
            "ytick.labelsize": 14, "legend.fontsize": 13, "lines.linewidth": 2.2, "lines.markersize": 8}


def load_test(analysis: str) -> pd.DataFrame:
    rows = []
    for path in sorted((RESULTS_DIR / "test").glob("sub-*.json")):
        r = json.loads(path.read_text())
        for precision in ("fp32", "int8"):
            s = r[analysis][precision]
            rows.append({"subject": r["subject"], "seed": r["seed"], "condition": r["condition"],
                         **parse_condition(r["condition"]), "precision": precision, "acc": s["acc"],
                         "f1": s["macro_f1"], **{f"recall_{c}": v for c, v in zip(data.CLASSES, s["recall"])},
                         "confusion": np.array(s["confusion"])})
    return pd.DataFrame(rows)


def per_participant(df: pd.DataFrame) -> pd.DataFrame:
    """Seeds averaged within each participant."""
    keys = ["condition", "method", "electrodes", "pruned", "precision", "subject"]
    return df.drop(columns=["seed", "confusion"]).groupby(keys).mean().reset_index()


def accuracies(pp: pd.DataFrame, condition: str, precision: str) -> pd.Series:
    """Participant accuracies; 'random-K' means the average of the three random subsets."""
    if condition.startswith("random-"):
        k = condition.split("-")[1]
        sel = pp[pp["condition"].isin([f"random{i}-{k}" for i in (1, 2, 3)]) & (pp["precision"] == precision)]
        return sel.groupby("subject")["acc"].mean()
    return pp[(pp["condition"] == condition) & (pp["precision"] == precision)].set_index("subject")["acc"]


def bootstrap(diffs: np.ndarray, plan: dict) -> tuple[float, float]:
    rng = np.random.default_rng(plan["seed"])
    means = diffs[rng.integers(0, len(diffs), size=(plan["resamples"], len(diffs)))].mean(1)
    tail = (1 - plan["level"]) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def paired(pp, plan, a, pa, b, pb) -> dict:
    d = (accuracies(pp, a, pa) - accuracies(pp, b, pb)).dropna().to_numpy()
    lo, hi = bootstrap(d, plan)
    return {"a": f"{a} {pa.upper()}", "b": f"{b} {pb.upper()}", "mean_diff": float(d.mean()), "ci_low": lo,
            "ci_high": hi, "n_a_better": int((d > 0).sum()), "n_b_better": int((d < 0).sum()),
            "n_tied": int((d == 0).sum()), "per_participant": [round(float(x), 4) for x in d]}


def comparisons(pp: pd.DataFrame, plan: dict, reduced_base: str) -> pd.DataFrame:
    rows = []
    for k in (16, 8):
        for prec in ("fp32", "int8"):
            for other in ("relevance", "weightnorm", "random"):
                rows.append({"family": f"proposed vs {other}", "electrodes": k, "precision": prec,
                             **paired(pp, plan, f"proposed-{k}", prec, f"{other}-{k}", prec)})
    for cond in sorted(pp["condition"].unique()):
        rows.append({"family": "INT8 vs FP32", "electrodes": parse_condition(cond)["electrodes"], "precision": "int8",
                     **paired(pp, plan, cond, "int8", cond, "fp32")})
    for base in (FULL, reduced_base):
        for frac in (25, 50):
            for prec in ("fp32", "int8"):
                rows.append({"family": "pruned vs unpruned", "electrodes": parse_condition(base)["electrodes"],
                             "precision": prec, **paired(pp, plan, f"{base}-prune{frac}", prec, base, prec)})
    return pd.DataFrame(rows)


def condition_table(pp: pd.DataFrame, plan: dict) -> pd.DataFrame:
    baseline = accuracies(pp, FULL, "fp32")
    rows = []
    for (cond, prec), g in pp.groupby(["condition", "precision"]):
        acc = g.set_index("subject")["acc"]
        diff = (acc - baseline).dropna().to_numpy()
        lo, hi = bootstrap(diff, plan) if (cond, prec) != (FULL, "fp32") else (0.0, 0.0)
        rows.append({"condition": cond, "precision": prec, **parse_condition(cond), "acc_mean": acc.mean(),
                     "acc_sd": acc.std(), "acc_min": acc.min(), "f1_mean": g["f1"].mean(),
                     **{f"recall_{c}": g[f"recall_{c}"].mean() for c in data.CLASSES},
                     "diff_vs_full_fp32": diff.mean(), "diff_ci_low": lo, "diff_ci_high": hi,
                     "n_losing_more_than_2pts": int((diff < -MARGIN).sum())})
    return pd.DataFrame(rows).sort_values(["electrodes", "pruned", "method", "precision"],
                                          ascending=[False, True, True, True])


def selection_stability() -> pd.DataFrame:
    """Jaccard overlap between the subsets chosen in the three validation folds and on all of session 1."""
    rows = []
    for subject in data.SUBJECTS:
        folds = [json.loads((RESULTS_DIR / "validation" / "rankings" / f"sub-{subject:02d}_fold-{f}.json").read_text())
                 for f in range(3)]
        final = json.loads((RESULTS_DIR / "final" / "rankings" / f"sub-{subject:02d}_seed-0.json").read_text())
        for method in ("proposed", "relevance", "weightnorm"):
            for k in (8, 16):
                sets = [set(r["subsets"][f"{method}-{k}"]) for r in folds + [final]]
                pairs = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1 :]]
                rows.append({"subject": subject, "method": method, "electrodes": k, "mean_jaccard": np.mean(pairs)})
    return pd.DataFrame(rows)


def proposed_equals_relevance() -> dict:
    """How often the redundancy penalty changed nothing: identical subsets, final rankings and folds."""
    out = {}
    for k in (8, 16):
        same = total = 0
        for path in list((RESULTS_DIR / "final" / "rankings").glob("sub-*_seed-0.json")) + \
                list((RESULTS_DIR / "validation" / "rankings").glob("sub-*.json")):
            s = json.loads(path.read_text())["subsets"]
            same += set(s[f"proposed-{k}"]) == set(s[f"relevance-{k}"])
            total += 1
        out[k] = {"identical": same, "of": total}
    return out


# ---------------------------------------------------------------- figures


def method_of(condition: str) -> str:
    m = parse_condition(condition)["method"]
    return "random" if m.startswith("random") else m


def fig_accuracy_vs_electrodes(pp: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, constrained_layout=True)
    for ax, prec in zip(axes, ("fp32", "int8")):
        full = accuracies(pp, FULL, prec)
        for i, method in enumerate(METHODS):
            accs = [accuracies(pp, f"{method}-{k}", prec) for k in (8, 16)] + [full]
            dodge = 0.25 * (i - 1.5)  # spread the methods so their error bars stay apart
            ax.errorbar(np.array([8, 16, 22]) + dodge, [100 * a.mean() for a in accs],
                        yerr=[100 * a.std() / np.sqrt(len(a)) for a in accs], marker="o", capsize=3,
                        color=METHOD_COLOR[method], label=METHOD_LABEL[method])
        ax.set(xticks=[8, 16, 22], xlabel="EEG electrodes (22 = all; chance = 25%)", title=f"{prec.upper()} model")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("session-2 accuracy, %\n(mean ± SE over 9 participants)")
    axes[0].legend(fontsize="x-small", loc="lower right")
    fig.savefig(out / "fig_accuracy_vs_electrodes.png", dpi=150)
    plt.close(fig)


def fig_forest(comp: pd.DataFrame, out: Path) -> None:
    sel = comp[comp["family"].str.startswith("proposed vs")].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(sel) + 1.2), constrained_layout=True)
    y = np.arange(len(sel))[::-1]
    for yi, (_, r) in zip(y, sel.iterrows()):
        other = r["family"].split(" vs ")[1]
        ax.errorbar(100 * r["mean_diff"], yi, xerr=[[100 * (r["mean_diff"] - r["ci_low"])],
                                                    [100 * (r["ci_high"] - r["mean_diff"])]],
                    fmt="o" if r["precision"] == "fp32" else "s", color=METHOD_COLOR[other], capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, [f"{r['electrodes']} el. {r['precision'].upper()}: proposed − {r['family'].split(' vs ')[1]}"
                      for _, r in sel.iterrows()], fontsize="small")
    ax.set(xlabel="paired accuracy difference (percentage points)\nmean and 95% participant bootstrap interval",
           title="Redundancy-aware selection against its controls, session 2")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(out / "fig_forest.png", dpi=150)
    plt.close(fig)


ELECTRODE_COLOR = {22: "#404040", 16: "#2e75b6", 8: "#c55a11"}


def fig_pareto(pp: pd.DataFrame, bench: dict, final_cfg: dict, out: Path) -> None:
    """Accuracy against parameter bytes and measured latency, one point per model and precision."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    seen = set()
    for r in bench["rows"]:
        cond, prec = r["condition"], r["precision"]
        if cond == "proposed-16":  # same architecture as the 16-electrode pruning base
            continue
        info = parse_condition(cond)
        acc = 100 * accuracies(pp, cond, prec).mean()
        final = (cond, prec) == (final_cfg["condition"], final_cfg["precision"])
        for ax, x in zip(axes, (r["parameter_bytes"] / 1024, r["latency_median_ms"])):
            ax.scatter(x, acc, marker="o" if prec == "fp32" else "s", s=150 if final else 70,
                       color=ELECTRODE_COLOR[info["electrodes"]], edgecolor="black", lw=1.6 if final else 0.5,
                       alpha=1 if info["pruned"] == 0 else 0.55, zorder=3)
            if final or (cond, prec) == (FULL, "fp32") or (cond == "proposed-8" and prec == "fp32"):
                label = "final configuration" if final else f"{cond} {prec.upper()}"
                ax.annotate(label, (x, acc), fontsize="x-small", xytext=(8, -12 if final else 6),
                            textcoords="offset points")
        seen.add(info["electrodes"])
    for k in sorted(seen, reverse=True):
        axes[0].scatter([], [], color=ELECTRODE_COLOR[k], label=f"{k} electrodes")
    axes[0].scatter([], [], marker="o", color="white", edgecolor="black", label="FP32")
    axes[0].scatter([], [], marker="s", color="white", edgecolor="black", label="INT8")
    axes[0].scatter([], [], marker="o", color="grey", alpha=0.55, label="pruned (25% or 50%)")
    axes[0].legend(fontsize="x-small", loc="center")
    axes[0].set(xlabel="parameter bytes in the ONNX file (KiB)", ylabel="session-2 accuracy, % (mean of 9)")
    axes[1].set(xlabel="median CPU latency per decision, batch 1, one thread (ms)")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Fewer electrodes cut latency; INT8 and pruning cut bytes, not time", fontsize="medium")
    fig.savefig(out / "fig_pareto.png", dpi=150)
    plt.close(fig)


def electrode_xy() -> np.ndarray:
    """2-D scalp positions of the 22 electrodes (MNE's topomap projection of standard 10-20)."""
    from mne.channels.layout import _find_topomap_coords

    info = mne.create_info(data.EEG_CHANNELS, data.SFREQ, "eeg")
    info.set_montage("standard_1020")
    return _find_topomap_coords(info, picks=list(range(len(data.EEG_CHANNELS))))


def electrode_map(ax, xy: np.ndarray, values: np.ndarray, cmap: str, title: str):
    """One dot per electrode coloured by its value: no interpolation between electrodes."""
    r = 0.084
    ax.add_patch(plt.Circle((0, 0), r, fill=False, lw=1.2, color="#555555"))
    ax.plot([-0.011, 0, 0.011], [r * 0.99, r + 0.012, r * 0.99], color="#555555", lw=1.2)
    dots = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap, vmin=0, vmax=1, s=430, edgecolors="#555555",
                      linewidths=0.6, zorder=3)
    for (x, y), name, v in zip(xy, data.EEG_CHANNELS, values):
        ax.text(x, y, name, ha="center", va="center", fontsize="xx-small", zorder=4,
                color="white" if v > 0.6 else "#222222")
    ax.set(xlim=(-0.09, 0.09), ylim=(-0.088, 0.1), aspect="equal")
    ax.axis("off")
    ax.set_title(title, fontsize="small")
    return dots


def fig_selection_maps(out: Path) -> None:
    xy = electrode_xy()
    rankings = [json.loads(p.read_text()) for p in sorted((RESULTS_DIR / "final" / "rankings").glob("sub-*_seed-0.json"))]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2), constrained_layout=True)
    for row, k in enumerate((8, 16)):
        for col, method in enumerate(("proposed", "relevance", "weightnorm")):
            freq = np.zeros(len(data.EEG_CHANNELS))
            for r in rankings:
                freq[r["subsets"][f"{method}-{k}"]] += 1
            dots = electrode_map(axes[row, col], xy, freq / len(rankings), "Oranges",
                                 f"{METHOD_LABEL[method]}\n{k} electrodes")
    relevance = np.mean([r["relevance_scores"] for r in rankings], axis=0)
    electrode_map(axes[0, 3], xy, relevance / relevance.max(), "Purples", "mean relevance\n(Fisher score, scaled)")
    axes[1, 3].axis("off")
    fig.colorbar(dots, ax=axes[:, :3], shrink=0.6, label="share of the 9 participants choosing the electrode")
    fig.savefig(out / "fig_selection_maps.png", dpi=150)
    plt.close(fig)


def fig_stability(stab: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    groups = [(m, k) for k in (8, 16) for m in ("proposed", "relevance", "weightnorm")]
    for i, (method, k) in enumerate(groups):
        vals = stab[(stab["method"] == method) & (stab["electrodes"] == k)]["mean_jaccard"]
        ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
                   boxprops={"facecolor": METHOD_COLOR[method], "alpha": 0.45}, medianprops={"color": "black"})
        ax.scatter(i + np.random.default_rng(i).uniform(-0.12, 0.12, len(vals)), vals, s=14, color="black", zorder=3)
    ax.set_xticks(range(len(groups)), [f"{m}\n{k} el." for m, k in groups], fontsize="small")
    ax.set(ylabel="mean pairwise Jaccard overlap", ylim=(0, 1.05),
           title="Selection stability across 3 validation folds and all of session 1")
    fig.savefig(out / "fig_selection_stability.png", dpi=150)
    plt.close(fig)


def fig_participants(pp: pd.DataFrame, final_cfg: dict, out: Path) -> None:
    shown = list(dict.fromkeys([(FULL, "fp32"), ("proposed-16", "fp32"), ("proposed-8", "fp32"),
                                ("relevance-8", "fp32"), ("weightnorm-8", "fp32"), ("random-8", "fp32"),
                                (final_cfg["condition"], final_cfg["precision"])]))
    fig, ax = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
    for subject in data.SUBJECTS:
        ys = [accuracies(pp, c, p).get(subject, np.nan) for c, p in shown]
        ax.plot(range(len(shown)), ys, marker="o", lw=1, alpha=0.85, label=f"P{subject}")
    ax.axhline(0.25, color="grey", lw=0.8, ls="--")
    ax.set_xticks(range(len(shown)), [f"{c}\n{p.upper()}" for c, p in shown], fontsize="x-small")
    ax.set(ylabel="session-2 accuracy\n(mean of 3 seeds)", title="Accuracy per participant")
    ax.legend(fontsize="x-small", ncol=1, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.savefig(out / "fig_participants.png", dpi=150)
    plt.close(fig)


def fig_confusion(raw: pd.DataFrame, final_cfg: dict, out: Path) -> None:
    pairs = list(dict.fromkeys([(FULL, "fp32"), (final_cfg["condition"], final_cfg["precision"])]))
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.4), constrained_layout=True, squeeze=False)
    for ax, (cond, prec) in zip(axes[0], pairs):
        cm = np.sum(raw[(raw["condition"] == cond) & (raw["precision"] == prec)]["confusion"].to_list(), axis=0)
        cm = cm / cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize="small",
                        color="white" if cm[i, j] > 0.5 else "black")
        ax.set_xticks(range(4), CLASS_LABELS, fontsize="x-small")
        ax.set_yticks(range(4), CLASS_LABELS, fontsize="x-small")
        ax.set(xlabel="decoded", ylabel="cued", title=f"{cond} {prec.upper()}")
    fig.savefig(out / "fig_confusion.png", dpi=150)
    plt.close(fig)


LAYER_GROUPS = ["temporal conv", "spatial conv", "separable depthwise", "separable pointwise", "everything else"]


def layer_times(profile: dict[str, float]) -> dict[str, float]:
    """Group ONNX Runtime node times by EEGNet layer. The exporter numbers the four
    convolutions in graph order (node_Conv_90, _92, ...), in FP32 and INT8 alike."""
    convs = sorted({int(m.group(1)) for k in profile if (m := re.search(r"node_Conv_(\d+)", k))})
    names = dict(zip(convs, LAYER_GROUPS[:4]))
    out = dict.fromkeys(LAYER_GROUPS, 0.0)
    for key, us in profile.items():
        m = re.search(r"node_Conv_(\d+)", key)
        out[names[int(m.group(1))] if m else "everything else"] += us
    return out


def fig_costs(bench: dict, reduced_base: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    names = None
    for offset, cond in ((-0.2, FULL), (0.2, reduced_base)):
        layers = bench["layer_costs"][cond]
        names = [r["layer"].replace("final_layer.linearconstraint", "dense").replace("conv_separable_", "sep_")
                 for r in layers]
        x = np.arange(len(layers)) + offset
        color = METHOD_COLOR[method_of(cond)]
        axes[0].bar(x, [r["macs"] / 1e6 for r in layers], width=0.4, label=cond, color=color)
        axes[1].bar(x, [4 * (r["in_elems"] + r["out_elems"]) / 1024 for r in layers], width=0.4, label=cond,
                    color=color)
    for ax, label, title in zip(axes[:2], ("MACs (millions)", "input + output tensors, FP32 (KiB)"),
                                ("Compute per layer", "Live activation memory per layer")):
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize="x-small")
        ax.set(ylabel=label, title=title)
        ax.legend(fontsize="x-small")
    times = {prec: layer_times(bench["op_profile_us"][f"{FULL} {prec}"]) for prec in ("fp32", "int8")}
    x = np.arange(len(LAYER_GROUPS))
    for offset, prec, color in ((-0.2, "fp32", "#9e9e9e"), (0.2, "int8", "#2e75b6")):
        vals = [times[prec][g] for g in LAYER_GROUPS]
        axes[2].bar(x + offset, vals, width=0.4, color=color, label=f"{prec.upper()}, total {sum(vals):.0f} µs")
    axes[2].set_xticks(x, LAYER_GROUPS, rotation=30, ha="right", fontsize="x-small")
    axes[2].set(ylabel="kernel time per inference (µs)", title=f"CPU kernel time by layer, {FULL}")
    axes[2].legend(fontsize="x-small")
    fig.savefig(out / "fig_costs.png", dpi=150)
    plt.close(fig)


def fig_int8_layers(bench: dict, out: Path) -> None:
    """CPU kernel time per layer, FP32 against INT8, for 22 and 8 electrodes."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, constrained_layout=True)
    x = np.arange(len(LAYER_GROUPS))
    for ax, cond in zip(axes, (FULL, "proposed-8")):
        for offset, prec, color in ((-0.2, "fp32", "#9e9e9e"), (0.2, "int8", "#2e75b6")):
            vals = [layer_times(bench["op_profile_us"][f"{cond} {prec}"])[g] for g in LAYER_GROUPS]
            ax.bar(x + offset, vals, width=0.4, color=color, label=f"{prec.upper()}: {sum(vals):.0f} µs in total")
        ax.set_xticks(x, [g.replace("separable ", "sep. ") for g in LAYER_GROUPS], rotation=25, ha="right",
                      fontsize="small")
        ax.set_title(f"{parse_condition(cond)['electrodes']} electrodes")
        ax.legend(fontsize="small")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("CPU kernel time per decision (µs)")
    fig.savefig(out / "fig_int8_layers.png", dpi=150)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, cols: list[str], pct: list[str]) -> str:
    out = df[cols].copy()
    for c in pct:
        out[c] = (100 * out[c]).map(lambda v: f"{v:.1f}")
    return out.to_markdown(index=False)


def main() -> None:
    frozen = json.loads((RESULTS_DIR / "frozen_config.json").read_text())
    plan, final_cfg = frozen["uncertainty"], frozen["final_configuration"]
    reduced_base = frozen["pruning"]["bases"][1]
    bench = json.loads((RESULTS_DIR / "benchmark" / "benchmark.json").read_text())

    raw = load_test("primary")
    pp = per_participant(raw)
    table = condition_table(pp, plan)
    secondary = condition_table(per_participant(load_test("secondary")), plan)
    comp = comparisons(pp, plan, reduced_base)
    stab = selection_stability()
    same = proposed_equals_relevance()
    eff = pd.DataFrame([{k: v for k, v in r.items() if k != "executed_ops"} for r in bench["rows"]])

    (OUT / "slides").mkdir(parents=True, exist_ok=True)
    pp.to_csv(OUT / "per_participant_primary.csv", index=False, float_format="%.4f")
    table.to_csv(OUT / "conditions_primary.csv", index=False, float_format="%.4f")
    secondary.to_csv(OUT / "conditions_secondary.csv", index=False, float_format="%.4f")
    comp.to_csv(OUT / "comparisons_primary.csv", index=False, float_format="%.4f")
    stab.to_csv(OUT / "selection_stability.csv", index=False, float_format="%.4f")
    eff.to_csv(OUT / "efficiency.csv", index=False, float_format="%.4f")
    # Small chart-ready tables for the report's native charts.
    acc_rows = []
    for prec in ("fp32", "int8"):
        for method in METHODS + ("full",):
            for k in ((22,) if method == "full" else (8, 16)):
                a = accuracies(pp, FULL if method == "full" else f"{method}-{k}", prec)
                acc_rows.append({"precision": prec.upper(), "method": METHOD_LABEL[method], "electrodes": k,
                                 "mean_pct": round(100 * a.mean(), 2), "se_pct": round(100 * a.std() / np.sqrt(len(a)), 2)})
    pd.DataFrame(acc_rows).to_csv(OUT / "chart_accuracy.csv", index=False)
    forest = comp[comp["family"].str.startswith("proposed vs")]
    pd.DataFrame({"control": forest["family"].str.split(" vs ").str[1], "electrodes": forest["electrodes"],
                  "precision": forest["precision"].str.upper(), "diff_pct": (100 * forest["mean_diff"]).round(2),
                  "ci_low_pct": (100 * forest["ci_low"]).round(2), "ci_high_pct": (100 * forest["ci_high"]).round(2)}
                 ).to_csv(OUT / "chart_forest.csv", index=False)

    for out, rc in ((OUT, {}), (OUT / "slides", SLIDE_RC)):
        with plt.rc_context(rc):
            fig_accuracy_vs_electrodes(pp, out)
            fig_forest(comp, out)
            fig_pareto(pp, bench, final_cfg, out)
            fig_selection_maps(out)
            fig_stability(stab, out)
            fig_participants(pp, final_cfg, out)
            fig_confusion(raw, final_cfg, out)
            fig_costs(bench, reduced_base, out)
            fig_int8_layers(bench, out)

    final_row = table[(table["condition"] == final_cfg["condition"]) & (table["precision"] == final_cfg["precision"])]
    md = [
        "# Session-2 results (generated by scripts/analyze.py)",
        f"Final configuration chosen on validation data: **{final_cfg['condition']} {final_cfg['precision'].upper()}**.",
        f"Proposed and relevance-only subsets identical (final rankings + validation folds): {same}.",
        "",
        "## Conditions, primary analysis (artifact-free trials; % accuracy)",
        markdown_table(table, ["condition", "precision", "acc_mean", "acc_sd", "acc_min", "f1_mean",
                               "diff_vs_full_fp32", "diff_ci_low", "diff_ci_high", "n_losing_more_than_2pts"],
                       ["acc_mean", "acc_sd", "acc_min", "f1_mean", "diff_vs_full_fp32", "diff_ci_low", "diff_ci_high"]),
        "",
        "## Final configuration",
        markdown_table(final_row, ["condition", "precision", "acc_mean", "acc_sd", "diff_vs_full_fp32",
                                   "n_losing_more_than_2pts"], ["acc_mean", "acc_sd", "diff_vs_full_fp32"]),
        "",
        "## Prespecified paired comparisons (percentage points)",
        markdown_table(comp, ["family", "a", "b", "mean_diff", "ci_low", "ci_high", "n_a_better", "n_b_better",
                              "n_tied"], ["mean_diff", "ci_low", "ci_high"]),
        "",
        "## Secondary analysis (all trials, flagged ones included)",
        markdown_table(secondary, ["condition", "precision", "acc_mean", "acc_sd", "diff_vs_full_fp32"],
                       ["acc_mean", "acc_sd", "diff_vs_full_fp32"]),
        "",
        "## Selection stability (mean pairwise Jaccard)",
        stab.groupby(["method", "electrodes"])["mean_jaccard"].agg(["mean", "min", "max"]).round(3).to_markdown(),
        "",
        "## Efficiency on this CPU (ONNX Runtime, batch 1, 1 thread)",
        eff[["condition", "precision", "electrodes", "F2", "params", "parameter_bytes", "file_bytes", "macs",
             "input_bytes", "peak_activation_bytes", "latency_median_ms", "latency_p95_ms"]].to_markdown(index=False),
        "",
        "Executed ops: " + "; ".join(f"{r['condition']} {r['precision']}: {r['executed_ops']}" for r in bench["rows"]),
        "",
        f"Preprocessing per 2-s window (median ms by electrode count): "
        f"{ {k: round(v['median_ms'], 4) for k, v in bench['preprocessing'].items()} }",
    ]
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:9]))


if __name__ == "__main__":
    main()
