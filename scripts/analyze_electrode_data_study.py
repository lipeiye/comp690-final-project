"""Curves, peaks and tests for the electrode-count x training-data study (scripts/electrode_data_study.py).

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only, never EEG:
  supplementary/electrode_data_study/{protocol.json,runs,test}   relevance-K models at every data amount
  supplementary/overfit_study/{protocol.json,runs,test}          full and reduced montages at the same splits
                                                                 (BCI IV 2a 22 and 16, Lee2019 62 and 45)

Writes to RESULTS_DIR/supplementary/electrode_data_study/: curves.csv, peaks.csv, fig_curves_by_data.png,
summary.md and analysis.json.

Per participant, a count's accuracy is the mean over seeds and over the folds of one data amount; the
curve is the mean over participants. The overfit study's fold models use seeds 0-9; only the seeds
this study trained at that amount (BCI IV 2a folds: 0-4) enter its points, so every point of a curve
averages the same seeds. The best count's uncertainty is a participant bootstrap of the argmax
(10,000 resamples, seed 690). Lee2019 tests D1 and D2 are the plan's; BCI IV 2a is exploratory.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mibci.artifacts import write_json
from mibci.paths import RESULTS_DIR
from mibci.stats import FROZEN_BOOTSTRAP, bootstrap_ci

OUT = RESULTS_DIR / "supplementary" / "electrode_data_study"
OVERFIT = RESULTS_DIR / "supplementary" / "overfit_study"
FULL = {"bnci2a": 22, "lee2019": 62}
REUSED = {"bnci2a": {"full-22": 22, "relevance-16": 16}, "lee2019": {"full-62": 62, "relevance-45": 45}}
LEVEL_NAMES = {
    "bnci2a": {"final": "6 of 6 runs", "fold": "4 of 6 runs", "rfold": "2 of 6 runs"},
    "lee2019": {"final": "2 of 2 runs", "fold": "1 of 2 runs"},
}


def records_json(df: pd.DataFrame) -> list[dict]:
    """Rows as dicts with NaN written as null (write_json refuses NaN)."""
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def records(folder, jobs, keep) -> list[dict]:
    rows, missing = [], 0
    for job in jobs:
        if not keep(job):
            continue
        run, test = folder / "runs" / f"{job['name']}.json", folder / "test" / f"{job['name']}.json"
        if not (run.exists() and test.exists()):
            missing += 1
            continue
        r, t = json.loads(run.read_text()), json.loads(test.read_text())
        rows.append(
            {
                "dataset": job["dataset"],
                "subject": job["subject"],
                "seed": job["seed"],
                "split": job["split"],
                "level": job["split"].split("-")[0],
                "condition": job["condition"],
                "within": 100 * r["within_day"]["acc"] if "within_day" in r else np.nan,
                "next_day": 100 * t["primary"]["acc"],
            }
        )
    if missing:
        raise SystemExit(f"{folder.name}: {missing} jobs lack a run or test record")
    return rows


def load() -> pd.DataFrame:
    plan = json.loads((OUT / "protocol.json").read_text())
    seeds = {
        (d, lvl): set(spec["final_seeds"] if lvl == "final" else spec["fold_seeds"])
        for d, spec in plan["datasets"].items()
        for lvl in ("final", "fold", "rfold")
    }
    rows = records(OUT, plan["jobs"], lambda j: True)
    overfit = json.loads((OVERFIT / "protocol.json").read_text())["jobs"]
    rows += records(
        OVERFIT,
        overfit,
        lambda j: (
            j["condition"] in REUSED[j["dataset"]] and j["seed"] in seeds[(j["dataset"], j["split"].split("-")[0])]
        ),
    )
    df = pd.DataFrame(rows)
    df["electrodes"] = [REUSED[d].get(c) or int(c.split("-")[1]) for d, c in zip(df.dataset, df.condition, strict=True)]
    return df


def argmax_bootstrap(table: pd.DataFrame) -> np.ndarray:
    """Best electrode count in each participant bootstrap resample (rows participants, columns counts)."""
    rng = np.random.default_rng(FROZEN_BOOTSTRAP["seed"])
    values, counts = table.to_numpy(), table.columns.to_numpy()
    idx = rng.integers(0, len(values), size=(FROZEN_BOOTSTRAP["resamples"], len(values)))
    return counts[values[idx].mean(1).argmax(1)]


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    df = load()
    pp = df.groupby(["dataset", "level", "electrodes", "subject"])[["within", "next_day"]].mean().reset_index()
    curves = pp.groupby(["dataset", "level", "electrodes"])[["within", "next_day"]].mean().reset_index()
    curves.to_csv(OUT / "curves.csv", index=False, float_format="%.3f")
    peaks, boot = [], {}
    for (dataset, level), g in pp.groupby(["dataset", "level"]):
        table = g.pivot(index="subject", columns="electrodes", values="next_day").dropna(axis=1)
        best = argmax_bootstrap(table)
        boot[(dataset, level)] = (table, best)
        counts, share = np.unique(best, return_counts=True)
        mean_curve = table.mean()
        peaks.append(
            {
                "dataset": dataset,
                "level": level,
                "data": LEVEL_NAMES[dataset][level],
                "best_count": int(mean_curve.idxmax()),
                "best_accuracy": float(mean_curve.max()),
                "full_accuracy": float(mean_curve[FULL[dataset]]),
                "bootstrap_best": {int(c): float(s / len(best)) for c, s in zip(counts, share, strict=True)},
            }
        )
    peaks_df = pd.DataFrame(peaks)
    peaks_df.to_csv(OUT / "peaks.csv", index=False)

    tests, gaps = {}, []
    for dataset, small in (("lee2019", 16), ("bnci2a", 16)):
        full = FULL[dataset]
        t_final = boot[(dataset, "final")][0]
        for level in [lv for lv in LEVEL_NAMES[dataset] if lv != "final"]:
            t_lvl = boot[(dataset, level)][0]
            d = ((t_lvl[small] - t_lvl[full]) - (t_final[small] - t_final[full])).dropna().to_numpy()
            lo, hi = bootstrap_ci(d)
            gaps.append(
                {
                    "dataset": dataset,
                    "level": level,
                    "contrast": f"({small} minus {full}) with {LEVEL_NAMES[dataset][level]} minus the same with all",
                    "mean": float(d.mean()),
                    "ci": [lo, hi],
                }
            )
    lee_d1 = next(g for g in gaps if g["dataset"] == "lee2019" and g["level"] == "fold")
    tests["D1"] = {**lee_d1, "rule": "interval lower bound above 0", "holds": bool(lee_d1["ci"][0] > 0)}
    # D2: paired argmax on the same resampled participants
    t_fold, t_final = boot[("lee2019", "fold")][0], boot[("lee2019", "final")][0]
    common = t_fold.columns.intersection(t_final.columns)
    rng = np.random.default_rng(FROZEN_BOOTSTRAP["seed"])
    idx = rng.integers(0, len(t_fold), size=(FROZEN_BOOTSTRAP["resamples"], len(t_fold)))
    a_fold = common.to_numpy()[t_fold[common].to_numpy()[idx].mean(1).argmax(1)]
    a_final = common.to_numpy()[t_final.loc[t_fold.index, common].to_numpy()[idx].mean(1).argmax(1)]
    share = float((a_fold < a_final).mean())
    tests["D2"] = {
        "argmax_fold": int(t_fold[common].mean().idxmax()),
        "argmax_final": int(t_final[common].mean().idxmax()),
        "share_lower": share,
        "rule": "argmax(fold) < argmax(final) in at least 95% of resamples",
        "holds": share >= 0.95,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = {"final": "#13294B", "fold": "#4B9CD3", "rfold": "#9ec5e8"}
    for ax, dataset in zip(axes, ("bnci2a", "lee2019"), strict=True):
        for level, name in LEVEL_NAMES[dataset].items():
            c = curves[(curves.dataset == dataset) & (curves.level == level)].sort_values("electrodes")
            ax.plot(c.electrodes, c.next_day, "o-", color=colors[level], label=f"trained on {name}")
            best = c.loc[c.next_day.idxmax()]
            ax.plot(best.electrodes, best.next_day, "*", color="#C8102E", ms=14)
        ax.set_xlabel("electrodes kept (relevance order, ranked on the training data)")
        ax.set_ylabel("next-day accuracy %")
        ax.set_title({"bnci2a": "BCI IV 2a (9 people)", "lee2019": "Lee 2019 (54 people)"}[dataset])
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_curves_by_data.png", dpi=200)
    plt.close(fig)

    write_json(
        OUT / "analysis.json",
        {"tests_lee2019": tests, "gap_changes": gaps, "peaks": peaks, "curves": records_json(curves)},
    )
    lines = [
        "# Electrode count x training data",
        "",
        "## Best count at each amount of training data",
        "",
        "| dataset | training data | best count | its accuracy % | all electrodes % | bootstrap share of best counts |",
        "|---|---|---:|---:|---:|---|",
    ]
    for p in peaks:
        shares = ", ".join(f"{k}: {v:.0%}" for k, v in sorted(p["bootstrap_best"].items(), key=lambda kv: -kv[1])[:4])
        lines.append(
            f"| {p['dataset']} | {p['data']} | {p['best_count']} | {p['best_accuracy']:.2f} | "
            f"{p['full_accuracy']:.2f} | {shares} |"
        )
    lines += ["", "## Gap changes", "", "| dataset | contrast | mean | 95% interval |", "|---|---|---:|---|"]
    lines += [
        f"| {g['dataset']} | {g['contrast']} | {g['mean']:+.2f} | [{g['ci'][0]:+.2f}, {g['ci'][1]:+.2f}] |"
        for g in gaps
    ]
    lines += [
        "",
        "## Lee2019 tests",
        "",
        f"- D1: {tests['D1']['mean']:+.2f} [{tests['D1']['ci'][0]:+.2f}, "
        f"{tests['D1']['ci'][1]:+.2f}]: {'holds' if tests['D1']['holds'] else 'does not hold'}",
        f"- D2: best count {tests['D2']['argmax_fold']} with half the data, {tests['D2']['argmax_final']} with "
        f"all; lower in {tests['D2']['share_lower']:.0%} of resamples: "
        f"{'holds' if tests['D2']['holds'] else 'does not hold'}",
        "",
        "## Curves (next-day accuracy %)",
        "",
        "| dataset | training data | " + " | ".join(f"{k}" for k in sorted(curves.electrodes.unique())) + " |",
    ]
    counts = sorted(curves.electrodes.unique())
    lines.append("|---|---|" + "---:|" * len(counts))
    for (dataset, level), g in curves.groupby(["dataset", "level"]):
        g = g.set_index("electrodes")["next_day"]
        lines.append(
            f"| {dataset} | {LEVEL_NAMES[dataset][level]} | "
            + " | ".join(f"{g[k]:.2f}" if k in g else "" for k in counts)
            + " |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
