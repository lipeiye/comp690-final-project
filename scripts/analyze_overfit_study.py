"""Tables, tests and figures for the overfitting study (scripts/overfit_study.py).

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only, never EEG:
  supplementary/overfit_study/protocol.json      the plan: predictions P1-P4, datasets, jobs
  supplementary/overfit_study/runs/<job>.json    training-set accuracy, and held-out runs of the training day
  supplementary/overfit_study/test/<job>.json    next-day (session 2) accuracy

Writes to RESULTS_DIR/supplementary/overfit_study/: per_participant.csv, contrasts.csv, levels.csv,
fig_overfit_contrasts.png, fig_overfit_levels.png, summary.md and analysis.json.

Every model's accuracy is averaged over seeds (and over the folds of one data amount) within a
participant; contrasts are paired over participants with the frozen bootstrap (mibci.stats). For
BCI IV 2a the seed error of each contrast is also given (the contrast computed seed by seed, its
standard deviation over seeds divided by the square root of their number). Lee2019 tests are the
plan's; BCI IV 2a is exploratory.
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
from mibci.stats import bootstrap_ci

OUT = RESULTS_DIR / "supplementary" / "overfit_study"
MONTAGES = {"bnci2a": ("full-22", "relevance-16"), "lee2019": ("full-62", "relevance-45")}
LEVEL_NAMES = {
    "bnci2a": {"final": "6 of 6 runs", "fold": "4 of 6 runs", "rfold": "2 of 6 runs"},
    "lee2019": {"final": "2 of 2 runs", "fold": "1 of 2 runs"},
}


def records_json(df: pd.DataFrame) -> list[dict]:
    """Rows as dicts with NaN written as null (write_json refuses NaN)."""
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def load() -> pd.DataFrame:
    plan = json.loads((OUT / "protocol.json").read_text())
    rows, missing = [], 0
    for job in plan["jobs"]:
        run, test = OUT / "runs" / f"{job['name']}.json", OUT / "test" / f"{job['name']}.json"
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
                "n_train": r["n_train"],
                "train": 100 * r["train_acc"],
                "within": 100 * r["within_day"]["acc"] if "within_day" in r else np.nan,
                "next_day": 100 * t["primary"]["acc"],
            }
        )
    if missing:
        raise SystemExit(f"{missing} of {len(plan['jobs'])} jobs lack a run or test record")
    return pd.DataFrame(rows)


def per_participant(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "level", "condition", "subject"])[["train", "within", "next_day"]].mean().reset_index()
    )


def table(pp: pd.DataFrame, dataset: str, level: str, condition: str, column: str = "next_day") -> pd.Series:
    sel = pp[(pp.dataset == dataset) & (pp.level == level) & (pp.condition == condition)]
    return sel.set_index("subject")[column]


def seed_table(df: pd.DataFrame, dataset: str, level: str, condition: str, column: str = "next_day") -> pd.DataFrame:
    sel = df[(df.dataset == dataset) & (df.level == level) & (df.condition == condition)]
    return sel.groupby(["seed", "subject"])[column].mean().unstack("subject")


def contrasts(df: pd.DataFrame, pp: pd.DataFrame) -> list[dict]:
    out = []
    for dataset, (full, red) in MONTAGES.items():
        levels = list(LEVEL_NAMES[dataset])

        def T(level, cond, col="next_day", dataset=dataset):
            return table(pp, dataset, level, cond, col)

        def S(level, cond, col="next_day", dataset=dataset):
            return seed_table(df, dataset, level, cond, col)

        items = {
            "reduced minus full, next day, all of the training day": (
                T("final", red) - T("final", full),
                S("final", red) - S("final", full),
            ),
        }
        for level in levels[1:]:
            items[f"reduced minus full, next day, {LEVEL_NAMES[dataset][level]}"] = (
                T(level, red) - T(level, full),
                S(level, red) - S(level, full),
            )
            items[f"P1: gap with {LEVEL_NAMES[dataset][level]} minus gap with all"] = (
                (T(level, red) - T(level, full)) - (T("final", red) - T("final", full)),
                (S(level, red) - S(level, full)) - (S("final", red) - S("final", full)),
            )
            items[f"P2: reduced minus full, held-out runs of the training day, {LEVEL_NAMES[dataset][level]}"] = (
                T(level, red, "within") - T(level, full, "within"),
                S(level, red, "within") - S(level, full, "within"),
            )
            items[f"next-day gap minus within-day gap, {LEVEL_NAMES[dataset][level]}"] = (
                (T(level, red) - T(level, full)) - (T(level, red, "within") - T(level, full, "within")),
                (S(level, red) - S(level, full)) - (S(level, red, "within") - S(level, full, "within")),
            )
        items |= {
            "P3: reduced plus noise minus reduced": (
                T("final", f"{red}-plus-noise") - T("final", red),
                S("final", f"{red}-plus-noise") - S("final", red),
            ),
            "information in the dropped electrodes: full minus reduced plus noise": (
                T("final", full) - T("final", f"{red}-plus-noise"),
                S("final", full) - S("final", f"{red}-plus-noise"),
            ),
            "reduced plus noise minus full": (
                T("final", f"{red}-plus-noise") - T("final", full),
                S("final", f"{red}-plus-noise") - S("final", full),
            ),
            "reduced plus shuffled minus reduced": (
                T("final", f"{red}-plus-shuffled") - T("final", red),
                S("final", f"{red}-plus-shuffled") - S("final", red),
            ),
            "reduced plus shuffled minus full": (
                T("final", f"{red}-plus-shuffled") - T("final", full),
                S("final", f"{red}-plus-shuffled") - S("final", full),
            ),
            "full with channel dropout minus full": (
                T("final", f"{full}-chdrop") - T("final", full),
                S("final", f"{full}-chdrop") - S("final", full),
            ),
            "reduced with channel dropout minus reduced": (
                T("final", f"{red}-chdrop") - T("final", red),
                S("final", f"{red}-chdrop") - S("final", red),
            ),
            "P4: dropout gain of full minus dropout gain of reduced": (
                (T("final", f"{full}-chdrop") - T("final", full)) - (T("final", f"{red}-chdrop") - T("final", red)),
                (S("final", f"{full}-chdrop") - S("final", full)) - (S("final", f"{red}-chdrop") - S("final", red)),
            ),
            "full with channel dropout minus reduced": (
                T("final", f"{full}-chdrop") - T("final", red),
                S("final", f"{full}-chdrop") - S("final", red),
            ),
        }
        for name, (per_person, per_seed) in items.items():
            d = per_person.dropna().to_numpy()
            lo, hi = bootstrap_ci(d)
            seed_means = per_seed.mean(axis=1).dropna().to_numpy()
            seed_error = float(seed_means.std(ddof=1) / np.sqrt(len(seed_means))) if len(seed_means) > 1 else np.nan
            out.append(
                {
                    "dataset": dataset,
                    "contrast": name,
                    "mean": float(d.mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                    "seed_error": seed_error,
                    "better": int((d > 0).sum()),
                    "worse": int((d < 0).sum()),
                    "n": len(d),
                }
            )
    return out


TESTS = {  # prediction -> (contrast name prefix, rule) for Lee2019, as in the plan
    "P1": ("P1: gap with 1 of 2 runs minus gap with all", "low>0"),
    "P2": ("P2: reduced minus full, held-out runs of the training day, 1 of 2 runs", "low>0"),
    "P3": ("P3: reduced plus noise minus reduced", "high<0"),
    "P4": ("P4: dropout gain of full minus dropout gain of reduced", "low>0"),
}


def figure_contrasts(rows: pd.DataFrame) -> None:
    keep = [c for c in rows.contrast.unique() if not c.startswith("next-day gap")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.42 * len(keep) + 1.4), sharey=False)
    for ax, dataset in zip(axes, MONTAGES, strict=True):
        sel = rows[(rows.dataset == dataset) & rows.contrast.isin(keep)].reset_index(drop=True)
        y = np.arange(len(sel))[::-1]
        ax.errorbar(
            sel["mean"],
            y,
            xerr=[sel["mean"] - sel["ci_low"], sel["ci_high"] - sel["mean"]],
            fmt="o",
            color="#13294B",
            ecolor="#4B9CD3",
            capsize=3,
        )
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_yticks(y, sel["contrast"], fontsize=7)
        ax.set_xlabel("percentage points (95% participant interval)")
        ax.set_title({"bnci2a": "BCI IV 2a (9 people, exploratory)", "lee2019": "Lee 2019 (54 people)"}[dataset])
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_overfit_contrasts.png", dpi=200)
    plt.close(fig)


def figure_levels(pp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, (full, red) in MONTAGES.items():
        for level in LEVEL_NAMES[dataset]:
            for cond in (full, red):
                g = pp[(pp.dataset == dataset) & (pp.level == level) & (pp.condition == cond)]
                rows.append(
                    {
                        "dataset": dataset,
                        "level": level,
                        "condition": cond,
                        "train": g["train"].mean(),
                        "within": g["within"].mean(),
                        "next_day": g["next_day"].mean(),
                    }
                )
    levels = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, dataset in zip(axes, MONTAGES, strict=True):
        sel = levels[levels.dataset == dataset]
        names = list(LEVEL_NAMES[dataset])[::-1]
        for cond, color in zip(MONTAGES[dataset], ("#7f8fa6", "#13294B"), strict=True):
            g = sel[sel.condition == cond].set_index("level").loc[names]
            x = np.arange(len(names))
            ax.plot(x, g["next_day"], "o-", color=color, label=f"{cond}, next day")
            ax.plot(x, g["within"], "s--", color=color, alpha=0.6, label=f"{cond}, held-out runs, same day")
        ax.set_xticks(np.arange(len(names)), [LEVEL_NAMES[dataset][n] for n in names])
        ax.set_xlabel("training data (runs of the training day)")
        ax.set_ylabel("accuracy %")
        ax.set_title(dataset)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_overfit_levels.png", dpi=200)
    plt.close(fig)
    return levels


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    plan = json.loads((OUT / "protocol.json").read_text())
    df = load()
    pp = per_participant(df)
    pp.to_csv(OUT / "per_participant.csv", index=False, float_format="%.3f")
    rows = pd.DataFrame(contrasts(df, pp))
    rows.to_csv(OUT / "contrasts.csv", index=False, float_format="%.3f")
    levels = figure_levels(pp)
    levels.to_csv(OUT / "levels.csv", index=False, float_format="%.3f")
    figure_contrasts(rows)
    lee = rows[rows.dataset == "lee2019"].set_index("contrast")
    tests = {}
    for key, (name, rule) in TESTS.items():
        r = lee.loc[name]
        holds = r.ci_low > 0 if rule == "low>0" else r.ci_high < 0
        tests[key] = {
            "claim": plan["predictions"][key]["claim"],
            "contrast": name,
            "mean": r["mean"],
            "ci": [r.ci_low, r.ci_high],
            "rule": rule,
            "holds": bool(holds),
        }
    write_json(
        OUT / "analysis.json",
        {"tests_lee2019": tests, "contrasts": records_json(rows), "levels": records_json(levels)},
    )
    lines = [
        "# Overfitting study: why can fewer electrodes decode the next day better?",
        "",
        "Accuracy in %, contrasts in percentage points with the 95% participant interval; seed error for BCI IV 2a "
        "(ten seeds). Lee2019 tests were fixed in the plan; BCI IV 2a is exploratory.",
        "",
        "## Lee2019 tests (54 people, seeds 0-2)",
        "",
        "| prediction | contrast | mean | 95% interval | holds |",
        "|---|---|---:|---|---|",
    ]
    for key, t in tests.items():
        lines.append(
            f"| {key}: {t['claim']} | {t['contrast']} | {t['mean']:+.2f} | [{t['ci'][0]:+.2f}, "
            f"{t['ci'][1]:+.2f}] | {'yes' if t['holds'] else 'no'} |"
        )
    for dataset in MONTAGES:
        lines += [
            "",
            f"## {dataset}: all contrasts",
            "",
            "| contrast | mean | 95% interval | seed error | better/worse |",
            "|---|---:|---|---:|---|",
        ]
        for r in rows[rows.dataset == dataset].itertuples():
            se = f"{r.seed_error:.2f}" if pd.notna(r.seed_error) else "-"
            lines.append(
                f"| {r.contrast} | {r.mean:+.2f} | [{r.ci_low:+.2f}, {r.ci_high:+.2f}] | {se} | {r.better}/{r.worse} |"
            )
        lines += [
            "",
            "| data | montage | training set | held-out runs, same day | next day |",
            "|---|---|---:|---:|---:|",
        ]
        for r in levels[levels.dataset == dataset].itertuples():
            within = f"{r.within:.2f}" if pd.notna(r.within) else "-"
            lines.append(
                f"| {LEVEL_NAMES[dataset][r.level]} | {r.condition} | {r.train:.2f} | {within} | {r.next_day:.2f} |"
            )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
