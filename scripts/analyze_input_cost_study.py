"""Dose-response tables, tests and a figure for the input-cost study (scripts/input_cost_study.py).

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only, never EEG:
  supplementary/input_cost_study/{protocol.json,runs,test}      relevance-16 plus k noise inputs
  supplementary/overfit_study/{protocol.json,runs,test}         BCI IV 2a relevance-16 (k = 0), and relevance-16
                                                                with noise in the six dropped places (another draw)
  supplementary/electrode_data_study/{protocol.json,runs,test}  Lee2019 relevance-16 (k = 0)

Writes to RESULTS_DIR/supplementary/input_cost_study/: curves.csv, costs.csv, fig_input_cost.png,
summary.md and analysis.json.

Per participant, accuracy is the mean over seeds and over the folds of one data amount; k = 0 uses
only the seeds this study trained at that amount. A cost is the accuracy with k added inputs minus the
accuracy with none, paired over participants (frozen bootstrap, mibci.stats), with its seed error (the
cost computed seed by seed, its standard deviation over seeds divided by the square root of their
number). The slope is each participant's least-squares cost per added input on a line through k = 0,
averaged. The share of the margin divides the mean cost by the mean k = 0 accuracy above chance.
Lee2019 tests C1 and C2 are the plan's; BCI IV 2a is exploratory.
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

SUPP = RESULTS_DIR / "supplementary"
OUT = SUPP / "input_cost_study"
BASELINES = {"bnci2a": SUPP / "overfit_study", "lee2019": SUPP / "electrode_data_study"}
CHANCE = {"bnci2a": 25.0, "lee2019": 50.0}
LEVEL_NAMES = {
    "bnci2a": {"final": "6 of 6 runs", "fold": "4 of 6 runs", "rfold": "2 of 6 runs"},
    "lee2019": {"final": "2 of 2 runs", "fold": "1 of 2 runs"},
}
TITLES = {"bnci2a": "BCI IV 2a (9 people, 4 classes)", "lee2019": "Lee 2019 (54 people, 2 classes)"}
COLORS = {"final": "#13294B", "fold": "#4B9CD3", "rfold": "#9ec5e8"}


def records_json(df: pd.DataFrame) -> list[dict]:
    """Rows as dicts with NaN written as null (write_json refuses NaN)."""
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def records(folder, keep) -> list[dict]:
    rows, missing = [], 0
    for job in json.loads((folder / "protocol.json").read_text())["jobs"]:
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
                "level": job["split"].split("-")[0],
                "added": job.get("added", 0),
                "train": 100 * r["train_acc"],
                "within": 100 * r["within_day"]["acc"] if "within_day" in r else np.nan,
                "next_day": 100 * t["primary"]["acc"],
            }
        )
    if missing:
        raise SystemExit(f"{folder.name}: {missing} jobs lack a run or test record")
    return rows


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The dose-response records (k = 0 from the earlier studies) and the overfit study's other noise draw."""
    plan = json.loads((OUT / "protocol.json").read_text())
    seeds = {
        (d, lvl): set(spec["final_seeds"] if lvl == "final" else spec["fold_seeds"])
        for d, spec in plan["datasets"].items()
        for lvl in LEVEL_NAMES[d]
    }
    rows = records(OUT, lambda j: True)
    for dataset, folder in BASELINES.items():
        rows += records(
            folder,
            lambda j, dataset=dataset: (
                j["dataset"] == dataset
                and j["condition"] == "relevance-16"
                and j["seed"] in seeds[(dataset, j["split"].split("-")[0])]
            ),
        )
    other = records(
        BASELINES["bnci2a"], lambda j: j["dataset"] == "bnci2a" and j["condition"] == "relevance-16-plus-noise"
    )
    return pd.DataFrame(rows), pd.DataFrame(other)


def wide(pp: pd.DataFrame, dataset: str, level: str, column: str = "next_day") -> pd.DataFrame:
    """Participants by number of added inputs."""
    sel = pp[(pp.dataset == dataset) & (pp.level == level)]
    return sel.pivot(index="subject", columns="added", values=column)


def seed_error(df: pd.DataFrame, dataset: str, level: str, k: int, column: str = "next_day") -> float:
    sel = df[(df.dataset == dataset) & (df.level == level)]
    per_seed = sel.groupby(["seed", "added", "subject"])[column].mean().unstack("subject")
    d = (per_seed.xs(k, level="added") - per_seed.xs(0, level="added")).mean(axis=1)
    return float(d.std(ddof=1) / np.sqrt(len(d)))


def paired(d: np.ndarray) -> dict:
    lo, hi = bootstrap_ci(d)
    return {"mean": float(d.mean()), "ci": [lo, hi], "worse": int((d < 0).sum()), "people": len(d)}


def cost_table(df: pd.DataFrame, pp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, levels in LEVEL_NAMES.items():
        for level, name in levels.items():
            t, w, tr = (wide(pp, dataset, level, c) for c in ("next_day", "within", "train"))
            margin = t[0].mean() - CHANCE[dataset]
            for k in [c for c in t.columns if c > 0]:
                c = paired((t[k] - t[0]).to_numpy())
                row = {
                    "dataset": dataset,
                    "level": level,
                    "data": name,
                    "added": int(k),
                    "inputs": 16 + int(k),
                    "accuracy": float(t[k].mean()),
                    "cost": c["mean"],
                    "ci_low": c["ci"][0],
                    "ci_high": c["ci"][1],
                    "seed_error": seed_error(df, dataset, level, k),
                    "worse": c["worse"],
                    "people": c["people"],
                    "share_of_margin": float(-c["mean"] / margin),
                    "train_change": float((tr[k] - tr[0]).mean()),
                    "within_cost": np.nan,
                    "within_ci_low": np.nan,
                    "within_ci_high": np.nan,
                }
                if w[k].notna().all():
                    cw = paired((w[k] - w[0]).to_numpy())
                    row |= {"within_cost": cw["mean"], "within_ci_low": cw["ci"][0], "within_ci_high": cw["ci"][1]}
                rows.append(row)
    return pd.DataFrame(rows)


def slopes(pp: pd.DataFrame) -> list[dict]:
    """Each participant's cost per added input (least squares through k = 0), averaged with its interval."""
    out = []
    for dataset, levels in LEVEL_NAMES.items():
        for level, name in levels.items():
            t = wide(pp, dataset, level)
            ks = [c for c in t.columns if c > 0]
            k = np.array(ks, dtype=float)
            per = t[ks].sub(t[0], axis=0).to_numpy() @ k / (k @ k)
            s = paired(per)
            margin = t[0].mean() - CHANCE[dataset]
            out.append(
                {
                    "dataset": dataset,
                    "level": level,
                    "data": name,
                    "slope": s["mean"],
                    "ci": s["ci"],
                    "share_of_margin_per_input": float(-s["mean"] / margin),
                    "accuracy_none": float(t[0].mean()),
                }
            )
    return out


def tests(pp: pd.DataFrame) -> dict:
    final, fold = wide(pp, "lee2019", "final"), wide(pp, "lee2019", "fold")
    c1 = paired((final[8] - final[46]).to_numpy())
    c2 = paired(((fold[17] - fold[0]) - (final[17] - final[0])).to_numpy())
    margins = {lvl: wide(pp, "lee2019", lvl)[0].mean() - CHANCE["lee2019"] for lvl in ("final", "fold")}
    return {
        "C1": {**c1, "rule": "interval lower bound above 0", "holds": bool(c1["ci"][0] > 0)},
        "C2": {
            **c2,
            "rule": "interval upper bound below 0",
            "holds": bool(c2["ci"][1] < 0),
            "share_of_margin": {
                "fold": float(-(fold[17] - fold[0]).mean() / margins["fold"]),
                "final": float(-(final[17] - final[0]).mean() / margins["final"]),
            },
        },
    }


def data_amount_2a(pp: pd.DataFrame) -> list[dict]:
    """BCI IV 2a, exploratory: does the cost of 16 added inputs change with less training data?"""
    final = wide(pp, "bnci2a", "final")
    out = []
    for level in ("fold", "rfold"):
        t = wide(pp, "bnci2a", level)
        d = paired(((t[16] - t[0]) - (final[16] - final[0])).to_numpy())
        out.append({"level": level, "data": LEVEL_NAMES["bnci2a"][level], **d})
    return out


def other_draw(pp: pd.DataFrame, other: pd.DataFrame) -> dict:
    """BCI IV 2a final models, 16 electrodes plus 6 noise inputs: this study's draw against the overfit study's."""
    t = wide(pp, "bnci2a", "final")
    o = other.groupby("subject")["next_day"].mean()
    return {
        "this_study": paired((t[6] - t[0]).to_numpy()),
        "overfit_study": paired((o - t[0]).to_numpy()),
        "difference": paired((t[6] - o).to_numpy()),
        "per_participant_difference": {int(s): round(float(v), 2) for s, v in (t[6] - o).items()},
    }


def figure(costs: pd.DataFrame, draw: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, dataset in zip(axes, ("bnci2a", "lee2019"), strict=True):
        for level, name in LEVEL_NAMES[dataset].items():
            c = costs[(costs.dataset == dataset) & (costs.level == level)].sort_values("added")
            x = np.r_[0, c.added]
            y = np.r_[0, c.cost]
            err = np.vstack([np.r_[0, c.cost - c.ci_low], np.r_[0, c.ci_high - c.cost]])
            ax.errorbar(x, y, yerr=err, fmt="o-", color=COLORS[level], capsize=3, label=f"next day, trained on {name}")
            if c.within_cost.notna().all():
                ax.plot(
                    x,
                    np.r_[0, c.within_cost],
                    "s--",
                    color=COLORS[level],
                    alpha=0.8,
                    ms=4,
                    label=f"held-out runs of the training day, {name}",
                )
        if dataset == "bnci2a":
            ax.plot(
                6,
                draw["overfit_study"]["mean"],
                "D",
                mfc="white",
                color="#C8102E",
                ms=8,
                label="6 noise inputs, another noise draw (overfit study)",
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("noise inputs added to the 16 electrodes")
        ax.set_ylabel("accuracy change vs. 16 electrodes alone (pp)")
        ax.set_title(TITLES[dataset])
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_input_cost.png", dpi=200)
    plt.close(fig)


def fmt(d: dict) -> str:
    return f"{d['mean']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    df, other = load()
    pp = df.groupby(["dataset", "level", "added", "subject"])[["train", "within", "next_day"]].mean().reset_index()
    curves = pp.groupby(["dataset", "level", "added"])[["train", "within", "next_day"]].mean().reset_index()
    curves.to_csv(OUT / "curves.csv", index=False, float_format="%.3f")
    costs = cost_table(df, pp)
    costs.to_csv(OUT / "costs.csv", index=False, float_format="%.3f")
    slope_rows, lee, by_amount, draw = slopes(pp), tests(pp), data_amount_2a(pp), other_draw(pp, other)
    figure(costs, draw)
    write_json(
        OUT / "analysis.json",
        {
            "tests_lee2019": lee,
            "slopes": slope_rows,
            "data_amount_bnci2a": by_amount,
            "other_noise_draw_bnci2a": draw,
            "costs": records_json(costs),
            "curves": records_json(curves),
        },
    )

    lines = [
        "# The cost of inputs that carry nothing",
        "",
        "Relevance-16 plus k inputs of band-limited noise; accuracy in %, costs in percentage points against the "
        "same 16 electrodes alone, with the 95% participant interval. Lee2019 tests were fixed in the plan; "
        "BCI IV 2a is exploratory.",
        "",
        "## Lee2019 tests (54 people, seeds 0-2)",
        "",
        f"- C1, the cost grows with the number of such inputs (8 added minus 46 added, both runs): {fmt(lee['C1'])}: "
        f"{'holds' if lee['C1']['holds'] else 'does not hold'}",
        f"- C2, the cost is larger with fewer training trials ((17 added minus none) with one run minus the same with "
        f"both): {fmt(lee['C2'])}: {'holds' if lee['C2']['holds'] else 'does not hold'}. Share of the accuracy above "
        f"chance that 17 added inputs remove: {lee['C2']['share_of_margin']['fold']:.0%} with one run, "
        f"{lee['C2']['share_of_margin']['final']:.0%} with both.",
        "",
        "## Cost per added input (least squares through k = 0)",
        "",
        "| dataset | training data | 16 electrodes alone % | pp per added input | 95% interval | share of the "
        "margin above chance per input |",
        "|---|---|---:|---:|---|---:|",
    ]
    lines += [
        f"| {s['dataset']} | {s['data']} | {s['accuracy_none']:.2f} | {s['slope']:+.2f} | "
        f"[{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}] | {s['share_of_margin_per_input']:.1%} |"
        for s in slope_rows
    ]
    lines += [
        "",
        "## Every point",
        "",
        "| dataset | training data | added | accuracy % | next-day cost | 95% interval | seed error | worse | "
        "held-out runs, same day | training-set change | share of margin |",
        "|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|",
    ]
    for r in costs.itertuples():
        within = (
            "-"
            if np.isnan(r.within_cost)
            else f"{r.within_cost:+.2f} [{r.within_ci_low:+.2f}, {r.within_ci_high:+.2f}]"
        )
        lines.append(
            f"| {r.dataset} | {r.data} | {r.added} | {r.accuracy:.2f} | {r.cost:+.2f} | [{r.ci_low:+.2f}, "
            f"{r.ci_high:+.2f}] | {r.seed_error:.2f} | {r.worse}/{r.people} | {within} | {r.train_change:+.2f} | "
            f"{r.share_of_margin:.0%} |"
        )
    lines += [
        "",
        "## BCI IV 2a: the cost of 16 added inputs with less training data, minus the same with all",
        "",
        "| training data | mean | 95% interval |",
        "|---|---:|---|",
    ]
    lines += [f"| {b['data']} | {b['mean']:+.2f} | [{b['ci'][0]:+.2f}, {b['ci'][1]:+.2f}] |" for b in by_amount]
    lines += [
        "",
        "## BCI IV 2a: two noise draws of the same condition",
        "",
        "Final models, the same 16 electrodes plus 6 noise inputs, seeds 0-9. The overfit study put its noise in "
        "the six dropped places; this study appends it. EEGNet treats input positions alike (its spatial filters "
        "span every input), so the two differ only in the noise drawn, one draw per participant shared by all seeds.",
        "",
        f"- this study's draw: {fmt(draw['this_study'])}",
        f"- the overfit study's draw: {fmt(draw['overfit_study'])}",
        f"- difference: {fmt(draw['difference'])}; per participant {draw['per_participant_difference']}",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
