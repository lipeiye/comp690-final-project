"""Tables, tests and a figure for the fresh-noise study (scripts/fresh_noise_study.py).

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only, never EEG:
  supplementary/fresh_noise_study/{protocol.json,runs,test,reference}  noise drawn afresh every epoch; the fixed-
                                                                       noise models measured the same way
  supplementary/input_cost_study/{protocol.json,runs,test}      relevance-16-plus-k: the same noise in every epoch
  supplementary/overfit_study/{protocol.json,runs,test}         BCI IV 2a relevance-16 (no noise input)
  supplementary/electrode_data_study/{protocol.json,runs,test}  Lee2019 relevance-16 (no noise input)

Writes to RESULTS_DIR/supplementary/fresh_noise_study/: conditions.csv, contrasts.csv, measures.csv,
fig_fresh_noise.png, summary.md and analysis.json.

Per participant, accuracy is the mean over seeds (final models, next day) or over seeds and folds (fold models,
held-out runs of the training day). Contrasts are paired over participants (frozen bootstrap, mibci.stats); on
BCI IV 2a also with the seed error. Lee2019 tests F1-F3 are the plan's; BCI IV 2a is exploratory.
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
OUT = SUPP / "fresh_noise_study"
NONE_FROM = {"bnci2a": "overfit_study", "lee2019": "electrode_data_study"}
MEASURES = ("train_acc_draw0", "train_acc_unseen", "noise_share", "weight_ratio")


def records_json(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def row(job: dict, label: str, added: int, run: dict, test: dict, measured: dict | None) -> dict:
    out = {
        "dataset": job["dataset"], "subject": job["subject"], "seed": job["seed"], "split": job["split"],
        "level": "final" if job["split"] == "final" else "fold", "label": label, "added": added,
        "train": 100 * run["train_acc"] if "train_acc" in run else np.nan,
        "within": 100 * run["within_day"]["acc"] if "within_day" in run else np.nan,
        "next_day": 100 * test["primary"]["acc"],
    }  # fmt: skip
    m = measured or {}
    out |= {
        "train_acc_draw0": 100 * m["train_acc_draw0"] if "train_acc_draw0" in m else np.nan,
        "train_acc_unseen": 100 * m["train_acc_unseen"] if "train_acc_unseen" in m else np.nan,
        "noise_share": m.get("noise_share", np.nan),
        "weight_ratio": m["weight_rms_noise"] / m["weight_rms_kept"] if "weight_rms_noise" in m else np.nan,
    }
    return out


def read(path) -> dict:
    return json.loads(path.read_text())


def load() -> tuple[pd.DataFrame, dict]:
    plan = read(OUT / "protocol.json")
    rows, missing = [], 0
    cost = SUPP / "input_cost_study"
    for job in plan["jobs"]:
        files = [OUT / "runs" / f"{job['name']}.json", OUT / "test" / f"{job['name']}.json",
                 cost / "runs" / f"{job['fixed']}.json", cost / "test" / f"{job['fixed']}.json",
                 OUT / "reference" / f"{job['fixed']}.json"]  # fmt: skip
        if not all(f.exists() for f in files):
            missing += 1
            continue
        run, test, frun, ftest, fref = (read(f) for f in files)
        rows.append(row(job, "fresh", job["added"], run, test, run))
        rows.append(row(job, "fixed", job["added"], frun, ftest, fref))
    if missing:
        raise SystemExit(f"{missing} jobs lack a record (run, test or reference) here or in the input-cost study")
    spec = plan["datasets"]
    for dataset, study in NONE_FROM.items():
        s = spec[dataset]
        for job in read(SUPP / study / "protocol.json")["jobs"]:
            if job["dataset"] != dataset or job["condition"] != "relevance-16":
                continue
            seeds = s["final_seeds"] if job["split"] == "final" else s["fold_seeds"]
            if job["seed"] not in seeds or job["split"] not in ("final", *s["splits"]):
                continue
            run = read(SUPP / study / "runs" / f"{job['name']}.json")
            test = read(SUPP / study / "test" / f"{job['name']}.json")
            rows.extend(row(job, "none", k, run, test, None) for k in s["added"])  # one reference for every k
    return pd.DataFrame(rows), plan


def per_person(df, dataset, added, level, column) -> pd.DataFrame:
    sel = df[(df.dataset == dataset) & (df.added == added) & (df.level == level)]
    return sel.groupby(["subject", "label"])[column].mean().unstack("label")


def seed_error(df, dataset, added, level, a, b, column) -> float:
    sel = df[(df.dataset == dataset) & (df.added == added) & (df.level == level) & df.label.isin([a, b])]
    per_seed = sel.groupby(["seed", "label", "subject"])[column].mean().unstack("subject")
    d = (per_seed.xs(a, level="label") - per_seed.xs(b, level="label")).mean(axis=1)
    return float(d.std(ddof=1) / np.sqrt(len(d)))


def paired(d) -> dict:
    d = np.asarray(d, float)
    lo, hi = bootstrap_ci(d)
    return {"mean": float(d.mean()), "ci": [lo, hi], "better": int((d > 0).sum()), "people": len(d)}


def contrasts(df: pd.DataFrame, plan: dict) -> list[dict]:
    out = []
    for dataset, spec in plan["datasets"].items():
        for k in spec["added"]:
            for level, column in (("final", "next_day"), ("fold", "within")):
                t = per_person(df, dataset, k, level, column)
                for a, b in (("fresh", "fixed"), ("fresh", "none"), ("fixed", "none")):
                    r = {"dataset": dataset, "added": k, "level": level, "measure": column, "a": a, "b": b}
                    r |= paired((t[a] - t[b]).to_numpy())
                    if dataset == "bnci2a":
                        r["seed_error"] = seed_error(df, dataset, k, level, a, b, column)
                    out.append(r)
    return out


def measures_table(df: pd.DataFrame, plan: dict) -> pd.DataFrame:
    rows = []
    for dataset, spec in plan["datasets"].items():
        for k in spec["added"]:
            sel = df[(df.dataset == dataset) & (df.added == k) & (df.level == "final")]
            for label in ("fixed", "fresh"):
                s = sel[sel.label == label]
                summary = {m: float(s[m].median() if m == "weight_ratio" else s[m].mean()) for m in MEASURES}
                rows.append({"dataset": dataset, "added": k, "noise": label, **summary})
    return pd.DataFrame(rows)


def tests(df: pd.DataFrame) -> dict:
    t = per_person(df, "lee2019", 46, "final", "next_day")
    f1, f2 = paired((t["fresh"] - t["fixed"]).to_numpy()), paired((t["fresh"] - t["none"]).to_numpy())
    w = per_person(df, "lee2019", 46, "final", "weight_ratio")
    f3 = paired((w["fresh"] - w["fixed"]).to_numpy())
    fresh_ratio = df[(df.dataset == "lee2019") & (df.level == "final") & (df.label == "fresh")].weight_ratio.median()
    return {
        "F1": {**f1, "rule": "interval lower bound above 0", "holds": bool(f1["ci"][0] > 0)},
        "F2": {**f2, "rule": "interval lower bound above -2", "holds": bool(f2["ci"][0] > -2)},
        "F3": {**f3, "rule": "interval upper bound below 0", "holds": bool(f3["ci"][1] < 0),
               "fresh_median_ratio": float(fresh_ratio), "below_one_third": bool(fresh_ratio < 1 / 3)},
    }  # fmt: skip


def figure(con: list[dict], meas: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    c = pd.DataFrame(con)
    c = c[(c.level == "final") & (c.b == "none")]
    groups = [(d, k) for d, k in c[["dataset", "added"]].drop_duplicates().itertuples(index=False)]
    x = np.arange(len(groups))
    for i, (label, color) in enumerate((("fixed", "#C8102E"), ("fresh", "#13294B"))):
        sub = c[c.a == label].set_index(["dataset", "added"]).loc[groups]
        err = [sub["mean"] - sub.ci.map(lambda v: v[0]), sub.ci.map(lambda v: v[1]) - sub["mean"]]
        axes[0].bar(x + (i - 0.5) * 0.38, sub["mean"], 0.38, yerr=err, capsize=3, color=color,
                    label=f"{label} noise")  # fmt: skip
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(x, [f"{'2a' if d == 'bnci2a' else 'Lee 2019'}\n+{k} noise" for d, k in groups])
    axes[0].set_ylabel("next-day change vs 16 electrodes alone (points)")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.3, axis="y")
    for i, (label, color) in enumerate((("fixed", "#C8102E"), ("fresh", "#13294B"))):
        sub = meas[meas.noise == label]
        axes[1].bar(np.arange(len(sub)) + (i - 0.5) * 0.38, sub.weight_ratio, 0.38, color=color, label=f"{label} noise")
    axes[1].axhline(1 / 3, color="#4B9CD3", ls="--", lw=1, label="1/3")
    axes[1].set_xticks(np.arange(len(groups)), [f"{'2a' if d == 'bnci2a' else 'Lee 2019'}\n+{k}" for d, k in groups])
    axes[1].set_ylabel("noise / electrode weight size (median)")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_fresh_noise.png", dpi=200)
    plt.close(fig)


def fmt(d: dict) -> str:
    return f"{d['mean']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    df, plan = load()
    con = contrasts(df, plan)
    meas = measures_table(df, plan)
    lee = tests(df)
    pd.DataFrame(
        [{k: v for k, v in r.items() if k != "ci"} | {"ci_low": r["ci"][0], "ci_high": r["ci"][1]} for r in con]
    ).to_csv(OUT / "contrasts.csv", index=False, float_format="%.3f")
    meas.to_csv(OUT / "measures.csv", index=False, float_format="%.4f")
    cond = df.groupby(["dataset", "added", "level", "label"])[["train", "within", "next_day"]].mean().reset_index()
    cond.to_csv(OUT / "conditions.csv", index=False, float_format="%.3f")
    figure(con, meas)
    write_json(OUT / "analysis.json", {"tests_lee2019": lee, "contrasts": con, "measures": records_json(meas),
                                       "conditions": records_json(cond)})  # fmt: skip
    verdict = {True: "holds", False: "does not hold"}
    lines = [
        "# Noise drawn afresh in every epoch",
        "",
        "relevance-16-plus-k with the noise drawn once (fixed, scripts/input_cost_study.py) or afresh in every "
        "training epoch (fresh); scored inputs identical. Points of accuracy, 95% participant intervals. Lee2019 "
        f"tests were fixed in the plan ({plan['created_at'][:16]}); BCI IV 2a is exploratory.",
        "",
        "## Lee2019 tests (54 people, final models, next day)",
        "",
        f"- F1, fresh noise costs less than fixed noise (fresh minus fixed): {fmt(lee['F1'])}: "
        f"{verdict[lee['F1']['holds']]}",
        f"- F2, fresh noise has no visible cost (fresh minus 16 alone): {fmt(lee['F2'])}: "
        f"{verdict[lee['F2']['holds']]} (lower bound above -2)",
        f"- F3, the network turns fresh noise down (weight ratio, fresh minus fixed): {fmt(lee['F3'])}: "
        f"{verdict[lee['F3']['holds']]}; fresh median ratio {lee['F3']['fresh_median_ratio']:.2f} "
        f"({'below' if lee['F3']['below_one_third'] else 'not below'} 1/3)",
        "",
        "## Contrasts",
        "",
        "| dataset | noise inputs | measure | contrast | mean | 95% interval | better | seed error |",
        "|---|---:|---|---|---:|---|---:|---:|",
    ]
    for r in con:
        se = f"{r['seed_error']:.2f}" if "seed_error" in r else ""
        lines.append(
            f"| {r['dataset']} | {r['added']} | {r['measure']} | {r['a']} - {r['b']} | {r['mean']:+.2f} | "
            f"[{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | {r['better']}/{r['people']} | {se} |"
        )
    lines += [
        "",
        "## What the networks learned (final models, training day)",
        "",
        "| dataset | noise inputs | noise | training trials, draw-0 noise % | training trials, unseen noise % | noise "
        "share of the spatial filters | noise / electrode weight size (median) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r.dataset} | {r.added} | {r.noise} | {r.train_acc_draw0:.1f} | {r.train_acc_unseen:.1f} | "
        f"{r.noise_share:.0%} | {r.weight_ratio:.2f} |"
        for r in meas.itertuples()
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
