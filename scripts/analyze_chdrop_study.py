"""Tables, tests and figures for the channel-dropout study (scripts/chdrop_study.py).

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only, never EEG:
  supplementary/chdrop_study/{protocol.json,runs,test}

Writes to RESULTS_DIR/supplementary/chdrop_study/: finals.csv, folds.csv, variance.csv, failure.csv,
fig_chdrop.png, summary.md and analysis.json.

p = 0 is the model without dropout and p = 0.2 the archived dropout model of scripts/overfit_study.py
(final models); the fold models at p = 0.2 are new. "Re-estimated" means the batch norms after the
spatial filters re-estimated on the training trials with every electrode present. Per participant,
accuracy is the mean over seeds (and folds); contrasts are paired over participants (frozen bootstrap,
mibci.stats). A failed electrode is set to its training mean on the next day; its cost is the intact
accuracy minus the failed one, averaged over electrodes (or over the five three-electrode sets), and the
worst single electrode. Lee2019 tests B1-B3 are the plan's; BCI IV 2a is exploratory.
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

OUT = RESULTS_DIR / "supplementary" / "chdrop_study"
VERSIONS = ("as_trained", "recalibrated")
TITLES = {"bnci2a": "BCI IV 2a (9 people)", "lee2019": "Lee 2019 (54 people)"}
FULL = {"bnci2a": "full-22", "lee2019": "full-62"}


def records_json(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def paired(d) -> dict:
    d = np.asarray(d, float)
    lo, hi = bootstrap_ci(d)
    return {"mean": float(d.mean()), "ci": [lo, hi], "better": int((d > 0).sum()), "people": len(d)}


def fmt(d: dict) -> str:
    return f"{d['mean']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}] ({d['better']}/{d['people']} positive)"


def verdict(holds: bool) -> str:
    return "holds" if holds else "does not hold"


def family(job: dict) -> str:
    """'full' or 'reduced' (relevance-16 / relevance-45) montage."""
    return "full" if job["condition"].startswith("full") else "reduced"


def load(plan: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    finals, folds, variance, failure = [], [], [], []
    checks = {"archived_match": 0, "archived": 0, "reference_match": 0, "reference": 0}
    missing = 0
    for job in plan["jobs"]:
        base = {
            "dataset": job["dataset"],
            "subject": job["subject"],
            "seed": job["seed"],
            "p": float(job["p"]),
            "montage": family(job),
            "kind": job["kind"],
        }
        if job["split"] == "final":
            path = OUT / "test" / f"{job['name']}.json"
            if not path.exists():
                missing += 1
                continue
            t = json.loads(path.read_text())
            if job["kind"] == "archived":
                checks["archived"] += 1
                checks["archived_match"] += int(t["matches_earlier_record"])
            for v in VERSIONS:
                finals.append({**base, "version": v, "next_day": 100 * t[v]["primary"]["acc"]})
                variance += [
                    {**base, "version": v, "layer": layer, "ratio": float(np.median(r))}
                    for layer, r in t[v]["variance_ratios"].items()
                ]
                if "failed" in t[v]:
                    intact = t[v]["primary"]["acc"]
                    single = 100 * (intact - np.array(t[v]["failed"]["single"]))
                    three = 100 * (intact - np.array(t[v]["failed"]["three"]))
                    failure.append(
                        {
                            **base,
                            "version": v,
                            "single_mean": float(single.mean()),
                            "single_worst": float(single.max()),
                            "three_mean": float(three.mean()),
                        }
                    )
        else:
            path = OUT / "runs" / f"{job['name']}.json"
            if not path.exists():
                missing += 1
                continue
            r = json.loads(path.read_text())
            if job["kind"] == "reference":
                checks["reference"] += 1
                checks["reference_match"] += int(r["matches_earlier_record"])
            folds.extend(
                {**base, "split": job["split"], "version": v, "held_out": 100 * r[v]["within_day"]["acc"]}
                for v in VERSIONS
            )
    if missing:
        raise SystemExit(f"{missing} jobs lack their record")
    return pd.DataFrame(finals), pd.DataFrame(folds), pd.DataFrame(variance), pd.DataFrame(failure), checks


def per_person(df: pd.DataFrame, column: str, **where) -> pd.Series:
    sel = df
    for key, value in where.items():
        sel = sel[sel[key] == value]
    return sel.groupby("subject")[column].mean()


def contrast(df: pd.DataFrame, column: str, a: dict, b: dict) -> dict:
    return paired((per_person(df, column, **a) - per_person(df, column, **b)).to_numpy())


def best_p(folds: pd.DataFrame, dataset: str) -> float:
    """The p (0.05, 0.1, 0.2) with the best re-estimated held-out accuracy of the training day."""
    sel = folds[(folds.dataset == dataset) & (folds.version == "recalibrated") & (folds.p > 0)]
    means = {p: per_person(sel, "held_out", p=p).mean() for p in sorted(sel.p.unique())}
    best = max(means.values())
    return min(p for p, v in means.items() if v == best)


def tests(finals: pd.DataFrame, folds: pd.DataFrame, failure: pd.DataFrame) -> dict:
    lee = finals[(finals.dataset == "lee2019") & (finals.montage == "full")]
    b1 = contrast(lee, "next_day", {"p": 0.2, "version": "recalibrated"}, {"p": 0.2, "version": "as_trained"})
    loss = contrast(lee, "next_day", {"p": 0.0, "version": "as_trained"}, {"p": 0.2, "version": "as_trained"})
    b2 = contrast(lee, "next_day", {"p": 0.05, "version": "as_trained"}, {"p": 0.2, "version": "as_trained"})
    b2r = contrast(lee, "next_day", {"p": 0.05, "version": "recalibrated"}, {"p": 0.2, "version": "recalibrated"})
    p_star = best_p(folds, "lee2019")
    fail = failure[failure.dataset == "lee2019"]
    b3 = contrast(fail, "single_mean", {"p": 0.0, "version": "as_trained"}, {"p": p_star, "version": "recalibrated"})
    return {
        "B1": {
            **b1,
            "rule": "interval lower bound above 0",
            "holds": bool(b1["ci"][0] > 0),
            "loss_without_dropout_minus_p02": loss,
            "share_recovered": b1["mean"] / loss["mean"],
        },
        "B2": {**b2, "rule": "interval lower bound above 0", "holds": bool(b2["ci"][0] > 0), "recalibrated": b2r},
        "B3": {**b3, "rule": "interval lower bound above 0", "holds": bool(b3["ci"][0] > 0), "p": p_star},
    }


def tables(finals, folds, variance, failure) -> tuple[pd.DataFrame, ...]:
    keys = ["dataset", "montage", "p", "version"]
    f = finals.groupby([*keys, "subject"]).next_day.mean().groupby(keys).mean().reset_index()
    h = folds.groupby([*keys, "subject"]).held_out.mean().groupby(keys).mean().reset_index()
    v = variance.groupby([*keys, "layer"]).ratio.median().unstack("layer").reset_index()
    c = failure.groupby([*keys, "subject"])[["single_mean", "single_worst", "three_mean"]].mean()
    c = c.groupby(keys).mean().reset_index()
    return f, h, v, c


def figure(f: pd.DataFrame, h: pd.DataFrame, c: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, dataset in enumerate(("bnci2a", "lee2019")):
        ax = axes[0, col]
        for version, style in (("as_trained", "o-"), ("recalibrated", "s--")):
            t = f[(f.dataset == dataset) & (f.montage == "full") & (f.version == version)].sort_values("p")
            ax.plot(t.p, t.next_day, style, color="#13294B", label=f"next day, {version.replace('_', ' ')}")
            t = h[(h.dataset == dataset) & (h.montage == "full") & (h.version == version)].sort_values("p")
            ax.plot(t.p, t.held_out, style, color="#4B9CD3", label=f"held-out runs, {version.replace('_', ' ')}")
        ax.set(xlabel="channel dropout p", ylabel="accuracy, %", title=f"{TITLES[dataset]}: all electrodes")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
        ax = axes[1, col]
        for version, style in (("as_trained", "o-"), ("recalibrated", "s--")):
            t = c[(c.dataset == dataset) & (c.version == version)].sort_values("p")
            ax.plot(t.p, t.single_mean, style, color="#13294B", label=f"one electrode, {version.replace('_', ' ')}")
            ax.plot(t.p, t.three_mean, style, color="#C8102E", label=f"three electrodes, {version.replace('_', ' ')}")
        ax.set(
            xlabel="channel dropout p",
            ylabel="accuracy lost when electrodes fail, points",
            title=f"{TITLES[dataset]}: failed electrodes, next day",
        )
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_chdrop.png", dpi=200)
    plt.close(fig)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    plan = json.loads((OUT / "protocol.json").read_text())
    finals, folds, variance, failure, checks = load(plan)
    f, h, v, c = tables(finals, folds, variance, failure)
    for name, df in (("finals", f), ("folds", h), ("variance", v), ("failure", c)):
        df.to_csv(OUT / f"{name}.csv", index=False, float_format="%.4f")
    lee = tests(finals, folds, failure)
    two_a = {}
    for montage in ("full", "reduced"):
        sel = finals[(finals.dataset == "bnci2a") & (finals.montage == montage)]
        two_a[f"{montage}: p 0.2 re-estimated minus as trained"] = contrast(
            sel, "next_day", {"p": 0.2, "version": "recalibrated"}, {"p": 0.2, "version": "as_trained"}
        )
        two_a[f"{montage}: p 0 re-estimated minus as trained"] = contrast(
            sel, "next_day", {"p": 0.0, "version": "recalibrated"}, {"p": 0.0, "version": "as_trained"}
        )
    sel = finals[(finals.dataset == "bnci2a") & (finals.montage == "full")]
    for p in (0.05, 0.1, 0.2):
        for version in VERSIONS:
            two_a[f"full: p {p} minus p 0, {version}"] = contrast(
                sel, "next_day", {"p": p, "version": version}, {"p": 0.0, "version": "as_trained"}
            )
    p2a = best_p(folds, "bnci2a")
    fail = failure[failure.dataset == "bnci2a"]
    two_a[f"one failed electrode: p 0 as trained minus p {p2a} re-estimated (cost)"] = contrast(
        fail, "single_mean", {"p": 0.0, "version": "as_trained"}, {"p": p2a, "version": "recalibrated"}
    )
    figure(f, h, c)
    write_json(
        OUT / "analysis.json",
        {
            "tests_lee2019": lee,
            "bnci2a": two_a,
            "checks": checks,
            "finals": records_json(f),
            "folds": records_json(h),
            "variance": records_json(v),
            "failure": records_json(c),
        },
    )
    lines = [
        "# Channel dropout: the batch norms and a failed electrode",
        "",
        "Accuracy in %, differences in points with the 95% participant interval. p = 0 without dropout; p = 0.2 final "
        "models are the archived ones of the overfitting study. Re-estimated: batch norms after the spatial filters "
        f"re-estimated on the training day with every electrode present. Lee2019 tests were fixed in the plan "
        f"({plan['created_at'][:16]}); BCI IV 2a is exploratory.",
        "",
        f"Checks: archived final models scored as trained reproduce their archived predictions in "
        f"{checks['archived_match']}/{checks['archived']}; fold models without dropout reproduce their held-out "
        f"predictions in {checks['reference_match']}/{checks['reference']}.",
        "",
        "## Lee2019 tests",
        "",
        f"- B1, re-estimating recovers the p = 0.2 models: {fmt(lee['B1'])}: {verdict(lee['B1']['holds'])}; it "
        f"recovers {lee['B1']['share_recovered']:.0%} of the {lee['B1']['loss_without_dropout_minus_p02']['mean']:.2f}"
        "-point loss",
        f"- B2, p = 0.05 costs less than p = 0.2 (as trained): {fmt(lee['B2'])}: {verdict(lee['B2']['holds'])}; "
        f"re-estimated: {fmt(lee['B2']['recalibrated'])}",
        f"- B3, a failed electrode costs the dropout model (p = {lee['B3']['p']}, chosen on held-out runs) less: "
        f"{fmt(lee['B3'])}: {verdict(lee['B3']['holds'])}",
        "",
        "## Next day (final models) and held-out runs of the training day (fold models)",
        "",
        "| dataset | montage | p | version | next day % | held-out runs % |",
        "|---|---|---:|---|---:|---:|",
    ]
    for r in f.itertuples():
        hh = h[(h.dataset == r.dataset) & (h.montage == r.montage) & (h.p == r.p) & (h.version == r.version)]
        held = f"{hh.held_out.iloc[0]:.2f}" if len(hh) else ""
        version = r.version.replace("_", " ")
        lines.append(f"| {r.dataset} | {r.montage} | {r.p} | {version} | {r.next_day:.2f} | {held} |")
    lines += [
        "",
        "## Stored over actual variance, per batch norm (median over channels, mean over models)",
        "",
        "| dataset | montage | p | version | temporal | after spatial (bnorm_1) | after separable (bnorm_2) |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r.dataset} | {r.montage} | {r.p} | {r.version.replace('_', ' ')} | {r.bnorm_temporal:.3f} | "
        f"{r.bnorm_1:.3f} | {r.bnorm_2:.3f} |"
        for r in v.itertuples()
    ]
    lines += [
        "",
        "## A failed electrode, next day (all-electrode models; points lost)",
        "",
        "| dataset | p | version | one electrode, mean | one electrode, worst | three electrodes, mean |",
        "|---|---:|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r.dataset} | {r.p} | {r.version.replace('_', ' ')} | {r.single_mean:.2f} | {r.single_worst:.2f} | "
        f"{r.three_mean:.2f} |"
        for r in c.itertuples()
    ]
    lines += ["", "## BCI IV 2a contrasts (exploratory)", "", "| contrast | mean | 95% interval |", "|---|---:|---|"]
    lines += [f"| {k} | {d['mean']:+.2f} | [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}] |" for k, d in two_a.items()]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
