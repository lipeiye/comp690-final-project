"""Tables, tests and figures for the scale-free penalty study (scripts/scalefree_penalty_study.py).

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only, never EEG:
  supplementary/scalefree_penalty_study/{protocol.json,selection.json,pilot,reference,runs,test}

`reference` holds the lam = 0 models (the earlier studies'), measured as this study's own models.

Writes to RESULTS_DIR/supplementary/scalefree_penalty_study/: folds.csv, finals.csv, fig_strengths.png,
summary.md and analysis.json.

Fold models (training day only): per dataset, condition and lam, the held-out accuracy (mean over seeds
and folds within each participant, then over participants), the median over models of the
noise-to-electrode ratio of the normalized norms g (and of the plain norms), and for all-electrode
models the overlap of the 16 inputs with the largest g with the relevance-16 electrodes of that
split. Final models, next day: at the lam selected on the training day, and, where lam = 0 was
selected, at the second-reading lam; contrasts are paired over participants (frozen bootstrap,
mibci.stats). Lee2019 tests N1-N3 are the plan's; BCI IV 2a is exploratory.
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

OUT = RESULTS_DIR / "supplementary" / "scalefree_penalty_study"
TITLES = {"bnci2a": "BCI IV 2a (9 people)", "lee2019": "Lee 2019 (54 people)"}
FULL = {"bnci2a": "full-22", "lee2019": "full-62"}
NOISE = {"bnci2a": ("relevance-16-plus-6", "relevance-16-plus-16"), "lee2019": ("relevance-16-plus-46",)}
CHANCE_OVERLAP = {"bnci2a": 16 * 16 / 22, "lee2019": 16 * 16 / 62}


def records_json(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def paired(d) -> dict:
    d = np.asarray(d, dtype=float)
    lo, hi = bootstrap_ci(d)
    return {"mean": float(d.mean()), "ci": [lo, hi], "better": int((d > 0).sum()), "people": len(d)}


def fmt(d: dict) -> str:
    return f"{d['mean']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}] ({d['better']}/{d['people']} better)"


def verdict(holds: bool) -> str:
    return "holds" if holds else "does not hold"


def model_rows(plan: dict) -> pd.DataFrame:
    """One row per model: this study's (lam > 0) and the references (lam = 0)."""
    rows, missing = [], 0
    for job in plan["jobs"]:
        path = OUT / "runs" / f"{job['name']}.json"
        if job["split"] != "final" and not path.exists():
            missing += 1
            continue
        if path.exists():
            r = json.loads(path.read_text())
            test = OUT / "test" / f"{job['name']}.json"
            nd = json.loads(test.read_text())["primary"]["acc"] if test.exists() else np.nan
            rows.append(row(job, r, nd))
    for path in sorted((OUT / "reference").glob("*.json")):
        r = json.loads(path.read_text())
        rows.append(row(r, r, r.get("next_day_acc", np.nan)))
    if missing:
        raise SystemExit(f"{missing} fold models lack a run record")
    return pd.DataFrame(rows)


def row(job: dict, r: dict, next_day: float) -> dict:
    return {
        "dataset": job["dataset"],
        "subject": job["subject"],
        "seed": job["seed"],
        "split": job["split"],
        "level": "final" if job["split"] == "final" else "fold",
        "condition": job["condition"],
        "lam": float(r["lam"]),
        "held_out": 100 * r["within_day"]["acc"] if "within_day" in r else np.nan,
        "next_day": 100 * next_day,
        "train": 100 * r["train_acc"] if r.get("train_acc") is not None else np.nan,
        "g_ratio": r.get("g_ratio", np.nan),
        "plain_ratio": r.get("plain_ratio", np.nan),
        "overlap": r.get("top16_overlap_with_relevance", np.nan),
    }


def fold_table(df: pd.DataFrame) -> pd.DataFrame:
    folds = df[df.level == "fold"]
    out = []
    for (dataset, condition, lam), c in folds.groupby(["dataset", "condition", "lam"]):
        per = c.groupby("subject").held_out.mean()
        zero = folds[(folds.dataset == dataset) & (folds.condition == condition) & (folds.lam == 0.0)]
        diff = per - zero.groupby("subject").held_out.mean()
        d = paired(diff.to_numpy()) if lam > 0 else None
        out.append(
            {
                "dataset": dataset,
                "condition": condition,
                "lam": lam,
                "models": len(c),
                "held_out": float(per.mean()),
                "held_out_minus_lam0": d["mean"] if d else 0.0,
                "ci_low": d["ci"][0] if d else np.nan,
                "ci_high": d["ci"][1] if d else np.nan,
                "g_ratio_median": float(c.g_ratio.median()) if c.g_ratio.notna().any() else np.nan,
                "plain_ratio_median": float(c.plain_ratio.median()) if c.plain_ratio.notna().any() else np.nan,
                "top16_overlap": float(c.overlap.mean()) if c.overlap.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(out)


def final_contrasts(df: pd.DataFrame, chosen: dict, reading: str) -> dict:
    """Next-day contrasts with each condition at its selected lam ('selected') or second-reading lam."""
    finals = df[df.level == "final"]

    def accuracy(dataset: str, condition: str, lam: float) -> pd.Series:
        c = finals[(finals.dataset == dataset) & (finals.condition == condition) & (finals.lam == lam)]
        return c.groupby("subject").next_day.mean()

    def lam_of(dataset: str, condition: str) -> float:
        cell = chosen[f"{dataset}|{condition}"]
        lam = cell["selected_lam"] if reading == "selected" else cell["second_reading_lam"]
        return cell["selected_lam"] if lam is None else lam

    out = {}
    for dataset in ("bnci2a", "lee2019"):
        full = FULL[dataset]
        base = accuracy(dataset, "relevance-16", lam_of(dataset, "relevance-16"))
        out[f"{dataset}|full minus relevance-16, penalized"] = paired(
            accuracy(dataset, full, lam_of(dataset, full)) - base
        )
        out[f"{dataset}|full minus relevance-16, unpenalized"] = paired(
            accuracy(dataset, full, 0.0) - accuracy(dataset, "relevance-16", 0.0)
        )
        for noise in NOISE[dataset]:
            pen = accuracy(dataset, noise, lam_of(dataset, noise)) - base
            free = accuracy(dataset, noise, 0.0) - accuracy(dataset, "relevance-16", 0.0)
            out[f"{dataset}|{noise} cost, penalized"] = paired(pen)
            out[f"{dataset}|{noise} cost, unpenalized"] = paired(free)
            out[f"{dataset}|{noise} cost change"] = paired(pen - free)
        for condition in (full, "relevance-16", *NOISE[dataset]):
            lam = lam_of(dataset, condition)
            if lam > 0:
                out[f"{dataset}|{condition} at lam {lam} minus lam 0"] = paired(
                    accuracy(dataset, condition, lam) - accuracy(dataset, condition, 0.0)
                )
    return out


def n3(folds: pd.DataFrame) -> dict:
    t = folds[(folds.dataset == "lee2019") & (folds.condition == "relevance-16-plus-46") & (folds.lam > 0)]
    rows = [
        {
            "lam": r.lam,
            "held_out_minus_lam0": r.held_out_minus_lam0,
            "g_ratio_median": r.g_ratio_median,
            "qualifies": bool(r.held_out_minus_lam0 >= -1.0 and r.g_ratio_median < 1 / 3),
        }
        for r in t.itertuples()
    ]
    return {
        "per_lam": rows,
        "rule": "some lam > 0 with accuracy difference >= -1 point and ratio < 1/3",
        "holds": any(r["qualifies"] for r in rows),
    }


def figure(folds: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, dataset in enumerate(("bnci2a", "lee2019")):
        t = folds[folds.dataset == dataset]
        for condition, c in t.groupby("condition"):
            c = c.sort_values("lam")
            x = np.where(c.lam == 0, c.lam[c.lam > 0].min() / 3, c.lam)
            axes[0, col].plot(x, c.held_out, "o-", label=condition)
            if c.g_ratio_median.notna().any():
                axes[1, col].plot(x, c.g_ratio_median, "o-", label=f"{condition}: normalized norms g")
                axes[1, col].plot(x, c.plain_ratio_median, "s:", alpha=0.6, label=f"{condition}: plain norms")
        axes[1, col].axhline(1 / 3, color="#C8102E", lw=1, ls="--", label="1/3 (N3)")
        for ax, ylabel in (
            (axes[0, col], "held-out runs of the training day, %"),
            (axes[1, col], "noise / electrode median norm"),
        ):
            ax.set_xscale("log")
            ax.set(xlabel="penalty strength lam (leftmost point: lam = 0)", ylabel=ylabel, title=TITLES[dataset])
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_strengths.png", dpi=200)
    plt.close(fig)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    plan = json.loads((OUT / "protocol.json").read_text())
    chosen = json.loads((OUT / "selection.json").read_text())["selection"]
    df = model_rows(plan)
    folds = fold_table(df)
    folds.to_csv(OUT / "folds.csv", index=False, float_format="%.4f")
    columns = ["next_day", "train", "g_ratio", "overlap"]
    finals = df[df.level == "final"].groupby(["dataset", "condition", "lam"])[columns].mean().reset_index()
    finals.to_csv(OUT / "finals.csv", index=False, float_format="%.4f")
    selected, second = final_contrasts(df, chosen, "selected"), final_contrasts(df, chosen, "second")
    lee = {
        "N1": {**selected["lee2019|relevance-16-plus-46 cost change"], "rule": "interval lower bound above 0"},
        "N2": {**selected["lee2019|full minus relevance-16, penalized"], "rule": "interval lower bound above 0"},
        "N3": n3(folds),
    }
    for key in ("N1", "N2"):
        lee[key]["holds"] = bool(lee[key]["ci"][0] > 0)
    figure(folds)
    write_json(
        OUT / "analysis.json",
        {
            "tests_lee2019": lee,
            "selection": chosen,
            "selected_reading": selected,
            "second_reading": second,
            "folds": records_json(folds),
            "finals": records_json(finals),
        },
    )

    lines = [
        "# A penalty no rescaling can lower",
        "",
        "Each spatial filter's weights scaled to unit length, then lam times the sum over inputs of each input's norm "
        "over the filters (mibci.scalefree_penalty). lam chosen per dataset and condition on held-out runs of the "
        f"training day. Accuracy in %, differences in points with the 95% participant interval. Lee2019 tests were "
        f"fixed in the plan ({plan['created_at'][:16]}); BCI IV 2a is exploratory.",
        "",
        "## Lee2019 tests",
        "",
        f"- N1, noise inputs cost less with the penalty: {fmt(lee['N1'])}: {verdict(lee['N1']['holds'])}",
        f"- N2, all 62 electrodes beat relevance-16 with the penalty: {fmt(lee['N2'])}: {verdict(lee['N2']['holds'])}",
        f"- N3, some strength switches the noise off (ratio below 1/3) within 1 point of lam = 0 on held-out runs: "
        f"{verdict(lee['N3']['holds'])}",
        "",
        "## Selected strengths (training day only)",
        "",
        "| dataset | condition | selected lam | second reading |",
        "|---|---|---:|---:|",
    ]
    lines += [
        f"| {k.split('|')[0]} | {k.split('|')[1]} | {v['selected_lam']} | {v['second_reading_lam'] or ''} |"
        for k, v in chosen.items()
    ]
    lines += [
        "",
        "## Fold models by strength (training day only)",
        "",
        "| dataset | condition | lam | held-out % | minus lam 0 | 95% interval | noise/electrode g | plain norms | "
        "top-16 overlap with relevance-16 |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for r in folds.itertuples():
        ci = "" if np.isnan(r.ci_low) else f"[{r.ci_low:+.2f}, {r.ci_high:+.2f}]"
        g = "" if np.isnan(r.g_ratio_median) else f"{r.g_ratio_median:.3f}"
        pl = "" if np.isnan(r.plain_ratio_median) else f"{r.plain_ratio_median:.3f}"
        ov = "" if np.isnan(r.top16_overlap) else f"{r.top16_overlap:.1f} (chance {CHANCE_OVERLAP[r.dataset]:.1f})"
        lines.append(
            f"| {r.dataset} | {r.condition} | {r.lam} | {r.held_out:.2f} | {r.held_out_minus_lam0:+.2f} | "
            f"{ci} | {g} | {pl} | {ov} |"
        )
    for name, reading in (("selected strengths", selected), ("second reading", second)):
        lines += ["", f"## Next day, {name}", "", "| contrast | mean | 95% interval |", "|---|---:|---|"]
        lines += [f"| {k} | {v['mean']:+.2f} | [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] |" for k, v in reading.items()]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
