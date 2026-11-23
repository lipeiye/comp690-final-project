"""Tables, tests and a figure for the input-kind study (scripts/input_kind_study.py).

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only, never EEG:
  supplementary/input_kind_study/{protocol.json,runs,test,predictability}  copies, mixes, one shared waveform
  supplementary/overfit_study/{protocol.json,runs,test}         BCI IV 2a relevance-16 (no extra input)
  supplementary/electrode_data_study/{protocol.json,runs,test}  Lee2019 relevance-16 (no extra input)
  supplementary/input_cost_study/{protocol.json,runs,test}      relevance-16-plus-k: independent noise (a = 1)

Writes to RESULTS_DIR/supplementary/input_kind_study/: conditions.csv, contrasts.csv, fig_input_kind.png,
summary.md and analysis.json.

Per participant, accuracy is the mean over seeds (final models, next day) or over seeds and folds (fold
models, held-out runs of the training day). References use only the seeds and splits this study trained.
A contrast is paired over participants (frozen bootstrap, mibci.stats); on BCI IV 2a its seed error is
the contrast computed seed by seed, its standard deviation over seeds over the square root of their number.
The unexplained share of a dropped electrode (predictability stage, session 1) is averaged over the
dropped electrodes and folds of each participant, then over participants; it places the real dropped
electrodes on the a axis of the mixes. Lee2019 tests K1-K3 are the plan's; BCI IV 2a is exploratory.
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
OUT = SUPP / "input_kind_study"
NONE_FROM = {"bnci2a": "overfit_study", "lee2019": "electrode_data_study"}
A = {"dup": 0.0, "mix25": 0.25, "mix50": 0.5, "mix75": 0.75, "noise": 1.0}
ORDER = ("none", "dup", "mix25", "mix50", "mix75", "noise", "shared")
NAMES = {
    "none": "16 electrodes alone",
    "dup": "copies of kept electrodes",
    "mix25": "copy + noise, a = 0.25",
    "mix50": "copy + noise, a = 0.5",
    "mix75": "copy + noise, a = 0.75",
    "noise": "independent noise (a = 1)",
    "shared": "one noise waveform, repeated",
}
TITLES = {"bnci2a": "BCI IV 2a (9 people), 16 + 6 inputs", "lee2019": "Lee 2019 (54 people), 16 + 46 inputs"}
CHANCE = {"bnci2a": 25.0, "lee2019": 50.0}


def records_json(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def load_rows(folder, keep, label) -> list[dict]:
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
                "split": job["split"],
                "level": "final" if job["split"] == "final" else "fold",
                "label": label(job),
                "train": 100 * r["train_acc"],
                "within": 100 * r["within_day"]["acc"] if "within_day" in r else np.nan,
                "next_day": 100 * t["primary"]["acc"],
                "extra_share": float(np.mean(r["extra_share"])) if "extra_share" in r else np.nan,
                "weight_ratio": r["weight_rms_extra"] / r["weight_rms_kept"] if "weight_rms_extra" in r else np.nan,
                "final_train_loss": r.get("final_train_loss", np.nan),
            }
        )
    if missing:
        raise SystemExit(f"{folder.name}: {missing} jobs lack a run or test record")
    return rows


def load() -> tuple[pd.DataFrame, dict]:
    plan = json.loads((OUT / "protocol.json").read_text())
    spec = plan["datasets"]

    def trained_here(job: dict) -> bool:
        s = spec[job["dataset"]]
        if job["split"] == "final":
            return job["seed"] in s["final_seeds"]
        return job["split"] in s["splits"] and job["seed"] in s["fold_seeds"]

    rows = load_rows(OUT, lambda j: True, lambda j: j["kind"])
    for dataset, study in NONE_FROM.items():
        rows += load_rows(
            SUPP / study,
            lambda j, d=dataset: j["dataset"] == d and j["condition"] == "relevance-16" and trained_here(j),
            lambda j: "none",
        )
        k = spec[dataset]["added"]
        rows += load_rows(
            SUPP / "input_cost_study",
            lambda j, d=dataset, k=k: (
                j["dataset"] == d and j["condition"] == f"relevance-16-plus-{k}" and trained_here(j)
            ),
            lambda j: "noise",
        )
    return pd.DataFrame(rows), plan


def per_person(df: pd.DataFrame, dataset: str, level: str, column: str) -> pd.DataFrame:
    """Participants by condition label (mean over seeds, and folds)."""
    sel = df[(df.dataset == dataset) & (df.level == level)]
    return sel.groupby(["subject", "label"])[column].mean().unstack("label")


def seed_error(df: pd.DataFrame, dataset: str, level: str, a: str, b: str, column: str) -> float:
    sel = df[(df.dataset == dataset) & (df.level == level) & df.label.isin([a, b])]
    per_seed = sel.groupby(["seed", "label", "subject"])[column].mean().unstack("subject")
    d = (per_seed.xs(a, level="label") - per_seed.xs(b, level="label")).mean(axis=1)
    return float(d.std(ddof=1) / np.sqrt(len(d)))


def paired(d: np.ndarray) -> dict:
    d = np.asarray(d, float)
    lo, hi = bootstrap_ci(d)
    return {"mean": float(d.mean()), "ci": [lo, hi], "worse": int((d < 0).sum()), "people": len(d)}


def contrast(df: pd.DataFrame, dataset: str, level: str, a: str, b: str, column: str) -> dict:
    t = per_person(df, dataset, level, column)
    out = paired((t[a] - t[b]).to_numpy())
    if dataset == "bnci2a":
        out["seed_error"] = seed_error(df, dataset, level, a, b, column)
    return out


def condition_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in ("bnci2a", "lee2019"):
        labels = [c for c in ORDER if c in set(df[df.dataset == dataset].label)]
        for label in labels:
            row = {"dataset": dataset, "condition": label, "a": A.get(label)}
            for level, column, key in (
                ("final", "next_day", "next_day"),
                ("fold", "within", "held_out"),
                ("final", "train", "train_final"),
            ):
                t = per_person(df, dataset, level, column)
                row[key] = float(t[label].mean())
                if label != "none":
                    c = paired((t[label] - t["none"]).to_numpy())
                    row |= {f"{key}_cost": c["mean"], f"{key}_ci_low": c["ci"][0], f"{key}_ci_high": c["ci"][1]}
                    if key == "next_day":
                        row["worse"] = c["worse"]
                        row["people"] = c["people"]
            sel = df[(df.dataset == dataset) & (df.label == label) & (df.level == "final")]
            row["extra_share"] = float(sel.extra_share.mean()) if sel.extra_share.notna().any() else np.nan
            row["weight_ratio"] = float(sel.weight_ratio.mean()) if sel.weight_ratio.notna().any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def contrasts(df: pd.DataFrame) -> list[dict]:
    pairs = [
        ("dup", "noise"),
        ("dup", "none"),
        ("dup", "mix50"),
        ("mix50", "noise"),
        ("shared", "noise"),
        ("shared", "none"),
        ("mix25", "dup"),
        ("mix75", "mix50"),
    ]
    out = []
    for dataset in ("bnci2a", "lee2019"):
        present = set(df[df.dataset == dataset].label)
        for a, b in pairs:
            if a not in present or b not in present:
                continue
            for level, column in (("final", "next_day"), ("fold", "within")):
                out.append(
                    {
                        "dataset": dataset,
                        "a": a,
                        "b": b,
                        "level": level,
                        "measure": column,
                        **contrast(df, dataset, level, a, b, column),
                    }
                )
    return out


def tests(df: pd.DataFrame) -> dict:
    def c(a, b):
        return contrast(df, "lee2019", "final", a, b, "next_day")

    k1, k1_none = c("dup", "noise"), c("dup", "none")
    k2a, k2b = c("dup", "mix50"), c("mix50", "noise")
    k3 = c("shared", "noise")
    return {
        "K1": {
            **k1,
            "rule": "interval lower bound above 0",
            "holds": bool(k1["ci"][0] > 0),
            "dup_minus_none": k1_none,
            "no_visible_cost": bool(k1_none["ci"][0] > -2),
        },
        "K2": {
            "dup_minus_mix50": k2a,
            "mix50_minus_noise": k2b,
            "rule": "both interval lower bounds above 0",
            "holds": bool(k2a["ci"][0] > 0 and k2b["ci"][0] > 0),
        },
        "K3": {**k3, "rule": "interval lower bound above 0", "holds": bool(k3["ci"][0] > 0)},
    }


def predictability() -> dict:
    out = {}
    for dataset in ("bnci2a", "lee2019"):
        per = {}
        for path in sorted((OUT / "predictability").glob(f"{dataset}_sub-*.json")):
            r = json.loads(path.read_text())
            values = [v for fold in r["folds"].values() for v in fold["unexplained"] if v is not None]
            missing = sum(v is None for fold in r["folds"].values() for v in fold["unexplained"])
            per[r["subject"]] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
                "undefined": missing,
            }
        means = np.array([v["mean"] for v in per.values()])
        out[dataset] = {
            "people": len(per),
            "mean": float(means.mean()),
            "ci": list(bootstrap_ci(means)),
            "range": [float(means.min()), float(means.max())],
            "median_of_medians": float(np.median([v["median"] for v in per.values()])),
            "per_participant": {int(s): v for s, v in per.items()},
        }
    return out


def a_equivalent(table: pd.DataFrame, pred: dict) -> dict:
    """The cost a mix of the dropped electrodes' unexplained share would have, read off the mixes' curve."""
    out = {}
    for dataset in ("bnci2a", "lee2019"):
        t = table[(table.dataset == dataset) & table.a.notna()].sort_values("a")
        a = pred[dataset]["mean"]
        out[dataset] = {
            "a": a,
            "next_day_cost_at_a": float(np.interp(a, t.a, t.next_day_cost)),
            "held_out_cost_at_a": float(np.interp(a, t.a, t.held_out_cost)),
            "points": [float(v) for v in t.a],
        }
    return out


def figure(table: pd.DataFrame, pred: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, dataset in zip(axes, ("bnci2a", "lee2019"), strict=True):
        t = table[table.dataset == dataset]
        mixes = t[t.a.notna()].sort_values("a")
        for key, color, label, marker in (
            ("next_day", "#13294B", "next day", "o"),
            ("held_out", "#4B9CD3", "held-out runs of the training day", "s"),
        ):
            err = np.vstack(
                [mixes[f"{key}_cost"] - mixes[f"{key}_ci_low"], mixes[f"{key}_ci_high"] - mixes[f"{key}_cost"]]
            )
            ax.errorbar(mixes.a, mixes[f"{key}_cost"], yerr=err, fmt=f"{marker}-", color=color, capsize=3, label=label)
            shared = t[t.condition == "shared"].iloc[0]
            ax.errorbar(
                [1.08],
                [shared[f"{key}_cost"]],
                yerr=[
                    [shared[f"{key}_cost"] - shared[f"{key}_ci_low"]],
                    [shared[f"{key}_ci_high"] - shared[f"{key}_cost"]],
                ],
                fmt=marker,
                mfc="white",
                color=color,
                capsize=3,
            )
        p = pred[dataset]
        ax.axvspan(p["ci"][0], p["ci"][1], color="#C8102E", alpha=0.15, lw=0)
        ax.axvline(
            p["mean"],
            color="#C8102E",
            lw=1.2,
            label="share of the real dropped electrodes' variance the kept ones cannot predict",
        )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0, 1.08], ["copies\n0", "0.25", "0.5", "0.75", "noise\n1", "shared"])
        ax.set_xlabel("a: share of each extra input that is independent noise")
        ax.set_ylabel("accuracy change vs. 16 electrodes alone (pp)")
        ax.set_title(TITLES[dataset])
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_input_kind.png", dpi=200)
    plt.close(fig)


def fmt(d: dict) -> str:
    return f"{d['mean']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"


def verdict(holds: bool) -> str:
    return "holds" if holds else "does not hold"


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    df, plan = load()
    table = condition_table(df)
    table.to_csv(OUT / "conditions.csv", index=False, float_format="%.3f")
    pairs = contrasts(df)
    pd.DataFrame(
        [{k: v for k, v in p.items() if k != "ci"} | {"ci_low": p["ci"][0], "ci_high": p["ci"][1]} for p in pairs]
    ).to_csv(OUT / "contrasts.csv", index=False, float_format="%.3f")
    lee, pred = tests(df), predictability()
    equiv = a_equivalent(table, pred)
    figure(table, pred)
    write_json(
        OUT / "analysis.json",
        {
            "tests_lee2019": lee,
            "contrasts": pairs,
            "predictability": pred,
            "a_equivalent": equiv,
            "conditions": records_json(table),
        },
    )

    lines = [
        "# Which extra inputs cost EEGNet accuracy",
        "",
        "Extra inputs after the 16 most relevant electrodes (2a: 6, Lee2019: 46). Accuracy in %, costs in percentage "
        "points against the same 16 electrodes alone, with the 95% participant interval. The next day uses the final "
        "models; held-out runs of the training day use the fold models. Lee2019 tests were fixed in the plan "
        f"({plan['created_at'][:16]}); BCI IV 2a is exploratory.",
        "",
        "## Lee2019 tests (54 people, final models, next day)",
        "",
        f"- K1, copies cost less than independent noise (dup minus noise): {fmt(lee['K1'])}: "
        f"{verdict(lee['K1']['holds'])}. "
        f"Copies against 16 electrodes alone: {fmt(lee['K1']['dup_minus_none'])}"
        + (
            "; no visible cost (lower bound above -2)."
            if lee["K1"]["no_visible_cost"]
            else "; a cost cannot be ruled out."
        ),
        f"- K2, half noise lies between (dup minus mix50 {fmt(lee['K2']['dup_minus_mix50'])}; mix50 minus noise "
        f"{fmt(lee['K2']['mix50_minus_noise'])}): {verdict(lee['K2']['holds'])}",
        f"- K3, one shared waveform costs less than independent noise (shared minus noise): {fmt(lee['K3'])}: "
        f"{verdict(lee['K3']['holds'])}",
        "",
        "## Every condition",
        "",
        "| dataset | extra inputs | next day % | cost | 95% interval | worse | held-out runs, cost | training set % | "
        "extra inputs' share of the spatial filters' output | extra / kept weight size |",
        "|---|---|---:|---:|---|---:|---|---:|---:|---:|",
    ]
    for r in table.itertuples():
        if r.condition == "none":
            lines.append(
                f"| {r.dataset} | {NAMES[r.condition]} | {r.next_day:.2f} | | | | {r.held_out:.2f} % | "
                f"{r.train_final:.1f} | | |"
            )
            continue
        share = "" if np.isnan(r.extra_share) else f"{r.extra_share:.0%}"
        ratio = "" if np.isnan(r.weight_ratio) else f"{r.weight_ratio:.2f}"
        lines.append(
            f"| {r.dataset} | {NAMES[r.condition]} | {r.next_day:.2f} | {r.next_day_cost:+.2f} | "
            f"[{r.next_day_ci_low:+.2f}, "
            f"{r.next_day_ci_high:+.2f}] | {r.worse}/{r.people} | {r.held_out_cost:+.2f} [{r.held_out_ci_low:+.2f}, "
            f"{r.held_out_ci_high:+.2f}] | {r.train_final:.1f} | {share} | {ratio} |"
        )
    lines += [
        "",
        "## Paired contrasts",
        "",
        "| dataset | contrast | measure | mean | 95% interval | seed error | worse |",
        "|---|---|---|---:|---|---:|---:|",
    ]
    for p in pairs:
        se = f"{p['seed_error']:.2f}" if "seed_error" in p else ""
        measure = "next day" if p["measure"] == "next_day" else "held-out runs"
        lines.append(
            f"| {p['dataset']} | {p['a']} minus {p['b']} | {measure} | {p['mean']:+.2f} | "
            f"[{p['ci'][0]:+.2f}, {p['ci'][1]:+.2f}] | {se} | {p['worse']}/{p['people']} |"
        )
    lines += [
        "",
        "## The real dropped electrodes on the same axis (session 1 only)",
        "",
        "Share of each dropped electrode's variance on the held-out runs that a least-squares fit from the 16 kept "
        "electrodes (fitted on the training runs) cannot predict: 0 for a copy, about a for a mix, about 1 for noise.",
        "",
        "| dataset | people | mean share | 95% interval | range over people | cost of mixes with that a, next day | "
        "same, held-out runs |",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    for dataset in ("bnci2a", "lee2019"):
        p, e = pred[dataset], equiv[dataset]
        lines.append(
            f"| {dataset} | {p['people']} | {p['mean']:.3f} | [{p['ci'][0]:.3f}, {p['ci'][1]:.3f}] | "
            f"{p['range'][0]:.3f}-{p['range'][1]:.3f} | {e['next_day_cost_at_a']:+.2f} | "
            f"{e['held_out_cost_at_a']:+.2f} |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
