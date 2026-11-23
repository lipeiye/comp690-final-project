"""Tables, tests and a figure for the residual-surrogate study (scripts/residual_surrogate_study.py).

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only, never EEG:
  supplementary/residual_surrogate_study/{protocol.json,runs,test}  real, pred and surr-dK extra inputs
  supplementary/overfit_study/{protocol.json,runs,test}         BCI IV 2a relevance-16; full-22 and full-62
  supplementary/electrode_data_study/{protocol.json,runs,test}  Lee2019 relevance-16 (no extra input)
  supplementary/input_kind_study/analysis.json                  per-person unexplained share; interpolated cost

Writes to RESULTS_DIR/supplementary/residual_surrogate_study/: conditions.csv, decomposition.csv,
fig_residual_surrogate.png, summary.md and analysis.json.

Per participant, accuracy is the mean over seeds and surrogate draws (final models, next day) or also over folds
(fold models, held-out runs of the training day). The decomposition: cost = surr - 16 alone, information =
real - surr, their sum = real - 16 alone; pred - 16 alone; real - all electrodes (the same electrodes in another
input order). On Lee2019's held-out runs real is not trained, and full-62 stands in for it. Contrasts are paired
over participants (frozen bootstrap, mibci.stats); on BCI IV 2a also with the seed error. S1-S4 are the plan's.
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

SUPP = RESULTS_DIR / "supplementary"
OUT = SUPP / "residual_surrogate_study"
NONE_FROM = {"bnci2a": "overfit_study", "lee2019": "electrode_data_study"}
FULL = {"bnci2a": "full-22", "lee2019": "full-62"}
NAMES = {
    "none": "16 electrodes alone",
    "real": "real dropped electrodes",
    "pred": "their prediction from the 16 alone",
    "surr": "prediction + another trial's residual (mean of draws)",
    "full": "all electrodes (another input order)",
}
TERMS = {  # term: (a, b) of the contrast a - b
    "cost": ("surr", "none"),
    "information": ("real", "surr"),
    "sum": ("real", "none"),
    "pred_cost": ("pred", "none"),
    "order": ("real", "full"),
}


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
        rows.append({
            "dataset": job["dataset"], "subject": job["subject"], "seed": job["seed"], "split": job["split"],
            "level": "final" if job["split"] == "final" else "fold", "label": label(job),
            "train": 100 * r["train_acc"],
            "within": 100 * r["within_day"]["acc"] if "within_day" in r else np.nan,
            "next_day": 100 * t["primary"]["acc"],
            "residual_share": r.get("residual_share_train", np.nan),
            "extra_share": float(np.mean(r["extra_share"])) if "extra_share" in r else np.nan,
        })  # fmt: skip
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

    rows = load_rows(OUT, lambda j: True, lambda j: j["kind"] + (f"-d{j['draw']}" if j["draw"] is not None else ""))
    for dataset, study in NONE_FROM.items():
        rows += load_rows(
            SUPP / study,
            lambda j, d=dataset: j["dataset"] == d and j["condition"] == "relevance-16" and trained_here(j),
            lambda j: "none",
        )
        rows += load_rows(
            SUPP / "overfit_study",
            lambda j, d=dataset: j["dataset"] == d and j["condition"] == FULL[d] and trained_here(j),
            lambda j: "full",
        )
    df = pd.DataFrame(rows)
    surr = df[df.label.str.startswith("surr-d")]
    means = surr.groupby(["dataset", "subject", "seed", "split", "level"], as_index=False).mean(numeric_only=True)
    return pd.concat([df, means.assign(label="surr")], ignore_index=True), plan


def per_person(df: pd.DataFrame, dataset: str, level: str, column: str) -> pd.DataFrame:
    sel = df[(df.dataset == dataset) & (df.level == level)]
    return sel.groupby(["subject", "label"])[column].mean().unstack("label")


def seed_error(df: pd.DataFrame, dataset: str, level: str, a: str, b: str, column: str) -> float:
    sel = df[(df.dataset == dataset) & (df.level == level) & df.label.isin([a, b])]
    per_seed = sel.groupby(["seed", "label", "subject"])[column].mean().unstack("subject")
    d = (per_seed.xs(a, level="label") - per_seed.xs(b, level="label")).mean(axis=1)
    return float(d.std(ddof=1) / np.sqrt(len(d)))


def paired(d) -> dict:
    d = np.asarray(d, float)
    lo, hi = bootstrap_ci(d)
    return {"mean": float(d.mean()), "ci": [lo, hi], "worse": int((d < 0).sum()), "people": len(d)}


def stand_in(dataset: str, level: str, a: str) -> str:
    """Lee2019's fold models have no 'real': full-62 holds the same electrodes."""
    return "full" if (dataset, level, a) == ("lee2019", "fold", "real") else a


def decomposition(df: pd.DataFrame) -> list[dict]:
    out = []
    for dataset in ("bnci2a", "lee2019"):
        for level, column in (("final", "next_day"), ("fold", "within")):
            t = per_person(df, dataset, level, column)
            draws = sorted(c for c in t.columns if c.startswith("surr-d"))
            terms = {**TERMS, **{f"cost_{d.removeprefix('surr-')}": (d, "none") for d in draws}}
            for term, (a, b) in terms.items():
                a = stand_in(dataset, level, a)
                if a not in t or b not in t or (term == "order" and a == "full"):
                    continue
                row = {"dataset": dataset, "level": level, "measure": column, "term": term, "a": a, "b": b}
                row |= paired((t[a] - t[b]).to_numpy())
                if dataset == "bnci2a":
                    row["seed_error"] = seed_error(df, dataset, level, a, b, column)
                out.append(row)
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def tests(df: pd.DataFrame, unexplained: dict[int, float]) -> dict:
    t = per_person(df, "lee2019", "final", "next_day")
    s1, s2, s3 = (paired((t[a] - t[b]).to_numpy()) for a, b in (("real", "surr"), ("surr", "none"), ("pred", "none")))
    people = [s for s in t.index if s in unexplained]
    x, y = np.array([unexplained[s] for s in people]), (t["surr"] - t["none"]).loc[people].to_numpy()
    rng = np.random.default_rng(FROZEN_BOOTSTRAP["seed"])
    draws = rng.integers(0, len(people), size=(FROZEN_BOOTSTRAP["resamples"], len(people)))
    rhos = np.array([spearman(x[i], y[i]) for i in draws])
    tail = (1 - FROZEN_BOOTSTRAP["level"]) / 2
    ci = [float(np.nanquantile(rhos, tail)), float(np.nanquantile(rhos, 1 - tail))]
    return {
        "S1": {**s1, "rule": "interval lower bound above 0", "holds": bool(s1["ci"][0] > 0)},
        "S2": {**s2, "rule": "interval upper bound below 0", "holds": bool(s2["ci"][1] < 0)},
        "S3": {**s3, "rule": "interval lower bound above -2", "holds": bool(s3["ci"][0] > -2)},
        "S4": {"rho": spearman(x, y), "ci": ci, "people": len(people), "rule": "interval upper bound below 0",
               "holds": bool(ci[1] < 0)},
    }  # fmt: skip


def seed_split(df: pd.DataFrame) -> dict:
    """Post hoc, for S4: does a person's cost repeat across seeds? Lee2019 final models, next day: the cost
    (surr - 16 alone) from seed 0 against the cost from seeds 1-2, Spearman over participants."""
    sel = df[(df.dataset == "lee2019") & (df.level == "final")]
    halves = []
    for seeds in ([0], [1, 2]):
        t = sel[sel.seed.isin(seeds)].groupby(["subject", "label"]).next_day.mean().unstack("label")
        halves.append(t["surr"] - t["none"])
    a, b = halves[0].align(halves[1], join="inner")
    return {"rho": spearman(a.to_numpy(), b.to_numpy()), "people": len(a), "halves": "seed 0 against seeds 1-2",
            "note": "post hoc; not one of the registered tests"}  # fmt: skip


def conditions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in ("bnci2a", "lee2019"):
        for label in ("none", "real", "pred", "surr", "full"):
            row = {"dataset": dataset, "condition": label}
            for level, column, key in (("final", "next_day", "next_day"), ("fold", "within", "held_out"),
                                       ("final", "train", "train_final")):  # fmt: skip
                t = per_person(df, dataset, level, column)
                row[key] = float(t[label].mean()) if label in t else np.nan
            sel = df[(df.dataset == dataset) & (df.label == label) & (df.level == "final")]
            row["residual_share"] = float(sel.residual_share.mean()) if sel.residual_share.notna().any() else np.nan
            row["extra_share"] = float(sel.extra_share.mean()) if sel.extra_share.notna().any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def figure(dec: pd.DataFrame, interpolated: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    terms = (
        ("cost", "cost\n(surr - 16)", "#C8102E"),
        ("information", "information\n(real - surr)", "#13294B"),
        ("sum", "sum\n(real - 16)", "#4B9CD3"),
        ("pred_cost", "prediction only\n(pred - 16)", "#9ec5e8"),
    )
    for ax, dataset in zip(axes, ("bnci2a", "lee2019"), strict=True):
        sub = dec[(dec.dataset == dataset) & (dec.level == "final")].set_index("term")
        for i, (term, _, color) in enumerate(terms):
            r = sub.loc[term]
            ax.bar(i, r["mean"], color=color, yerr=[[r["mean"] - r.ci_low], [r.ci_high - r["mean"]]], capsize=4)
        cost = interpolated[dataset]["next_day_cost_at_a"]
        ax.plot([-0.4, 0.4], [cost, cost], "k--", lw=1.2, label="input-kind interpolation (cost)")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(terms)), [t[1] for t in terms], fontsize=8)
        ax.set_ylabel("next-day accuracy (points)")
        ax.set_title({"bnci2a": "BCI IV 2a, 6 dropped", "lee2019": "Lee 2019, 46 dropped"}[dataset])
        ax.grid(alpha=0.3, axis="y")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_residual_surrogate.png", dpi=200)
    plt.close(fig)


def fmt(d: dict) -> str:
    return f"{d['mean']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    df, plan = load()
    kind = json.loads((SUPP / "input_kind_study" / "analysis.json").read_text())
    unexplained = {int(s): v["mean"] for s, v in kind["predictability"]["lee2019"]["per_participant"].items()}
    dec_rows = decomposition(df)
    dec = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "ci"} | {"ci_low": r["ci"][0], "ci_high": r["ci"][1]} for r in dec_rows]
    )
    dec.to_csv(OUT / "decomposition.csv", index=False, float_format="%.3f")
    table = conditions(df)
    table.to_csv(OUT / "conditions.csv", index=False, float_format="%.3f")
    lee = tests(df, unexplained)
    split = seed_split(df)
    figure(dec, kind["a_equivalent"])
    write_json(OUT / "analysis.json", {"tests_lee2019": lee, "decomposition": dec_rows,
                                       "conditions": records_json(table), "seed_split_cost": split,
                                       "interpolated_input_kind": kind["a_equivalent"]})  # fmt: skip
    lines = [
        "# How much the dropped electrodes add, and what they cost",
        "",
        "Extra inputs after the 16 kept electrodes (2a: the 6 dropped, Lee2019: the 46 dropped): the real "
        "electrodes; their least-squares prediction from the 16 (fitted on training trials only); and the "
        "prediction plus another trial's residual (surr, mean of the draws). Points of accuracy, 95% participant "
        f"intervals. Lee2019 tests were fixed in the plan ({plan['created_at'][:16]}); BCI IV 2a is exploratory.",
        "",
        "## Lee2019 tests (54 people, final models, next day)",
        "",
        f"- S1, the unpredictable part carries information (real minus surr): {fmt(lee['S1'])}: "
        f"{'holds' if lee['S1']['holds'] else 'does not hold'}",
        f"- S2, a residual unrelated to the trial still costs (surr minus 16 alone): {fmt(lee['S2'])}: "
        f"{'holds' if lee['S2']['holds'] else 'does not hold'}",
        f"- S3, a fully predictable input has no visible cost (pred minus 16 alone): {fmt(lee['S3'])}: "
        f"{'holds' if lee['S3']['holds'] else 'does not hold'} (lower bound above -2)",
        f"- S4, more unexplained, larger cost (Spearman over {lee['S4']['people']} people): rho "
        f"{lee['S4']['rho']:+.3f} [{lee['S4']['ci'][0]:+.3f}, {lee['S4']['ci'][1]:+.3f}]: "
        f"{'holds' if lee['S4']['holds'] else 'does not hold'}",
        f"- Post hoc, for S4: a person's cost from seed 0 against seeds 1-2, Spearman over {split['people']} "
        f"people: rho {split['rho']:+.3f} (does a person's cost repeat at all?)",
        "",
        "## Decomposition",
        "",
        "| dataset | measure | term | contrast | mean | 95% interval | worse | seed error |",
        "|---|---|---|---|---:|---|---:|---:|",
    ]
    for r in dec_rows:
        se = f"{r['seed_error']:.2f}" if "seed_error" in r else ""
        lines.append(
            f"| {r['dataset']} | {r['measure']} | {r['term']} | {r['a']} - {r['b']} | {r['mean']:+.2f} | "
            f"[{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | {r['worse']}/{r['people']} | {se} |"
        )
    lines += [
        "",
        "Interpolated in the input-kind study (cost of the real electrodes, next day): "
        + ", ".join(f"{d} {v['next_day_cost_at_a']:+.2f}" for d, v in kind["a_equivalent"].items())
        + ". On Lee2019's held-out runs, full-62 stands in for real.",
        "",
        "## Every condition",
        "",
        "| dataset | condition | next day % | held-out runs % | training set % | residual share (train) | extra "
        "inputs' share of the spatial filters |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r.dataset} | {NAMES[r.condition]} | {r.next_day:.2f} | {r.held_out:.2f} | {r.train_final:.1f} | "
        f"{r.residual_share:.3f} | {r.extra_share:.0%} |".replace("| nan |", "| |").replace("nan%", "")
        for r in table.itertuples()
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
