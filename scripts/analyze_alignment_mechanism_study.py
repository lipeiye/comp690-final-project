"""Tables, readings and figures for the alignment-mechanism study (scripts/alignment_mechanism_study.py).

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only, never EEG:
  supplementary/alignment_mechanism_study/{protocol.json,spectra,runs,test}
  supplementary/overfit_study/{runs,test}        unaligned full-22 / full-62 models, seeds 0-2
  supplementary/alignment_study/test             2a plain alignment (full-22-ea), and the 2a day-2 run and
                                                 artifact flags for scoring on runs 2-6
  supplementary/replication_study/test           Lee2019 plain alignment (full-62-ea)

Writes to RESULTS_DIR/supplementary/alignment_mechanism_study/: spectra.csv, finals.csv, folds.csv,
fig_alignment_mechanism.png, summary.md and analysis.json.

Per participant, accuracy is the mean over seeds (and folds). Final models are scored the next day with
that day aligned with all of its trials (2a on artifact-free trials, Lee2019 on all), and secondarily
with its first run (2a, scored on runs 2-6) or offline run (Lee2019, scored on the online run). Fold
models are scored on held-out runs of the training day, each side aligned with its own covariance.
Contrasts are paired over participants (frozen bootstrap, mibci.stats). Exploratory on both datasets;
M1-M3 are the plan's expectations.
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
OUT = SUPP / "alignment_mechanism_study"
VARIANTS = ("ea", "ea-shrink10", "ea-lw", "ea-top", "ea-floor")
FULL = {"bnci2a": "full-22", "lee2019": "full-62"}
SECONDARY = {"bnci2a": "first-run", "lee2019": "offline-run"}
TITLES = {"bnci2a": "BCI IV 2a (9 people, 22 electrodes)", "lee2019": "Lee 2019 (54 people, 62 electrodes)"}


def records_json(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def paired(d) -> dict:
    d = np.asarray(d, float)
    lo, hi = bootstrap_ci(d)
    return {"mean": float(d.mean()), "ci": [lo, hi], "better": int((d > 0).sum()), "people": len(d)}


def fmt(d: dict) -> str:
    return f"{d['mean']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}] ({d['better']}/{d['people']} better)"


def read(path) -> dict:
    return json.loads(path.read_text())


def finals(plan: dict) -> pd.DataFrame:
    rows, missing = [], 0
    for job in plan["jobs"]:
        if job["kind"] != "train" or job["split"] != "final":
            continue
        path = OUT / "test" / f"{job['name']}.json"
        if not path.exists():
            missing += 1
            continue
        a = json.loads(path.read_text())["aligned_with"]
        rows.append(
            {
                "dataset": job["dataset"],
                "subject": job["subject"],
                "seed": job["seed"],
                "variant": job["variant"],
                "all": 100 * a["all"]["acc"],
                "secondary": 100 * a[SECONDARY[job["dataset"]]]["acc"],
            }
        )
    if missing:
        raise SystemExit(f"{missing} final models lack a test record")
    seeds = set(plan["seeds"])
    for subject in range(1, 10):  # 2a references
        for seed in seeds:
            ea = read(SUPP / "alignment_study" / "test" / f"sub-{subject:02d}_seed-{seed}_full-22-ea.json")
            un = read(SUPP / "overfit_study" / "test" / f"bnci2a_sub-{subject:02d}_seed-{seed}_full-22.json")
            clean, later = ~np.array(ea["artifact"], bool), np.array(ea["run"]) > 0
            y, pred = np.array(un["y"]), np.array(un["pred"])
            rows.append(
                {
                    "dataset": "bnci2a",
                    "subject": subject,
                    "seed": seed,
                    "variant": "ea",
                    "all": 100 * ea["acc"]["ea-all"],
                    "secondary": 100 * ea["acc"]["ea-first-run"],
                }
            )
            rows.append(
                {
                    "dataset": "bnci2a",
                    "subject": subject,
                    "seed": seed,
                    "variant": "none",
                    "all": 100 * un["primary"]["acc"],
                    "secondary": 100 * float((pred[clean & later] == y[clean & later]).mean()),
                }
            )
    for subject in range(1, 55):  # Lee2019 references
        for seed in seeds:
            ea = read(SUPP / "replication_study" / "test" / f"sub-{subject:02d}_seed-{seed}_full-62-ea.json")
            un = read(SUPP / "replication_study" / "test" / f"sub-{subject:02d}_seed-{seed}_full-62.json")
            rows.append(
                {
                    "dataset": "lee2019",
                    "subject": subject,
                    "seed": seed,
                    "variant": "ea",
                    "all": 100 * ea["scores"]["ea-all"]["acc_all"],
                    "secondary": 100 * ea["scores"]["ea-offline"]["acc_online"],
                }
            )
            rows.append(
                {
                    "dataset": "lee2019",
                    "subject": subject,
                    "seed": seed,
                    "variant": "none",
                    "all": 100 * un["scores"]["fp32"]["acc_all"],
                    "secondary": 100 * un["scores"]["fp32"]["acc_online"],
                }
            )
    return pd.DataFrame(rows)


def folds(plan: dict) -> pd.DataFrame:
    rows, missing = [], 0
    seeds = set(plan["seeds"])
    for job in plan["jobs"]:
        if job["kind"] != "train" or job["split"] == "final":
            continue
        path = OUT / "runs" / f"{job['name']}.json"
        if not path.exists():
            missing += 1
            continue
        r = json.loads(path.read_text())
        rows.append(
            {
                "dataset": job["dataset"],
                "subject": job["subject"],
                "seed": job["seed"],
                "split": job["split"],
                "variant": job["variant"],
                "held_out": 100 * r["within_day"]["acc"],
            }
        )
    if missing:
        raise SystemExit(f"{missing} fold models lack a run record")
    for job in json.loads((SUPP / "overfit_study" / "protocol.json").read_text())["jobs"]:
        wanted = job["condition"] in ("full-22", "full-62") and job["split"].startswith("fold-")
        if not wanted or job["seed"] not in seeds:
            continue
        r = json.loads((SUPP / "overfit_study" / "runs" / f"{job['name']}.json").read_text())
        rows.append(
            {
                "dataset": job["dataset"],
                "subject": job["subject"],
                "seed": job["seed"],
                "split": job["split"],
                "variant": "none",
                "held_out": 100 * r["within_day"]["acc"],
            }
        )
    return pd.DataFrame(rows)


def spectra(plan: dict) -> pd.DataFrame:
    rows = []
    for job in plan["jobs"]:
        if job["kind"] != "spectrum":
            continue
        r = json.loads((OUT / "spectra" / f"{job['name']}.json").read_text())
        values = np.array(r["eigenvalues"])
        rows.append(
            {
                "dataset": job["dataset"],
                "subject": job["subject"],
                "montage": job["montage"],
                "electrodes": len(values),
                "condition_number": r["condition_number"],
                "below_1pct": r["below_1pct_of_largest"],
                "top_share": float(values[0] / values.sum()),
                "ledoit_wolf": r["ledoit_wolf"],
                "split_half": r["split_half_whitener_difference"],
            }
        )
    return pd.DataFrame(rows)


def per_person(df: pd.DataFrame, dataset: str, variant: str, column: str) -> pd.Series:
    sel = df[(df.dataset == dataset) & (df.variant == variant)]
    return sel.groupby("subject")[column].mean()


def contrasts(fin: pd.DataFrame, fol: pd.DataFrame) -> dict:
    out = {}
    for dataset in ("bnci2a", "lee2019"):
        for v in VARIANTS:
            for name, df, column in (
                ("next day", fin, "all"),
                ("next day, secondary", fin, "secondary"),
                ("held-out runs", fol, "held_out"),
            ):
                if v not in set(df[df.dataset == dataset].variant):
                    continue
                out[f"{dataset}|{name}|{v} minus unaligned"] = paired(
                    per_person(df, dataset, v, column) - per_person(df, dataset, "none", column)
                )
                if v != "ea":
                    out[f"{dataset}|{name}|{v} minus ea"] = paired(
                        per_person(df, dataset, v, column) - per_person(df, dataset, "ea", column)
                    )
    return out


def readings(fin: pd.DataFrame, spec: pd.DataFrame, pairs: dict) -> dict:
    m1 = pairs["lee2019|held-out runs|ea minus unaligned"]
    harm = per_person(fin, "lee2019", "none", "all") - per_person(fin, "lee2019", "ea", "all")
    shares = {}
    for v in ("ea-top", "ea-shrink10", "ea-floor", "ea-lw"):
        gain = per_person(fin, "lee2019", v, "all") - per_person(fin, "lee2019", "ea", "all")
        shares[v] = {"share_of_harm_removed": float(gain.mean() / harm.mean()), "gain_over_ea": paired(gain)}
    lw = spec[(spec.dataset == "lee2019") & (spec.montage == "full-62")].ledoit_wolf
    return {
        "M1": {**m1, "reading": "interval upper bound below 0", "holds": bool(m1["ci"][1] < 0)},
        "M2": {
            "harm": paired(harm),
            "variants": shares,
            "reading": "at least 0.5 for ea-top and ea-shrink10",
            "holds": bool(
                shares["ea-top"]["share_of_harm_removed"] >= 0.5
                and shares["ea-shrink10"]["share_of_harm_removed"] >= 0.5
            ),
        },
        "M3": {
            "mean_coefficient": float(lw.mean()),
            "max_coefficient": float(lw.max()),
            "reading": "below 0.01",
            "holds": bool(lw.mean() < 0.01),
        },
    }


def figure(fin: pd.DataFrame, fol: pd.DataFrame, spec: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    order = ("none", *VARIANTS)
    for i, dataset in enumerate(("bnci2a", "lee2019")):
        nd = [per_person(fin, dataset, v, "all").mean() for v in order]
        ho = [per_person(fol, dataset, v, "held_out").mean() for v in order]
        x = np.arange(len(order))
        axes[i].bar(x - 0.2, nd, 0.4, color="#13294B", label="next day (final models)")
        axes[i].bar(x + 0.2, ho, 0.4, color="#4B9CD3", label="held-out runs of the training day (fold models)")
        axes[i].set_xticks(x, ["unaligned", *VARIANTS], rotation=20)
        axes[i].set_ylim(min(nd + ho) - 8, max(nd + ho) + 4)
        axes[i].set(ylabel="accuracy, %", title=TITLES[dataset])
        axes[i].grid(alpha=0.3, axis="y")
        axes[i].legend(frameon=False, fontsize=7)
    ax = axes[2]
    for (dataset, montage), c in spec.groupby(["dataset", "montage"]):
        ax.scatter(
            c.electrodes + np.random.default_rng(0).normal(0, 0.4, len(c)),
            c.below_1pct,
            s=12,
            label=f"{dataset} {montage}",
        )
    ax.plot([0, 62], [0, 62], color="grey", lw=0.8, ls=":")
    ax.set(
        xlabel="electrodes",
        ylabel="directions below 1% of the largest eigenvalue",
        title="Training-day covariance of each participant",
    )
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_alignment_mechanism.png", dpi=200)
    plt.close(fig)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    plan = json.loads((OUT / "protocol.json").read_text())
    fin, fol, spec = finals(plan), folds(plan), spectra(plan)
    keys = ["dataset", "variant"]
    fin_t = fin.groupby([*keys, "subject"])[["all", "secondary"]].mean().groupby(keys).mean().reset_index()
    fol_t = fol.groupby([*keys, "subject"]).held_out.mean().groupby(keys).mean().reset_index()
    spec_t = spec.groupby(["dataset", "montage"])[
        ["electrodes", "condition_number", "below_1pct", "top_share", "ledoit_wolf", "split_half"]
    ].agg(["mean", "median", "max"])
    fin_t.to_csv(OUT / "finals.csv", index=False, float_format="%.3f")
    fol_t.to_csv(OUT / "folds.csv", index=False, float_format="%.3f")
    spec.to_csv(OUT / "spectra.csv", index=False, float_format="%.6g")
    pairs = contrasts(fin, fol)
    read = readings(fin, spec, pairs)
    harm = per_person(fin, "lee2019", "none", "all") - per_person(fin, "lee2019", "ea", "all")
    lee_spec = spec[(spec.dataset == "lee2019") & (spec.montage == "full-62")].set_index("subject")
    relation = {
        c: float(np.corrcoef(lee_spec.loc[harm.index, c], harm)[0, 1])
        for c in ("condition_number", "below_1pct", "top_share", "ledoit_wolf", "split_half")
    }
    figure(fin, fol, spec)
    write_json(
        OUT / "analysis.json",
        {
            "readings": read,
            "contrasts": pairs,
            "harm_vs_spectrum_correlation": relation,
            "finals": records_json(fin_t),
            "folds": records_json(fol_t),
        },
    )
    lines = [
        "# Why alignment hurts with 62 electrodes",
        "",
        "Accuracy in %, differences in points with the 95% participant interval. Exploratory on both datasets; the "
        f"readings M1-M3 were written in the plan ({plan['created_at'][:16]}) before training.",
        "",
        "## Readings",
        "",
        f"- M1, plain alignment hurts within the training day too (Lee2019 fold models, ea minus unaligned): "
        f"{fmt(read['M1'])}: {'yes' if read['M1']['holds'] else 'no'}",
        f"- M2, harm on Lee2019 (unaligned minus ea, next day): {fmt(read['M2']['harm'])}; share removed: "
        + ", ".join(f"{v} {s['share_of_harm_removed']:.0%}" for v, s in read["M2"]["variants"].items())
        + f": {'yes' if read['M2']['holds'] else 'no'} (ea-top and ea-shrink10 both at least half)",
        f"- M3, Ledoit-Wolf coefficient (Lee2019, 62 electrodes): mean {read['M3']['mean_coefficient']:.5f}, largest "
        f"{read['M3']['max_coefficient']:.5f}: {'yes' if read['M3']['holds'] else 'no'} (below 0.01)",
        "",
        "## The training-day covariance",
        "",
        "| dataset | montage | condition number (median) | directions below 1% of the largest (mean) | largest "
        "eigenvalue's share | Ledoit-Wolf (mean) | split-half whitener difference (mean) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (dataset, montage), r in spec_t.iterrows():
        lines.append(
            f"| {dataset} | {montage} | {r[('condition_number', 'median')]:.0f} | "
            f"{r[('below_1pct', 'mean')]:.1f} of {r[('electrodes', 'mean')]:.0f} | "
            f"{r[('top_share', 'mean')]:.2f} | {r[('ledoit_wolf', 'mean')]:.5f} | {r[('split_half', 'mean')]:.3f} |"
        )
    lines += [
        "",
        "## Accuracy",
        "",
        "| dataset | alignment | next day (all trials aligned) | next day (secondary) | held-out runs |",
        "|---|---|---:|---:|---:|",
    ]
    for r in fin_t.itertuples():
        ho = fol_t[(fol_t.dataset == r.dataset) & (fol_t.variant == r.variant)].held_out
        lines.append(
            f"| {r.dataset} | {r.variant} | {r.all:.2f} | {r.secondary:.2f} | {ho.iloc[0]:.2f} |"
            if len(ho)
            else f"| {r.dataset} | {r.variant} | {r.all:.2f} | {r.secondary:.2f} | |"
        )
    lines += [
        "",
        "## Contrasts",
        "",
        "| dataset | measure | contrast | mean | 95% interval | better |",
        "|---|---|---|---:|---|---:|",
    ]
    for key, d in pairs.items():
        dataset, measure, what = key.split("|")
        lines.append(
            f"| {dataset} | {measure} | {what} | {d['mean']:+.2f} | [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}] | "
            f"{d['better']}/{d['people']} |"
        )
    lines += [
        "",
        "Lee2019, across participants: correlation of the harm (unaligned minus ea, next day) with "
        + ", ".join(f"{k} {v:+.2f}" for k, v in relation.items()),
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
