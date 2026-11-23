"""Tables, tests and figures for the input-penalty study (scripts/input_penalty_study.py).

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only, never EEG:
  supplementary/input_penalty_study/{protocol.json,selection.json,runs,test}   penalized models
  supplementary/{overfit_study,input_cost_study,electrode_data_study}/{runs,test}  the same conditions
                                                                                 without the penalty (lam = 0)

Writes to RESULTS_DIR/supplementary/input_penalty_study/: selection.csv, finals.csv, norms.csv,
fig_selection.png, fig_finals.png, summary.md and analysis.json.

lam was chosen per dataset and condition on held-out runs of the training day (selection.json).
Next-day accuracy is averaged over seeds within each participant; contrasts are paired over
participants with the frozen bootstrap (mibci.stats). Lee2019 tests G1-G3 are the plan's; BCI IV 2a
is exploratory.
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
OUT = SUPP / "input_penalty_study"
TITLES = {"bnci2a": "BCI IV 2a (9 people)", "lee2019": "Lee 2019 (54 people)"}
FULL = {"bnci2a": "full-22", "lee2019": "full-62"}
NOISE = {"bnci2a": ("relevance-16-plus-6", "relevance-16-plus-16"), "lee2019": ("relevance-16-plus-46",)}


def records_json(df: pd.DataFrame) -> list[dict]:
    """Rows as dicts with NaN written as null (write_json refuses NaN)."""
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def paired(d) -> dict:
    d = np.asarray(d, dtype=float)
    lo, hi = bootstrap_ci(d)
    return {"mean": float(d.mean()), "ci": [lo, hi], "better": int((d > 0).sum()), "people": len(d)}


def fmt(d: dict) -> str:
    return f"{d['mean']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}] ({d['better']}/{d['people']} better)"


def unpenalized(plan: dict, job: dict) -> tuple[str, str]:
    study, pattern = plan["unpenalized"][f"{job['dataset']}|{job['condition']}"]
    split = job["split"]
    suffix = "" if split == "final" else f"_{split}"
    return study, pattern.format(s=job["subject"], seed=job["seed"], split=split, suffix=suffix)


def finals(plan: dict, chosen: dict) -> pd.DataFrame:
    """Next-day accuracy of every final model, without the penalty and at the selected lam."""
    rows = []
    for job in plan["jobs"]:
        if job["split"] != "final" or job["lam"] != plan["lams"][0]:
            continue
        lam = chosen[f"{job['dataset']}|{job['condition']}"]["selected_lam"]
        study, name = unpenalized(plan, job)
        old = json.loads((SUPP / study / "test" / f"{name}.json").read_text())["primary"]["acc"]
        if lam:
            selected = {**job, "lam": lam}
            selected["name"] = job["name"].replace(f"_lam-{job['lam']}_", f"_lam-{lam}_")
            new = json.loads((OUT / "test" / f"{selected['name']}.json").read_text())["primary"]["acc"]
        else:
            new = old
        rows.append(
            {
                "dataset": job["dataset"],
                "condition": job["condition"],
                "subject": job["subject"],
                "seed": job["seed"],
                "lam": lam,
                "unpenalized": 100 * old,
                "penalized": 100 * new,
            }
        )
    return pd.DataFrame(rows)


def norms(plan: dict) -> pd.DataFrame:
    """Input norms of the penalized models: noise inputs against electrodes, per lam (fold and final models)."""
    rows = []
    for job in plan["jobs"]:
        path = OUT / "runs" / f"{job['name']}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        n = np.array(r["input_norms"])
        electrodes = n[: len(job["channels"])]
        noise = n[len(job["channels"]) :]
        rows.append(
            {
                "dataset": job["dataset"],
                "condition": job["condition"],
                "lam": job["lam"],
                "final": job["split"] == "final",
                "electrode_norm": float(np.median(electrodes)),
                "noise_norm": float(np.median(noise)) if len(noise) else np.nan,
                "electrodes_off": float((electrodes < 0.05 * electrodes.max()).mean()),
            }
        )
    return pd.DataFrame(rows)


def contrasts(f: pd.DataFrame) -> dict:
    pp = f.groupby(["dataset", "condition", "subject"])[["unpenalized", "penalized"]].mean()
    out = {}
    for dataset in ("bnci2a", "lee2019"):
        t = pp.loc[dataset]

        def col(condition, which, t=t):
            return t.loc[condition][which]

        full = FULL[dataset]
        out[dataset] = {
            f"{full} minus relevance-16, unpenalized": paired(
                col(full, "unpenalized") - col("relevance-16", "unpenalized")
            ),
            f"{full} minus relevance-16, penalized": paired(col(full, "penalized") - col("relevance-16", "penalized")),
            "relevance-16: penalized minus unpenalized": paired(
                col("relevance-16", "penalized") - col("relevance-16", "unpenalized")
            ),
            f"{full}: penalized minus unpenalized": paired(col(full, "penalized") - col(full, "unpenalized")),
        }
        for noisy in NOISE[dataset]:
            cost_new = col(noisy, "penalized") - col("relevance-16", "penalized")
            cost_old = col(noisy, "unpenalized") - col("relevance-16", "unpenalized")
            out[dataset][f"{noisy}: cost unpenalized"] = paired(cost_old)
            out[dataset][f"{noisy}: cost penalized"] = paired(cost_new)
            out[dataset][f"{noisy}: penalized cost minus unpenalized cost"] = paired(cost_new - cost_old)
    return out


def figures(plan: dict, sel: pd.DataFrame, means: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, dataset in zip(axes, ("bnci2a", "lee2019"), strict=True):
        for condition, g in sel[sel.dataset == dataset].groupby("condition"):
            g = g.sort_values("lam")
            x = np.where(g.lam == 0, plan["lams"][0] / 3, g.lam)
            ax.plot(x, g.held_out, "o-", label=condition)
        ax.set_xscale("log")
        ax.set_xlabel("penalty strength lam (leftmost point: no penalty)")
        ax.set_ylabel("held-out runs of the training day, accuracy %")
        ax.set_title(TITLES[dataset])
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_selection.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, dataset in zip(axes, ("bnci2a", "lee2019"), strict=True):
        g = means[means.dataset == dataset].reset_index(drop=True)
        x = np.arange(len(g))
        ax.bar(x - 0.2, g.unpenalized, 0.4, color="#9ec5e8", label="as frozen (no penalty)")
        ax.bar(x + 0.2, g.penalized, 0.4, color="#13294B", label="group-lasso penalty, lam chosen on day 1")
        ax.set_xticks(x, [f"{cnd}\nlam = {lam:g}" for cnd, lam in zip(g.condition, g.lam, strict=True)], fontsize=8)
        low = min(g.unpenalized.min(), g.penalized.min())
        ax.set_ylim(low - 8, max(g.unpenalized.max(), g.penalized.max()) + 3)
        ax.set_ylabel("next-day accuracy %")
        ax.set_title(TITLES[dataset])
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_finals.png", dpi=200)
    plt.close(fig)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    plan = json.loads((OUT / "protocol.json").read_text())
    chosen = json.loads((OUT / "selection.json").read_text())["selection"]
    sel = pd.DataFrame(
        [
            {"dataset": k.split("|")[0], "condition": k.split("|")[1], "lam": float(lam), "held_out": 100 * acc}
            for k, v in chosen.items()
            for lam, acc in v["held_out_accuracy"].items()
        ]
    )
    sel.to_csv(OUT / "selection.csv", index=False, float_format="%.3f")
    f = finals(plan, chosen)
    f.to_csv(OUT / "finals.csv", index=False, float_format="%.3f")
    nm = norms(plan)
    nm_summary = nm.groupby(["dataset", "condition", "lam", "final"]).mean().reset_index()
    nm_summary.to_csv(OUT / "norms.csv", index=False, float_format="%.4f")
    c = contrasts(f)
    lee = c["lee2019"]
    g3_rows = nm_summary[
        (nm_summary.dataset == "lee2019")
        & (nm_summary.condition == "relevance-16-plus-46")
        & nm_summary.final
        & (nm_summary.lam == chosen["lee2019|relevance-16-plus-46"]["selected_lam"])
    ]
    g3_ratio = float(g3_rows.noise_norm.iloc[0] / g3_rows.electrode_norm.iloc[0]) if len(g3_rows) else np.nan
    tests = {
        "G1": {
            **lee["relevance-16-plus-46: penalized cost minus unpenalized cost"],
            "rule": "interval lower bound above 0",
        },
        "G2": {**lee["full-62 minus relevance-16, penalized"], "rule": "interval lower bound above 0"},
        "G3": {
            "noise_over_electrode_norm": None if np.isnan(g3_ratio) else g3_ratio,
            "rule": "below 1/3 (only if a lam > 0 is selected)",
        },
    }
    tests["G1"]["holds"] = bool(tests["G1"]["ci"][0] > 0)
    tests["G2"]["holds"] = bool(tests["G2"]["ci"][0] > 0)
    tests["G3"]["holds"] = bool(g3_ratio < 1 / 3) if not np.isnan(g3_ratio) else None
    g3_verdict = {True: "holds", False: "does not hold", None: "not applicable"}[tests["G3"]["holds"]]

    means = f.groupby(["dataset", "condition"])[["unpenalized", "penalized", "lam"]].mean().reset_index()
    figures(plan, sel, means)

    write_json(
        OUT / "analysis.json",
        {
            "tests_lee2019": tests,
            "contrasts": c,
            "selection": chosen,
            "finals": records_json(means),
            "norms": records_json(nm_summary),
        },
    )
    lines = [
        "# Letting EEGNet switch inputs off: a group-lasso penalty on the spatial weights",
        "",
        "## lam chosen on held-out runs of the training day (accuracy %)",
        "",
        "| dataset | condition | " + " | ".join(f"lam {lam:g}" for lam in (0.0, *plan["lams"])) + " | selected |",
        "|---|---|" + "---:|" * (len(plan["lams"]) + 2),
    ]
    for key, v in chosen.items():
        d, cnd = key.split("|")
        accs = " | ".join(f"{100 * v['held_out_accuracy'][str(lam)]:.2f}" for lam in (0.0, *plan["lams"]))
        lines.append(f"| {d} | {cnd} | {accs} | {v['selected_lam']:g} |")
    lines += [
        "",
        "## Next day, final models (accuracy %)",
        "",
        "| dataset | condition | selected lam | no penalty | penalty |",
        "|---|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r.dataset} | {r.condition} | {r.lam:g} | {r.unpenalized:.2f} | {r.penalized:.2f} |"
        for r in means.itertuples()
    ]
    lines += ["", "## Contrasts (next day, pp, 95% participant interval)", ""]
    for dataset, items in c.items():
        lines += [f"### {dataset}", ""]
        lines += [f"- {name}: {fmt(v)}" for name, v in items.items()]
        lines.append("")
    lines += [
        "## Lee2019 tests",
        "",
        f"- G1 (inputs that carry nothing cost less with the penalty): {fmt(tests['G1'])}: "
        f"{'holds' if tests['G1']['holds'] else 'does not hold'}",
        f"- G2 (with the penalty all 62 electrodes beat the best 16): {fmt(tests['G2'])}: "
        f"{'holds' if tests['G2']['holds'] else 'does not hold'}",
        f"- G3 (noise inputs switched off): median noise-input norm / electrode norm = {g3_ratio:.3f} "
        f"(nan: lam = 0 was selected, so no penalized model): {g3_verdict}",
        "",
        "## Input norms of the penalized models (medians)",
        "",
        "| dataset | condition | lam | final | electrodes | noise inputs | electrodes switched off |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r.dataset} | {r.condition} | {r.lam:g} | {'yes' if r.final else 'folds'} | {r.electrode_norm:.3f} | "
        f"{'' if np.isnan(r.noise_norm) else f'{r.noise_norm:.3f}'} | {r.electrodes_off:.0%} |"
        for r in nm_summary.itertuples()
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
