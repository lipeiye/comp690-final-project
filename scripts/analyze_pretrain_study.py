"""Tables, tests and figures for the pretraining study (scripts/pretrain_study.py).

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only, never EEG:
  supplementary/pretrain_study/{protocol.json,budget.json,pretrain,runs,test}
  supplementary/overfit_study/test      the all-electrode models trained from scratch (full-22, full-62)

Writes to RESULTS_DIR/supplementary/pretrain_study/: accuracy.csv, curves.csv, fig_pretrain.png,
summary.md and analysis.json.

Per target participant, next-day accuracy is the mean over seeds (2a 0-4, Lee2019 0-2) of each
configuration (all electrodes, the shared group-16, group-16 plus noise) and training (scratch,
pretrain only, pretrain then fine-tune). Contrasts are paired over participants (frozen bootstrap,
mibci.stats). group-16 is one electrode set shared by everyone in a pretraining run, not each person's
own 16. Lee2019 tests T1-T3 are the plan's; BCI IV 2a is exploratory.
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
OUT = SUPP / "pretrain_study"
TRAINING = ("scratch", "pretrain", "finetune")
TITLES = {
    "bnci2a": "BCI IV 2a (9 targets, pretrained on the other 8)",
    "lee2019": "Lee 2019 (54 targets, pretrained on 45 others)",
}
COLORS = {"full": "#13294B", "group-16": "#4B9CD3", "noise": "#C8102E"}


def records_json(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def paired(d) -> dict:
    d = np.asarray(d, float)
    lo, hi = bootstrap_ci(d)
    return {"mean": float(d.mean()), "ci": [lo, hi], "better": int((d > 0).sum()), "people": len(d)}


def fmt(d: dict) -> str:
    return f"{d['mean']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}] ({d['better']}/{d['people']} better)"


def verdict(holds: bool) -> str:
    return "holds" if holds else "does not hold"


def role(config: str) -> str:
    return "full" if config.startswith("full") else "noise" if "plus" in config else "group-16"


def load(plan: dict) -> pd.DataFrame:
    rows, missing = [], 0
    seeds = {d: set(s["seeds"]) for d, s in plan["datasets"].items()}
    for job in plan["jobs"]:
        if job["kind"] == "pretrain":
            continue
        path = OUT / "test" / f"{job['name']}.json"
        if not path.exists():
            missing += 1
            continue
        t = json.loads(path.read_text())
        training = {"pretrain-only": "pretrain", "finetune": "finetune", "scratch": "scratch"}[job["kind"]]
        rows.append(
            {
                "dataset": job["dataset"],
                "subject": job["subject"],
                "seed": job["seed"],
                "config": job["config"],
                "role": role(job["config"]),
                "training": training,
                "next_day": 100 * t["primary"]["acc"],
            }
        )
    if missing:
        raise SystemExit(f"{missing} jobs lack a test record")
    # all electrodes from scratch: the overfit study's final models at the same seeds
    for job in json.loads((SUPP / "overfit_study" / "protocol.json").read_text())["jobs"]:
        if job["split"] != "final" or job["condition"] not in ("full-22", "full-62"):
            continue
        if job["seed"] not in seeds[job["dataset"]]:
            continue
        t = json.loads((SUPP / "overfit_study" / "test" / f"{job['name']}.json").read_text())
        rows.append(
            {
                "dataset": job["dataset"],
                "subject": job["subject"],
                "seed": job["seed"],
                "config": job["condition"],
                "role": "full",
                "training": "scratch",
                "next_day": 100 * t["primary"]["acc"],
            }
        )
    return pd.DataFrame(rows)


def cell(df: pd.DataFrame, dataset: str, role_: str, training: str) -> pd.Series:
    sel = df[(df.dataset == dataset) & (df.role == role_) & (df.training == training)]
    return sel.groupby("subject").next_day.mean()


def contrasts(df: pd.DataFrame, dataset: str) -> dict:
    c = {(r, t): cell(df, dataset, r, t) for r in COLORS for t in TRAINING}
    out = {}
    for t in TRAINING:
        out[f"{t}: all electrodes minus group-16"] = paired(c[("full", t)] - c[("group-16", t)])
        out[f"{t}: noise cost (group-16 plus noise minus group-16)"] = paired(c[("noise", t)] - c[("group-16", t)])
    for r in COLORS:
        out[f"{r}: finetune minus scratch"] = paired(c[(r, "finetune")] - c[(r, "scratch")])
        out[f"{r}: pretrain only minus scratch"] = paired(c[(r, "pretrain")] - c[(r, "scratch")])
        out[f"{r}: finetune minus pretrain only"] = paired(c[(r, "finetune")] - c[(r, "pretrain")])
    noise_ft = c[("noise", "finetune")] - c[("group-16", "finetune")]
    noise_sc = c[("noise", "scratch")] - c[("group-16", "scratch")]
    out["noise cost after finetune minus from scratch"] = paired(noise_ft - noise_sc)
    gap_ft = c[("full", "finetune")] - c[("group-16", "finetune")]
    gap_sc = c[("full", "scratch")] - c[("group-16", "scratch")]
    out["(all minus group-16) after finetune minus from scratch"] = paired(gap_ft - gap_sc)
    return out


def curves(plan: dict) -> pd.DataFrame:
    rows = []
    for job in plan["jobs"]:
        if job["kind"] != "pretrain":
            continue
        r = json.loads((OUT / "pretrain" / f"{job['name']}.json").read_text())
        rows += [
            {
                "dataset": job["dataset"],
                "config": job["config"],
                "seed": job["seed"],
                "group": job["group"],
                "epoch": i + 1,
                "target_day1": 100 * a,
                "train_loss": tl,
            }
            for i, (a, tl) in enumerate(zip(r["history"]["val_acc"], r["history"]["train_loss"], strict=True))
        ]
    return pd.DataFrame(rows)


def figure(table: pd.DataFrame, cur: pd.DataFrame, budget: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, dataset in enumerate(("bnci2a", "lee2019")):
        ax = axes[0, col]
        mean = cur[cur.dataset == dataset].groupby(["config", "epoch"]).target_day1.mean().reset_index()
        for config, c in mean.groupby("config"):
            ax.plot(c.epoch, c.target_day1, color=COLORS[role(config)], label=config)
        ax.axvline(budget[dataset]["epochs"], color="grey", ls="--", lw=1, label=f"chosen: {budget[dataset]['epochs']}")
        ax.set(xlabel="pretraining epoch", ylabel="targets' training day, % (no fine-tuning)", title=TITLES[dataset])
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
        ax = axes[1, col]
        t = table[table.dataset == dataset]
        x = np.arange(len(TRAINING))
        for i, r in enumerate(COLORS):
            vals = [t[(t.role == r) & (t.training == tr)].next_day.mean() for tr in TRAINING]
            ax.bar(x + (i - 1) * 0.27, vals, width=0.27, color=COLORS[r], label=r)
        ax.set_xticks(x, ["from scratch", "pretrained only", "pretrained + fine-tuned"])
        ax.set(ylabel="next day, %", title=f"{TITLES[dataset].split(' (')[0]}: next day")
        lo = np.nanmin(t.next_day) - 5
        ax.set_ylim(max(lo, 0), None)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pretrain.png", dpi=200)
    plt.close(fig)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    plan = json.loads((OUT / "protocol.json").read_text())
    budget = json.loads((OUT / "budget.json").read_text())["budget"]
    df = load(plan)
    pp = df.groupby(["dataset", "config", "role", "training", "subject"]).next_day.mean().reset_index()
    table = pp.groupby(["dataset", "config", "role", "training"]).next_day.mean().reset_index()
    table.to_csv(OUT / "accuracy.csv", index=False, float_format="%.3f")
    cur = curves(plan)
    cur.groupby(["dataset", "config", "epoch"])[["target_day1", "train_loss"]].mean().reset_index().to_csv(
        OUT / "curves.csv", index=False, float_format="%.4f"
    )
    lee, two_a = contrasts(df, "lee2019"), contrasts(df, "bnci2a")
    tests = {
        "T1": {**lee["full: finetune minus scratch"], "rule": "interval lower bound above 0"},
        "T2": {**lee["noise cost after finetune minus from scratch"], "rule": "interval lower bound above 0"},
        "T3": {**lee["finetune: all electrodes minus group-16"], "rule": "interval lower bound above 0"},
    }
    for t in tests.values():
        t["holds"] = bool(t["ci"][0] > 0)
    figure(pp, cur, budget)
    write_json(
        OUT / "analysis.json",
        {
            "tests_lee2019": tests,
            "lee2019": lee,
            "bnci2a": two_a,
            "budget": {d: {k: v for k, v in b.items() if k != "smoothed"} for d, b in budget.items()},
            "accuracy": records_json(table),
        },
    )
    lines = [
        "# Pretraining on other people, then fine-tuning",
        "",
        "Accuracy in %, differences in points with the 95% participant interval. group-16 is the 16 electrodes with "
        "the best mean relevance rank among the pretraining participants, shared by everyone in that run. Lee2019 "
        f"tests were fixed in the plan ({plan['created_at'][:16]}); BCI IV 2a is exploratory.",
        "",
        "Pretraining epochs chosen on the targets' training days (the project's budget rule): "
        + "; ".join(
            f"{d} {b['epochs']} (smoothed maximum {100 * b['smoothed_max']:.2f}% at epoch {b['smoothed_max_epoch']})"
            for d, b in budget.items()
        ),
        "",
        "## Lee2019 tests",
        "",
        f"- T1, with all 62 electrodes, pretraining and fine-tuning beat training from scratch: {fmt(tests['T1'])}: "
        f"{verdict(tests['T1']['holds'])}",
        f"- T2, after pretraining and fine-tuning the 46 noise inputs cost less than from scratch: {fmt(tests['T2'])}: "
        f"{verdict(tests['T2']['holds'])}",
        f"- T3, after pretraining and fine-tuning, all 62 electrodes beat the shared 16: {fmt(tests['T3'])}: "
        f"{verdict(tests['T3']['holds'])}",
        "",
        "## Next-day accuracy",
        "",
        "| dataset | configuration | from scratch | pretrained only | pretrained + fine-tuned |",
        "|---|---|---:|---:|---:|",
    ]
    for (dataset, config), c in table.groupby(["dataset", "config"]):
        c = c.set_index("training").next_day
        lines.append(f"| {dataset} | {config} | " + " | ".join(f"{c.get(t, np.nan):.2f}" for t in TRAINING) + " |")
    for dataset, block in (("lee2019", lee), ("bnci2a", two_a)):
        lines += [
            "",
            f"## Contrasts, {dataset}",
            "",
            "| contrast | mean | 95% interval | better |",
            "|---|---:|---|---:|",
        ]
        lines += [
            f"| {k} | {v['mean']:+.2f} | [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] | {v['better']}/{v['people']} |"
            for k, v in block.items()
        ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
