"""Tables, tests and figures for the long-training study (scripts/long_training_study.py).

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only, never EEG:
  supplementary/long_training_study/{protocol.json,runs,test}

Writes to RESULTS_DIR/supplementary/long_training_study/: checkpoints.csv, curves.csv,
fig_long_training.png, summary.md and analysis.json.

Per participant, next-day accuracy is the mean over seeds of the final models at one checkpoint
(475, 950 or 1,900 epochs; all from this study, so the 1x reference is the same run stopped earlier).
Held-out accuracy of the training day comes from the fold models, tracked after every epoch, averaged
over seeds and folds within each participant. Contrasts are paired over participants (frozen
bootstrap, mibci.stats); on BCI IV 2a the seed error is the contrast computed seed by seed, its
standard deviation over seeds over the square root of their number. Lee2019 tests L1-L3 are the plan's;
BCI IV 2a is exploratory.
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

OUT = RESULTS_DIR / "supplementary" / "long_training_study"
NOISE = {"bnci2a": "relevance-16-plus-6", "lee2019": "relevance-16-plus-46"}
FULL = {"bnci2a": "full-22", "lee2019": "full-62"}
TITLES = {"bnci2a": "BCI IV 2a (9 people)", "lee2019": "Lee 2019 (54 people)"}
COLORS = {"relevance-16": "#13294B", "noise": "#C8102E", "full": "#4B9CD3"}
EARLY, LATE = (425, 525), (1800, 1900)  # L3's epoch windows, inclusive


def records_json(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def load(plan: dict) -> tuple[pd.DataFrame, list[dict], dict]:
    """Next-day accuracy per final model and checkpoint; fold-model histories; the 475-epoch identity check."""
    finals, folds, identity, missing = [], [], {"found": 0, "identical": 0, "max_abs_diff": 0.0}, 0
    for job in plan["jobs"]:
        run = OUT / "runs" / f"{job['name']}.json"
        if not run.exists():
            missing += 1
            continue
        r = json.loads(run.read_text())
        e = r["earlier_475"]
        if e.get("found"):
            identity["found"] += 1
            identity["identical"] += int(e.get("identical", False))
            identity["max_abs_diff"] = max(identity["max_abs_diff"], float(e.get("max_abs_diff", np.inf)))
        if job["split"] == "final":
            test = OUT / "test" / f"{job['name']}.json"
            if not test.exists():
                missing += 1
                continue
            t = json.loads(test.read_text())
            for epoch, at in t["at"].items():
                finals.append(
                    {
                        "dataset": job["dataset"],
                        "condition": job["condition"],
                        "subject": job["subject"],
                        "seed": job["seed"],
                        "epoch": int(epoch),
                        "next_day": 100 * at["primary"]["acc"],
                        "train": 100 * r["checkpoints"][epoch]["train_acc"],
                    }
                )
        else:
            folds.append(
                {
                    **{k: job[k] for k in ("dataset", "condition", "subject", "seed", "split")},
                    "val_acc": np.array(r["history"]["val_acc"]) * 100,
                    "train_loss": np.array(r["history"]["train_loss"]),
                }
            )
    if missing:
        raise SystemExit(f"{missing} jobs lack a run or test record")
    return pd.DataFrame(finals), folds, identity


def per_person(df: pd.DataFrame, dataset: str, epoch: int) -> pd.DataFrame:
    sel = df[(df.dataset == dataset) & (df.epoch == epoch)]
    return sel.groupby(["subject", "condition"]).next_day.mean().unstack("condition")


def seed_error(df: pd.DataFrame, dataset: str, terms: list[tuple[float, str, int]]) -> float:
    """Seed error of sum(weight * accuracy[condition, epoch]) over the given terms (2a: seeds 0-9)."""
    sel = df[df.dataset == dataset]
    by_seed = sel.groupby(["seed", "condition", "epoch", "subject"]).next_day.mean()
    total = 0
    for weight, condition, epoch in terms:
        total = total + weight * by_seed.xs((condition, epoch), level=("condition", "epoch")).unstack("subject")
    d = total.mean(axis=1)
    return float(d.std(ddof=1) / np.sqrt(len(d)))


def paired(d) -> dict:
    d = np.asarray(d, float)
    lo, hi = bootstrap_ci(d)
    return {"mean": float(d.mean()), "ci": [lo, hi], "worse": int((d < 0).sum()), "people": len(d)}


def combo(df: pd.DataFrame, dataset: str, terms: list[tuple[float, str, int]]) -> dict:
    """A paired contrast sum(weight * accuracy[condition, epoch]) over participants."""
    total = 0
    for weight, condition, epoch in terms:
        total = total + weight * per_person(df, dataset, epoch)[condition]
    out = paired(total.to_numpy())
    if dataset == "bnci2a":
        out["seed_error"] = seed_error(df, dataset, terms)
    return out


def contrasts(df: pd.DataFrame, epochs: list[int]) -> list[dict]:
    out = []
    for dataset in ("bnci2a", "lee2019"):
        noise, full = NOISE[dataset], FULL[dataset]
        for e in epochs:
            out.append(
                {
                    "dataset": dataset,
                    "what": "noise cost",
                    "epoch": e,
                    **combo(df, dataset, [(1, noise, e), (-1, "relevance-16", e)]),
                }
            )
            out.append(
                {
                    "dataset": dataset,
                    "what": "all electrodes minus relevance-16",
                    "epoch": e,
                    **combo(df, dataset, [(1, full, e), (-1, "relevance-16", e)]),
                }
            )
        first, last = min(epochs), max(epochs)
        out.extend(
            {
                "dataset": dataset,
                "what": f"{condition}: {last} minus {first} epochs",
                "epoch": last,
                **combo(df, dataset, [(1, condition, last), (-1, condition, first)]),
            }
            for condition in ("relevance-16", noise, full)
        )
        out.append(
            {
                "dataset": dataset,
                "what": f"noise cost at {last} minus at {first} epochs",
                "epoch": last,
                **combo(
                    df,
                    dataset,
                    [(1, noise, last), (-1, "relevance-16", last), (-1, noise, first), (1, "relevance-16", first)],
                ),
            }
        )
    return out


def window_gain(folds: list[dict], dataset: str, condition: str) -> dict:
    """Per participant: mean held-out accuracy over epochs LATE minus over EARLY, averaged over folds and seeds."""
    per: dict[int, list[float]] = {}
    for f in folds:
        if (f["dataset"], f["condition"]) != (dataset, condition):
            continue
        acc = f["val_acc"]
        late, early = acc[LATE[0] - 1 : LATE[1]].mean(), acc[EARLY[0] - 1 : EARLY[1]].mean()
        per.setdefault(f["subject"], []).append(float(late - early))
    return paired([np.mean(v) for v in per.values()])


def mean_curves(folds: list[dict]) -> pd.DataFrame:
    """Held-out accuracy and training loss after every epoch, mean over participants (each over folds and seeds)."""
    rows = []
    for dataset in ("bnci2a", "lee2019"):
        for condition in sorted({f["condition"] for f in folds if f["dataset"] == dataset}):
            cell = [f for f in folds if (f["dataset"], f["condition"]) == (dataset, condition)]
            by_person: dict[int, list] = {}
            for f in cell:
                by_person.setdefault(f["subject"], []).append(f)
            acc = np.mean([np.mean([f["val_acc"] for f in fs], 0) for fs in by_person.values()], 0)
            loss = np.mean([np.mean([f["train_loss"] for f in fs], 0) for fs in by_person.values()], 0)
            rows += [
                {"dataset": dataset, "condition": condition, "epoch": i + 1, "held_out": a, "train_loss": x}
                for i, (a, x) in enumerate(zip(acc, loss, strict=True))
            ]
    return pd.DataFrame(rows)


def role(dataset: str, condition: str) -> str:
    return "noise" if condition == NOISE[dataset] else "full" if condition == FULL[dataset] else "relevance-16"


def figure(curves: pd.DataFrame, table: pd.DataFrame, epochs: list[int]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, dataset in enumerate(("bnci2a", "lee2019")):
        ax = axes[0, col]
        for condition, c in curves[curves.dataset == dataset].groupby("condition"):
            smooth = c.held_out.rolling(25, center=True, min_periods=1).mean()
            ax.plot(c.epoch, smooth, color=COLORS[role(dataset, condition)], label=condition)
        for e in epochs:
            ax.axvline(e, color="grey", lw=0.8, ls=":")
        ax.set(xlabel="epoch", ylabel="held-out runs of the training day, % (25-epoch mean)", title=TITLES[dataset])
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        ax = axes[1, col]
        t = table[table.dataset == dataset]
        for condition, c in t.groupby("condition"):
            ax.plot(c.epoch, c.next_day, "o-", color=COLORS[role(dataset, condition)], label=condition)
        ax.set_xscale("log", base=2)
        ax.set_xticks(epochs, [str(e) for e in epochs])
        ax.set(xlabel="training epochs (final models)", ylabel="next day, %", title=f"{TITLES[dataset]}: next day")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_long_training.png", dpi=200)
    plt.close(fig)


def fmt(d: dict) -> str:
    return f"{d['mean']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"


def verdict(holds: bool) -> str:
    return "holds" if holds else "does not hold"


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    plan = json.loads((OUT / "protocol.json").read_text())
    epochs = sorted(plan["checkpoints"])
    df, folds, identity = load(plan)
    pp = df.groupby(["dataset", "condition", "epoch", "subject"])[["next_day", "train"]].mean().reset_index()
    table = pp.groupby(["dataset", "condition", "epoch"])[["next_day", "train"]].mean().reset_index()
    table.to_csv(OUT / "checkpoints.csv", index=False, float_format="%.3f")
    curves = mean_curves(folds)
    curves.to_csv(OUT / "curves.csv", index=False, float_format="%.4f")
    pairs = contrasts(df, epochs)
    first, last = epochs[0], epochs[-1]
    l1 = combo(
        df,
        "lee2019",
        [
            (1, NOISE["lee2019"], last),
            (-1, "relevance-16", last),
            (-1, NOISE["lee2019"], first),
            (1, "relevance-16", first),
        ],
    )
    l2 = combo(df, "lee2019", [(1, FULL["lee2019"], last), (-1, "relevance-16", last)])
    l3 = window_gain(folds, "lee2019", NOISE["lee2019"])
    lee = {
        "L1": {**l1, "rule": "interval lower bound above 0", "holds": bool(l1["ci"][0] > 0)},
        "L2": {**l2, "rule": "interval lower bound above 0", "holds": bool(l2["ci"][0] > 0)},
        "L3": {**l3, "rule": "interval lower bound above 0", "holds": bool(l3["ci"][0] > 0)},
    }
    gains = {
        f"{d}|{c}": window_gain(folds, d, c) for d in ("bnci2a", "lee2019") for c in ("relevance-16", NOISE[d], FULL[d])
    }
    figure(curves, table, epochs)
    write_json(
        OUT / "analysis.json",
        {
            "tests_lee2019": lee,
            "contrasts": pairs,
            "held_out_window_gain": gains,
            "identity_475": identity,
            "checkpoints": records_json(table),
        },
    )

    lines = [
        "# Training four times as long",
        "",
        f"The frozen recipe run for {last} epochs; checkpoints at {', '.join(map(str, epochs))}. Accuracy in %, "
        "differences in percentage points with the 95% participant interval. Next day: final models; held-out runs: "
        f"fold models of the training day. Lee2019 tests were fixed in the plan ({plan['created_at'][:16]}); BCI IV "
        "2a is exploratory.",
        "",
        f"475-epoch checkpoints with exactly the earlier study's weights: {identity['identical']} of "
        f"{identity['found']} compared (largest difference {identity['max_abs_diff']:.3g}).",
        "",
        "## Lee2019 tests (54 people)",
        "",
        f"- L1, the noise inputs cost less after {last} epochs ((plus-46 minus relevance-16) at {last} minus at "
        f"{first}): {fmt(lee['L1'])}: {verdict(lee['L1']['holds'])}",
        f"- L2, all 62 electrodes beat relevance-16 after {last} epochs: {fmt(lee['L2'])}: "
        f"{verdict(lee['L2']['holds'])}",
        f"- L3, held-out accuracy with noise inputs still rises after epoch 475 (epochs {LATE[0]}-{LATE[1]} minus "
        f"{EARLY[0]}-{EARLY[1]}): {fmt(lee['L3'])}: {verdict(lee['L3']['holds'])}",
        "",
        "## Next-day accuracy at each checkpoint (final models)",
        "",
        "| dataset | condition | " + " | ".join(f"{e} epochs" for e in epochs) + f" | training set at {last} |",
        "|---|---|" + "---:|" * (len(epochs) + 1),
    ]
    for (dataset, condition), c in table.groupby(["dataset", "condition"]):
        c = c.set_index("epoch")
        lines.append(
            f"| {dataset} | {condition} | "
            + " | ".join(f"{c.next_day[e]:.2f}" for e in epochs)
            + f" | {c.train[last]:.1f} |"
        )
    lines += [
        "",
        "## Contrasts",
        "",
        "| dataset | contrast | epochs | mean | 95% interval | seed error | worse |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for p in pairs:
        se = f"{p['seed_error']:.2f}" if "seed_error" in p else ""
        lines.append(
            f"| {p['dataset']} | {p['what']} | {p['epoch']} | {p['mean']:+.2f} | [{p['ci'][0]:+.2f}, "
            f"{p['ci'][1]:+.2f}] | {se} | {p['worse']}/{p['people']} |"
        )
    lines += [
        "",
        f"## Held-out runs of the training day: epochs {LATE[0]}-{LATE[1]} minus {EARLY[0]}-{EARLY[1]}",
        "",
        "| dataset | condition | mean | 95% interval |",
        "|---|---|---:|---|",
    ]
    lines += [
        f"| {k.split('|')[0]} | {k.split('|')[1]} | {v['mean']:+.2f} | [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] |"
        for k, v in gains.items()
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
