"""Tables and figures for the noise-mechanism study (scripts/noise_mechanism_study.py).

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only, never EEG:
  supplementary/noise_mechanism_study/protocol.json   the plan: questions M1-M3, B1-B2, jobs
  supplementary/noise_mechanism_study/heldout/        mechanism on held-out runs of the training day
  supplementary/noise_mechanism_study/runs/           the trained noise-draw arms (with every epoch's loss)
  supplementary/noise_mechanism_study/test/           next-day scores of the draw arms and the references

Writes to RESULTS_DIR/supplementary/noise_mechanism_study/: mechanism.csv, shares.csv, draws.csv,
fig_mechanism.png, fig_noise_share.png, fig_draws.png, fig_loss_curves.png, summary.md, analysis.json.
A part whose records are not all there yet is reported as pending.

Accuracy is averaged over seeds and folds within a participant; intervals are the frozen participant
bootstrap (mibci.stats). Exploratory throughout.
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

OUT = RESULTS_DIR / "supplementary" / "noise_mechanism_study"
LEVEL_NAMES = {
    "bnci2a": {"fold": "4 of 6 runs", "rfold": "2 of 6 runs"},
    "lee2019": {"fold": "1 of 2 runs"},
}
TITLES = {"bnci2a": "BCI IV 2a (9 people)", "lee2019": "Lee 2019 (54 people)"}
ACCURACIES = ("as_trained", "recalibrated", "noise_removed", "noise_removed_recalibrated")
DRAW_ARMS = ("input-cost", "overfit-draw-appended", "draw-1", "draw-2", "draw-3", "draw-4")


def records_json(df: pd.DataFrame) -> list[dict]:
    """Rows as dicts with NaN written as null (write_json refuses NaN)."""
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def load(kind: str, jobs: list[dict]) -> pd.DataFrame | None:
    """One row per job of the stage, or None while records are missing."""
    paths = [OUT / kind / f"{j['name']}.json" for j in jobs]
    if not all(p.exists() for p in paths):
        return None
    rows = []
    for p in paths:
        r = json.loads(p.read_text())
        row = {k: r.get(k) for k in ("name", "kind", "dataset", "subject", "seed", "split", "arm", "added")}
        row |= {k: 100 * r[k] if k in r else np.nan for k in ACCURACIES}
        for k in ("noise_share_trained", "noise_share_init"):
            row[k] = float(np.mean(r[k])) if k in r else np.nan
        for k, v in r.get("weight_rms", {}).items():
            row[f"rms_{k}"] = v
        for k in ("recorded_within_day_acc", "recorded_next_day_acc"):
            row[k] = 100 * r[k] if k in r else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df["level"] = df["split"].str.split("-").str[0]
    return df


def paired(d: np.ndarray) -> dict:
    lo, hi = bootstrap_ci(d)
    return {"mean": float(np.mean(d)), "ci": [lo, hi], "people": len(d)}


def fmt(d: dict) -> str:
    return f"{d['mean']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"


def mechanism_table(h: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pp = h.groupby(["dataset", "level", "added", "subject"])[list(ACCURACIES)].mean().reset_index()
    rows, m1 = [], {}
    for (dataset, level), g in pp.groupby(["dataset", "level"]):
        base = g[g.added == 0].set_index("subject")
        for k, gk in g[g.added > 0].groupby("added"):
            t = gk.set_index("subject")
            cost = paired((t.as_trained - base.as_trained).to_numpy())
            recovered = paired((t.noise_removed_recalibrated - t.recalibrated).to_numpy())
            residual = paired((t.noise_removed_recalibrated - base.recalibrated).to_numpy())
            lost = (base.recalibrated - t.recalibrated).mean()
            share = float(recovered["mean"] / lost) if lost > 0 else np.nan
            rows.append(
                {
                    "dataset": dataset,
                    "level": level,
                    "data": LEVEL_NAMES[dataset][level],
                    "added": int(k),
                    "none_as_trained": float(base.as_trained.mean()),
                    **{a: float(t[a].mean()) for a in ACCURACIES},
                    "none_recalibrated": float(base.recalibrated.mean()),
                    "cost": cost["mean"],
                    "cost_ci": cost["ci"],
                    "recovered": recovered["mean"],
                    "recovered_ci": recovered["ci"],
                    "residual": residual["mean"],
                    "residual_ci": residual["ci"],
                    "share_recovered": share,
                }
            )
            if dataset == "bnci2a" and level == "fold" and k in (6, 16):
                m1[f"k={k}"] = {"share_recovered": share, "recovered": recovered, "residual": residual, "cost": cost}
    return pd.DataFrame(rows), m1


def share_table(h: pd.DataFrame) -> pd.DataFrame:
    noisy = h[h.added > 0]
    g = noisy.groupby(["dataset", "added"])
    out = g[["noise_share_trained", "noise_share_init", "rms_noise", "rms_electrodes", "rms_noise_init"]].mean()
    out["trained_over_init"] = out.noise_share_trained / out.noise_share_init
    out["noise_over_electrode_weights"] = out.rms_noise / out.rms_electrodes
    return out.reset_index()


def within_correlation(h: pd.DataFrame, x: str, y: str) -> dict:
    """Correlation of x and y after removing each participant-condition mean, with a participant bootstrap."""
    d = h[h.added > 0].copy()
    keys = ["dataset", "level", "added", "subject"]
    for c in (x, y):
        d[c] = d[c] - d.groupby(keys)[c].transform("mean")
    people = d.subject.unique()
    xs = {s: g[x].to_numpy() for s, g in d.groupby("subject")}
    ys = {s: g[y].to_numpy() for s, g in d.groupby("subject")}

    def r(sample) -> float:
        return float(
            np.corrcoef(np.concatenate([xs[s] for s in sample]), np.concatenate([ys[s] for s in sample]))[0, 1]
        )

    rng = np.random.default_rng(FROZEN_BOOTSTRAP["seed"])
    boots = [r(rng.choice(people, len(people))) for _ in range(2000)]
    return {"r": r(people), "ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}


def draw_table(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pp = t.groupby(["arm", "subject"])["as_trained"].mean().unstack("arm")
    none = pp["relevance-16"]
    rows = []
    for arm in [*DRAW_ARMS, "overfit-draw-in-place", "input-cost-draw-in-place"]:
        c = paired((pp[arm] - none).to_numpy())
        rows.append({"arm": arm, "accuracy": float(pp[arm].mean()), "cost": c["mean"], "ci": c["ci"]})
    costs = pp[list(DRAW_ARMS)].sub(none, axis=0)
    seeds = t[t.arm.isin(DRAW_ARMS)].groupby(["arm", "subject"])["as_trained"].agg(["var", "count"])
    seed_var = (seeds["var"] / seeds["count"]).groupby("subject").mean()
    between = costs.var(axis=1, ddof=1)
    draw_var = between - seed_var

    def sd(sample) -> float:
        return float(np.sqrt(max(draw_var.loc[sample].mean(), 0.0)))

    rng = np.random.default_rng(FROZEN_BOOTSTRAP["seed"])
    people = draw_var.index.to_numpy()
    boots = [sd(rng.choice(people, len(people))) for _ in range(FROZEN_BOOTSTRAP["resamples"])]
    b1 = {
        "draw_sd": sd(people),
        "ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "seed_sd_of_a_draw_mean": float(np.sqrt(seed_var.mean())),
        "per_participant_range": {
            int(s): [round(float(v.min()), 2), round(float(v.max()), 2)] for s, v in costs.iterrows()
        },
    }
    b2 = {
        "original": paired((pp["input-cost"] - pp["overfit-draw-in-place"]).to_numpy()),
        "swapped": paired((pp["input-cost-draw-in-place"] - pp["overfit-draw-appended"]).to_numpy()),
        "input_cost_draw_moved": paired((pp["input-cost-draw-in-place"] - pp["input-cost"]).to_numpy()),
        "overfit_draw_moved": paired((pp["overfit-draw-appended"] - pp["overfit-draw-in-place"]).to_numpy()),
    }
    return pd.DataFrame(rows), {"B1": b1, "B2": b2, "costs_by_participant": costs.round(2).to_dict()}


def fig_mechanism(mech: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [("bnci2a", "fold"), ("bnci2a", "rfold"), ("lee2019", "fold")]
    for ax, (dataset, level) in zip(axes, panels, strict=True):
        g = mech[(mech.dataset == dataset) & (mech.level == level)].sort_values("added")
        ax.axhline(g.none_as_trained.iloc[0], color="black", lw=1, ls=":", label="16 electrodes, no noise")
        for col, label, color in (
            ("as_trained", "as trained", "#13294B"),
            ("noise_removed_recalibrated", "noise weights set to 0, batch norms re-estimated", "#C8102E"),
            ("recalibrated", "batch norms re-estimated only", "#9ec5e8"),
        ):
            ax.plot(g.added, g[col], "o-", color=color, label=label)
        ax.set_title(f"{TITLES[dataset]}, trained on {LEVEL_NAMES[dataset][level]}")
        ax.set_xlabel("noise inputs added to the 16 electrodes")
        ax.set_ylabel("held-out runs of the training day, accuracy %")
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mechanism.png", dpi=200)
    plt.close(fig)


def fig_shares(shares: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, dataset in zip(axes, ("bnci2a", "lee2019"), strict=True):
        g = shares[shares.dataset == dataset]
        ax.plot(g.added, 100 * g.noise_share_init, "o--", color="#9ec5e8", label="at initialization")
        ax.plot(g.added, 100 * g.noise_share_trained, "o-", color="#13294B", label="after training")
        ax.set_title(TITLES[dataset])
        ax.set_xlabel("noise inputs added to the 16 electrodes")
        ax.set_ylabel("share of a spatial filter's output that is noise, %")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_noise_share.png", dpi=200)
    plt.close(fig)


def fig_draws(t: pd.DataFrame) -> None:
    pp = t.groupby(["arm", "subject"])["as_trained"].mean().unstack("arm")
    none = pp["relevance-16"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    arms = [*DRAW_ARMS, "overfit-draw-in-place", "input-cost-draw-in-place"]
    for i, arm in enumerate(arms):
        d = pp[arm] - none
        ax.scatter(np.full(len(d), i) + np.linspace(-0.2, 0.2, len(d)), d, s=14, color="#4B9CD3")
        ax.plot([i - 0.3, i + 0.3], [d.mean()] * 2, color="#C8102E", lw=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(arms)), [a.replace("-", "\n", 1) for a in arms], fontsize=7)
    ax.set_ylabel("next-day accuracy change vs. 16 electrodes (pp)")
    ax.set_title("BCI IV 2a, 16 electrodes plus 6 noise inputs: each dot a participant (10 seeds), bar the mean")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_draws.png", dpi=200)
    plt.close(fig)


def fig_losses(runs: dict, t: pd.DataFrame) -> None:
    acc = t.set_index("name")["as_trained"]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
    for ax, (subject, g) in zip(axes.flat, sorted(runs.items()), strict=False):
        accs = np.array([acc[n] for n in g])
        norm = plt.Normalize(accs.min(), accs.max())
        for (_, loss), a in zip(g.items(), accs, strict=True):
            ax.plot(loss, color=plt.cm.viridis(norm(a)), lw=0.6, alpha=0.8)
        ax.set_yscale("log")
        ax.set_title(f"participant {subject}: next day {accs.min():.0f}-{accs.max():.0f}% (yellow = best)", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("training loss")
    fig.tight_layout()
    fig.savefig(OUT / "fig_loss_curves.png", dpi=150)
    plt.close(fig)


def figures(mech: pd.DataFrame | None, shares: pd.DataFrame | None, t: pd.DataFrame | None, runs) -> None:
    if mech is not None:
        fig_mechanism(mech)
    if shares is not None:
        fig_shares(shares)
    if t is not None:
        fig_draws(t)
    if runs is not None and t is not None:
        fig_losses(runs, t)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    plan = json.loads((OUT / "protocol.json").read_text())
    jobs = plan["jobs"]
    h = load("heldout", [j for j in jobs if j["kind"] == "heldout"])
    t = load("test", [j for j in jobs if j["kind"] in ("draw", "reference")])
    trained = [j for j in jobs if j["kind"] == "draw"]
    runs = None
    if all((OUT / "runs" / f"{j['name']}.json").exists() for j in trained):
        runs = {}
        for j in trained:
            r = json.loads((OUT / "runs" / f"{j['name']}.json").read_text())
            runs.setdefault(j["subject"], {})[j["name"]] = r["train_loss"]
    summary: dict = {}
    lines = ["# Why an input that carries nothing costs EEGNet accuracy", ""]

    mech = shares = None
    if h is None:
        lines += ["## Mechanism (held-out runs of the training day)", "", "Pending: not every record is there.", ""]
    else:
        same = int((h.as_trained.round(6) == h.recorded_within_day_acc.round(6)).sum())
        mech, m1 = mechanism_table(h)
        shares = share_table(h)
        m2 = {
            dataset: within_correlation(h[h.dataset == dataset], "noise_share_trained", "as_trained")
            for dataset in ("bnci2a", "lee2019")
        }
        mech.to_csv(OUT / "mechanism.csv", index=False, float_format="%.3f")
        shares.to_csv(OUT / "shares.csv", index=False, float_format="%.4f")
        summary |= {"reproduced": [same, len(h)], "M1": m1, "M2": m2, "mechanism": records_json(mech)}
        summary["shares"] = records_json(shares)
        lines += [
            "## Mechanism (held-out runs of the training day)",
            "",
            f"Held-out accuracy recomputed from the saved models equals the record for {same} of {len(h)} models.",
            "",
            "| dataset | training data | added | none % | as trained | cost | batch norms re-estimated | noise "
            "weights 0 | noise weights 0, re-estimated | recovered | still lost | share of the cost recovered |",
            "|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---:|",
        ]
        for r in mech.itertuples():
            lines.append(
                f"| {r.dataset} | {r.data} | {r.added} | {r.none_as_trained:.2f} | {r.as_trained:.2f} | "
                f"{r.cost:+.2f} [{r.cost_ci[0]:+.2f}, {r.cost_ci[1]:+.2f}] | {r.recalibrated:.2f} | "
                f"{r.noise_removed:.2f} | {r.noise_removed_recalibrated:.2f} | {r.recovered:+.2f} "
                f"[{r.recovered_ci[0]:+.2f}, {r.recovered_ci[1]:+.2f}] | {r.residual:+.2f} [{r.residual_ci[0]:+.2f}, "
                f"{r.residual_ci[1]:+.2f}] | {r.share_recovered:.0%} |"
            )
        lines += [
            "",
            '"Recovered": noise weights 0 and batch norms re-estimated, minus batch norms re-estimated only. "Still '
            'lost": the former minus the no-noise models with re-estimated batch norms.',
            "",
            "## How much of a spatial filter's output is noise",
            "",
            "| dataset | added | after training | at initialization | trained / initial | noise weight rms / "
            "electrode weight rms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        lines += [
            f"| {r.dataset} | {r.added} | {r.noise_share_trained:.1%} | {r.noise_share_init:.1%} | "
            f"{r.trained_over_init:.2f} | {r.noise_over_electrode_weights:.2f} |"
            for r in shares.itertuples()
        ]
        lines += [
            "",
            "Within participant and condition, correlation of the noise share after training with held-out accuracy "
            "(M2): "
            + "; ".join(f"{d} r = {v['r']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]" for d, v in m2.items()),
            "",
        ]

    draws = None
    if t is None:
        lines += ["## Noise draws (next day, BCI IV 2a final models)", "", "Pending: not every record is there."]
    else:
        refs = t[t.kind == "reference"]
        same = int((refs.as_trained.round(6) == refs.recorded_next_day_acc.round(6)).sum())
        draws, b = draw_table(t)
        draws.to_csv(OUT / "draws.csv", index=False, float_format="%.3f")
        b1, b2 = b["B1"], b["B2"]
        summary |= {"references_reproduced": [same, len(refs)], "draws": records_json(draws), **b}
        lines += [
            "## Noise draws (next day, BCI IV 2a final models, seeds 0-9)",
            "",
            f"Reference models rescored: {same} of {len(refs)} equal their recorded next-day accuracy.",
            "",
            "| arm | accuracy % | cost vs. 16 electrodes | 95% interval |",
            "|---|---:|---:|---|",
        ]
        lines += [
            f"| {r.arm} | {r.accuracy:.2f} | {r.cost:+.2f} | [{r.ci[0]:+.2f}, {r.ci[1]:+.2f}] |"
            for r in draws.itertuples()
        ]
        lines += [
            "",
            f"- B1: spread between the six appended draws beyond seed noise: SD {b1['draw_sd']:.2f} pp "
            f"[{b1['ci'][0]:.2f}, {b1['ci'][1]:.2f}] (seed SD of one draw's mean {b1['seed_sd_of_a_draw_mean']:.2f}); "
            f"per participant, lowest and highest draw cost: {b1['per_participant_range']}",
            f"- B2: original difference (input-cost appended minus overfit-study in place) {fmt(b2['original'])}; with "
            f"the placements swapped {fmt(b2['swapped'])}; the input-cost draw moved into the dropped places "
            f"{fmt(b2['input_cost_draw_moved'])}; the overfit-study draw appended {fmt(b2['overfit_draw_moved'])}",
            "",
        ]
        nxt = t[t.added > 0].groupby("arm")[list(ACCURACIES)].mean()
        lines += [
            "Next day, the same mechanism measurements (mean over participants and seeds, accuracy %):",
            "",
            "| arm | " + " | ".join(ACCURACIES) + " |",
            "|---|" + "---:|" * len(ACCURACIES),
        ]
        lines += [f"| {arm} | " + " | ".join(f"{v:.2f}" for v in row) + " |" for arm, row in nxt.iterrows()]
    figures(mech, shares, t, runs)
    write_json(OUT / "analysis.json", summary)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
