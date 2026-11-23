"""Tables and figures for the supplementary electrode-count study.

Combines the archived session-2 records (seeds 0-2) with the post-test runs of
scripts/run_electrode_study.py and, when present, the second wave of
scripts/run_electrode_curve.py (seeds 3-9 for 20, 18, 14, 12 and 10 electrodes). Seeds are
averaged within each participant, then the nine participants are summarized; intervals are
the participant-level percentile bootstrap of the frozen plan. Everything here is
exploratory: the test results were known before these runs were planned.

Two kinds of uncertainty are reported and should not be mixed up. The bootstrap interval
resamples participants: it says how far the mean could move with nine other people. The
"seed standard error" holds the nine participants fixed and says how far the mean could
move if the same models were trained again with other random seeds.

It also describes which electrodes the selection rule keeps (from the archived rankings)
and compares kept and dropped electrodes on both recording days. That description reads
session 2, so it is a post-test description, not something that fed any selection.

One what-if is included because it bears on the 16-versus-22 question: the same models are
scored again with the session-2 inputs rescaled to the training amplitude (one gain for the
whole recording, or each electrode re-standardized with session-2 statistics). No labels
are used, but session-2 statistics are, so this is outside the frozen protocol. It is a
diagnostic of where the difference comes from, not a proposed change to the pipeline.
The same pass also averages the class probabilities of the ten seeds of each model
(a seed ensemble): if a difference came from run-to-run variance it would shrink there.

Writes summary.md, CSV tables, analysis.json and two figures to
results/supplementary/electrode_count/.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from mibci import data, deploy, selection
from mibci.artifacts import require_files, write_json
from mibci.experiment import MODELS_DIR, RESULTS_DIR, session1
from mibci.model import load_bundle
from mibci.stats import bootstrap_ci
from mibci.train import ChannelStandardizer, scores

OUT = RESULTS_DIR / "supplementary/electrode_count"
WAVE2 = RESULTS_DIR / "supplementary/electrode_count_wave2"
FULL = "full-22"
GROUP_THRESHOLD = 60.0  # % with all 22 electrodes; no participant lies between 50 and 73
SCALP_ROWS = {
    "frontal (Fz)": ["Fz"],
    "fronto-central (FC)": ["FC3", "FC1", "FCz", "FC2", "FC4"],
    "central (C)": ["C5", "C3", "C1", "Cz", "C2", "C4", "C6"],
    "centro-parietal (CP)": ["CP3", "CP1", "CPz", "CP2", "CP4"],
    "parietal (P)": ["P1", "Pz", "P2"],
    "parieto-occipital (POz)": ["POz"],
}


def electrodes_of(condition: str) -> int:
    return int(condition.split("-")[1])


def load_runs(plan: dict, plan2: dict | None) -> pd.DataFrame:
    rows = []

    def add(record, source, condition=None):
        for precision in ("fp32", "int8"):
            train = record.get("train", {}).get(precision)
            rows.append(
                {
                    "subject": record["subject"],
                    "seed": record["seed"],
                    "condition": condition or record["condition"],
                    "precision": precision,
                    "source": source,
                    "acc": record["primary"][precision]["acc"],
                    "loss": record["primary"][precision]["loss"],
                    "acc_all": record["secondary"][precision]["acc"],
                    "train_acc": train["acc"] if train else np.nan,
                    "train_loss": train["loss"] if train else np.nan,
                }
            )

    for path in sorted((RESULTS_DIR / "test").glob("sub-*.json")):
        add(json.loads(path.read_text()), "archived")
    expected = [OUT / "test" / j["wave"] / f"{j['name']}.json" for j in plan["jobs"]]
    require_files(expected, "electrode-count analysis (run --stage train and --stage evaluate first)")
    for job, path in zip(plan["jobs"], expected, strict=True):
        record = json.loads(path.read_text())
        if job["wave"] == "repro":
            continue
        add(record, job["wave"])
        # Where the redundancy-aware subset equals the relevance subset, the same model serves both.
        for condition, subjects in plan["same_subset_as_relevance"].items():
            k = electrodes_of(condition)
            if job["wave"] == "seeds" and job["condition"] == f"relevance-{k}" and job["subject"] in subjects:
                add(record, "seeds (same subset as relevance)", condition)
    if plan2:
        expected = [WAVE2 / "test" / j["wave"] / f"{j['name']}.json" for j in plan2["jobs"]]
        require_files(expected, "second-wave analysis (run scripts/run_electrode_curve.py train and evaluate first)")
        for path in expected:
            add(json.loads(path.read_text()), "second wave")
    return pd.DataFrame(rows)


def paired(df, boot, a, b, precision="fp32", seeds=None, label="") -> dict:
    sel = df[(df.precision == precision) & df.condition.isin([a, b]) & df.seed.isin(seeds)]
    table = sel.pivot(index=["subject", "seed"], columns="condition", values="acc") * 100
    by_seed = (table[a] - table[b]).unstack("seed")  # participants x seeds, same seed in both models
    if by_seed.isna().any().any():
        raise RuntimeError(f"{a} or {b} lacks runs for seeds {label}")
    d = by_seed.mean(axis=1).to_numpy()
    lo, hi = bootstrap_ci(d, boot)
    # participants held fixed: standard error of the mean difference from training seeds alone
    seed_se = float(np.sqrt((by_seed.var(axis=1, ddof=1) / by_seed.shape[1]).sum()) / len(d))
    return {
        "a": a,
        "b": b,
        "precision": precision,
        "seeds": label,
        "mean_diff_pp": float(d.mean()),
        "ci_low_pp": lo,
        "ci_high_pp": hi,
        "seed_se_pp": seed_se,
        "median_diff_pp": float(np.median(d)),
        "n_a_better": int((d > 0).sum()),
        "n_b_better": int((d < 0).sum()),
        "per_participant_pp": [round(float(x), 2) for x in d],
    }


def seed_luck(df, a, b, k: int, draws: int = 20000, seed: int = 690) -> dict:
    """How far the nine-participant mean difference moves if only k of the available seeds are
    trained per model: draw k seeds per participant and condition at random, many times."""
    rng = np.random.default_rng(seed)
    tables = []
    for condition in (a, b):
        sel = df[(df.condition == condition) & (df.precision == "fp32")]
        tables.append(sel.pivot(index="subject", columns="seed", values="acc").to_numpy() * 100)
    n_seeds = tables[0].shape[1]
    out = np.empty(draws)
    for i in range(draws):
        means = [np.take_along_axis(t, np.argsort(rng.random(t.shape), axis=1)[:, :k], axis=1).mean(1) for t in tables]
        out[i] = (means[0] - means[1]).mean()
    return {
        "k": k,
        "available_seeds": int(n_seeds),
        "mean_pp": float(out.mean()),
        "sd_pp": float(out.std()),
        "p2_5_pp": float(np.quantile(out, 0.025)),
        "p97_5_pp": float(np.quantile(out, 0.975)),
        "share_at_or_below_zero": float((out <= 0).mean()),
    }


def archived_train_fit(conditions) -> pd.DataFrame:
    """Training-day accuracy of the archived models (seeds 0-2); needs the original models locally."""
    cache = OUT / "archived_train_fit.json"
    if cache.exists():
        return pd.DataFrame(json.loads(cache.read_text()))
    rows = []
    for subject in data.SUBJECTS:
        X, y, tr, _ = session1(subject, "final")
        for seed in (0, 1, 2):
            for condition in conditions:
                bundle = MODELS_DIR / "final" / f"sub-{subject:02d}_seed-{seed}_{condition}.pt"
                if not bundle.exists():
                    return pd.DataFrame()
                _, scaler, b = load_bundle(bundle)
                logits = deploy.logits(
                    deploy.session(bundle.with_suffix(".onnx")), scaler.transform(X[tr][:, b["channels"]])
                )
                s = scores(logits, y[tr])
                rows.append(
                    {
                        "subject": subject,
                        "seed": seed,
                        "condition": condition,
                        "train_acc": s["acc"],
                        "train_loss": s["loss"],
                    }
                )
    write_json(cache, rows)
    return pd.DataFrame(rows)


def rescore(plan: dict, conditions) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every seed of the ten-seed models again from the stored ONNX graphs.

    Returns (what_if, ensemble). what_if holds the session-2 accuracy of each run under three
    input scalings: 'protocol' (training-day statistics, as evaluated), 'one_gain' (the
    normalized session-2 recording divided by its overall standard deviation) and
    'per_electrode' (z-scored with session-2 statistics). ensemble holds, per participant and
    model, the mean single-run accuracy, the accuracy of the ten seeds' averaged class
    probabilities, and how often two seeds give the same label. Needs the models locally;
    cached after the first run."""
    caches = OUT / "amplitude_what_if.json", OUT / "seed_ensemble.json"
    if all(c.exists() for c in caches):
        return pd.DataFrame(json.loads(caches[0].read_text())), pd.DataFrame(json.loads(caches[1].read_text()))
    new_jobs = {(j["subject"], j["seed"], j["condition"]): j for j in plan["jobs"] if j["wave"] == "seeds"}
    rows, ensembles = [], []
    for subject in data.SUBJECTS:
        ses = data.load_session(subject, 2, final_evaluation=True)  # post-test diagnostic
        clean = ~ses["artifact"]
        y = ses["y"][clean]
        for condition in conditions:
            probabilities = []
            for seed in plan["historical_seeds"] + plan["new_seeds"]:
                if seed in plan["historical_seeds"]:
                    bundle = MODELS_DIR / "final" / f"sub-{subject:02d}_seed-{seed}_{condition}.pt"
                else:
                    job = new_jobs[(subject, seed, condition)]
                    bundle = MODELS_DIR / "supplementary/electrode_count" / job["wave"] / f"{job['name']}.pt"
                if not bundle.exists() or not bundle.with_suffix(".onnx").exists():
                    return pd.DataFrame(), pd.DataFrame()
                _, scaler, b = load_bundle(bundle)
                X = ses["X"][clean][:, b["channels"]]
                protocol = scaler.transform(X)
                inputs = {
                    "protocol": protocol,
                    "one_gain": (protocol / protocol.std()).astype(np.float32),
                    "per_electrode": ChannelStandardizer().fit(X).transform(X),
                }
                sess = deploy.session(bundle.with_suffix(".onnx"))
                row = {
                    "subject": subject,
                    "seed": seed,
                    "condition": condition,
                    "amplitude_ratio": float(protocol.std()),
                }
                for k, v in inputs.items():
                    logits = deploy.logits(sess, v)
                    row[k] = float((logits.argmax(1) == y).mean())
                    if k == "protocol":
                        e = np.exp(logits - logits.max(1, keepdims=True))
                        probabilities.append(e / e.sum(1, keepdims=True))
                rows.append(row)
            P = np.array(probabilities)
            labels = P.argmax(2)
            ensembles.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "n_seeds": len(P),
                    "single": float((labels == y).mean()),
                    "ensemble": float((P.mean(0).argmax(1) == y).mean()),
                    "agreement": float(
                        np.mean([(labels[i] == labels[j]).mean() for i in range(len(P)) for j in range(i)])
                    ),
                }
            )
    write_json(caches[0], rows)
    write_json(caches[1], ensembles)
    return pd.DataFrame(rows), pd.DataFrame(ensembles)


def selection_description() -> tuple[pd.DataFrame, dict]:
    """Which electrodes relevance-16 keeps, and how kept and dropped electrodes compare on both days."""
    names = data.EEG_CHANNELS
    kept16, kept8, rows = np.zeros((9, 22), bool), np.zeros((9, 22), bool), []
    for i, subject in enumerate(data.SUBJECTS):
        ranking = json.loads((RESULTS_DIR / f"final/rankings/sub-{subject:02d}_seed-0.json").read_text())
        kept16[i, ranking["subsets"]["relevance-16"]] = True
        kept8[i, ranking["subsets"]["relevance-8"]] = True
        day1 = data.load_session(subject, 1)
        day2 = data.load_session(subject, 2, final_evaluation=True)  # post-test description only
        c1, c2 = ~day1["artifact"], ~day2["artifact"]
        p1, p2 = selection.band_log_power(day1["X"][c1]), selection.band_log_power(day2["X"][c2])
        f1 = selection.fisher_score(p1, day1["y"][c1]).mean(-1)
        f2 = selection.fisher_score(p2, day2["y"][c2]).mean(-1)
        shift = (np.abs(p2.mean(0) - p1.mean(0)) / np.sqrt((p1.var(0) + p2.var(0)) / 2)).mean(-1)
        keep = kept16[i]
        rows.append(
            {
                "subject": subject,
                "dropped": " ".join(n for n, k in zip(names, keep, strict=True) if not k),
                "fisher_day1_kept": f1[keep].mean(),
                "fisher_day1_dropped": f1[~keep].mean(),
                "fisher_day2_kept": f2[keep].mean(),
                "fisher_day2_dropped": f2[~keep].mean(),
                "dropped_share_of_fisher_day1": f1[~keep].sum() / f1.sum(),
                "dropped_share_of_fisher_day2": f2[~keep].sum() / f2.sum(),
                "day_shift_kept": shift[keep].mean(),
                "day_shift_dropped": shift[~keep].mean(),
                "amplitude_ratio_day2_over_day1": float(
                    np.exp(np.log(day2["X"][c2].std(axis=(0, 2)) / day1["X"][c1].std(axis=(0, 2))).mean())
                ),
                "rank_correlation_of_relevance_between_days": stats.spearmanr(f1, f2).statistic,
            }
        )
    per_subject = pd.DataFrame(rows)
    by_row = {}
    for row, members in SCALP_ROWS.items():
        idx = [names.index(m) for m in members]
        by_row[row] = {
            "electrodes": len(idx),
            "dropped_at_16": int((~kept16[:, idx]).sum()),
            "possible": 9 * len(idx),
            "share_of_all_drops": float((~kept16[:, idx]).sum() / (~kept16).sum()),
            "kept_at_8_rate": float(kept8[:, idx].mean()),
        }
    summary = {
        "kept_at_16_count_per_electrode": dict(zip(names, kept16.sum(0).tolist(), strict=True)),
        "kept_at_8_count_per_electrode": dict(zip(names, kept8.sum(0).tolist(), strict=True)),
        "by_scalp_row": by_row,
    }
    return per_subject, summary


def group_of(baseline: pd.Series) -> dict:
    """Participants split by their ten-seed accuracy with all 22 electrodes (a post-hoc split)."""
    below = [int(s) for s in baseline.index[baseline < GROUP_THRESHOLD]]
    return {"below": below, "above": [int(s) for s in baseline.index if int(s) not in below]}


def by_group(df: pd.DataFrame, groups: dict) -> pd.DataFrame:
    """Accuracy of each condition for all nine participants and for the two groups, using every
    available run, and its difference from all 22 electrodes (ten seeds) with a seed standard error."""
    fp = df[df.precision == "fp32"]

    def runs(conditions):
        return {int(s): g.acc.to_numpy() * 100 for s, g in fp[fp.condition.isin(conditions)].groupby("subject")}

    base = runs([FULL])
    listing = (
        [(FULL, [FULL])]
        + [(f"relevance-{k}", [f"relevance-{k}"]) for k in (20, 18, 16, 14, 12, 10, 8)]
        + [
            ("random-16 (three fixed subsets)", [f"random{i}-16" for i in (1, 2, 3)]),
            ("leastrelevant-16", ["leastrelevant-16"]),
        ]
    )
    rows = []
    for label, members in listing:
        r = runs(members)
        row = {"condition": label, "runs_per_participant": int(np.mean([len(v) for v in r.values()]))}
        for name, subjects in {"all": sorted(r), **groups}.items():
            acc = np.mean([r[s].mean() for s in subjects])
            se = np.sqrt(
                sum(r[s].var(ddof=1) / len(r[s]) + base[s].var(ddof=1) / len(base[s]) for s in subjects)
            ) / len(subjects)
            row[f"{name}_acc"] = acc
            row[f"{name}_minus_full_pp"] = acc - np.mean([base[s].mean() for s in subjects])
            row[f"{name}_seed_se_pp"] = 0.0 if label == FULL else se
        rows.append(row)
    return pd.DataFrame(rows)


def participant_names(subjects) -> str:
    return ", ".join(f"P{s}" for s in subjects)


def figures(
    df: pd.DataFrame, curve: pd.DataFrame, groups_table: pd.DataFrame, groups: dict, baseline: pd.Series, plan: dict
) -> None:
    counts = [22, 20, 18, 16, 14, 12, 10, 8]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    three = curve[curve.seeds == "0-2"].set_index("electrodes")
    ten = curve[curve.seeds == "0-9"].set_index("electrodes")
    left.errorbar(
        three.index,
        three.acc_mean,
        yerr=three.seed_se,
        marker="o",
        ms=4,
        lw=1.2,
        color="#9db7d5",
        capsize=2,
        label="seeds 0-2 (3 runs per model)",
    )
    complete = set(counts) <= set(ten.index)
    left.errorbar(
        ten.index,
        ten.acc_mean,
        yerr=ten.seed_se,
        marker="s",
        ms=5,
        color="#1f4e79",
        capsize=3,
        ls="-" if complete else "none",
        lw=1.8,
        label="seeds 0-9 (10 runs per model)",
    )
    left.set(
        xlabel="electrodes kept (most relevant first)",
        ylabel="held-out-day accuracy (%)",
        xticks=counts,
        title="All nine participants (FP32)",
    )
    left.invert_xaxis()
    left.grid(alpha=0.3)
    left.legend(frameon=False, loc="lower left")
    table = groups_table[groups_table.condition.str.fullmatch(r"relevance-\d+")]
    x = [22, *table.condition.map(electrodes_of)]
    for name, color, where in (("below", "#c55a11", "below"), ("above", "#2e75b6", "above")):
        right.errorbar(
            x,
            [0.0, *table[f"{name}_minus_full_pp"]],
            yerr=[0.0, *table[f"{name}_seed_se_pp"]],
            marker="o",
            ms=4,
            lw=1.6,
            color=color,
            capsize=3,
            label=f"{where} {GROUP_THRESHOLD:.0f}% with 22 electrodes ({participant_names(groups[name])})",
        )
    right.axhline(0, color="#404040", lw=0.8)
    right.set(
        xlabel="electrodes kept (most relevant first)",
        ylabel="difference from all 22 electrodes (pp)",
        xticks=counts,
        title="Same curve, participants split by 22-electrode accuracy",
    )
    right.invert_xaxis()
    right.grid(alpha=0.3)
    right.legend(frameon=False, loc="lower left")
    fig.text(0.995, 0.012, "bars: standard error from training seeds", ha="right", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / "fig_electrode_count.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    seeds = plan["historical_seeds"] + plan["new_seeds"]
    order = [int(s) for s in baseline.sort_values().index]  # weakest participant on the left
    for offset, (condition, color, label) in enumerate(
        [(FULL, "#404040", "all 22 electrodes"), ("relevance-16", "#2e75b6", "16 by relevance")]
    ):
        sel = df[(df.condition == condition) & (df.precision == "fp32") & df.seed.isin(seeds)]
        for position, subject in enumerate(order):
            g = sel[sel.subject == subject]
            x = position + (offset - 0.5) * 0.3
            ax.scatter(np.full(len(g), x), g.acc * 100, s=10, color=color, alpha=0.45, linewidths=0)
            ax.plot(
                [x - 0.1, x + 0.1],
                [g.acc.mean() * 100] * 2,
                color=color,
                lw=2.2,
                label=label if position == 0 else None,
            )
    ax.axvline(len(groups["below"]) - 0.5, color="#999999", lw=0.8, ls=":")
    ax.set(
        xlabel="participant, ordered by accuracy with all 22 electrodes",
        ylabel="held-out-day accuracy (%)",
        xticks=range(len(order)),
        xticklabels=[f"P{s}" for s in order],
        title="Ten training seeds per model: dots are runs, bars are means",
    )
    ax.grid(alpha=0.3, axis="y")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_16_vs_22_seeds.png", dpi=200)
    plt.close(fig)


def condition_table(df: pd.DataFrame, seed_sets) -> pd.DataFrame:
    """Accuracy of each condition and precision for every seed set that all participants were trained with."""
    rows = []
    for condition in sorted(df.condition.unique(), key=lambda c: (-electrodes_of(c), c)):
        for precision in ("fp32", "int8"):
            sel = df[(df.condition == condition) & (df.precision == precision)]
            for label, seeds in seed_sets:
                part = sel[sel.seed.isin(seeds)]
                counts = part.groupby("subject").seed.nunique()
                if len(counts) != len(data.SUBJECTS) or counts.min() != len(seeds):
                    continue  # this condition was not trained with this set of seeds
                pp = part.groupby("subject").acc.mean() * 100
                run_sd = part.groupby("subject").acc.std().mean() * 100
                rows.append(
                    {
                        "condition": condition,
                        "electrodes": electrodes_of(condition),
                        "precision": precision,
                        "seeds": label,
                        "n_seeds": len(seeds),
                        "acc_mean": pp.mean(),
                        "acc_sd_participants": pp.std(),
                        "run_sd_within_participant": run_sd,
                        "seed_se": run_sd / np.sqrt(len(seeds) * pp.size),
                        "acc_all_trials": part.groupby("subject").acc_all.mean().mean() * 100,
                    }
                )
    return pd.DataFrame(rows)


def paired_comparisons(df: pd.DataFrame, plan: dict, seed_sets, available: set) -> list[dict]:
    """Every paired comparison, in the order of comparisons.csv and analysis.json."""
    boot = plan["bootstrap"]
    comparisons = []
    for a in ("relevance-16", "proposed-16", "relevance-8", "proposed-8"):
        for precision in ("fp32", "int8"):
            for label, seeds in seed_sets:
                comparisons.append(paired(df, boot, a, FULL, precision, seeds, label))
    for k in (16, 8):
        for label, seeds in seed_sets:
            comparisons.append(paired(df, boot, f"proposed-{k}", f"relevance-{k}", "fp32", seeds, label))
    for condition in [f"relevance-{k}" for k in plan["sweep_counts"]] + ["leastrelevant-16"]:
        for label, seeds in seed_sets:
            if (condition, "fp32", label) in available:
                comparisons.append(paired(df, boot, condition, FULL, "fp32", seeds, label))
    old = plan["historical_seeds"]
    comparisons.append(paired(df, boot, "leastrelevant-16", "relevance-16", "fp32", old, "0-2"))
    return comparisons


def ten_seed_by_participant(df: pd.DataFrame, ten_seed: list[str], all_seeds: list[int]) -> pd.DataFrame:
    """Per participant: mean and seed standard error of each ten-seed model, and the paired
    difference of 16 and 8 relevance-ranked electrodes from all 22."""
    per = []
    for subject in data.SUBJECTS:
        row = {"subject": subject}
        runs = {}
        for condition in ten_seed:
            sel = (
                df[
                    (df.condition == condition)
                    & (df.precision == "fp32")
                    & (df.subject == subject)
                    & df.seed.isin(all_seeds)
                ]
                .sort_values("seed")
                .acc.to_numpy()
                * 100
            )
            runs[condition] = sel
            row[f"{condition}_mean"], row[f"{condition}_se"] = sel.mean(), sel.std(ddof=1) / np.sqrt(len(sel))
        for condition in ("relevance-16", "relevance-8"):
            d = runs[condition] - runs[FULL]  # same seed in both models
            row[f"{condition}_minus_full_pp"] = d.mean()
            row[f"{condition}_minus_full_se"] = d.std(ddof=1) / np.sqrt(len(d))
        per.append(row)
    return pd.DataFrame(per)


def train_fit_by_participant(
    df: pd.DataFrame, archived_fit: pd.DataFrame, per: pd.DataFrame, new: list[int], all_seeds: list[int]
) -> list[dict]:
    """Training-day and held-out-day accuracy per participant over all ten seeds, written to
    train_fit_by_participant.csv; empty when the archived models' training-day fit is missing."""
    if not len(archived_fit):
        return []
    newer = df[(df.precision == "fp32") & df.seed.isin(new)][["subject", "seed", "condition", "train_acc"]]
    fit = pd.concat([archived_fit[["subject", "seed", "condition", "train_acc"]], newer])
    fit_rows = []
    for subject in data.SUBJECTS:
        row = {"subject": subject}
        for condition in (FULL, "relevance-16"):
            t = fit[(fit.subject == subject) & (fit.condition == condition)]
            if t.seed.nunique() != len(all_seeds):
                raise RuntimeError(f"Training-day fit incomplete for participant {subject}, {condition}")
            row[f"{condition}_train"] = t.train_acc.mean() * 100
            row[f"{condition}_test"] = per.set_index("subject").loc[subject, f"{condition}_mean"]
        fit_rows.append(row)
    pd.DataFrame(fit_rows).to_csv(OUT / "train_fit_by_participant.csv", index=False)
    return fit_rows


def amplitude_what_if(plan: dict) -> tuple[list[dict], dict, pd.Series | None]:
    """The rescaled-input what-if and the seed ensemble, from the rescored models.

    Returns the what-if comparisons (also written to amplitude_what_if.csv), the ensemble
    summary and each participant's session-2 amplitude ratio; empty without the models."""
    what_if, ensemble = rescore(plan, [FULL, "relevance-16", "relevance-8"])
    if not len(what_if):
        return [], {}, None
    boot = plan["bootstrap"]
    pp = what_if.groupby(["condition", "subject"])[["protocol", "one_gain", "per_electrode", "amplitude_ratio"]].mean()
    what_if_rows = []
    for scaling in ("protocol", "one_gain", "per_electrode"):
        table = pp[scaling].unstack(0) * 100
        for condition in ("relevance-16", "relevance-8"):
            d = (table[condition] - table[FULL]).to_numpy()
            lo, hi = bootstrap_ci(d, boot)
            what_if_rows.append(
                {
                    "scaling": scaling,
                    "a": condition,
                    "b": FULL,
                    "acc_a": float(table[condition].mean()),
                    "acc_b": float(table[FULL].mean()),
                    "mean_diff_pp": float(d.mean()),
                    "ci_low_pp": lo,
                    "ci_high_pp": hi,
                    "per_participant_pp": [round(float(x), 2) for x in d],
                    "per_participant_acc_a": [float(x) for x in table[condition]],
                    "per_participant_acc_b": [float(x) for x in table[FULL]],
                }
            )
    pd.DataFrame(what_if_rows).to_csv(OUT / "amplitude_what_if.csv", index=False)
    ratios = pp["amplitude_ratio"].unstack(0)[FULL]
    e = ensemble.groupby("condition")[["single", "ensemble", "agreement"]].mean() * 100
    ensemble_summary = {c: {k: float(v) for k, v in e.loc[c].items()} for c in e.index}
    return what_if_rows, ensemble_summary, ratios


def repro_check(plan: dict) -> dict:
    """The pipeline-check runs against the archived runs they repeat."""
    repro = {}
    for job in plan["jobs"]:
        if job["wave"] == "repro":
            record = json.loads((OUT / "test" / "repro" / f"{job['name']}.json").read_text())
            archived = json.loads((RESULTS_DIR / "test" / f"{job['name']}.json").read_text())
            repro[job["name"]] = {
                "prediction_agreement": record["historical_prediction_agreement"],
                "acc_new": record["primary"]["fp32"]["acc"],
                "acc_archived": archived["primary"]["fp32"]["acc"],
            }
    return repro


# ---------------------------------------------------------------- summary.md, one function per section


def comparison_row(comp: pd.DataFrame, a: str, b: str, seeds: str, precision: str = "fp32") -> pd.Series:
    return comp[(comp.a == a) & (comp.b == b) & (comp.seeds == seeds) & (comp.precision == precision)].iloc[0]


def condition_row(conditions: pd.DataFrame, condition: str, seeds: str, precision: str = "fp32") -> pd.Series:
    return conditions[
        (conditions.condition == condition) & (conditions.seeds == seeds) & (conditions.precision == precision)
    ].iloc[0]


def interval_text(r: pd.Series) -> str:
    """Mean difference and its 95% participant bootstrap interval."""
    return f"{r.mean_diff_pp:+.2f} [{r.ci_low_pp:+.2f}, {r.ci_high_pp:+.2f}]"


def intro_lines(plan: dict, plan2: dict | None, n_new: int) -> list[str]:
    return [
        "# Electrode-count study (supplementary, post-test; generated by scripts/analyze_electrode_study.py)",
        "",
        f"New runs: {n_new}"
        + (f" ({len(plan['jobs'])} in the first wave, {len(plan2['jobs'])} in the second)" if plan2 else "")
        + ". Seeds 0-2 of the archived conditions come from results/test; everything else is new. Accuracy is on "
        "artifact-free session-2 trials, FP32 unless stated, seeds averaged within participant. "
        "Intervals resample the nine participants; the seed standard error holds the participants fixed and "
        "reflects training randomness only.",
    ]


def versus_full_lines(comp: pd.DataFrame) -> list[str]:
    lines = [
        "",
        "## 16 or 8 electrodes against all 22 (percentage points; 95% participant bootstrap interval)",
        "",
        "| comparison | seeds 0-2 (archived) | seeds 3-9 (new, independent) | all ten seeds "
        "| seed standard error (ten seeds) | participants better/worse (ten seeds) |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    pairs = [(a, FULL) for a in ("relevance-16", "proposed-16", "relevance-8", "proposed-8")]
    pairs += [(f"proposed-{k}", f"relevance-{k}") for k in (16, 8)]
    for a, b in pairs:
        archived, independent, r = (comparison_row(comp, a, b, seeds) for seeds in ("0-2", "3-9", "0-9"))
        lines.append(
            f"| {a} minus {b} | {interval_text(archived)} | {interval_text(independent)} | {interval_text(r)} | "
            f"{r.seed_se_pp:.2f} | {r.n_a_better}/{r.n_b_better} |"
        )
    return lines


def ten_seed_lines(conditions: pd.DataFrame, ten_seed: list[str]) -> list[str]:
    lines = [
        "",
        "## Mean accuracy with ten seeds (%)",
        "",
        "| condition | FP32 | INT8 | spread of single runs within a participant (SD, pp) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for c in ten_seed:
        fp32, int8 = condition_row(conditions, c, "0-9"), condition_row(conditions, c, "0-9", "int8")
        lines.append(f"| {c} | {fp32.acc_mean:.2f} | {int8.acc_mean:.2f} | {fp32.run_sd_within_participant:.2f} |")
    return lines


def seed_luck_lines(luck: dict, ensemble_summary: dict) -> list[str]:
    lines = [
        "",
        "## Is it seed luck?",
        "",
        "Drawing 3 of the 10 seeds at random for each participant and model, the nine-participant mean "
        "difference comes out as:",
        "",
    ]
    for name, s in luck.items():
        lines.append(
            f"- {name}: mean {s['mean_pp']:+.2f} pp, SD {s['sd_pp']:.2f} pp, central 95% "
            f"[{s['p2_5_pp']:+.2f}, {s['p97_5_pp']:+.2f}], "
            f"at or below zero in {100 * s['share_at_or_below_zero']:.0f}% of draws"
        )
    if ensemble_summary:
        f, r = ensemble_summary[FULL], ensemble_summary["relevance-16"]
        lines += [
            "",
            "Averaging the class probabilities of the ten seeds of each model into one prediction removes most "
            "run-to-run variance. A difference caused by that variance would shrink:",
            "",
            f"- mean single run: {f['single']:.2f}% with 22 electrodes, {r['single']:.2f}% with 16 "
            f"({r['single'] - f['single']:+.2f} pp)",
            f"- ten-seed average: {f['ensemble']:.2f}% with 22 electrodes, {r['ensemble']:.2f}% with 16 "
            f"({r['ensemble'] - f['ensemble']:+.2f} pp)",
            f"- two seeds of the same model give the same label on {f['agreement']:.1f}% of trials with 22 electrodes "
            f"and {r['agreement']:.1f}% with 16",
        ]
    return lines


def per_participant_lines(per: pd.DataFrame) -> list[str]:
    lines = [
        "",
        "## Per participant, ten seeds (FP32, %; difference with its standard error from seeds)",
        "",
        "| participant | 22 electrodes | 16 by relevance | 16 minus 22 | 8 by relevance | 8 minus 22 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in per.iterrows():
        lines.append(
            f"| {int(r.subject)} | {r[f'{FULL}_mean']:.1f} | {r['relevance-16_mean']:.1f} | "
            f"{r['relevance-16_minus_full_pp']:+.1f} ± {r['relevance-16_minus_full_se']:.1f} | "
            f"{r['relevance-8_mean']:.1f} | "
            f"{r['relevance-8_minus_full_pp']:+.1f} ± {r['relevance-8_minus_full_se']:.1f} |"
        )
    return lines


def group_lines(groups: dict, rho, groups_table: pd.DataFrame) -> list[str]:
    lines = [
        "",
        "## Who gains: participants grouped by their accuracy with all 22 electrodes",
        "",
        f"Below {GROUP_THRESHOLD:.0f}% with 22 electrodes: {participant_names(groups['below'])}. "
        f"Above: {participant_names(groups['above'])}. "
        "The split was chosen after seeing the results, and the groups have four and five people, so this describes "
        "these participants and is not a tested hypothesis. "
        "Rank correlation between a participant's 22-electrode accuracy "
        f"and the gain from 16 electrodes: {rho.statistic:+.2f} (n = 9, p = {rho.pvalue:.2f}).",
        "",
        "Differences are against all 22 electrodes with ten seeds; ± is the standard error from training seeds.",
        "",
        "| condition | runs per participant | all nine: accuracy % | difference (pp) "
        "| lower group: accuracy % | difference (pp) | upper group: accuracy % | difference (pp) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in groups_table.iterrows():
        cells = [
            f"{r[f'{g}_acc']:.1f} | "
            + ("0" if r.condition == FULL else f"{r[f'{g}_minus_full_pp']:+.1f} ± {r[f'{g}_seed_se_pp']:.1f}")
            for g in ("all", "below", "above")
        ]
        lines.append(f"| {r.condition} | {int(r.runs_per_participant)} | " + " | ".join(cells) + " |")
    return lines


def curve_lines(comp: pd.DataFrame, conditions: pd.DataFrame, curve_counts: list[int], available: set) -> list[str]:
    lines = [
        "",
        "## Accuracy against electrode count (relevance-ranked subsets)",
        "",
        "| electrodes | seeds 0-2: accuracy % | minus 22 electrodes (pp) [95% interval] "
        "| all ten seeds: accuracy % | minus 22 electrodes (pp) [95% interval] | seed standard error |",
        "| ---: | ---: | --- | ---: | --- | ---: |",
        f"| 22 | {condition_row(conditions, FULL, '0-2').acc_mean:.2f} | 0 | "
        f"{condition_row(conditions, FULL, '0-9').acc_mean:.2f} | 0 | |",
    ]
    for k in curve_counts:
        name = f"relevance-{k}"
        r = comparison_row(comp, name, FULL, "0-2")
        if (name, "fp32", "0-9") in available:
            r10 = comparison_row(comp, name, FULL, "0-9")
            ten = f"{condition_row(conditions, name, '0-9').acc_mean:.2f} | {interval_text(r10)} | {r10.seed_se_pp:.2f}"
        else:
            ten = "not trained | | "
        lines.append(f"| {k} | {condition_row(conditions, name, '0-2').acc_mean:.2f} | {interval_text(r)} | {ten} |")
    return lines


def subset_lines(comp: pd.DataFrame, conditions: pd.DataFrame) -> list[str]:
    r = comparison_row(comp, "leastrelevant-16", FULL, "0-2")
    r2 = comparison_row(comp, "leastrelevant-16", "relevance-16", "0-2")
    randoms = np.mean([condition_row(conditions, f"random{i}-16", "0-2").acc_mean for i in (1, 2, 3)])
    return [
        "",
        "## Which six electrodes are removed matters (16 electrodes, seeds 0-2)",
        "",
        f"- 16 most relevant: {condition_row(conditions, 'relevance-16', '0-2').acc_mean:.2f}%",
        f"- three fixed random subsets (archived): {randoms:.2f}% on average",
        "- 16 least relevant (the six most relevant removed): "
        f"{condition_row(conditions, 'leastrelevant-16', '0-2').acc_mean:.2f}%, "
        f"{interval_text(r)} against 22 electrodes and {interval_text(r2)} against the 16 most relevant",
    ]


def train_fit_lines(
    conditions: pd.DataFrame,
    ten_seed: list[str],
    fit_new: pd.DataFrame,
    archived_fit: pd.DataFrame,
    fit_rows: list[dict],
) -> list[str]:
    lines = [
        "",
        "## Training-day fit against held-out-day accuracy (FP32, %)",
        "",
        "| condition | seeds | training day | held-out day | gap |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {c} | 3-9 | {fit_new.loc[c, 'train_acc']:.1f} | {fit_new.loc[c, 'acc']:.1f} | "
        f"{fit_new.loc[c, 'train_acc'] - fit_new.loc[c, 'acc']:.1f} |"
        for c in ten_seed
        if c in fit_new.index
    )
    if len(archived_fit):
        fit_old = archived_fit.groupby(["condition", "subject"]).train_acc.mean().groupby("condition").mean() * 100
        for c in fit_old.index:
            test = condition_row(conditions, c, "0-2").acc_mean
            lines.append(f"| {c} | 0-2 | {fit_old[c]:.1f} | {test:.1f} | {fit_old[c] - test:.1f} |")
    if fit_rows:
        lines += [
            "",
            "Per participant, ten seeds:",
            "",
            "| participant | 22 electrodes: training day | held-out day | gap "
            "| 16 by relevance: training day | held-out day | gap |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in fit_rows:
            cells = [
                f"{r[f'{c}_train']:.1f} | {r[f'{c}_test']:.1f} | {r[f'{c}_train'] - r[f'{c}_test']:.1f}"
                for c in (FULL, "relevance-16")
            ]
            lines.append(f"| {r['subject']} | " + " | ".join(cells) + " |")
    return lines


def removal_lines(per_subject_sel: pd.DataFrame, selection_summary: dict) -> list[str]:
    means = per_subject_sel.mean(numeric_only=True)
    lines = [
        "",
        "## What the rule removes (relevance-16, archived session-1 rankings)",
        "",
        "| scalp row | electrodes | times dropped (of participants x electrodes) | share of all 54 drops |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row, v in selection_summary["by_scalp_row"].items():
        lines.append(
            f"| {row} | {v['electrodes']} | {v['dropped_at_16']}/{v['possible']} | "
            f"{100 * v['share_of_all_drops']:.0f}% |"
        )
    lines += [
        "",
        f"- Mean Fisher score per electrode, training day: kept {means.fisher_day1_kept:.3f}, dropped "
        f"{means.fisher_day1_dropped:.3f}; held-out day: kept {means.fisher_day2_kept:.3f}, "
        f"dropped {means.fisher_day2_dropped:.3f}.",
        "- The six dropped electrodes (27% of the montage) hold "
        f"{100 * means.dropped_share_of_fisher_day1:.0f}% of the summed Fisher score on the training day and "
        f"{100 * means.dropped_share_of_fisher_day2:.0f}% on the held-out day.",
        f"- Day-to-day change of band power, in within-day standard deviations: kept {means.day_shift_kept:.2f}, "
        f"dropped {means.day_shift_dropped:.2f}.",
        "- Rank correlation of electrode relevance between the two days, by participant: "
        + ", ".join(
            f"P{int(r.subject)} {r.rank_correlation_of_relevance_between_days:+.2f}"
            for _, r in per_subject_sel.iterrows()
        ),
        "",
    ]
    return lines


def what_if_lines(what_if_rows: list[dict], ratios: pd.Series | None) -> list[str]:
    if not what_if_rows:
        return []
    name = {
        "protocol": "as evaluated (training-day statistics)",
        "one_gain": "one gain for the whole recording",
        "per_electrode": "each electrode re-standardized on session 2",
    }
    lines = [
        "## What-if: session-2 inputs rescaled to the training amplitude (all ten seeds, no labels used)",
        "",
        "Session-2 amplitude relative to the training day, after the training-day normalization "
        "(1.00 means unchanged): " + ", ".join(f"P{s} {ratios[s]:.2f}" for s in data.SUBJECTS),
        "",
        "| input scaling | 22 electrodes % | 16 by relevance % | 16 minus 22 (pp) | 95% interval "
        "| P2: 22 electrodes % | P2: 16 by relevance % |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {name[r['scaling']]} | {r['acc_b']:.2f} | {r['acc_a']:.2f} | {r['mean_diff_pp']:+.2f} | "
        f"[{r['ci_low_pp']:+.2f}, {r['ci_high_pp']:+.2f}] | {r['per_participant_acc_b'][1]:.1f} | "
        f"{r['per_participant_acc_a'][1]:.1f} |"
        for r in what_if_rows
        if r["a"] == "relevance-16"
    )
    lines.append("")
    return lines


def repro_lines(repro: dict) -> list[str]:
    lines = ["## Pipeline check", ""]
    for name, v in repro.items():
        lines.append(
            f"- {name} trained again: predictions agree with the archived run on "
            f"{100 * v['prediction_agreement']['fp32']:.1f}% of session-2 trials "
            f"(accuracy {100 * v['acc_new']:.2f}% now, {100 * v['acc_archived']:.2f}% archived)"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--first-wave-only",
        action="store_true",
        help="ignore the second wave (for a look at the first wave while the second is still training)",
    )
    args = parser.parse_args()
    plan = json.loads((OUT / "protocol.json").read_text())
    use_second = (WAVE2 / "protocol.json").exists() and not args.first_wave_only
    plan2 = json.loads((WAVE2 / "protocol.json").read_text()) if use_second else None
    df = load_runs(plan, plan2)
    old, new = plan["historical_seeds"], plan["new_seeds"]
    all_seeds = old + new
    seed_sets = (("0-2", old), ("3-9", new), ("0-9", all_seeds))
    ten_seed = [FULL, "relevance-16", "proposed-16", "relevance-8", "proposed-8"]
    curve_counts = sorted([16, 8, *plan["sweep_counts"]], reverse=True)

    conditions = condition_table(df, seed_sets)
    conditions.to_csv(OUT / "conditions.csv", index=False)
    available = set(zip(conditions.condition, conditions.precision, conditions.seeds, strict=True))

    comparisons = paired_comparisons(df, plan, seed_sets, available)
    comp = pd.DataFrame(comparisons)
    comp.to_csv(OUT / "comparisons.csv", index=False)

    per = ten_seed_by_participant(df, ten_seed, all_seeds)
    per.to_csv(OUT / "per_participant.csv", index=False)

    # --- who gains: participants grouped by their accuracy with all 22 electrodes
    baseline = per.set_index("subject")[f"{FULL}_mean"]
    gain = per.set_index("subject")["relevance-16_minus_full_pp"]
    groups = group_of(baseline)
    rho = stats.spearmanr(baseline, gain)
    groups_table = by_group(df, groups)
    groups_table.to_csv(OUT / "by_baseline_group.csv", index=False)

    # --- electrode-count curve
    curve = conditions[
        (conditions.precision == "fp32")
        & conditions.seeds.isin(["0-2", "0-9"])
        & conditions.condition.isin([FULL, *(f"relevance-{k}" for k in curve_counts)])
    ]
    curve = curve.sort_values(["seeds", "electrodes"], ascending=[True, False])
    curve.to_csv(OUT / "electrode_count_curve.csv", index=False)

    # --- training-day fit against held-out-day score
    fit_new = (
        df[(df.precision == "fp32") & df.seed.isin(new) & df.condition.isin(ten_seed)]
        .groupby(["condition", "subject"])[["train_acc", "acc"]]
        .mean()
        .groupby("condition")
        .mean()
        * 100
    )
    archived_fit = archived_train_fit([FULL, "relevance-16", "relevance-8", "random1-16", "random2-16", "random3-16"])
    fit_rows = train_fit_by_participant(df, archived_fit, per, new, all_seeds)

    luck = {
        f"{a} minus {FULL}": seed_luck(df[df.seed.isin(all_seeds)], a, FULL, k=3)
        for a in ("relevance-16", "relevance-8")
    }
    per_subject_sel, selection_summary = selection_description()
    per_subject_sel.to_csv(OUT / "selection_by_participant.csv", index=False)

    what_if_rows, ensemble_summary, ratios = amplitude_what_if(plan)
    repro = repro_check(plan)

    figures(df, curve, groups_table, groups, baseline, plan)

    n_new = len(plan["jobs"]) + (len(plan2["jobs"]) if plan2 else 0)
    lines = [
        *intro_lines(plan, plan2, n_new),
        *versus_full_lines(comp),
        *ten_seed_lines(conditions, ten_seed),
        *seed_luck_lines(luck, ensemble_summary),
        *per_participant_lines(per),
        *group_lines(groups, rho, groups_table),
        *curve_lines(comp, conditions, curve_counts, available),
        *subset_lines(comp, conditions),
        *train_fit_lines(conditions, ten_seed, fit_new, archived_fit, fit_rows),
        *removal_lines(per_subject_sel, selection_summary),
        *what_if_lines(what_if_rows, ratios),
        *repro_lines(repro),
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    write_json(
        OUT / "analysis.json",
        {
            "status": "supplementary_post_test",
            "second_wave_included": bool(plan2),
            "new_runs": n_new,
            "comparisons": comparisons,
            "seed_luck": luck,
            "seed_ensemble": ensemble_summary,
            "groups": {
                "threshold_pct": GROUP_THRESHOLD,
                **groups,
                "spearman_baseline_vs_gain": float(rho.statistic),
                "spearman_p": float(rho.pvalue),
            },
            "by_baseline_group": groups_table.to_dict("records"),
            "train_fit_by_participant": fit_rows,
            "selection": selection_summary,
            "repro": repro,
            "amplitude_what_if": what_if_rows,
            "note": "Exploratory. Seeds 3-9 were trained after the seed 0-2 results were known; they are an "
            "independent draw of training randomness, not new participants. The second wave was planned after the "
            "first-wave results were known. The participant split by 22-electrode accuracy is post hoc.",
        },
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
