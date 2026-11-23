"""Tests of the Lee2019 replication hypotheses, as fixed in the plan (scripts/replication_study.py).

Reads results/supplementary/replication_study/protocol.json and its test records (no EEG) and
writes, in the same directory:

  hypotheses.csv       one row per hypothesis: the contrast, estimate, interval, the plan's test
                       and whether the data meet it
  conditions.csv       every condition against full-62 (FP32), all session-2 trials
  per_participant.csv  accuracy per participant and condition (seeds averaged)
  fig_curve.png        accuracy against the number of electrodes
  summary.md, analysis.json

Every number follows the plan: all session-2 trials (the dataset has no artifact flags), seeds
averaged within each participant (random-K: its three lists), then the 54 participants
summarized with the frozen percentile bootstrap (10,000 resamples, 95 %, seed 690). Seed
spread is reported apart from that interval: it says how much retraining the same people
moves the mean, not how other people would fare.
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

from mibci.artifacts import sha256, write_json
from mibci.paths import RESULTS_DIR, display
from mibci.stats import FROZEN_BOOTSTRAP, bootstrap_ci

OUT = RESULTS_DIR / "supplementary/replication_study"
PLAN = OUT / "protocol.json"
FULL = "full-62"
MATERIAL_PP = 2.0
TOLERANCE = 1e-9  # a loss of exactly 2 points (whole trials) is not more than 2 points
PERMUTATIONS = 10000
BLUE, ORANGE, GREY, NAVY = "#4B9CD3", "#eb6834", "#8a8a8a", "#13294B"


def load() -> tuple[dict, pd.DataFrame]:
    plan = json.loads(PLAN.read_text())
    digest = sha256(PLAN)
    missing = [j["name"] for j in plan["jobs"] if not (OUT / "test" / f"{j['name']}.json").is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(plan['jobs'])} test records missing (first {missing[0]}); "
                         "finish scripts/replication_study.py --stage evaluate")  # fmt: skip
    rows = []
    for job in plan["jobs"]:
        path = OUT / "test" / f"{job['name']}.json"
        record = json.loads(path.read_text())
        if record["protocol_sha256"] != digest or record["name"] != job["name"]:
            raise SystemExit(f"{display(path)} does not belong to this plan")
        for variant, s in record["scores"].items():
            rows.append({"subject": job["subject"], "seed": job["seed"], "condition": job["condition"],
                         "variant": variant, "all": 100 * s["acc_all"], "offline": 100 * s["acc_offline"],
                         "online": 100 * s["acc_online"]})  # fmt: skip
    return plan, pd.DataFrame(rows)


def family(condition: str) -> str:
    """random1-16 -> random-16; other conditions are their own family."""
    if condition.startswith("random"):
        return "random-" + condition.split("-")[1]
    return condition


def per_person(df: pd.DataFrame, variant: str, column: str = "all") -> pd.DataFrame:
    """Participants x condition family, seeds (random: lists) averaged within each participant."""
    part = df[df.variant == variant].assign(family=lambda d: d.condition.map(family))
    return part.groupby(["subject", "family"])[column].mean().unstack("family")


def seed_spread(df: pd.DataFrame, variant_a: str, cond_a: str, variant_b: str, cond_b: str, column: str) -> float:
    """SD over seeds of the participant-mean difference, pairing seed s with seed s."""
    a = df[(df.variant == variant_a) & (df.condition == cond_a)].set_index(["subject", "seed"])[column]
    b = df[(df.variant == variant_b) & (df.condition == cond_b)].set_index(["subject", "seed"])[column]
    by_seed = (a - b).groupby(level="seed").mean()
    return float(by_seed.std(ddof=1)) if len(by_seed) > 1 else float("nan")


def contrast(a: pd.Series, b: pd.Series) -> dict:
    diff = (a - b).to_numpy()
    low, high = bootstrap_ci(diff, FROZEN_BOOTSTRAP)
    return {"estimate_pp": float(diff.mean()), "low_pp": low, "high_pp": high,
            "better": int((diff > TOLERANCE).sum()), "worse": int((diff < -TOLERANCE).sum()),
            "lose_more_than_2": int((diff < -(MATERIAL_PP + TOLERANCE)).sum()), "worst_pp": float(diff.min()),
            "n": len(diff)}  # fmt: skip


def spearman_test(baseline: pd.Series, gain: pd.Series) -> dict:
    rho = stats.spearmanr(baseline, gain).statistic
    rng = np.random.default_rng(FROZEN_BOOTSTRAP["seed"])
    g = gain.to_numpy()
    null = np.array([stats.spearmanr(baseline, rng.permutation(g)).statistic for _ in range(PERMUTATIONS)])
    return {"rho": float(rho), "p_one_sided": float(((null <= rho).sum() + 1) / (PERMUTATIONS + 1)), "n": len(g)}


def test_hypotheses(df: pd.DataFrame) -> list[dict]:
    fp = per_person(df, "fp32")
    rows = []

    def add(key: str, result: dict, met: bool, **extra) -> None:
        rows.append({"hypothesis": key, **extra, **result, "met": bool(met)})

    for key, k in (("H1", 45), ("H1b", 16)):
        r = contrast(fp[f"relevance-{k}"], fp[FULL])
        spread = seed_spread(df, "fp32", f"relevance-{k}", "fp32", FULL, "all")
        add(key, r, r["low_pp"] > -MATERIAL_PP, seed_sd_pp=spread)
    for key, k in (("H2", 16), ("H2b", 45)):
        r = contrast(fp[f"relevance-{k}"], fp[f"random-{k}"])
        add(key, r, r["low_pp"] > 0)
    offline, online = per_person(df, "fp32", "offline"), per_person(df, "fp32", "online")
    for key, k in (("H3", 45), ("H3b", 16)):
        r = spearman_test(offline[FULL], online[f"relevance-{k}"] - online[FULL])
        add(key, r, r["rho"] < 0 and r["p_one_sided"] < 0.05)
    sim_float, w8 = per_person(df, "float"), per_person(df, "8/8")
    for condition in (FULL, "relevance-45", "relevance-16"):
        r = contrast(w8[condition], sim_float[condition])
        add("H4", r, r["low_pp"] > -0.5 and r["lose_more_than_2"] == 0, condition=condition)
    mapped = per_person(df, "map-2a")
    for condition in (FULL, "relevance-16"):
        r = contrast(mapped[condition], w8[condition])
        add("H4b", r, r["low_pp"] > -MATERIAL_PP, condition=condition)
    ea_online, ea_all = per_person(df, "ea-offline", "online"), per_person(df, "ea-all")
    for condition in (FULL, "relevance-45", "relevance-16"):
        r = contrast(ea_online[f"{condition}-ea"], online[condition])
        key = "H5" if condition == FULL else "H5b"
        add(key, r, r["low_pp"] > 0, condition=condition, alignment="session 2 aligned on its offline run; online run")
        r = contrast(ea_all[f"{condition}-ea"], fp[condition])
        add("H5b", r, r["low_pp"] > 0, condition=condition, alignment="session 2 aligned with all its trials; all")
    return rows


def condition_table(df: pd.DataFrame) -> pd.DataFrame:
    fp = per_person(df, "fp32")
    rows = [{"condition": c, "accuracy": float(fp[c].mean()), **contrast(fp[c], fp[FULL])} for c in fp.columns]
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)


def figure(df: pd.DataFrame) -> None:
    fp = per_person(df, "fp32")
    counts = [8, 16, 20, 32, 45, 62]
    rel = [fp[FULL if k == 62 else f"relevance-{k}"] for k in counts]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    means = [s.mean() for s in rel]
    lows, highs = zip(*(bootstrap_ci(s.to_numpy(), FROZEN_BOOTSTRAP) for s in rel), strict=True)
    ax.fill_between(counts, lows, highs, color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(counts, means, "o-", color=BLUE, label="most relevant K (session 1)")
    for k in (16, 45):
        ax.plot([k], [fp[f"random-{k}"].mean()], "s", color=GREY, label="random K" if k == 16 else None)
    ax.plot([20], [fp["motor-20"].mean()], "D", color=ORANGE, label="20 sensorimotor electrodes (no data)")
    ax.set_xlabel("electrodes")
    ax.set_ylabel("session-2 accuracy, % (54 people)")
    ax.set_xticks(counts)
    ax.axhline(50, color=GREY, linewidth=0.8, linestyle=":")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_curve.png", dpi=200)
    plt.close(fig)


def fmt(r: dict) -> str:
    return f"{r['estimate_pp']:+.2f} [{r['low_pp']:+.2f}, {r['high_pp']:+.2f}]"


def summary(plan: dict, hyp: list[dict], table: pd.DataFrame, fp: pd.DataFrame) -> str:
    lines = [
        "# Lee2019 replication (54 people, two days; generated by scripts/analyze_replication_study.py)", "",
        f"Plan written {plan['created_at']}, before any Lee2019 session-2 trial was scored; the hypotheses and "
        "tests below are copied from it. They were chosen knowing the BCI IV 2a results they carry over.", "",
        "Accuracy: share of all session-2 trials classified correctly (two classes, chance 50%), seeds averaged "
        "within each participant, then the 54 participants averaged. Differences in percentage points (pp) with "
        "the 95% participant bootstrap interval (10,000 resamples, seed 690). 'Met' means the data meet the "
        "plan's test; it is not a statement that two conditions are the same.", "",
        "## Hypotheses", "",
        "| | claim | contrast | estimate [95%] | better/worse | lose > 2 pp | test | met |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]  # fmt: skip
    for r in hyp:
        h = plan["hypotheses"][r["hypothesis"]]
        what = r.get("condition", "")
        if "alignment" in r:
            what += f" ({r['alignment']})"
        if "rho" in r:
            est, bw, lose = f"rho {r['rho']:+.3f}, p = {r['p_one_sided']:.4f}", "", ""
        else:
            est, bw, lose = fmt(r), f"{r['better']}/{r['worse']}", str(r["lose_more_than_2"])
        lines.append(f"| {r['hypothesis']} | {h['claim']} | {h['contrast']}{' — ' + what if what else ''} | {est} | "
                     f"{bw} | {lose} | {h['test']} | {'yes' if r['met'] else 'no'} |")  # fmt: skip
    lines += ["", "## Every condition against all 62 electrodes (FP32)", "",
              "| condition | accuracy % | minus full-62 (pp) [95%] | better/worse | lose > 2 pp |",
              "| --- | ---: | --- | --- | ---: |"]  # fmt: skip
    for _, r in table.iterrows():
        lines.append(f"| {r.condition} | {r.accuracy:.2f} | {fmt(r)} | {r.better}/{r.worse} | "
                     f"{r.lose_more_than_2} |")  # fmt: skip
    above = int((fp[FULL] > 60).sum())
    lines += ["", f"Full-62 accuracy ranges from {fp[FULL].min():.1f}% to {fp[FULL].max():.1f}% across people; "
              f"{above} of {len(fp)} are above 60%.", "",
              "## Boundaries", "",
              "- One dataset, one two-class task, 54 people: the results describe this cohort, not every user.",
              "- An interval that contains zero is not evidence of no difference; a 'met' non-inferiority test "
              "bounds the loss, it does not show the conditions are equal.",
              "- Bit widths below 8 are simulated integer grids; no low-bit kernel ran and nothing was timed.",
              "- Alignment uses unlabeled session-2 trials (the offline run, or all of them), which the frozen "
              "BCI IV 2a protocol did not.", ""]  # fmt: skip
    return "\n".join(lines)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    plan, df = load()
    hyp = test_hypotheses(df)
    table = condition_table(df)
    fp = per_person(df, "fp32")
    pd.DataFrame(hyp).to_csv(OUT / "hypotheses.csv", index=False)
    table.to_csv(OUT / "conditions.csv", index=False)
    pd.concat({v: per_person(df, v) for v in ("fp32", "8/8", "map-2a", "ea-all")}, axis=1).to_csv(
        OUT / "per_participant.csv"
    )
    figure(df)
    (OUT / "summary.md").write_text(summary(plan, hyp, table, fp))
    write_json(OUT / "analysis.json", {"hypotheses": hyp, "plan_sha256": sha256(PLAN)})
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
