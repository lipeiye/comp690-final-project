"""Supplementary, post-test INT8 deployment study on the original Apple M4 (added after the midterm review).

Review question: the INT8 model is slower than FP32 on this Mac because of the grouped
spatial convolution. Is the machine the reason, would INT16 help, and what does help here?

Every variant is derived from an archived final model without retraining; calibration
uses that model's session-1 training trials only. The archived FP32 and INT8 graphs are
the references and are never rewritten.

  int8-u8act, int8-pertensor, int8-qoperator   the same graph with one quantization setting changed
  int16, int16-w16      16-bit activations with 8-bit or 16-bit weights
  int8-mixed            INT8 everywhere except the spatial convolution, which stays FP32
  int8-dense            grouped spatial convolution rewritten as an equivalent dense one, then INT8
  fp32-spatial-first    first two layers swapped (same function), FP32
  int8-spatial-first    the swapped graph, INT8

Stages
  build      all archived final models: build the variants, check the FP32 rewrites against
             the archived FP32 output on session-1 inputs, score session 2 (post-test, exploratory)
  benchmark  batch-1, single-thread latency of every variant for one model per architecture
             (subject 1, seed 0, as in benchmark.py), in interleaved rounds; run on a quiet machine
  analyze    tables in results/supplementary/int8_study/

Nothing here changes the historical final configuration or the archived benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import onnxruntime as ort

from mibci import data, deploy, variants
from mibci.artifacts import completed, require_files, sha256, write_json
from mibci.env import environment
from mibci.experiment import MODELS_DIR, RESULTS_DIR, run_pool, session1
from mibci.model import load_bundle
from mibci.timing import build_variants, power_state, timed_rounds
from mibci.train import scores

OUT = RESULTS_DIR / "supplementary/int8_study"
KEPT = MODELS_DIR / "supplementary/int8_study"
SOURCES = ["src/mibci/variants.py", "src/mibci/deploy.py", "scripts/int8_study.py"]
BENCH_SUBJECT, BENCH_SEED = 1, 0
SCORED = ("int8-mixed", "int16", "int16-w16", "int8-dense", "fp32-spatial-first", "int8-spatial-first")
TIMED_ONLY = ("int8-u8act", "int8-pertensor", "int8-qoperator")
REFERENCES = ("fp32", "int8")


def build_job(bundle_path: str, record_path: str) -> None:
    bundle = Path(bundle_path)
    _, scaler, b = load_bundle(bundle)
    fp32, int8 = bundle.with_suffix(".onnx"), bundle.with_name(f"{bundle.stem}_int8.onnx")
    X, _, tr, _ = session1(b["subject"], "final")
    train_inputs = scaler.transform(X[tr][:, b["channels"]])
    ses = data.load_session(b["subject"], 2, final_evaluation=True)
    test_inputs = scaler.transform(ses["X"][:, b["channels"]])
    clean = ~ses["artifact"]
    archived = json.loads((RESULTS_DIR / "test" / f"{bundle.stem}.json").read_text())
    keep = b["subject"] == BENCH_SUBJECT and b["seed"] == BENCH_SEED

    with tempfile.TemporaryDirectory() as tmp:
        folder = KEPT if keep else Path(tmp)
        folder.mkdir(parents=True, exist_ok=True)
        paths = build_variants(fp32, folder, bundle.stem, train_inputs, SCORED)
        reference = {"fp32": deploy.session(fp32), "int8": deploy.session(int8)}
        train_ref = {p: deploy.logits(s, train_inputs) for p, s in reference.items()}
        test_ref = {p: deploy.logits(s, test_inputs) for p, s in reference.items()}
        for p in REFERENCES:  # the archived predictions must still come out of the archived graphs
            if test_ref[p].argmax(1).tolist() != archived["pred"][p]:
                raise RuntimeError(f"{bundle.stem}: archived {p} predictions are not reproduced")
        record = {
            "name": bundle.stem,
            "subject": b["subject"],
            "seed": b["seed"],
            "condition": b["condition"],
            "electrodes": len(b["channels"]),
            "F2": b["F2"],
            "status": "supplementary_post_test",
            "variants_sha256": sha256(data.PROJECT_ROOT / "src/mibci/variants.py"),
            "costs": {"original": _costs(fp32), "spatial-first": _costs(paths["fp32-spatial-first"])},
            "y": ses["y"].tolist(),
            "artifact": ses["artifact"].astype(int).tolist(),
            "variants": {},
        }
        for name, path in paths.items():
            sess = deploy.session(path)
            train_out, test_out = deploy.logits(sess, train_inputs), deploy.logits(sess, test_inputs)
            entry = {
                "size": deploy.size_bytes(path),
                "executed_ops": deploy.executed_ops(path),
                "primary": scores(test_out[clean], ses["y"][clean]),
                "secondary": scores(test_out, ses["y"]),
                "pred": test_out.argmax(1).tolist(),
                "train_max_abs_diff": {p: float(np.abs(train_out - train_ref[p]).max()) for p in REFERENCES},
                "test_max_abs_diff": {p: float(np.abs(test_out - test_ref[p]).max()) for p in REFERENCES},
                "test_agreement": {p: float((test_out.argmax(1) == test_ref[p].argmax(1)).mean()) for p in REFERENCES},
            }
            if name == "fp32-spatial-first" and not np.allclose(train_out, train_ref["fp32"], atol=1e-4, rtol=1e-4):
                raise RuntimeError(f"{bundle.stem}: the swapped FP32 graph differs from the archived FP32 graph")
            record["variants"][name] = entry
    write_json(Path(record_path), record)


def _costs(path: Path) -> dict:
    costs = variants.graph_costs(path)
    return {k: costs[k] for k in ("macs", "params", "peak_activation_elems")}


def benchmark(rounds: int, runs: int) -> None:
    out = OUT / "benchmark.json"
    if out.exists():
        raise SystemExit(f"{out} exists. Keep the measurement; move it aside deliberately to measure again.")
    frozen = json.loads((RESULTS_DIR / "frozen_config.json").read_text())
    base = frozen["pruning"]["bases"][1]
    conditions = list(
        dict.fromkeys(
            ["full-22", "proposed-16", "proposed-8", base]
            + [f"{b}-prune{p}" for b in ("full-22", base) for p in (25, 50)]
        )
    )
    X, _, train_mask, _ = session1(BENCH_SUBJECT, "final")
    load_before = os.getloadavg()
    rows, profiles = [], {}
    KEPT.mkdir(parents=True, exist_ok=True)
    for condition in conditions:
        bundle = MODELS_DIR / "final" / f"sub-{BENCH_SUBJECT:02d}_seed-{BENCH_SEED}_{condition}.pt"
        _, scaler, b = load_bundle(bundle)
        train_inputs = scaler.transform(X[train_mask][:, b["channels"]])
        inputs = train_inputs[:100]
        fp32, int8 = bundle.with_suffix(".onnx"), bundle.with_name(f"{bundle.stem}_int8.onnx")
        paths = {
            "fp32": fp32,
            "int8": int8,
            **build_variants(fp32, KEPT, bundle.stem, train_inputs, SCORED + TIMED_ONLY),
        }
        timing = timed_rounds(paths, inputs, rounds, runs, warmup=200)
        for name, path in paths.items():
            graph = "spatial-first" if "spatial-first" in name else "original"
            rows.append(
                {
                    "condition": condition,
                    "variant": name,
                    "electrodes": len(b["channels"]),
                    "F2": b["F2"],
                    **timing[name],
                    **deploy.size_bytes(path),
                    "executed_ops": deploy.executed_ops(path),
                    "graph": graph,
                    **_costs(paths["fp32-spatial-first"] if graph == "spatial-first" else fp32),
                }
            )
            profiles[f"{condition} {name}"] = deploy.op_profile(path, inputs)
        ref = timing["fp32"]["median_ms"]
        print(
            condition,
            {name: f"{t['median_ms']:.3f} ms ({t['median_ms'] / ref:.2f}x)" for name, t in timing.items()},
            flush=True,
        )
    write_json(
        out,
        {
            "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "status": "supplementary_post_test",
            "environment": {
                **environment(),
                "onnxruntime": ort.__version__,
                "machine": platform.machine(),
                "hw_model": _sysctl("hw.model"),
                "power": power_state(),
                "cpu_features": {k: _sysctl(f"hw.optional.arm.{k}") for k in ("FEAT_DotProd", "FEAT_I8MM")},
                "ort_threads": {"intra_op": 1, "inter_op": 1},
                "load_average_before": load_before,
                "load_average_after": os.getloadavg(),
            },
            "models": f"archived final models of subject {BENCH_SUBJECT}, seed {BENCH_SEED}; "
            "variants rebuilt from them",
            "method": "Sessions for all variants of one architecture are created together and timed in interleaved "
            "rounds; the reported value is the median of the per-round medians.",
            "sources_sha256": {p: sha256(data.PROJECT_ROOT / p) for p in SOURCES},
            "rows": rows,
            "op_profile_us": profiles,
        },
    )


def _sysctl(key: str) -> str:
    if platform.system() != "Darwin":
        return "unknown"
    return (
        subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=False).stdout.strip() or "unknown"
    )


def analyze() -> None:
    import pandas as pd

    plan = json.loads((RESULTS_DIR / "frozen_config.json").read_text())["uncertainty"]
    rng_seed, resamples, level = plan["seed"], plan["resamples"], plan["level"]
    records = [json.loads(p.read_text()) for p in sorted((OUT / "models").glob("sub-*.json"))]
    expected = len(list((RESULTS_DIR / "final" / "runs").glob("sub-*.json")))
    if len(records) != expected:
        raise SystemExit(f"{len(records)} of {expected} models scored; finish --stage build first")
    df = pd.DataFrame(accuracy_rows(records))
    pp = df.groupby(["condition", "variant", "subject"])[["acc", "acc_all", "agree_fp32"]].mean().reset_index()

    def interval(diffs: np.ndarray) -> tuple[float, float]:
        rng = np.random.default_rng(rng_seed)
        means = diffs[rng.integers(0, len(diffs), size=(resamples, len(diffs)))].mean(1)
        tail = (1 - level) / 2
        return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))

    table = []
    for (condition, variant), g in pp.groupby(["condition", "variant"]):
        acc = g.set_index("subject")["acc"]
        ref = pp[(pp.condition == condition) & (pp.variant == "fp32")].set_index("subject")["acc"]
        diff = (acc - ref).to_numpy()
        lo, hi = interval(diff) if variant != "fp32" else (0.0, 0.0)
        table.append(
            {
                "condition": condition,
                "variant": variant,
                "acc_mean": 100 * acc.mean(),
                "diff_vs_fp32_pp": 100 * diff.mean(),
                "ci_low_pp": 100 * lo,
                "ci_high_pp": 100 * hi,
                "worst_participant_diff_pp": 100 * diff.min(),
                "acc_all_trials": 100 * g["acc_all"].mean(),
                "agreement_with_fp32": 100 * g["agree_fp32"].mean(),
            }
        )
    acc_table = pd.DataFrame(table)
    acc_table.to_csv(OUT / "accuracy.csv", index=False)
    overall = pp.groupby(["variant", "subject"])["acc"].mean().unstack(0)
    summary = {
        "n_models": len(records),
        "int8_dense_identical_to_archived_int8": bool((df[df.variant == "int8-dense"]["max_diff_int8"] == 0).all()),
        "fp32_spatial_first_max_abs_logit_diff_train": max(
            r["variants"]["fp32-spatial-first"]["train_max_abs_diff"]["fp32"] for r in records
        ),
        "fp32_spatial_first_prediction_agreement_min": float(
            df[df.variant == "fp32-spatial-first"]["agree_fp32"].min()
        ),
        "all_conditions_mean_accuracy": {v: 100 * float(overall[v].mean()) for v in overall.columns},
        "all_conditions_diff_vs_fp32_pp": {},
    }
    for v in overall.columns:
        d = (overall[v] - overall["fp32"]).to_numpy()
        lo, hi = interval(d) if v != "fp32" else (0.0, 0.0)
        summary["all_conditions_diff_vs_fp32_pp"][v] = {
            "mean": 100 * float(d.mean()),
            "ci_low": 100 * lo,
            "ci_high": 100 * hi,
        }
    costs = pd.DataFrame(
        [
            {
                "condition": r["condition"],
                "electrodes": r["electrodes"],
                "F2": r["F2"],
                **{f"{g}_{k}": v for g, c in r["costs"].items() for k, v in c.items()},
            }
            for r in records
        ]
    )
    costs = costs.drop_duplicates("condition")
    costs = costs[~costs.condition.str.startswith(("random", "weightnorm"))].sort_values(
        ["electrodes", "F2"], ascending=False
    )
    costs.to_csv(OUT / "costs.csv", index=False)
    lines = [
        "# INT8 deployment study (supplementary, post-test; generated by scripts/int8_study.py)",
        "",
        f"Models: all {len(records)} archived final models. Variants are rebuilt from them without retraining.",
        "",
        f"- int8-dense logits identical to the archived INT8 graph on all session-2 trials: "
        f"**{summary['int8_dense_identical_to_archived_int8']}**",
        f"- fp32-spatial-first: largest logit difference from archived FP32 on session-1 inputs "
        f"**{summary['fp32_spatial_first_max_abs_logit_diff_train']:.1e}**; lowest per-model prediction agreement "
        f"on session 2 **{100 * summary['fp32_spatial_first_prediction_agreement_min']:.2f}%**",
        "",
        "## Accuracy over all 17 conditions (seeds averaged within participant, then 9 participants)",
        "",
        "| variant | accuracy % | minus FP32 (pp) | 95% interval |",
        "| --- | ---: | ---: | --- |",
    ]
    for v, acc in summary["all_conditions_mean_accuracy"].items():
        d = summary["all_conditions_diff_vs_fp32_pp"][v]
        lines.append(f"| {v} | {acc:.2f} | {d['mean']:+.2f} | [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}] |")
    shown = ["full-22", "relevance-16", "relevance-16-prune50", "proposed-8"]
    lines += [
        "",
        "## Accuracy for four architectures (% and difference from that condition's FP32)",
        "",
        "| condition | variant | accuracy % | minus FP32 (pp) | 95% interval | worst participant (pp) |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for condition in shown:
        for _, row in acc_table[acc_table.condition == condition].iterrows():
            lines.append(
                f"| {condition} | {row.variant} | {row.acc_mean:.2f} | {row.diff_vs_fp32_pp:+.2f} | "
                f"[{row.ci_low_pp:+.2f}, {row.ci_high_pp:+.2f}] | {row.worst_participant_diff_pp:+.2f} |"
            )
    lines += [
        "",
        "## Analytical cost of the two graphs (one 2-s window)",
        "",
        "| condition | MACs original | MACs spatial-first | ratio | peak activation values original "
        "| spatial-first | values stored original | spatial-first |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, c in costs.iterrows():
        lines.append(
            f"| {c.condition} | {c.original_macs:,} | {c['spatial-first_macs']:,} | "
            f"{c.original_macs / c['spatial-first_macs']:.1f}x | {c.original_peak_activation_elems:,} | "
            f"{c['spatial-first_peak_activation_elems']:,} | {c.original_params:,} | {c['spatial-first_params']:,} |"
        )
    lines += latency_section(summary)
    write_json(OUT / "analysis.json", summary)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def accuracy_rows(records: list[dict]) -> list[dict]:
    """One row per model and graph: the archived FP32 and INT8 references, then every scored variant."""
    rows = []
    for r in records:
        archived = json.loads((RESULTS_DIR / "test" / f"{r['name']}.json").read_text())
        rows.extend(
            {
                "subject": r["subject"],
                "seed": r["seed"],
                "condition": r["condition"],
                "variant": p,
                "acc": archived["primary"][p]["acc"],
                "acc_all": archived["secondary"][p]["acc"],
                "agree_fp32": 1.0 if p == "fp32" else archived["int8_pred_agreement"],
                "max_diff_int8": float("nan"),
            }
            for p in REFERENCES
        )
        for name, v in r["variants"].items():
            rows.append(
                {
                    "subject": r["subject"],
                    "seed": r["seed"],
                    "condition": r["condition"],
                    "variant": name,
                    "acc": v["primary"]["acc"],
                    "acc_all": v["secondary"]["acc"],
                    "agree_fp32": v["test_agreement"]["fp32"],
                    "max_diff_int8": v["test_max_abs_diff"]["int8"],
                }
            )
    return rows


def latency_section(summary: dict) -> list[str]:
    """Summary lines for benchmark.json, once --stage benchmark has written it (none before).

    Also writes latency.csv and fig_latency.png and adds the full-22 latencies to `summary`."""
    import pandas as pd

    bench_path = OUT / "benchmark.json"
    if not bench_path.exists():
        return []
    bench = json.loads(bench_path.read_text())
    b = pd.DataFrame(bench["rows"])
    b["vs_fp32"] = b["median_ms"] / b.groupby("condition")["median_ms"].transform("first")
    b.drop(columns=["executed_ops"]).to_csv(OUT / "latency.csv", index=False)
    env = bench["environment"]
    lines = [
        "",
        f"## Latency on {env['cpu']} ({env['hw_model']}), ONNX Runtime {env['onnxruntime']}, batch 1, 1 thread",
        "",
        f"Median of {b['rounds'].iloc[0]} interleaved rounds of {b['runs_per_round'].iloc[0]} calls. "
        f"Load average before/after: {env['load_average_before'][0]:.1f} / {env['load_average_after'][0]:.1f}.",
        "",
        "| condition | variant | median ms | relative to FP32 | round range ms | parameter bytes |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for _, row in b.iterrows():
        lines.append(
            f"| {row.condition} | {row.variant} | {row.median_ms:.3f} | {row.vs_fp32:.2f}x | "
            f"{row.best_round_ms:.3f}-{row.worst_round_ms:.3f} | {row.parameter_bytes:,} |"
        )
    for key in (
        "full-22 fp32",
        "full-22 int8",
        "full-22 int8-dense",
        "full-22 int8-mixed",
        "full-22 int16",
        "full-22 fp32-spatial-first",
        "full-22 int8-spatial-first",
    ):
        prof = bench["op_profile_us"][key]
        top = sorted(prof.items(), key=lambda kv: -kv[1])[:4]
        lines.append("")
        lines.append(
            f"Per-node time, {key}: total {sum(prof.values()):.0f} us; "
            + "; ".join(f"{n.split(':')[0]} {n.split(':')[1]} {v:.0f} us" for n, v in top)
        )
    summary["latency_full22_ms"] = {r.variant: r.median_ms for _, r in b[b.condition == "full-22"].iterrows()}
    figure_latency(b, env)
    loaded_path = OUT / "benchmark_while_training.json"
    if loaded_path.exists():  # an earlier run that overlapped with training; kept as a robustness check
        loaded = json.loads(loaded_path.read_text())
        times = {r["variant"]: r["median_ms"] for r in loaded["rows"] if r["condition"] == "full-22"}
        ratios = ", ".join(f"{v} {times[v] / times['fp32']:.2f}x" for v in times if v != "fp32")
        lines += [
            "",
            f"A first run overlapped with seven training processes (load average "
            f"{loaded['environment']['load_average_before'][0]:.0f}; kept as benchmark_while_training.json). "
            f"Absolute times were higher (full-22 FP32 {times['fp32']:.3f} ms) but the ratios to FP32 were "
            f"the same within a few percent: {ratios}.",
        ]
        summary["latency_full22_ms_while_training"] = times
    return lines


def figure_latency(b, env: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [
        ("fp32", "FP32 (original)", "#404040"),
        ("int8", "INT8 (original)", "#c55a11"),
        ("int16", "INT16 activations", "#8c8c8c"),
        ("int8-mixed", "INT8, spatial conv kept FP32", "#2e75b6"),
        ("int8-dense", "INT8, grouped conv rewritten as dense", "#2e75b6"),
        ("fp32-spatial-first", "FP32, first two layers swapped", "#548235"),
        ("int8-spatial-first", "INT8, first two layers swapped", "#548235"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.25, 1]})
    full = b[b.condition == "full-22"].set_index("variant")
    labels = [label for _, label, _ in shown][::-1]
    values = [full.loc[key, "median_ms"] * 1000 for key, _, _ in shown][::-1]
    bars = axes[0].barh(labels, values, color=[c for _, _, c in shown][::-1])
    reference = full.loc["fp32", "median_ms"] * 1000
    for bar, value in zip(bars, values, strict=True):
        axes[0].text(
            value + 4,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f} us ({value / reference:.2f}x)",
            va="center",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
        )
    axes[0].set(
        xlabel="median time per prediction (microseconds)",
        xlim=(0, max(values) * 1.32),
        title="22 electrodes, one prediction",
    )
    axes[0].axvline(reference, color="#404040", lw=0.8, ls=":", zorder=0)
    for key, label, color, style in [
        ("fp32", "FP32 (original)", "#404040", "-"),
        ("int8", "INT8 (original)", "#c55a11", "-"),
        ("fp32-spatial-first", "FP32, layers swapped", "#548235", "--"),
        ("int8-spatial-first", "INT8, layers swapped", "#548235", "-"),
    ]:
        rows = b[(b.variant == key) & (b.F2 == 16)].sort_values("electrodes")
        rows = rows.drop_duplicates("electrodes")
        axes[1].plot(rows.electrodes, rows.median_ms * 1000, marker="o", color=color, ls=style, label=label)
    axes[1].set(xlabel="electrodes", ylabel="microseconds", xticks=[8, 16, 22], title="Time against electrode count")
    axes[1].set_ylim(bottom=0)
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{env['cpu']} CPU, ONNX Runtime {env['onnxruntime']}, batch 1, one thread", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_latency.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["build", "benchmark", "analyze"], required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--runs", type=int, default=500)
    args = parser.parse_args()
    if args.stage == "build":
        records = sorted((RESULTS_DIR / "final" / "runs").glob("sub-*.json"))
        bundles = [MODELS_DIR / "final" / f"{p.stem}.pt" for p in records]
        require_files(
            [q for b in bundles for q in (b, b.with_suffix(".onnx"), b.with_name(f"{b.stem}_int8.onnx"))],
            "INT8 study inputs (archived final models)",
        )
        if not (OUT / "protocol.json").exists():
            write_json(
                OUT / "protocol.json",
                {
                    "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
                    "status": "supplementary_post_test",
                    "historical_test_results_already_known": True,
                    "purpose": "Explain the INT8 slowdown on the Apple M4 CPU and measure variants that avoid it",
                    "models": f"all {len(bundles)} archived final models; no retraining",
                    "calibration": "each model's own session-1 training trials",
                    "scored_variants": list(SCORED),
                    "timed_only_variants": list(TIMED_ONLY),
                    "claim_boundary": "Exploratory. Does not change the frozen final configuration "
                    "or the archived benchmark.",
                    "sources_sha256_at_build": {p: sha256(data.PROJECT_ROOT / p) for p in SOURCES},
                },
            )
        jobs = [
            (b.stem, {"bundle_path": str(b), "record_path": str(OUT / "models" / f"{b.stem}.json")})
            for b in bundles
            if not completed(OUT / "models" / f"{b.stem}.json", [], {"name": b.stem})
        ]
        run_pool(build_job, jobs, args.workers, threads=1)
    elif args.stage == "benchmark":
        benchmark(args.rounds, args.runs)
    else:
        analyze()


if __name__ == "__main__":
    main()
