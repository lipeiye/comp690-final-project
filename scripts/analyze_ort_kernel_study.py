"""Tables and a figure for the two missing-kernel studies (scripts/ort_kernel_study.py and
scripts/ort_kernel_swapped_study.py).

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only, never EEG:
  supplementary/ort_kernel_study/{protocol.json,equivalence/,benchmark.json}          the grouped-convolution
                                                                                     kernel (original order)
  supplementary/ort_kernel_swapped_study/{protocol.json,equivalence/,benchmark.json}  the spatial-first kernel
                                                                                     and both kernels timed together

Writes to RESULTS_DIR/supplementary/ort_kernel_study/: latency.csv, layers.csv, fig_kernel_layers.png,
summary.md and analysis.json. The latency and layer tables come from the second study's benchmark, which
times every variant in one interleaved set; the first study's benchmark is listed as the earlier
measurement. Latency is one machine's (single thread, batch 1); exploratory.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mibci.artifacts import write_json
from mibci.paths import RESULTS_DIR

OUT = RESULTS_DIR / "supplementary" / "ort_kernel_study"
SECOND = RESULTS_DIR / "supplementary" / "ort_kernel_swapped_study"
NAMES = {
    "fp32": "FP32",
    "int8": "INT8 (ONNX Runtime)",
    "int8-kernel": "INT8 + new kernel",
    "int8-dense": "INT8, dense rewrite",
    "fp32-spatial-first": "FP32, layers swapped",
    "int8-spatial-first": "INT8, layers swapped",
    "int8-spatial-first-kernel": "INT8, layers swapped + new kernel",
}
ORDER = ["fp32", "int8", "int8-kernel", "fp32-spatial-first", "int8-spatial-first", "int8-spatial-first-kernel"]


def layer_of(node: str) -> str:
    """Temporal, spatial or the rest, from ONNX Runtime's node names, which the swapped graphs keep
    (Conv_90: the temporal convolution, Conv_92: the spatial one, in either order)."""
    if "spatial_grouped_int8" in node or "spatial_first_int8" in node or "Conv_92" in node:
        return "spatial"
    if "Conv_90" in node:
        return "temporal"
    return "rest"


def equivalence(folder, prediction_key: str) -> dict:
    plan = json.loads((folder / "protocol.json").read_text())
    eq = [json.loads((folder / "equivalence" / f"{j['name']}.json").read_text()) for j in plan["jobs"]]
    return {
        "models": len(eq),
        "logits_identical": sum(r["logits_identical"] for r in eq),
        "max_abs_logit_difference": max(r["max_abs_logit_difference"] for r in eq),
        "windows": sum(r["n_windows"] for r in eq),
        "predictions_equal_archived": sum(r[prediction_key] for r in eq),
    }


def layer_table(bench: dict) -> pd.DataFrame:
    rows = []
    for key, nodes in bench["op_profile_us"].items():
        condition, variant = key.split(" ")
        totals: dict[str, float] = {"temporal": 0.0, "spatial": 0.0, "rest": 0.0}
        for node, us in nodes.items():
            totals[layer_of(node)] += us
        rows.append({"condition": condition, "variant": variant, **totals})
    return pd.DataFrame(rows)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    first = json.loads((OUT / "benchmark.json").read_text())
    second = json.loads((SECOND / "benchmark.json").read_text())
    eq = {
        "grouped": equivalence(OUT, "predictions_equal_archived_int8"),
        "swapped": equivalence(SECOND, "predictions_equal_archived"),
    }
    latency = pd.DataFrame(second["rows"])
    ref = latency[latency.variant == "fp32"].set_index("condition")["median_ms"]
    latency["us"] = latency["median_ms"] * 1000
    latency["vs_fp32"] = latency["median_ms"] / latency["condition"].map(ref)
    columns = ["condition", "variant", "electrodes", "F2", "us", "vs_fp32", "best_round_ms", "worst_round_ms"]
    latency[[*columns, "parameter_bytes", "file_bytes"]].to_csv(OUT / "latency.csv", index=False, float_format="%.4f")
    layers = layer_table(second)
    layers.to_csv(OUT / "layers.csv", index=False, float_format="%.2f")
    earlier = pd.DataFrame(first["rows"])
    earlier["us"] = earlier["median_ms"] * 1000

    fig, ax = plt.subplots(figsize=(10, 4.4))
    g = layers[layers.condition == "full-22"].set_index("variant").loc[ORDER]
    bottom = pd.Series(0.0, index=g.index)
    labels = {"temporal": "temporal convolution", "spatial": "spatial convolution", "rest": "other layers"}
    for part, color in (("temporal", "#13294B"), ("spatial", "#C8102E"), ("rest", "#9ec5e8")):
        ax.bar(range(len(ORDER)), g[part], 0.6, bottom=bottom, color=color, label=labels[part])
        bottom += g[part]
    ticks = [NAMES[v].replace(", ", ",\n").replace(" + ", "\n+ ") for v in ORDER]
    ax.set_xticks(range(len(ORDER)), ticks, fontsize=8)
    ax.set_ylabel("microseconds per window (ONNX Runtime profile)")
    ax.set_title(f"full-22, Apple M4, ONNX Runtime {second['environment']['onnxruntime']}, 1 thread")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_kernel_layers.png", dpi=200)
    plt.close(fig)

    env = second["environment"]
    keep = ("onnxruntime", "machine", "power", "load_average_before", "load_average_after")
    summary = {
        "equivalence": eq,
        "environment": {k: env.get(k) for k in keep},
        "latency_us": {f"{r.condition}|{r.variant}": round(r.us, 1) for r in latency.itertuples()},
        "earlier_latency_us": {f"{r.condition}|{r.variant}": round(r.us, 1) for r in earlier.itertuples()},
        "layers_us": layers.round(2).to_dict("records"),
    }
    write_json(OUT / "analysis.json", summary)
    lines = ["# The missing INT8 kernels (ONNX Runtime custom operators)", ""]
    what = {
        "grouped": "grouped spatial convolution, original order (kernels/ort_grouped_conv.c)",
        "swapped": "spatial convolution after the swap (kernels/ort_spatial_first_conv.c)",
    }
    for kind, e in eq.items():
        lines.append(
            f"- {what[kind]}: {e['logits_identical']} of {e['models']} archived models give bit-identical logits on "
            f"all {e['windows']:,} session-2 windows (largest difference {e['max_abs_logit_difference']}); "
            f"{e['predictions_equal_archived']:,} predictions equal the archived ones."
        )
    lines += [
        "",
        f"Latency: ONNX Runtime {env['onnxruntime']}, {env['machine']}, one thread, batch 1, 15 interleaved rounds of "
        f"500 runs, every variant in one set; load average {env['load_average_before'][0]:.1f} before and "
        f"{env['load_average_after'][0]:.1f} after. Microseconds per 2-s window.",
        "",
        "| condition | variant | microseconds | relative to FP32 | parameter bytes |",
        "|---|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r.condition} | {NAMES[r.variant]} | {r.us:.0f} | {r.vs_fp32:.2f}x | {r.parameter_bytes:,} |"
        for r in latency.itertuples()
    ]
    lines += [
        "",
        "## Where the time goes (ONNX Runtime profiler, microseconds per window)",
        "",
        "| condition | variant | temporal | spatial | other | total |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r.condition} | {NAMES[r.variant]} | {r.temporal:.1f} | {r.spatial:.1f} | {r.rest:.1f} | "
        f"{r.temporal + r.spatial + r.rest:.1f} |"
        for r in layers.itertuples()
    ]
    lines += [
        "",
        "## Earlier measurement (scripts/ort_kernel_study.py, without the second kernel)",
        "",
        "| condition | variant | microseconds |",
        "|---|---|---:|",
    ]
    lines += [f"| {r.condition} | {NAMES[r.variant]} | {r.us:.0f} |" for r in earlier.itertuples()]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
