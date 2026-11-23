"""Cross-machine table and figure for the INT8 latency runs of scripts/hw_latency.py.

Question: is the slow original INT8 graph on the Apple M4 a property of that chip, or of the path
ONNX Runtime takes for EEGNet's grouped spatial convolution? Every machine timed the same variants
of four architectures (participant 1, seed 0) one prediction at a time (batch 1) on one thread. The
M4 reference is the measurement of scripts/int8_study.py, made with the same timing code
(mibci.timing); its relevance-8 row is the int8 study's proposed-8 graph, which for participant 1,
seed 0 holds the same eight electrodes (the archived models give identical predictions).

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only:
  supplementary/int8_study/benchmark.json   the M4 reference (required)
  supplementary/hw_latency/<machine>.json   one file per machine, if any (names starting with "smoke" are ignored)

Writes to RESULTS_DIR/supplementary/hw_latency/: latency.csv (machine x architecture x variant: median
microseconds and the ratio to that machine's FP32 graph), fig_latency_by_machine.png and summary.md,
which says on which machines the original INT8 graph is slower than FP32, whether the layer-swapped
INT8 graph is faster than FP32 everywhere, and which integer extensions each CPU reports. Post-test
and exploratory. When the M4 reference is missing, the script says so and exits with status 1
before writing anything.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter

from mibci.paths import RESULTS_DIR, display

OUT = RESULTS_DIR / "supplementary/hw_latency"
REFERENCE = RESULTS_DIR / "supplementary/int8_study/benchmark.json"
REFERENCE_NAME = "m4-int8-study"
ARCHITECTURES = ("full-22", "relevance-16", "relevance-16-prune50", "relevance-8")
# The int8 study timed proposed-8, not relevance-8: same electrodes and graph for participant 1, seed 0.
REFERENCE_ALIASES = {"relevance-8": "proposed-8"}
# variant -> label and colour, in the order of the bars
VARIANTS = {
    "fp32": ("FP32, original graph", "#2a78d6"),
    "int8": ("INT8, original graph", "#eb6834"),
    "int8-spatial-first": ("INT8, first two layers swapped", "#1baf7a"),
    "fp32-spatial-first": ("FP32, first two layers swapped", "#eda100"),
    "int8-dense": ("INT8, grouped convolution as dense", "#e87ba4"),
    "int8-mixed": ("INT8, spatial convolution kept FP32", "#008300"),
    "int16": ("INT16 activations", "#6250d6"),
}
# feature key in a record (Arm keys without their "hw.optional.arm." prefix) -> name in the tables
EXTENSIONS = {
    "avx2": "AVX2",
    "avx512f": "AVX-512F",
    "avx512_vnni": "AVX-512 VNNI",
    "avx_vnni": "AVX-VNNI",
    "amx_int8": "AMX-INT8",
    "amx_tile": "AMX-TILE",
    "FEAT_DotProd": "DotProd",
    "FEAT_I8MM": "I8MM",
}
DOT_PRODUCT = ("AVX-512 VNNI", "AVX-VNNI", "AMX-INT8", "DotProd", "I8MM")  # 8-bit dot products in hardware
DISTINCT_INPUTS = {REFERENCE_NAME: 100}  # session-1 trials cycled through; scripts/hw_latency.py uses 64
MUTED, GRID = "#52514e", "#e6e6e6"


# ---------------------------------------------------------------- inputs


def isa(os_name: str, features: dict) -> str:
    text = os_name.lower()
    if "arm64" in text or "aarch64" in text:
        return "Arm"
    if "x86_64" in text or "amd64" in text:
        return "x86-64"
    return "Arm" if any(k.startswith(("hw.optional.arm", "FEAT_")) for k in features) else "x86-64"


def extensions(features: dict) -> dict[str, list[str]]:
    """Extension names the record reports as present, absent, or not readable."""
    found = {"present": [], "absent": [], "unknown": []}
    for key, value in features.items():
        name = EXTENSIONS.get(key.removeprefix("hw.optional.arm."), key)
        state = str(value).strip().lower()
        found["present" if state in ("true", "1") else "absent" if state in ("false", "0") else "unknown"].append(name)
    return found


def timed_rows(machine: str, rows: list[dict], timed: dict[str, str]) -> list[dict]:
    """The architectures and variants that every machine times, in microseconds."""
    kept = []
    for condition in ARCHITECTURES:
        source = timed.get(condition, condition)
        kept += [
            {
                "machine": machine,
                "condition": condition,
                "timed_condition": source,
                "variant": r["variant"],
                "median_us": 1000 * r["median_ms"],
                "best_round_us": 1000 * r["best_round_ms"],
                "worst_round_us": 1000 * r["worst_round_ms"],
                "rounds": r["rounds"],
                "runs_per_round": r["runs_per_round"],
            }
            for r in rows
            if r["condition"] == source and r["variant"] in VARIANTS
        ]
    return kept


def load_reference() -> tuple[dict, list[dict]]:
    bench = json.loads(REFERENCE.read_text())
    env = bench["environment"]
    info = {
        "machine": REFERENCE_NAME,
        "cpu": env["cpu"],
        "isa": isa(env.get("os", ""), env["cpu_features"]),
        "onnxruntime": env["onnxruntime"],
        "load_average": float(env["load_average_before"][0]),
        "features": env["cpu_features"],
        "source": display(REFERENCE),
    }
    return info, timed_rows(REFERENCE_NAME, bench["rows"], REFERENCE_ALIASES)


def load_machine(path: Path) -> tuple[dict, list[dict]]:
    record = json.loads(path.read_text())
    lacking = [key for key in ("cpu", "rows", "onnxruntime", "load_average", "environment") if key not in record]
    if lacking:
        raise SystemExit(f"{display(path)} lacks {', '.join(lacking)}; is it a scripts/hw_latency.py record?")
    info = {
        "machine": path.stem,
        "cpu": record["cpu"]["model"] or record["environment"].get("cpu") or "unknown CPU",
        "isa": isa(record["environment"].get("os", ""), record["cpu"]["features"]),
        "onnxruntime": record["onnxruntime"],
        "load_average": float(record["load_average"][0]),
        "features": record["cpu"]["features"],
        "source": display(path),
    }
    return info, timed_rows(path.stem, record["rows"], {})


def ordered(machines: pd.DataFrame) -> pd.DataFrame:
    """Arm machines first, the M4 reference first among them, then by name."""
    keys = [(m["isa"] != "Arm", name != REFERENCE_NAME, name) for name, m in machines.iterrows()]
    return machines.iloc[sorted(range(len(keys)), key=keys.__getitem__)]


def latency_table(rows: list[dict], machines: pd.DataFrame) -> pd.DataFrame:
    """latency.csv: one row per machine, architecture and variant, with the ratio to that machine's FP32."""
    table = pd.DataFrame(rows)
    if table.duplicated(["machine", "condition", "variant"]).any():
        raise SystemExit("A machine reports the same architecture and variant twice")
    fp32 = table[table.variant == "fp32"].set_index(["machine", "condition"])["median_us"]
    keys = pd.MultiIndex.from_frame(table[["machine", "condition"]])
    if not keys.isin(fp32.index).all():
        raise SystemExit("Every timed architecture needs its FP32 graph on the same machine")
    table["ratio_to_fp32"] = table["median_us"].to_numpy() / fp32.reindex(keys).to_numpy()
    table = table.merge(machines[["cpu", "isa", "onnxruntime"]], left_on="machine", right_index=True)
    table["machine"] = pd.Categorical(table["machine"], categories=list(machines.index), ordered=True)
    table["condition"] = pd.Categorical(table["condition"], categories=ARCHITECTURES, ordered=True)
    table["variant"] = pd.Categorical(table["variant"], categories=list(VARIANTS), ordered=True)
    columns = ["machine", "cpu", "isa", "onnxruntime", "condition", "timed_condition", "variant", "median_us"]
    columns += ["ratio_to_fp32", "best_round_us", "worst_round_us", "rounds", "runs_per_round"]
    return table.sort_values(["machine", "condition", "variant"])[columns].reset_index(drop=True)


# ---------------------------------------------------------------- figure


def short_cpu(model: str) -> str:
    text = re.sub(r"\((R|TM)\)|\bCPU\b|\bProcessor\b|\d+-Core|@.*$", "", model)
    return " ".join(text.split())[:28]


def figure(table: pd.DataFrame, machines: pd.DataFrame) -> None:
    names = list(machines.index)
    x = np.arange(len(names))
    width = 0.84 / len(VARIANTS)
    low, high = table["median_us"].min() / 1.6, table["median_us"].max() * 2.2
    fig, axes = plt.subplots(2, 2, figsize=(max(9.0, 3.4 + 1.9 * len(names)), 7.8), sharey=True)
    for ax, condition in zip(axes.flat, ARCHITECTURES, strict=True):
        part = table[table.condition == condition].set_index(["machine", "variant"])
        for i, (variant, (label, colour)) in enumerate(VARIANTS.items()):
            keys = pd.MultiIndex.from_tuples([(n, variant) for n in names])
            heights = part["median_us"].reindex(keys).to_numpy()  # NaN where a machine did not time it
            xs = x - 0.42 + width * (i + 0.5)
            ax.bar(xs, heights, width * 0.88, color=colour, label=label)
            if variant in ("int8", "int8-spatial-first"):  # the two answers, as ratios to FP32
                ratios = part["ratio_to_fp32"].reindex(keys).to_numpy()
                for xi, h, r in zip(xs, heights, ratios, strict=True):
                    if np.isfinite(h):
                        ax.text(xi, h * 1.08, f"{r:.2f}x", ha="center", va="bottom", rotation=90, fontsize=6.5)
        ax.set_yscale("log")
        ax.set_ylim(low, high)
        ticks = [t for t in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000) if low <= t <= high]
        ax.set_yticks(ticks, labels=[str(t) for t in ticks])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(x, labels=[f"{n}\n{short_cpu(machines.loc[n, 'cpu'])}" for n in names], fontsize=7)
        ax.set_title(condition, fontsize=10)
        ax.grid(axis="y", which="major", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("median time per prediction (us, log scale)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8)
    fig.text(
        0.995,
        0.005,
        "numbers above the INT8 bars: time relative to the same machine's FP32 graph; batch 1, one thread",
        ha="right",
        fontsize=7,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(OUT / "fig_latency_by_machine.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- summary.md


def spread(ratios: pd.Series) -> str:
    low, high = ratios.min(), ratios.max()
    return f"{low:.2f}x" if f"{low:.2f}" == f"{high:.2f}" else f"{low:.2f}x to {high:.2f}x"


def machine_lines(machines: pd.DataFrame, table: pd.DataFrame, ignored: list[str]) -> list[str]:
    lines = [
        "",
        "## Machines and the integer instructions their CPUs report",
        "",
        "| machine | CPU | instruction set | integer extensions present | checked, absent | ONNX Runtime "
        "| load average at start (1 min) | rounds x calls |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for name, m in machines.iterrows():
        found = extensions(m["features"])
        absent = ", ".join(found["absent"]) or "none"
        if found["unknown"]:
            absent += f"; not readable: {', '.join(found['unknown'])}"
        timing = table[table.machine == name][["rounds", "runs_per_round"]].drop_duplicates()
        plan = ", ".join(f"{r} x {c}" for r, c in zip(timing["rounds"], timing["runs_per_round"], strict=True))
        lines.append(
            f"| {name} | {m['cpu']} | {m['isa']} | {', '.join(found['present']) or 'none reported'} | {absent} "
            f"| {m['onnxruntime']} | {m['load_average']:.1f} | {plan} |"
        )
    lines += [
        "",
        "AVX-512 VNNI and AVX-VNNI (x86), AMX (Intel's matrix tiles), and DotProd and I8MM (Arm) are instructions "
        "that multiply and add 8-bit integers in hardware; AVX2 and AVX-512F are general vector instructions. "
        "A record lists only the extensions its operating system reports, so a missing entry is not proof of "
        "absence.",
    ]
    if ignored:
        lines.append(f"Ignored as smoke tests: {', '.join(ignored)}.")
    return lines


def int8_lines(table: pd.DataFrame, machines: pd.DataFrame) -> list[str]:
    lines = ["", "## Is the original INT8 graph slower than FP32?", ""]
    verdict = {"slower": [], "faster": [], "mixed": []}
    for name in machines.index:
        r = table[(table.machine == name) & (table.variant == "int8")]
        if r.empty:
            lines.append(f"- {name}: the original INT8 graph was not timed.")
            continue
        slower = int((r["ratio_to_fp32"] > 1).sum())
        kind = "slower" if slower == len(r) else "faster" if slower == 0 else "mixed"
        verdict[kind].append(name)
        lines.append(
            f"- {name} ({machines.loc[name, 'cpu']}): slower than FP32 for {slower} of {len(r)} architectures; "
            f"INT8 takes {spread(r['ratio_to_fp32'])} the FP32 time."
        )
    lines += [
        "",
        f"Slower for every architecture on: {', '.join(verdict['slower']) or 'no machine'}. "
        f"Faster for every architecture on: {', '.join(verdict['faster']) or 'no machine'}. "
        f"Mixed: {', '.join(verdict['mixed']) or 'no machine'}.",
    ]
    return lines


def swapped_lines(table: pd.DataFrame) -> list[str]:
    indexed = table.set_index(["machine", "condition", "variant"])["median_us"]
    swapped = table[table.variant == "int8-spatial-first"]
    lines = ["", "## Is the layer-swapped INT8 graph faster than FP32 everywhere?", ""]
    if swapped.empty:
        return [*lines, "The layer-swapped INT8 graph was not timed."]
    slow = swapped[swapped["ratio_to_fp32"] >= 1]
    pairs = len(swapped)
    if slow.empty:
        lines.append(
            f"Yes, on every machine and architecture measured ({pairs} pairs): it takes "
            f"{spread(swapped['ratio_to_fp32'])} the time of the original FP32 graph on the same machine."
        )
    else:
        listed = "; ".join(f"{r.machine} {r.condition} ({r.ratio_to_fp32:.2f}x)" for r in slow.itertuples())
        lines.append(f"No. It is not faster than the original FP32 graph in {len(slow)} of {pairs} pairs: {listed}.")
    keys = [(r.machine, r.condition, "fp32-spatial-first") for r in swapped.itertuples()]
    same_swap = swapped["median_us"].to_numpy() / indexed.reindex(keys).to_numpy()
    measured = np.isfinite(same_swap)
    if measured.any():
        lines.append(
            "Against the FP32 graph with the same swap (the fastest FP32 version measured), the swapped INT8 graph "
            f"is faster in {int((same_swap[measured] < 1).sum())} of {int(measured.sum())} pairs, taking "
            f"{spread(pd.Series(same_swap[measured]))} its time."
        )
    return lines


def table_lines(table: pd.DataFrame) -> list[str]:
    shown = [v for v in VARIANTS if v != "fp32"]
    lines = [
        "",
        "## Time per prediction (microseconds; in brackets, relative to the same machine's FP32 graph)",
        "",
        "| machine | architecture | FP32 | " + " | ".join(VARIANTS[v][0] for v in shown) + " |",
        "| --- | --- | ---: |" + " ---: |" * len(shown),
    ]
    for (machine, condition), group in table.groupby(["machine", "condition"], observed=True, sort=True):
        g = group.set_index("variant")
        cells = [
            f"{g.loc[v, 'median_us']:.0f} ({g.loc[v, 'ratio_to_fp32']:.2f}x)" if v in g.index else "not timed"
            for v in shown
        ]
        lines.append(f"| {machine} | {condition} | {g.loc['fp32', 'median_us']:.0f} | " + " | ".join(cells) + " |")
    return lines


def reading_lines(table: pd.DataFrame, machines: pd.DataFrame) -> list[str]:
    int8 = table[table.variant == "int8"]
    slower_everywhere = bool(len(int8)) and bool((int8["ratio_to_fp32"] > 1).all())
    with_dot = [n for n, m in machines.iterrows() if set(extensions(m["features"])["present"]) & set(DOT_PRODUCT)]
    lines = ["", "## Reading the result", ""]
    if len(machines) == 1:
        lines.append(
            "Only the M4 reference is available so far, so this cannot yet separate the chip from the software "
            "path. Run scripts/hw_latency.py on other CPUs (Longleaf nodes: python3 tools/longleaf.py) and run "
            "this script again."
        )
    elif slower_everywhere:
        lines.append(
            "The original INT8 graph is slower than FP32 on every machine, including those whose CPUs multiply "
            f"8-bit integers in hardware ({', '.join(with_dot) or 'none of them reports it'}). That points at the "
            "path ONNX Runtime takes for this graph (its grouped spatial convolution), not at a missing 8-bit unit "
            "in one chip: quantizing to 8 bits did not by itself make this model faster on any machine measured."
        )
    else:
        lines.append(
            "The original INT8 graph is not slower everywhere, so the slowdown depends on the machine as well as on "
            "the graph; compare the extensions above with the machines where INT8 is slower."
        )
    lines += [
        "",
        "Notes:",
        "",
        "- The M4 reference comes from scripts/int8_study.py (cycling through 100 session-1 trials); the other "
        "machines ran scripts/hw_latency.py (64 trials). Both time with mibci.timing: interleaved rounds, the median "
        "of the per-round medians.",
        "- The M4 relevance-8 row is the int8 study's proposed-8 graph: for participant 1, seed 0, both rules chose "
        "the same eight electrodes and the archived models give identical predictions.",
        "- These are compute times for one prediction. They do not include the 2-second EEG window a prediction "
        "needs, and machines shared with other jobs (see the load average) add noise; ratios within one machine "
        "are more reliable than times across machines.",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    if not (REFERENCE.is_file() and REFERENCE.stat().st_size > 0):
        raise SystemExit(
            "analyze_hw_latency.py: inputs are missing; nothing was written.\n"
            f"  M4 reference (scripts/int8_study.py --stage benchmark): {display(REFERENCE)}"
        )
    files = sorted(OUT.glob("*.json")) if OUT.is_dir() else []
    ignored = [p.name for p in files if p.name.startswith("smoke")]
    loaded = [load_reference(), *(load_machine(p) for p in files if not p.name.startswith("smoke"))]
    machines = pd.DataFrame([info for info, _ in loaded]).set_index("machine")
    if machines.index.duplicated().any():
        raise SystemExit(f"Two inputs share a machine name: {', '.join(machines.index[machines.index.duplicated()])}")
    machines = ordered(machines)
    table = latency_table([row for _, rows in loaded for row in rows], machines)
    lines = [
        "# INT8 latency across machines (supplementary, post-test; generated by scripts/analyze_hw_latency.py)",
        "",
        f"{len(machines)} machine(s). Each timed the same deployment variants of four architectures (participant 1, "
        "seed 0): one prediction at a time (batch 1) on one CPU thread. INT8 means 8-bit integer weights and "
        'activations; "first two layers swapped" runs EEGNet\'s spatial convolution before its temporal one, '
        "which gives the same predictions (int8 study) with far fewer multiply-adds. FP32 is the original 32-bit "
        "floating-point graph. Post-test and exploratory: timing does not change any accuracy.",
        *machine_lines(machines, table, ignored),
        *int8_lines(table, machines),
        *swapped_lines(table),
        *table_lines(table),
        *reading_lines(table, machines),
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    figure(table, machines)
    table.to_csv(OUT / "latency.csv", index=False)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
