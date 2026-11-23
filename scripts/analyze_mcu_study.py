"""Tables and figures for the microcontroller study (scripts/mcu_study.py).

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only, never EEG:
  supplementary/mcu_study/protocol.json        the plan (cores, boards, tools, jobs)
  supplementary/mcu_study/emulate/<job>.json   one emulated build: instructions per layer and window,
                                               linker sections, scratch and stack, the logits
  supplementary/mcu_study/reference/<m>.json   session-2 accuracy of FP32 and of the CMSIS-NN integer
                                               arithmetic for every model and layer order

Writes to RESULTS_DIR/supplementary/mcu_study/: instructions.csv, layers.csv, memory.csv, accuracy.csv,
fig_mcu_cores.png, fig_mcu_layers.png, summary.md and analysis.json.

Instruction counts are QEMU's, per 2-s window: not cycles, time or energy. Accuracy averages seeds
within each participant, then the nine participants, with the frozen participant bootstrap interval.
Exploratory: the study was planned after the test results were known.
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

OUT = RESULTS_DIR / "supplementary" / "mcu_study"
LAYERS = ("l1", "l2", "elu1", "pool1", "conv_d", "conv_p", "elu2", "pool2", "fc")
CORES = ("m0", "m4", "m4-nodsp", "m55", "m55-nomve")
CORE_NAMES = {
    "m0": "Cortex-M0 (no FPU, no SIMD)",
    "m4": "Cortex-M4F (FPU, DSP)",
    "m4-nodsp": "M4 without DSP",
    "m55": "Cortex-M55 (FPU, Helium)",
    "m55-nomve": "M55 without Helium",
}
FROZEN = "relevance-16-prune50"
# multiply-accumulates per instruction of each core's widest MAC instruction, as instructions per MAC
BEST_CASE = {
    "M0 int8 (MULS + ADDS)": 2.0,
    "M4 int8 (SMLAD, 2 per instruction)": 0.5,
    "M4 float (VFMA.F32)": 1.0,
    "M55 int8 (VMLADAV.S8, 16 per instruction)": 1 / 16,
    "M55 float (VFMA.F32 on 4 lanes)": 0.25,
}
LAYER_NAMES = {
    "l1": "first layer",
    "l2": "second layer",
    "elu1": "ELU",
    "pool1": "pool",
    "conv_d": "depthwise 1x16",
    "conv_p": "pointwise 1x1",
    "elu2": "ELU",
    "pool2": "pool",
    "fc": "dense",
}


def load() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    plan = json.loads((OUT / "protocol.json").read_text())
    runs = [
        json.loads((OUT / "emulate" / f"{j['name']}.json").read_text())
        for j in plan["jobs"]
        if (OUT / "emulate" / f"{j['name']}.json").exists()
    ]
    missing = len(plan["jobs"]) - len(runs)
    if missing:
        raise SystemExit(f"{missing} emulation records missing: run scripts/mcu_study.py --stage emulate")
    rows = []
    for r in runs:
        row = {
            k: r[k]
            for k in (
                "subject",
                "condition",
                "order",
                "precision",
                "core",
                "flash_bytes",
                "ram_bytes",
                "scratch_bytes",
                "stack_used",
                "bit_exact",
                "max_abs_difference",
            )
        }
        row["instructions"] = float(np.mean(r["instructions_per_window"]))
        row["instructions_min"], row["instructions_max"] = (
            min(r["instructions_per_window"]),
            max(r["instructions_per_window"]),
        )
        row |= {f"i_{layer}": float(np.mean(r["instructions"][layer])) for layer in LAYERS}
        row["bss"] = r["sections"].get(".bss", 0)
        rows.append(row)
    refs = [json.loads(p.read_text()) for p in sorted((OUT / "reference").glob("*.json"))]
    acc = pd.DataFrame(
        [
            {
                **{k: r[k] for k in ("subject", "seed", "condition", "order")},
                **{f"acc_{k}": v for k, v in r["acc_clean"].items()},
                "macs": r["macs"],
                "int8_bytes": r["int8_bytes"],
                "f32_bytes": r["f32_bytes"],
            }
            for r in refs
        ]
    )
    return pd.DataFrame(rows), acc, refs


def accuracy_table(acc: pd.DataFrame) -> pd.DataFrame:
    pp = acc.groupby(["condition", "order", "subject"]).mean(numeric_only=True).reset_index()
    rows = []
    for (condition, order), g in pp.groupby(["condition", "order"]):
        d = (g["acc_mcu_int8"] - g["acc_torch_fp32"]).to_numpy() * 100
        lo, hi = bootstrap_ci(d)
        rows.append(
            {
                "condition": condition,
                "order": order,
                "fp32": 100 * g["acc_torch_fp32"].mean(),
                "int8_cmsis": 100 * g["acc_mcu_int8"].mean(),
                "int8_onnx": 100 * g["acc_onnx_int8"].mean(),
                "int8_minus_fp32": d.mean(),
                "ci_low": lo,
                "ci_high": hi,
                "worst_participant": d.min(),
                "losing_2pp": int((d < -2).sum()),
            }
        )
    return pd.DataFrame(rows)


def figure_cores(df: pd.DataFrame) -> None:
    p1 = df[df.subject == 1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cores = ["m0", "m4", "m55"]
    for ax, condition in zip(axes, ["full-22", FROZEN], strict=True):
        x = np.arange(len(cores))
        for i, (order, precision, color) in enumerate(
            [
                ("original", "f32", "#9ec5e8"),
                ("original", "int8", "#4B9CD3"),
                ("swapped", "f32", "#7f8fa6"),
                ("swapped", "int8", "#13294B"),
            ]
        ):
            sel = p1[(p1.condition == condition) & (p1.order == order) & (p1.precision == precision)].set_index("core")
            vals = [sel.loc[c, "instructions"] / 1e6 for c in cores]
            ax.bar(x + (i - 1.5) * 0.2, vals, 0.2, color=color, label=f"{precision.upper()}, {order} order")
        ax.set_yscale("log")
        ax.set_xticks(x, [CORE_NAMES[c].split(" (")[0] for c in cores])
        ax.set_title(condition)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("million instructions per 2-s window (QEMU)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mcu_cores.png", dpi=200)
    plt.close(fig)


def figure_layers(df: pd.DataFrame) -> None:
    p1 = df[(df.subject == 1) & (df.condition == FROZEN)]
    bars = [
        (core, order, prec) for core in ("m4", "m55") for order in ("original", "swapped") for prec in ("f32", "int8")
    ]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    bottom = np.zeros(len(bars))
    groups = {
        "first two layers": ("l1", "l2"),
        "ELU": ("elu1", "elu2"),
        "pooling": ("pool1", "pool2"),
        "depthwise + pointwise + dense": ("conv_d", "conv_p", "fc"),
    }
    colors = ["#13294B", "#4B9CD3", "#9ec5e8", "#c9d6e3"]
    for (label, layers), color in zip(groups.items(), colors, strict=True):
        vals = np.array(
            [
                p1[(p1.core == c) & (p1.order == o) & (p1.precision == p)][[f"i_{x}" for x in layers]].to_numpy().sum()
                / 1e6
                for c, o, p in bars
            ]
        )
        ax.bar(range(len(bars)), vals, 0.7, bottom=bottom, color=color, label=label)
        bottom += vals
    ax.set_xticks(range(len(bars)), [f"{c.upper()}\n{o}\n{p.upper()}" for c, o, p in bars], fontsize=8)
    ax.set_ylabel("million instructions per window")
    ax.set_title(f"{FROZEN} (the frozen model), participant 1")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mcu_layers.png", dpi=200)
    plt.close(fig)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    df, acc, refs = load()
    plan = json.loads((OUT / "protocol.json").read_text())
    p1 = df[df.subject == 1]
    pivot = p1.pivot_table(index=["condition", "order", "precision"], columns="core", values="instructions") / 1e6
    pivot.to_csv(OUT / "instructions.csv", float_format="%.4f")
    macs = {
        (r["condition"], r["order"]): {row["name"]: row["macs"] for row in r["layers"]}
        for r in refs
        if r["subject"] == 1
    }
    layer_rows = []
    for _, r in p1.iterrows():
        for layer in LAYERS:
            m = macs[(r.condition, r.order)][layer]
            layer_rows.append(
                {
                    "condition": r.condition,
                    "order": r.order,
                    "precision": r.precision,
                    "core": r.core,
                    "layer": layer,
                    "instructions": r[f"i_{layer}"],
                    "macs": m,
                    "instructions_per_mac": r[f"i_{layer}"] / m if m else None,
                }
            )
    layers = pd.DataFrame(layer_rows)
    layers.to_csv(OUT / "layers.csv", index=False, float_format="%.4f")
    memory = p1[
        ["condition", "order", "precision", "core", "flash_bytes", "ram_bytes", "bss", "scratch_bytes", "stack_used"]
    ].sort_values(["condition", "order", "precision", "core"])
    memory.to_csv(OUT / "memory.csv", index=False)
    accuracy = accuracy_table(acc)
    accuracy.to_csv(OUT / "accuracy.csv", index=False, float_format="%.3f")
    figure_cores(df)
    figure_layers(df)

    int8 = df[df.precision == "int8"]
    ratio = {
        core: {
            cond: {
                o: float(pivot.loc[(cond, o, "f32"), core] / pivot.loc[(cond, o, "int8"), core])
                for o in ("original", "swapped")
            }
            for cond in ("full-22", FROZEN)
        }
        for core in ("m0", "m4", "m55", "m55-nomve")
    }
    swap = {
        core: {
            p: float(pivot.loc[("full-22", "original", p), core] / pivot.loc[("full-22", "swapped", p), core])
            for p in ("f32", "int8")
        }
        for core in ("m0", "m4", "m55")
    }
    total_macs = {(r["condition"], r["order"]): r["macs"] for r in refs if r["subject"] == 1 and r["seed"] == 0}
    per_mac = {
        f"{c}|{o}|{p}": {core: float(v * 1e6 / total_macs[(c, o)]) for core, v in row.items() if pd.notna(v)}
        for (c, o, p), row in pivot.iterrows()
    }
    summary = {
        "tools": plan["tools"],
        "int8_builds": len(int8),
        "int8_bit_exact_builds": int(int8["bit_exact"].sum()),
        "int8_windows_compared": int(len(int8) * 8),
        "f32_max_abs_difference": float(df[df.precision == "f32"]["max_abs_difference"].max()),
        "instructions_million_p1": {
            f"{c}|{o}|{p}": {k: v for k, v in row.items() if pd.notna(v)} for (c, o, p), row in pivot.iterrows()
        },
        "instructions_per_mac_p1": per_mac,
        "best_case_instructions_per_mac": BEST_CASE,
        "fp32_over_int8": ratio,
        "original_over_swapped_full22": swap,
        "accuracy": accuracy.to_dict("records"),
    }
    write_json(OUT / "analysis.json", summary)

    lines = [
        "# Microcontroller study (CMSIS-NN on emulated Cortex-M cores)",
        "",
        f"Tools: {plan['tools']['gcc']}; {plan['tools']['qemu']}; CMSIS-NN {plan['tools']['cmsis_nn_commit'][:12]}.",
        "Counts are instructions executed per 2-s window in QEMU (participant 1, seed 0; mean of 8 windows). "
        "QEMU models no timing: these are not cycles, time or energy.",
        "",
        f"Bit-exactness: {summary['int8_bit_exact_builds']} of {summary['int8_builds']} INT8 builds gave logits "
        f"identical to the NumPy integer reference on all {summary['int8_windows_compared']} windows; FP32 builds "
        f"differ from NumPy float by at most {summary['f32_max_abs_difference']:.1e}.",
        "",
        "## Million instructions per window",
        "",
        "| condition | order | precision | " + " | ".join(CORES) + " |",
        "|---|---|---|" + "---:|" * len(CORES),
    ]
    for (c, o, p), row in pivot.iterrows():
        lines.append(
            f"| {c} | {o} | {p} | "
            + " | ".join(f"{row[k]:.2f}" if k in row and pd.notna(row[k]) else "-" for k in CORES)
            + " |"
        )
    lines += [
        "",
        "## Instructions per multiply-accumulate, whole network (participant 1)",
        "",
        "Best case if every multiply-accumulate went through the core's widest MAC instruction and nothing else ran: "
        + "; ".join(f"{k} {v}" for k, v in BEST_CASE.items())
        + ".",
        "",
        "| condition | order | precision | " + " | ".join(CORES) + " |",
        "|---|---|---|" + "---:|" * len(CORES),
    ]
    for key, cores in per_mac.items():
        c, o, prec = key.split("|")
        lines.append(
            f"| {c} | {o} | {prec} | " + " | ".join(f"{cores[k]:.2f}" if k in cores else "-" for k in CORES) + " |"
        )
    lines += [
        "",
        "## FP32 / INT8 instruction ratio",
        "",
        "| core | full-22 original | full-22 swapped | frozen original | frozen swapped |",
        "|---|---:|---:|---:|---:|",
    ]
    for core, v in ratio.items():
        lines.append(
            f"| {core} | {v['full-22']['original']:.2f} | {v['full-22']['swapped']:.2f} | "
            f"{v[FROZEN]['original']:.2f} | {v[FROZEN]['swapped']:.2f} |"
        )
    lines += [
        "",
        "## Per layer (frozen model, instructions per multiply-accumulate)",
        "",
        "| core | order | precision | " + " | ".join(LAYERS) + " |",
        "|---|---|---|" + "---:|" * len(LAYERS),
    ]
    fz = layers[layers.condition == FROZEN]
    for (core, order, prec), g in fz.groupby(["core", "order", "precision"]):
        g = g.set_index("layer")
        cells = [
            f"{g.loc[x, 'instructions'] / 1e3:.0f}k"
            + (f" ({g.loc[x, 'instructions_per_mac']:.1f})" if pd.notna(g.loc[x, "instructions_per_mac"]) else "")
            for x in LAYERS
        ]
        lines.append(f"| {core} | {order} | {prec} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Memory (participant 1): flash = code + constants; RAM = data + bss + measured stack",
        "",
        "| condition | order | precision | core | flash bytes | RAM bytes | scratch | stack |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, r in memory.iterrows():
        lines.append(
            f"| {r.condition} | {r.order} | {r.precision} | {r.core} | {r.flash_bytes:,} | {r.ram_bytes:,} | "
            f"{r.scratch_bytes:,} | {r.stack_used:,} |"
        )
    lines += [
        "",
        "## Session-2 accuracy (artifact-free trials; 9 participants x 3 seeds)",
        "",
        "| condition | order | FP32 % | INT8 CMSIS-NN % | INT8 ONNX % | INT8 minus FP32 (pp) | 95% interval | "
        "worst participant | losing > 2 pp |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for _, r in accuracy.iterrows():
        lines.append(
            f"| {r.condition} | {r.order} | {r.fp32:.2f} | {r.int8_cmsis:.2f} | {r.int8_onnx:.2f} | "
            f"{r.int8_minus_fp32:+.2f} | [{r.ci_low:+.2f}, {r.ci_high:+.2f}] | {r.worst_participant:+.2f} | "
            f"{r.losing_2pp} |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
