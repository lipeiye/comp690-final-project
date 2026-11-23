"""Tables and a figure for the microcontroller kernel study (scripts/mcu_kernel_study.py).

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only:
  supplementary/mcu_kernel_study/{protocol.json,emulate}   the network with the FIR kernel
  supplementary/mcu_study/emulate                           the same builds with CMSIS-NN's kernels

Writes to RESULTS_DIR/supplementary/mcu_kernel_study/: builds.csv, fig_mcu_kernel.png, summary.md and
analysis.json.

Instructions are the mean per 2-s window over the 8 emulated windows. The temporal layer is the first
layer in the original order and the second in the swapped one; instructions per multiply-accumulate set
it beside each core's best case (one widest MAC instruction per MAC: M0 MULS + ADDS 2, M4 SMLAD 0.5, M55
VMLADAV.S8 1/16). The other layers run the same CMSIS-NN calls in both builds; their counts are compared
to check that. QEMU counts instructions; these are not cycles, time or energy.
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

SUPP = RESULTS_DIR / "supplementary"
OUT = SUPP / "mcu_kernel_study"
LIBRARY = SUPP / "mcu_study" / "emulate"
LAYERS = ("l1", "l2", "elu1", "pool1", "conv_d", "conv_p", "elu2", "pool2", "fc")
BEST_CASE = {"m0": 2.0, "m4": 0.5, "m4-nodsp": 2.0, "m55": 1 / 16, "m55-nomve": 0.5}
CORE_NAMES = {
    "m0": "Cortex-M0",
    "m4": "Cortex-M4 (DSP)",
    "m4-nodsp": "M4 without DSP",
    "m55": "Cortex-M55 (Helium)",
    "m55-nomve": "M55 without Helium",
}


def records_json(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def load(plan: dict) -> pd.DataFrame:
    rows, missing = [], 0
    for job in plan["jobs"]:
        mine, lib = OUT / "emulate" / f"{job['name']}.json", LIBRARY / f"{job['library_job']}.json"
        if not (mine.exists() and lib.exists()):
            missing += 1
            continue
        a, b = json.loads(mine.read_text()), json.loads(lib.read_text())
        temporal = "l1" if job["order"] == "original" else "l2"
        other = [layer for layer in LAYERS if layer != temporal]
        row = {k: job[k] for k in ("subject", "condition", "order", "core")}
        row |= {
            "bit_exact": a["bit_exact"],
            "library_bit_exact": b["bit_exact"],
            "same_logits_as_library": a["emulated"] == b["emulated"],
            "macs_temporal": a["macs"][temporal],
            "macs_total": sum(a["macs"].values()),
            "fir_temporal": float(np.mean(a["instructions"][temporal])),
            "library_temporal": float(np.mean(b["instructions"][temporal])),
            "fir_total": float(np.mean(a["instructions_per_window"])),
            "library_total": float(np.mean(b["instructions_per_window"])),
            "other_layers_max_difference": max(
                abs(float(np.mean(a["instructions"][x])) - float(np.mean(b["instructions"][x]))) for x in other
            ),
            "fir_flash": a["flash_bytes"],
            "library_flash": b["flash_bytes"],
            "fir_ram": a["ram_bytes"],
            "library_ram": b["ram_bytes"],
        }
        rows.append(row)
    if missing:
        raise SystemExit(f"{missing} builds lack a record here or in mcu_study")
    df = pd.DataFrame(rows)
    df["fir_per_mac"] = df.fir_temporal / df.macs_temporal
    df["library_per_mac"] = df.library_temporal / df.macs_temporal
    df["best_case_per_mac"] = df.core.map(BEST_CASE)
    df["temporal_ratio"] = df.library_temporal / df.fir_temporal
    df["total_ratio"] = df.library_total / df.fir_total
    return df


def figure(p1: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    cores = ["m0", "m4", "m55"]
    for ax, order in zip(axes, ("original", "swapped"), strict=True):
        t = p1[(p1.order == order) & (p1.condition == "full-22")].set_index("core").loc[cores]
        x = np.arange(len(cores))
        ax.bar(x - 0.2, t.library_total / 1e6, 0.4, color="#9ec5e8", label="CMSIS-NN kernels")
        ax.bar(x + 0.2, t.fir_total / 1e6, 0.4, color="#13294B", label="with the FIR kernel")
        for i, (lib, fir) in enumerate(zip(t.library_total, t.fir_total, strict=True)):
            ax.text(i + 0.2, fir / 1e6, f"{lib / fir:.1f}x", ha="center", va="bottom", fontsize=8)
        ax.set_yscale("log")
        ax.set_xticks(x, [CORE_NAMES[c] for c in cores])
        ax.set(ylabel="million instructions per 2-s window", title=f"22 electrodes, {order} layer order")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mcu_kernel.png", dpi=200)
    plt.close(fig)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    plan = json.loads((OUT / "protocol.json").read_text())
    df = load(plan)
    df.to_csv(OUT / "builds.csv", index=False, float_format="%.4f")
    p1 = df[df.subject == 1]
    checks = {
        "builds": len(df),
        "bit_exact": int(df.bit_exact.sum()),
        "same_logits_as_library": int(df.same_logits_as_library.sum()),
        "other_layers_max_difference": float(df.other_layers_max_difference.max()),
    }
    figure(p1)
    write_json(OUT / "analysis.json", {"checks": checks, "builds": records_json(df)})
    lines = [
        "# The int8 FIR kernel for the temporal layer on Cortex-M",
        "",
        f"{checks['bit_exact']} of {checks['builds']} builds give logits bit-identical to the NumPy integer reference "
        f"({checks['same_logits_as_library']} identical to the CMSIS-NN-only build). The layers both builds share "
        f"differ by at most {checks['other_layers_max_difference']:.0f} instructions per window. Instruction counts "
        "from QEMU, mean per 2-s window; not cycles, time or energy.",
        "",
        "## Participant 1, seed 0",
        "",
        "| condition | order | core | temporal layer, CMSIS-NN | with the FIR kernel | per MAC: CMSIS-NN / FIR / "
        "best case | whole network, CMSIS-NN | with the FIR kernel | fewer | flash bytes (CMSIS-NN / FIR) | "
        "RAM bytes (CMSIS-NN / FIR) |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---|---|",
    ]
    lines.extend(
        f"| {r.condition} | {r.order} | {r.core} | {r.library_temporal / 1e6:.2f} M | "
        f"{r.fir_temporal / 1e6:.2f} M | {r.library_per_mac:.2f} / {r.fir_per_mac:.2f} / {r.best_case_per_mac:.3g} | "
        f"{r.library_total / 1e6:.2f} M | {r.fir_total / 1e6:.2f} M | {r.total_ratio:.2f}x | "
        f"{r.library_flash:,} / {r.fir_flash:,} | {r.library_ram:,} / {r.fir_ram:,} |"
        for r in p1.sort_values(["condition", "order", "core"]).itertuples()
    )
    others = df[df.subject > 1].groupby(["condition", "order", "core"])[["library_total", "fir_total"]].mean()
    lines += [
        "",
        "## Participants 2-9 (mean per window)",
        "",
        "| condition | order | core | CMSIS-NN | with the FIR kernel | fewer |",
        "|---|---|---|---:|---:|---:|",
    ]
    for (condition, order, core), r in others.iterrows():
        lines.append(
            f"| {condition} | {order} | {core} | {r.library_total / 1e6:.2f} M | {r.fir_total / 1e6:.2f} M | "
            f"{r.library_total / r.fir_total:.2f}x |"
        )
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
