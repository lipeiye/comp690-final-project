"""Tables and a figure for the INT8 path map (scripts/int8_paths_study.py).

Reads RESULTS_DIR/supplementary/int8_paths_study/{protocol.json,map/*.json}; writes shapes.csv,
fig_int8_paths.png, summary.md and analysis.json there.

For each machine and convolution shape: FP32 and INT8 time per window (median of interleaved rounds),
time per multiply-accumulate, and INT8 over FP32 (below 1: INT8 is faster). EEGNet's own layers are
marked: the spatial layer (spatial, G = 8, M = 2), the swapped network's first layer (spatial, G = 1,
M = 16), the temporal layer (temporal, G = 1, M = 8) and the swapped temporal layer (temporal, G = 16,
M = 1). Synthetic shapes on shared machines; not the decoders' end-to-end latency.
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

OUT = RESULTS_DIR / "supplementary" / "int8_paths_study"
EEGNET = {"spatial_g8_m2": "EEGNet spatial", "spatial_g1_m16": "swapped first layer",
          "temporal_g1_m8": "EEGNet temporal", "temporal_g16_m1": "swapped temporal"}  # fmt: skip


def load() -> pd.DataFrame:
    rows = []
    for path in sorted((OUT / "map").glob("*.json")):
        r = json.loads(path.read_text())
        keys = ("name", "kernel", "groups", "per_group", "macs", "fp32_ms", "int8_ms", "fp32_ns_per_mac",
                "int8_ns_per_mac", "int8_over_fp32")  # fmt: skip
        rows.extend(
            {
                "machine": r["label"],
                "cpu": r["cpu"]["model"],
                "load_before": r["load_before"][0],
                **{k: s[k] for k in keys},
                "qlinearconv": s["ops_int8"].get("QLinearConv", 0),
                "eegnet_layer": EEGNET.get(s["name"], ""),
            }
            for s in r["shapes"]
        )
    if not rows:
        raise SystemExit("no map/*.json yet")
    return pd.DataFrame(rows)


def figure(df: pd.DataFrame) -> None:
    machines = sorted(df.machine.unique())
    fig, axes = plt.subplots(2, len(machines), figsize=(3.2 * len(machines), 6), squeeze=False)
    sizes = (1, 2, 4, 8, 16)
    for col, machine in enumerate(machines):
        for row, kernel in enumerate(("spatial", "temporal")):
            ax = axes[row, col]
            grid = np.full((len(sizes), len(sizes)), np.nan)
            t = df[(df.machine == machine) & (df.kernel == kernel)]
            for r in t.itertuples():
                grid[sizes.index(r.groups), sizes.index(r.per_group)] = r.int8_over_fp32
            im = ax.imshow(np.log2(grid), cmap="RdBu_r", vmin=-2, vmax=2)
            for i in range(len(sizes)):
                for j in range(len(sizes)):
                    if not np.isnan(grid[i, j]):
                        ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7)
            ax.set_xticks(range(len(sizes)), sizes)
            ax.set_yticks(range(len(sizes)), sizes)
            ax.set(xlabel="filters per group M", ylabel="groups G", title=f"{machine}\n{kernel} kernel")
    fig.colorbar(im, ax=axes, shrink=0.6, label="log2(INT8 time / FP32 time)")
    fig.savefig(OUT / "fig_int8_paths.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    df = load()
    df.to_csv(OUT / "shapes.csv", index=False, float_format="%.5f")
    figure(df)
    summary = []
    for machine, t in df.groupby("machine"):
        slow = t[t.int8_over_fp32 > 1]
        summary.append({"machine": machine, "cpu": t.cpu.iloc[0], "shapes": len(t), "int8_slower": len(slow),
                        "int8_slower_shapes": slow.name.tolist(),
                        "eegnet_layers": {r.name: r.int8_over_fp32 for r in t[t.eegnet_layer != ""].itertuples()},
                        "every_int8_graph_runs_qlinearconv": bool((t.qlinearconv == 1).all())})  # fmt: skip
    write_json(OUT / "analysis.json", {"machines": summary, "shapes": df.to_dict("records")})
    lines = ["# Where ONNX Runtime's INT8 convolution is fast", "",
             "INT8 time over FP32 time per window (below 1: INT8 faster), batch 1, one thread. Single synthetic "
             "convolutions with G groups of M filters on a G x 22 x 500 input; spatial kernel 22 x 1, temporal 1 x 64.",
             ""]  # fmt: skip
    for s in summary:
        lines += [f"## {s['machine']} ({s['cpu']})", "",
                  f"INT8 slower than FP32 for {s['int8_slower']} of {s['shapes']} shapes; every INT8 graph runs one "
                  f"QLinearConv: {s['every_int8_graph_runs_qlinearconv']}. EEGNet's layers: "
                  + ", ".join(f"{EEGNET[k]} {v:.2f}" for k, v in s["eegnet_layers"].items()), "",
                  "| shape | MACs | FP32 ms | INT8 ms | INT8 / FP32 | FP32 ns/MAC | INT8 ns/MAC |",
                  "|---|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
        t = df[df.machine == s["machine"]]
        lines += [f"| {r.name}{' (' + r.eegnet_layer + ')' if r.eegnet_layer else ''} | {r.macs:,} | {r.fp32_ms:.3f} | "
                  f"{r.int8_ms:.3f} | {r.int8_over_fp32:.2f} | {r.fp32_ns_per_mac:.3f} | {r.int8_ns_per_mac:.3f} |"
                  for r in t.itertuples()]  # fmt: skip
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
