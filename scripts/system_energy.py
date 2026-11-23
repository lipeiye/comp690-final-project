"""Energy budget of one decision on a wearable decoder: electrodes, radio and arithmetic (added after the midterm).

Order-of-magnitude model, not a measurement. It asks which of the project's levers matters
for battery life: fewer electrodes, decoding on the device instead of streaming, or a
cheaper network (quantization, the layer swap). Every constant is listed with its source
in CONSTANTS and printed into summary.md, with a low and a high value so the conclusion
can be checked against both ends.

Per 2-s decision window (the decoder makes one decision per window):
  front end   electrodes x analog front-end power x 2 s
  streaming   electrodes x 250 Hz x 24 bits x 2 s x radio energy per bit (raw data to a phone)
  on device   MACs x energy per MAC, for the original and the layer-swapped graph, in 32-bit
              float and 8-bit integer arithmetic; plus a radio message carrying the decision

MAC counts come from results/supplementary/int8_study/costs.csv. Writes table.csv,
summary.md and fig_energy.png to results/supplementary/system_energy/.
"""

from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mibci.artifacts import write_json
from mibci.paths import RESULTS_DIR

OUT = RESULTS_DIR / "supplementary/system_energy"
WINDOW_S, SFREQ, BITS_PER_SAMPLE = 2.0, 250, 24
DECISION_BITS = 64  # one small radio message per decision
CONSTANTS = {
    "front_end_w_per_channel": {
        "low": 10e-6,
        "high": 5e-3,
        "source": "high: TI ADS1299 datasheet, 'low power: 5 mW/channel'; low: research-grade EEG front ends "
        "reported at around 10 uW per channel",
    },
    "radio_j_per_bit": {
        "low": 20e-9,
        "high": 100e-9,
        "source": "Bluetooth Low Energy: about 20 nJ/bit for the radio alone at 1 Mbit/s (nRF52-class transmit "
        "current near 6 mA at 3 V); about 100 nJ/bit once protocol overhead is included",
    },
    "mcu_j_per_mac_fp32": {
        "low": 1e-9,
        "high": 4e-9,
        "source": "Cortex-M4F-class microcontroller: about 0.1-0.3 nJ per cycle at roughly 0.25-0.5 float MACs per "
        "cycle (order of magnitude)",
    },
    "mcu_j_per_mac_int8": {
        "low": 0.2e-9,
        "high": 1e-9,
        "source": "the same core with CMSIS-NN SIMD int8 kernels, roughly 1-2 MACs per cycle (order of magnitude)",
    },
}


def read_costs() -> list[dict]:
    with (RESULTS_DIR / "supplementary/int8_study/costs.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def rows_for(costs: list[dict], end: str) -> list[dict]:
    c = {k: v[end] for k, v in CONSTANTS.items()}
    rows = []
    for row in costs:
        n = int(row["electrodes"])
        front = n * c["front_end_w_per_channel"] * WINDOW_S
        stream = n * SFREQ * BITS_PER_SAMPLE * WINDOW_S * c["radio_j_per_bit"]
        message = DECISION_BITS * c["radio_j_per_bit"]
        compute = {
            f"{graph}_{precision}": int(row[f"{graph}_macs"]) * c[f"mcu_j_per_mac_{precision}"]
            for graph in ("original", "spatial-first")
            for precision in ("fp32", "int8")
        }
        best = front + message + compute["spatial-first_int8"]
        rows.append({
            "condition": row["condition"], "electrodes": n, "assumptions": end, "front_end_j": front,
            "stream_raw_j": stream, "decision_message_j": message, **{f"compute_{k}_j": v for k, v in compute.items()},
            "total_streaming_j": front + stream,
            "total_on_device_best_j": best,
            "total_on_device_worst_j": front + message + compute["original_fp32"],
            "compute_share_on_device_best": compute["spatial-first_int8"] / best,
        })  # fmt: skip
    return rows


def figure(rows: list[dict]) -> None:
    chosen = [r for r in rows if r["condition"] in ("full-22", "relevance-16", "relevance-8")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True, sharey=True)
    parts = [("front_end_j", "front end"), ("stream_raw_j", "radio, raw stream"),
             ("compute_original_fp32_j", "network, original FP32"),
             ("compute_spatial-first_int8_j", "network, swapped INT8")]  # fmt: skip
    for ax, end in zip(axes, ("high", "low"), strict=True):
        subset = [r for r in chosen if r["assumptions"] == end]
        for i, (key, label) in enumerate(parts):
            ax.bar([x + 0.2 * i for x in range(len(subset))], [r[key] * 1e3 for r in subset], 0.2, label=label)
        ax.set_xticks([x + 0.3 for x in range(len(subset))], [f"{r['electrodes']} electrodes" for r in subset])
        ax.set_yscale("log")
        ax.set_title(f"{end} assumptions")
        ax.set_ylabel("energy per 2-s decision (mJ)")
    axes[0].legend(fontsize=8)
    fig.savefig(OUT / "fig_energy.png", dpi=150)
    plt.close(fig)


def summary(rows: list[dict]) -> str:
    def pick(condition, end):
        return next(r for r in rows if r["condition"] == condition and r["assumptions"] == end)

    lines = ["# Energy budget of one decision (order of magnitude, not measured)", ""]
    for end in ("high", "low"):
        full, sixteen = pick("full-22", end), pick("relevance-16", end)
        saved = full["front_end_j"] - sixteen["front_end_j"]
        lines += [
            f"## {end.capitalize()} assumptions",
            "",
            f"- Front end, 22 electrodes: {full['front_end_j'] * 1e3:.3g} mJ per decision; 16 electrodes save "
            f"{saved * 1e3:.3g} mJ.",
            f"- Streaming the raw 22-channel signal: {full['stream_raw_j'] * 1e3:.3g} mJ per decision.",
            f"- The network on the device: original FP32 {full['compute_original_fp32_j'] * 1e3:.3g} mJ, "
            f"layer-swapped INT8 {full['compute_spatial-first_int8_j'] * 1e3:.3g} mJ.",
            f"- Share of the network in the best on-device total: {full['compute_share_on_device_best']:.1%}.",
            f"- Removing 6 electrodes saves {saved / max(full['compute_original_fp32_j'], 1e-30):.3g} times the "
            "energy of running the original FP32 network.",
            "",
        ]
    lines += ["## Constants", ""]
    lines += [f"- {k}: {v['low']:.3g} to {v['high']:.3g}. {v['source']}." for k, v in CONSTANTS.items()]
    lines += ["", "MAC counts from results/supplementary/int8_study/costs.csv. Memory traffic, sleep current and "
              "the 2-s acquisition by the converter are not modelled."]  # fmt: skip
    return "\n".join(lines) + "\n"


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    costs = read_costs()
    rows = rows_for(costs, "high") + rows_for(costs, "low")
    with (OUT / "table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(OUT / "analysis.json", {"constants": CONSTANTS, "rows": rows})
    (OUT / "summary.md").write_text(summary(rows))
    figure(rows)
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
