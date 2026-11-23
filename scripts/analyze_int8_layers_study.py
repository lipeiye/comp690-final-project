"""Tables for the INT8 layer comparison (scripts/int8_layers_study.py).

Reads RESULTS_DIR/supplementary/int8_layers_study/{protocol.json,compare/*.json}; writes layers.csv,
variants.csv, summary.md and analysis.json there.

Per comparison (a VNNI reference against a CPU without VNNI): for the archived graphs, the share of values
that differ in each int8 tensor and the largest difference (in integer steps), the first tensor that differs,
and in that layer the overlap of the outputs that differ with those the int16-pairing model predicts to
saturate; for each 7-bit variant, how many models then agree in every tensor and in their predictions.
Training-day windows only.
"""

from __future__ import annotations

import argparse
import collections
import json

import pandas as pd

from mibci.artifacts import write_json
from mibci.paths import RESULTS_DIR

OUT = RESULTS_DIR / "supplementary" / "int8_layers_study"
NAMES = {
    "view_1_QuantizeLinear_Output": "input",
    "getitem_QuantizeLinear_Output": "temporal conv",
    "getitem_3_QuantizeLinear_Output": "spatial conv",
    "avg_pool2d_QuantizeLinear_Output": "pool 1",
    "conv2d_2_QuantizeLinear_Output": "depthwise conv",
    "getitem_6_QuantizeLinear_Output": "pointwise conv",
    "view_2_QuantizeLinear_Output": "pool 2",
    "logits_QuantizeLinear_Output": "logits (int8)",
    "logits": "logits (float)",
}


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    files = sorted((OUT / "compare").glob("*.json"))
    if not files:
        raise SystemExit("no compare/*.json yet")
    layer_rows, variant_rows, summaries = [], [], []
    for path in files:
        c = json.loads(path.read_text())
        pair = f"{c['other']} vs {c['reference']}"
        rows = c["rows"]
        for r in rows:
            if r["variant"] == "int8":
                layer_rows.extend(
                    {"comparison": pair, "model": r["model"], "tensor": NAMES.get(t, t), **v}
                    for t, v in r["tensors"].items()
                )
            identical = all(v["differ"] == 0 for v in r["tensors"].values())
            variant_rows.append(
                {
                    "comparison": pair,
                    "model": r["model"],
                    "variant": r["variant"],
                    "identical": identical,
                    "first_differing": NAMES.get(r["first_differing"]),
                    "same_predictions": r["same_predictions"],
                    "windows": r["windows"],
                }
            )
        archived = [r for r in rows if r["variant"] == "int8"]
        first = collections.Counter(NAMES.get(r["first_differing"], "none") for r in archived)
        sat = [r["saturation"] for r in archived if "saturation" in r]
        totals = {
            k: sum(s[k] for s in sat)
            for k in (
                "outputs",
                "differ",
                "predicted",
                "differ_and_predicted",
                "differ_not_predicted",
                "predicted_not_differ",
            )
        }
        summaries.append(
            {
                "comparison": pair,
                "models": len(archived),
                "first_differing": dict(first),
                "saturation_totals": totals,
                "share_of_differing_predicted": totals["differ_and_predicted"] / max(totals["differ"], 1),
                "share_of_predicted_differing": totals["differ_and_predicted"] / max(totals["predicted"], 1),
            }
        )
    layers = pd.DataFrame(layer_rows)
    variants = pd.DataFrame(variant_rows)
    layers.to_csv(OUT / "layers.csv", index=False)
    variants.to_csv(OUT / "variants.csv", index=False)
    by_variant = (
        variants.groupby(["comparison", "variant"])
        .agg(
            models=("model", "count"),
            identical=("identical", "sum"),
            same_predictions=("same_predictions", "sum"),
            windows=("windows", "sum"),
        )
        .reset_index()
    )
    write_json(OUT / "analysis.json", {"summaries": summaries, "variants": by_variant.to_dict("records")})
    lines = [
        "# Where INT8 computes differently without VNNI",
        "",
        "Archived INT8 graphs and 7-bit variants on training-day windows; a VNNI CPU as reference.",
        "",
        "**Caution.** This capture added the int8 tensors to the graph's outputs, which on x86 kept ONNX Runtime from "
        "rewriting the graph as U8S8 before fusing it, so these tensors are not what a normal session computes there "
        "(a normal session on the Gold 6140 changes 144 of P1's 273 training-day predictions; the capture changes "
        "none). scripts/int8_capture_study.py repeats the comparison on the graphs each machine actually runs.",
        "",
    ]
    for s in summaries:
        t = s["saturation_totals"]
        lines += [
            f"## {s['comparison']}",
            "",
            f"First tensor that differs (archived graphs, {s['models']} models): "
            + ", ".join(f"{k} {v}" for k, v in s["first_differing"].items()),
            "",
            f"In that layer {t['differ']:,} of {t['outputs']:,} outputs differ; the int16-pairing model predicts "
            f"{t['predicted']:,} to saturate; {t['differ_and_predicted']:,} differ and are predicted "
            f"({s['share_of_differing_predicted']:.0%} of those that differ, {s['share_of_predicted_differing']:.0%} "
            "of those predicted).",
            "",
            "| tensor | values differing (all models) | largest difference |",
            "|---|---:|---:|",
        ]
        sub = layers[layers.comparison == s["comparison"]].groupby("tensor", sort=False)
        lines += [f"| {name} | {g.differ.sum() / g['values'].sum():.2%} | {g.max_abs.max():g} |" for name, g in sub]
        lines += ["", "| variant | models identical in every tensor | predictions the same |", "|---|---:|---:|"]
        v = by_variant[by_variant.comparison == s["comparison"]]
        lines += [
            f"| {r.variant} | {r.identical}/{r.models} | {r.same_predictions}/{r.windows} |" for r in v.itertuples()
        ]
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
