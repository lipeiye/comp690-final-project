"""Tables for the INT8 tensors of the deployed x86 graphs (scripts/int8_capture_study.py).

Reads RESULTS_DIR/supplementary/int8_capture_study/{protocol.json,capture/*.json,compare/*.json}; writes
layers.csv, variants.csv, summary.md and analysis.json there.

Per comparison (a VNNI reference against a CPU without VNNI): whether every capture reproduced a normal session's
logits on its machine; for the archived graphs, the share of values that differ in each integer tensor and the
largest difference (in integer steps), the first tensor that differs, and in that layer the overlap of the
outputs that differ with those the u8 x s8 int16-pairing model predicts to saturate; for each 7-bit variant, how
many models then agree in every tensor, and how many training-day predictions the two machines' normal sessions
share. Training-day windows only.
"""

from __future__ import annotations

import argparse
import collections
import json

import pandas as pd

from mibci.artifacts import write_json
from mibci.paths import RESULTS_DIR

OUT = RESULTS_DIR / "supplementary" / "int8_capture_study"
# x86 only: the QuantizeLinear nodes ONNX Runtime 1.31.1 adds when it rewrites the S8S8 graph as U8S8, by the
# float tensor each one requantizes (the eeg input, pool 1, pool 2); see the optimized graphs in data/int8_capture/
U8_REQUANTIZED = {
    "qdq_s8_to_u8_quant": "input (as uint8)",
    "qdq_s8_to_u8_quant_token_9": "pool 1 (as uint8)",
    "qdq_s8_to_u8_quant_token_15": "pool 2 (as uint8)",
}
TOTALS = ("outputs", "differ", "predicted", "differ_and_predicted", "differ_not_predicted", "predicted_not_differ")


def captures() -> dict[str, dict]:
    out = {}
    for path in sorted((OUT / "capture").glob("*.json")):
        c = json.loads(path.read_text())
        out[c["label"]] = {
            "cpu": c["cpu"]["model"],
            "host": c["host"],
            "captures": len(c["rows"]),
            "capture_equals_plain": sum(r["capture_equals_plain"] for r in c["rows"]),
        }
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    files = sorted((OUT / "compare").glob("*.json"))
    if not files:
        raise SystemExit("no compare/*.json yet")
    machines = captures()
    layer_rows, variant_rows, summaries = [], [], []
    for path in files:
        c = json.loads(path.read_text())
        pair = f"{c['other']} vs {c['reference']}"
        for r in c["rows"]:
            for v in r["tensors"].values():
                base = v.get("label", "").removesuffix(" (layout)")
                if base in U8_REQUANTIZED:
                    v["label"] = U8_REQUANTIZED[base] + (" (layout)" if v["label"].endswith(" (layout)") else "")
            first_label = r["tensors"][r["first_differing"]].get("label") if r["first_differing"] else None
            if r["variant"] == "int8":
                layer_rows.extend(
                    {"comparison": pair, "model": r["model"], "tensor": t, "label": v.get("label", t),
                     **{k: v[k] for k in ("values", "differ", "max_abs")}}
                    for t, v in r["tensors"].items()
                )  # fmt: skip
            variant_rows.append({
                "comparison": pair, "model": r["model"], "variant": r["variant"],
                "identical": all(v["differ"] == 0 for v in r["tensors"].values()),
                "first_differing": first_label,
                "same_predictions_all": r["same_predictions_all"], "windows_all": r["windows_all"],
            })  # fmt: skip
        archived = [r for r in c["rows"] if r["variant"] == "int8"]
        first = collections.Counter(
            r["tensors"][r["first_differing"]].get("label", r["first_differing"]) if r["first_differing"] else "none"
            for r in archived
        )
        sat = [r["saturation"] for r in archived if r.get("saturation")]
        totals = {k: sum(s[k] for s in sat) for k in TOTALS}
        summaries.append({
            "comparison": pair, "reference": c["reference"], "other": c["other"], "models": len(archived),
            "first_differing": dict(first), "saturation_checked": len(sat), "saturation_totals": totals,
            "share_of_differing_predicted": totals["differ_and_predicted"] / max(totals["differ"], 1),
            "share_of_predicted_differing": totals["differ_and_predicted"] / max(totals["predicted"], 1),
            "only_reference": sorted({n for r in c["rows"] for n in r["only_reference"]}),
            "only_other": sorted({n for r in c["rows"] for n in r["only_other"]}),
        })  # fmt: skip
    layers, variants = pd.DataFrame(layer_rows), pd.DataFrame(variant_rows)
    layers.to_csv(OUT / "layers.csv", index=False)
    variants.to_csv(OUT / "variants.csv", index=False)
    by_variant = (
        variants.groupby(["comparison", "variant"])
        .agg(
            models=("model", "count"),
            identical=("identical", "sum"),
            same_predictions_all=("same_predictions_all", "sum"),
            windows_all=("windows_all", "sum"),
        )
        .reset_index()
    )
    write_json(
        OUT / "analysis.json",
        {"machines": machines, "summaries": summaries, "variants": by_variant.to_dict("records")},
    )
    lines = [
        "# Where INT8 computes differently without VNNI (the graphs each machine runs)",
        "",
        "Archived INT8 graphs and 7-bit variants, optimized on each machine as a normal session does; integer "
        "tensors read from that optimized graph for the first 32 artifact-free training-day windows; predictions "
        "of normal sessions on all artifact-free training-day windows. A VNNI CPU is the reference.",
        "",
        "| machine | CPU | captures equal to a normal session's logits |",
        "|---|---|---:|",
    ]
    lines += [f"| {k} | {m['cpu']} | {m['capture_equals_plain']}/{m['captures']} |" for k, m in machines.items()]
    lines.append("")
    for s in summaries:
        t = s["saturation_totals"]
        lines += [
            f"## {s['comparison']}",
            "",
            f"First tensor that differs (archived graphs, {s['models']} models): "
            + ", ".join(f"{k} {v}" for k, v in s["first_differing"].items()),
            "",
            f"In that layer ({s['saturation_checked']} models checked) {t['differ']:,} of {t['outputs']:,} outputs "
            f"differ; the u8 x s8 int16-pairing model predicts {t['predicted']:,} to saturate; "
            f"{t['differ_and_predicted']:,} differ and are predicted ({s['share_of_differing_predicted']:.0%} of "
            f"those that differ, {s['share_of_predicted_differing']:.0%} of those predicted).",
            "",
        ]
        if s["only_reference"] or s["only_other"]:
            lines += [f"Tensors only in one machine's graph: {s['only_reference']} / {s['only_other']}", ""]
        lines += ["| tensor | values differing (all models) | largest difference |", "|---|---:|---:|"]
        sub = layers[layers.comparison == s["comparison"]].groupby("label", sort=False)
        lines += [f"| {name} | {g.differ.sum() / g['values'].sum():.2%} | {g.max_abs.max():g} |" for name, g in sub]
        lines += [
            "",
            "| variant | models identical in every tensor | training-day predictions the same |",
            "|---|---:|---:|",
        ]
        v = by_variant[by_variant.comparison == s["comparison"]]
        lines += [
            f"| {r.variant} | {r.identical}/{r.models} | {r.same_predictions_all}/{r.windows_all} "
            f"({r.same_predictions_all / r.windows_all:.1%}) |"
            for r in v.itertuples()
        ]
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
