"""Tables for the INT8 portability check: same INT8 files, different CPUs (post-test, exploratory).

Reads every results/supplementary/int8_portability/*.json written by
scripts/int8_portability.py (one per machine; files starting with "rr-" also hold the 7-bit
weight re-quantization) and writes, in the same directory:

  machines.csv         one row per machine: CPU, integer extensions, FP32 and INT8 agreement
                       with the archived M4 predictions, accuracy change, 7-bit results
  per_participant.csv  INT8 accuracy change per participant and machine (seeds and conditions
                       averaged within the participant)
  summary.md           the answer in plain words

Accuracies are on artifact-free session-2 trials. Everything here was run after the main
study's test results were known; it describes deployment portability, not model quality.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from mibci.paths import RESULTS_DIR

OUT = RESULTS_DIR / "supplementary/int8_portability"


def extensions(cpu: dict) -> str:
    names = {"avx512_vnni": "AVX-512 VNNI", "avx_vnni": "AVX-VNNI", "amx_int8": "AMX", "avx512f": "AVX-512",
             "avx2": "AVX2", "hw.optional.arm.FEAT_DotProd": "DotProd", "hw.optional.arm.FEAT_I8MM": "I8MM",
             "asimddp": "DotProd", "i8mm": "I8MM"}  # fmt: skip
    return ", ".join(sorted({names[k] for k, v in cpu["features"].items() if v in (True, "1") and k in names}))


def machine_row(name: str, record: dict) -> dict:
    s = record["summary"]
    row = {
        "file": name, "cpu": record["cpu"]["model"], "integer_extensions": extensions(record["cpu"]),
        "onnxruntime": record["onnxruntime"],
        "fp32_models_changed": s["fp32"]["models_with_any_change"], "fp32_mean_agreement": s["fp32"]["mean_agreement"],
        "int8_models_changed": s["int8"]["models_with_any_change"], "int8_mean_agreement": s["int8"]["mean_agreement"],
        "int8_min_agreement": s["int8"]["min_agreement"],
        "int8_acc_change_pp": s["int8"]["mean_primary_acc_change_pp"],
    }  # fmt: skip
    rr = s.get("int8_reduce_range")
    if rr:
        row |= {"rr_mean_agreement_with_fp32": rr["mean_agreement_with_archived_fp32"],
                "rr_min_agreement_with_fp32": rr["min_agreement_with_archived_fp32"],
                "rr_acc_change_vs_fp32_pp": rr["mean_primary_acc_change_vs_fp32_pp"]}  # fmt: skip
    return row


def per_participant(name: str, record: dict) -> list[dict]:
    by = defaultdict(list)
    for r in record["rows"]:
        by[r["subject"]].append(100 * (r["int8"]["primary_acc"] - r["int8"]["archived_primary_acc"]))
    return [{"file": name, "subject": s, "int8_acc_change_pp": float(np.mean(v))} for s, v in sorted(by.items())]


def same_as_m4(records: dict[str, dict], reference: str = "rr-m4") -> dict[str, int]:
    """Per machine with 7-bit weights: models whose 7-bit scores equal the M4's exactly."""
    if reference not in records:
        return {}
    key = ("agreement_with_archived_fp32", "primary_acc")
    m4 = {r["model"]: [r["int8_reduce_range"][k] for k in key] for r in records[reference]["rows"]}
    return {
        name: sum([r["int8_reduce_range"][k] for k in key] == m4[r["model"]] for r in record["rows"])
        for name, record in records.items()
        if name != reference and "int8_reduce_range" in record["summary"]
    }


def write_csv(path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary(machines: list[dict], participants: list[dict], same_7bit: dict[str, int]) -> str:
    lines = ["# INT8 portability: the same INT8 files on different CPUs", "",
             "Reference: the archived predictions made on the original Apple M4. FP32 is the control.", "",
             "| machine | CPU | 8-bit related extensions | FP32 models changed | INT8 models changed | INT8 mean "
             "agreement | INT8 accuracy change (pp) | 7-bit weights: accuracy vs FP32 (pp) |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]  # fmt: skip
    for m in machines:
        rr = f"{m['rr_acc_change_vs_fp32_pp']:+.2f}" if "rr_acc_change_vs_fp32_pp" in m else ""
        lines.append(
            f"| {m['file']} | {m['cpu']} | {m['integer_extensions']} | {m['fp32_models_changed']}/459 | "
            f"{m['int8_models_changed']}/459 | {m['int8_mean_agreement']:.4f} | "
            f"{m['int8_acc_change_pp']:+.2f} | {rr} |"
        )
    changed = [m for m in machines if m["int8_models_changed"] > 0]
    exact = [m for m in machines if m["int8_models_changed"] == 0]
    lines += ["", f"Machines that reproduce every INT8 prediction: {', '.join(sorted({m['cpu'] for m in exact}))}."]
    if changed:
        lines.append(f"Machines that change INT8 predictions: {', '.join(sorted({m['cpu'] for m in changed}))}.")
        worst = min(participants, key=lambda r: r["int8_acc_change_pp"])
        lines.append(f"Largest per-participant INT8 accuracy change on those machines: participant {worst['subject']}, "
                     f"{worst['int8_acc_change_pp']:+.2f} pp ({worst['file']}).")  # fmt: skip
    if same_7bit:
        listed = ", ".join(f"{name} {n}/459" for name, n in sorted(same_7bit.items()))
        lines.append(f"7-bit weights: models whose accuracy and agreement with FP32 equal the M4's to the last trial: "
                     f"{listed}.")  # fmt: skip
    lines += ["", "FP32 agrees on every machine, so the environment is not the cause. On x86 CPUs without VNNI, "
              "ONNX Runtime's 8-bit kernels multiply with VPMADDUBSW, which adds pairs of products in 16 bits and can "
              "saturate; Arm dot products and VNNI accumulate in 32 bits. ONNX Runtime documents 7-bit weights "
              "(reduce_range) as the remedy for non-VNNI machines; the last column tests it.", "",
              "Exploratory: run after the main study's test results were known; it concerns deployment, "
              "not the frozen analysis."]  # fmt: skip
    return "\n".join(lines) + "\n"


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    paths = sorted(OUT.glob("*.json"))
    if not paths:
        raise SystemExit(f"no machine files in {OUT}; run scripts/int8_portability.py first")
    records = {p.stem: json.loads(p.read_text()) for p in paths}
    machines = [machine_row(name, r) for name, r in records.items()]
    participants = [row for name, r in records.items() for row in per_participant(name, r)]
    write_csv(OUT / "machines.csv", machines)
    write_csv(OUT / "per_participant.csv", participants)
    affected = {m["file"] for m in machines if m["int8_models_changed"] > 0}
    (OUT / "summary.md").write_text(
        summary(machines, [p for p in participants if p["file"] in affected] or participants, same_as_m4(records))
    )
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
