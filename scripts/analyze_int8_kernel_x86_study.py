"""Tables for the INT8 kernels on x86 (scripts/int8_kernel_x86_study.py).

Reads RESULTS_DIR/supplementary/int8_kernel_x86_study/{protocol.json,run/*.json}; writes machines.csv,
latency.csv, summary.md and analysis.json there.

Per machine: how many of the 459 archived INT8 graphs give logits identical to ONNX Runtime's own INT8 once
the grouped-convolution kernel is swapped in; how many models' next-day predictions differ from the archived
M4 INT8 predictions, with ONNX Runtime alone and with the kernel; mean primary accuracy over the models of
both against the archived; and the median latency of every version for the four timed architectures.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from mibci.artifacts import write_json
from mibci.paths import RESULTS_DIR

OUT = RESULTS_DIR / "supplementary" / "int8_kernel_x86_study"


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    files = sorted((OUT / "run").glob("*.json"))
    if not files:
        raise SystemExit("no run/*.json yet")
    machines, latency = [], []
    for path in files:
        r = json.loads(path.read_text())
        eq = pd.DataFrame(r["equivalence"])
        machines.append({
            "machine": r["label"], "cpu": r["cpu"]["model"], "models": len(eq),
            "kernel_equals_runtime": int(eq.kernel_equals_runtime.sum()),
            "max_abs_kernel_minus_runtime": float(eq.max_abs_kernel_minus_runtime.max()),
            "runtime_models_changed_vs_m4": int((eq.runtime_changed_vs_archived > 0).sum()),
            "kernel_models_changed_vs_m4": int((eq.kernel_changed_vs_archived > 0).sum()),
            "runtime_acc_minus_m4_pp": float(100 * (eq.runtime_primary_acc - eq.archived_primary_acc).mean()),
            "kernel_acc_minus_m4_pp": float(100 * (eq.kernel_primary_acc - eq.archived_primary_acc).mean()),
        })  # fmt: skip
        for arch, t in r["timing"].items():
            latency.extend(
                {"machine": r["label"], "architecture": arch, "variant": v, "us": 1000 * x["median_ms"],
                 "swapped_kernel_exact": t["swapped_kernel_equals_runtime_on_8_windows"]}
                for v, x in t["timed"].items()
            )  # fmt: skip
    m, lat = pd.DataFrame(machines), pd.DataFrame(latency)
    m.to_csv(OUT / "machines.csv", index=False, float_format="%.4f")
    lat.to_csv(OUT / "latency.csv", index=False, float_format="%.2f")
    write_json(OUT / "analysis.json", {"machines": machines, "latency": lat.to_dict("records")})
    lines = ["# The two INT8 kernels on x86 CPUs", "",
             "459 archived INT8 graphs, next day; the grouped-convolution kernel's portable C path swapped into each "
             "machine's optimized graph. 'Changed' counts models with any next-day prediction unlike the archived M4 "
             "INT8 one.", "", "| machine | CPU | kernel = runtime | changed vs M4: runtime / kernel | accuracy vs M4 "
             "(points): runtime / kernel |", "|---|---|---:|---|---|"]  # fmt: skip
    lines += [f"| {r.machine} | {r.cpu} | {r.kernel_equals_runtime}/{r.models} | {r.runtime_models_changed_vs_m4} / "
              f"{r.kernel_models_changed_vs_m4} | {r.runtime_acc_minus_m4_pp:+.2f} / {r.kernel_acc_minus_m4_pp:+.2f} |"
              for r in m.itertuples()]  # fmt: skip
    lines += ["", "## Latency (microseconds per window, one thread)", ""]
    pivot = lat.pivot_table(index=["machine", "architecture"], columns="variant", values="us")
    cols = ["fp32", "int8", "int8-kernel", "fp32-spatial-first", "int8-spatial-first", "int8-spatial-first-kernel"]
    lines += ["| machine | architecture | " + " | ".join(cols) + " |", "|---|---|" + "---:|" * len(cols)]
    lines += [f"| {i[0]} | {i[1]} | " + " | ".join(f"{row[c]:.0f}" for c in cols) + " |" for i, row in pivot.iterrows()]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
