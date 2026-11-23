"""Where every post-midterm study ran, how many models it trained and how many CPU-hours that took.

Reads, under RESULTS_DIR (results/, or MIBCI_RESULTS_DIR when set); records only:
  supplementary/<study>/runs/**/*.json       one training record per model (train_time_s, device)
  supplementary/<study>/environments/*.json  the operating system each stage ran on
  supplementary/<study>/protocol.json        the plan (number of jobs, created_at)

Writes RESULTS_DIR/supplementary/compute_summary.json and compute_summary.md. CPU-hours add up each
model's wall-clock training time with one thread, so they count training only, not scoring, timing or
emulation. A study whose environments are Linux ran on UNC Longleaf; Darwin means the M4 Mac mini. A study without
environment records (no training stage) is placed by the host names its other records carry (*.ll.unc.edu is
Longleaf), and a folder holding only a plan is marked as not run.
"""

from __future__ import annotations

import argparse
import json

from mibci.artifacts import write_json
from mibci.paths import RESULTS_DIR

SUPP = RESULTS_DIR / "supplementary"
# Studies whose records name no machine but ran as Longleaf SLURM jobs (their logs are in ~/comp690/logs there):
# the bit-width grid and fine-tuning, the precision-map check, cross-CPU timing and INT8 re-scoring.
LONGLEAF_WITHOUT_ENV = {"quant_study", "quant_rule_study", "hw_latency", "int8_portability"}


def hosts(folder) -> set[str]:
    """Where the records of a study without a training stage were written: Longleaf, the M4, or nothing."""
    found = set()
    for path in folder.rglob("*.json"):
        if path.name in ("protocol.json", "analysis.json"):
            continue
        record = json.loads(path.read_text())
        host = record.get("host") if isinstance(record, dict) else None
        if isinstance(host, str):
            found.add("Longleaf" if host.endswith(".ll.unc.edu") else "M4 Mac mini")
    return found


def no_training_stage(folder) -> str:
    if not any(p.name != "protocol.json" for p in folder.rglob("*") if p.is_file()):
        return "plan only, not run"
    found = hosts(folder)
    place = " and ".join(sorted(found, reverse=True)) if found else "M4 Mac mini"
    return f"{place} (no training stage)"


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    rows = []
    for folder in sorted(p for p in SUPP.iterdir() if p.is_dir()):
        runs = sorted((folder / "runs").rglob("*.json")) if (folder / "runs").is_dir() else []
        envs = sorted((folder / "environments").glob("*.json")) if (folder / "environments").is_dir() else []
        plan = json.loads((folder / "protocol.json").read_text()) if (folder / "protocol.json").exists() else {}
        systems = {json.loads(p.read_text()).get("os", "")[:5] for p in envs}
        trained = [
            r for r in (json.loads(p.read_text()) for p in runs) if r.get("train_time_s")
        ]  # scoring-only records skipped
        seconds = sum(r.get("train_time_s") or 0 for r in trained)
        where = "Longleaf" if systems == {"Linux"} else "M4 Mac mini" if systems == {"macOS"} else "mixed/unknown"
        if folder.name in LONGLEAF_WITHOUT_ENV:
            systems, where = {"Linux"}, "Longleaf"
        fine_tunes = len(list((folder / "qat").glob("*.json"))) if (folder / "qat").is_dir() else 0
        rows.append(
            {
                "study": folder.name,
                "where": where if systems else no_training_stage(folder),
                "planned_jobs": len(plan.get("jobs", [])),
                "models_trained": len(trained),
                "fine_tunes": fine_tunes,
                "cpu_hours": round(seconds / 3600, 1),
                "planned_at": plan.get("created_at", "")[:16],
            }
        )
    longleaf = [r for r in rows if r["where"] == "Longleaf"]
    total = {
        "longleaf_models": sum(r["models_trained"] for r in longleaf),
        "longleaf_cpu_hours": round(sum(r["cpu_hours"] for r in longleaf), 1),
    }
    write_json(SUPP / "compute_summary.json", {"studies": rows, **total})
    lines = [
        "# Compute per post-midterm study",
        "",
        f"Longleaf in total: {total['longleaf_models']:,} models, "
        f"{total['longleaf_cpu_hours']:,} CPU-hours of training.",
        "",
        "| study | where | planned jobs | models trained | fine-tunes | CPU-hours | plan written |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    lines += [
        f"| {r['study']} | {r['where']} | {r['planned_jobs']:,} | {r['models_trained']:,} | {r['fine_tunes']:,} | "
        f"{r['cpu_hours']:,} | "
        f"{r['planned_at']} |"
        for r in rows
    ]
    (SUPP / "compute_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
