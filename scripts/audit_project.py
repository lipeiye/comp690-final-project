"""Audit existing results without PyTorch, downloads, training or changing old evidence.

Run with system Python: python3 scripts/audit_project.py [--deep]. Deep mode
recomputes trial metrics and cross-checks every stored test prediction. A normal
audit checks coverage, frozen hashes, tables and required model/data artifacts.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # runs with a bare python3: mibci.artifacts and mibci.paths need no packages
from mibci.artifacts import sha256, write_json  # noqa: E402  (needs the sys.path line above)
from mibci.paths import DATA_DIR  # noqa: E402  (same reason)


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_bytes_retry(path):
    """Retry transient iCloud hydration cancellation without changing the source."""
    for attempt in range(2):
        try:
            return path.read_bytes()
        except OSError as error:  # noqa: PERF203  (retrying is the point of this loop)
            if error.errno not in {11, 35, 60, 89}:
                raise
            time.sleep(0.2 * (attempt + 1))
    return path.read_bytes()  # third and last attempt: every error propagates


def metrics(pred, y, artifact, clean):
    if not (len(pred) == len(y) == len(artifact)):
        raise ValueError("prediction, target and artifact lengths differ")
    pairs = [(int(a), int(b)) for a, b, f in zip(pred, y, artifact) if not clean or not f]
    if not pairs or any(not (0 <= a < 4 and 0 <= b < 4) for a, b in pairs):
        raise ValueError("empty selection or invalid four-class label")
    cm = [[sum(a == j and b == i for a, b in pairs) for j in range(4)] for i in range(4)]
    recalls, f1s = [], []
    for i in range(4):
        tp = cm[i][i]
        actual, predicted = sum(cm[i]), sum(row[i] for row in cm)
        recalls.append(tp / actual if actual else 0.0)
        f1s.append(2 * tp / (actual + predicted) if actual + predicted else 0.0)
    return {
        "acc": sum(a == b for a, b in pairs) / len(pairs),
        "macro_f1": statistics.mean(f1s),
        "recall": recalls,
        "confusion": cm,
        "n": len(pairs),
    }


def coverage(results: Path, data_dir: Path, frozen: dict, conditions: list[str]) -> tuple[dict, dict, list[str]]:
    """Expected against actual record files per phase, and the model artifacts that are missing."""
    checks, missing = {}, []
    counts = {}
    for phase, axis, indices in [("validation", "fold", range(3)), ("final", "seed", frozen["seeds"])]:
        expected = {f"sub-{s:02d}_{axis}-{i}_{c}" for s in range(1, 10) for i in indices for c in conditions}
        for group in ["runs", "quant"]:
            actual = {p.stem for p in (results / phase / group).glob("*.json")}
            name = f"{phase}/{group}"
            counts[name] = len(actual)
            checks[name] = {"missing": sorted(expected - actual), "unexpected": sorted(actual - expected)}
        if phase == "final":
            actual = {p.stem for p in (results / "test").glob("*.json")}
            counts["test"] = len(actual)
            checks["test"] = {"missing": sorted(expected - actual), "unexpected": sorted(actual - expected)}
        for stem in sorted(expected):
            for suffix in [".pt", ".onnx", "_int8.onnx"]:
                path = data_dir / "models" / phase / (stem + suffix)
                if not path.is_file() or not path.stat().st_size:
                    missing.append(str(path))
    return checks, counts, missing


def check_record(name: str, r: dict, metric_errors: list[str], recomputed: dict) -> None:
    """Recompute one test record's metrics from its stored predictions and compare them with the record."""
    for precision in ["fp32", "int8"]:
        for analysis in ["primary", "secondary"]:
            m = metrics(r["pred"][precision], r["y"], r["artifact"], analysis == "primary")
            metric_errors.extend(
                f"{name}: {analysis}/{precision}/{key}"
                for key in ["acc", "macro_f1"]
                if not math.isclose(m[key], r[analysis][precision][key], abs_tol=1e-10)
            )
            metric_errors.extend(
                f"{name}: {analysis}/{precision}/{key}"
                for key in ["recall", "confusion"]
                if m[key] != r[analysis][precision][key]
            )
            if m["n"] != r["n_clean" if analysis == "primary" else "n_trials"]:
                metric_errors.append(f"{name}: {analysis} count")
            recomputed[(analysis, r["condition"], precision, r["subject"])].append(m)


def deep_audit(results: Path) -> tuple[list[str], list[str], str]:
    """Cross-check every stored test prediction against its record and against the condition tables."""
    metric_errors, table_errors = [], []
    recomputed = collections.defaultdict(list)
    source_digest = hashlib.sha256()
    paths = sorted((results / "test").glob("*.json"))
    # iCloud may hydrate independent files slowly; bounded concurrent reads
    # avoid serial hydration while retaining deterministic filename order.
    reader = ThreadPoolExecutor(max_workers=12)
    contents = reader.map(read_bytes_retry, paths)
    for number, (path, content) in enumerate(zip(paths, contents), 1):
        source_digest.update(path.name.encode() + b"\0" + hashlib.sha256(content).digest())
        check_record(path.name, json.loads(content), metric_errors, recomputed)
        if number % 50 == 0 or number == len(paths):
            print(f"Prediction audit: {number}/{len(paths)}", flush=True)
    reader.shutdown()
    for analysis, filename in [("primary", "conditions_primary.csv"), ("secondary", "conditions_secondary.csv")]:
        for row in read_csv(results / "analysis" / filename):
            for metric, column in [("acc", "acc_mean"), ("macro_f1", "f1_mean")]:
                value = statistics.mean(
                    statistics.mean(m[metric] for m in recomputed[(analysis, row["condition"], row["precision"], s)])
                    for s in range(1, 10)
                )
                if abs(value - float(row[column])) > 0.000051:
                    table_errors.append(
                        f"predictions-to-table mismatch: {analysis}/{row['condition']}/{row['precision']}/{metric}"
                    )
    return metric_errors, table_errors, source_digest.hexdigest()


def audit(root: Path, data_dir: Path, deep: bool) -> dict:
    results = root / "results"
    frozen = json.loads((results / "frozen_config.json").read_text())
    conditions = ["full-22"] + [f"{method}-{k}" for k in frozen["channel_counts"] for method in frozen["methods"]]
    conditions += [
        f"{base}-prune{round(100 * f)}" for base in frozen["pruning"]["bases"] for f in frozen["pruning"]["fractions"]
    ]
    checks, counts, missing = coverage(results, data_dir, frozen, conditions)
    table_errors = []
    processed_missing = [
        str(data_dir / "processed" / f"sub-{s:02d}_ses-{day}.npz")
        for s in range(1, 10)
        for day in [1, 2]
        if not (data_dir / "processed" / f"sub-{s:02d}_ses-{day}.npz").is_file()
    ]
    hash_ok = sha256(results / "validation/summary.csv") == frozen["validation_summary_sha256"]
    pp = read_csv(results / "analysis/per_participant_primary.csv")
    groups = collections.defaultdict(list)
    for r in pp:
        groups[(r["condition"], r["precision"])].append(float(r["acc"]))
    summary = read_csv(results / "analysis/conditions_primary.csv")
    for r in summary:
        values = groups[(r["condition"], r["precision"])]
        if len(values) != 9 or abs(statistics.mean(values) - float(r["acc_mean"])) > 0.00011:
            table_errors.append(f"participant-to-condition mismatch: {r['condition']} {r['precision']}")
    final = frozen["final_configuration"]
    baseline = {
        int(r["subject"]): float(r["acc"]) for r in pp if r["condition"] == "full-22" and r["precision"] == "fp32"
    }
    selected = [r for r in pp if r["condition"] == final["condition"] and r["precision"] == final["precision"]]
    losses = [
        {"subject": int(r["subject"]), "difference_pp": 100 * (float(r["acc"]) - baseline[int(r["subject"])])}
        for r in selected
        if float(r["acc"]) - baseline[int(r["subject"])] < -0.02
    ]
    metric_errors, deep_table_errors, digest = deep_audit(results) if deep else ([], [], None)
    table_errors += deep_table_errors
    coverage_ok = all(not r["missing"] and not r["unexpected"] for r in checks.values())
    return {
        "audited_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "root": str(root),
        "data_dir": str(data_dir),
        "deep": deep,
        "counts": counts,
        "coverage": checks,
        "historical_evidence_consistent": coverage_ok and hash_ok and not metric_errors and not table_errors,
        "frozen_validation_hash_matches": hash_ok,
        "prediction_metric_errors": metric_errors,
        "table_errors": table_errors,
        "test_prediction_aggregate_sha256": digest,
        "replay_and_training_ready": not missing and not processed_missing,
        "missing_model_artifact_count": len(missing),
        "missing_model_artifacts": missing,
        "missing_processed_data": processed_missing,
        "final_configuration": final,
        "final_participants_losing_more_than_2pp": losses,
        "limits": [
            "Consistency checks do not independently reproduce training or validate original data provenance.",
            "Historical frozen time and hashes do not by themselves prove prospective registration.",
            "This audit does not deserialize model files, verify raw downloads or measure runtime memory.",
            "The 100-epoch matched fine-tuning control is supplementary and not in the original 17 conditions.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="default: MIBCI_DATA_DIR or data.nosync")
    parser.add_argument("--deep", action="store_true")
    parser.add_argument(
        "--require-artifacts", action="store_true", help="Fail also when original models/data are missing"
    )
    parser.add_argument("--output", type=Path, help="default: <root>/results/maintenance/audit.json")
    args = parser.parse_args()
    report = audit(args.root.resolve(), args.data_dir.resolve(), args.deep)
    write_json(args.output or args.root.resolve() / "results/maintenance/audit.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "counts",
                    "historical_evidence_consistent",
                    "replay_and_training_ready",
                    "missing_model_artifact_count",
                    "final_participants_losing_more_than_2pp",
                ]
            },
            indent=2,
        )
    )
    if not report["historical_evidence_consistent"] or (
        args.require_artifacts and not report["replay_and_training_ready"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
