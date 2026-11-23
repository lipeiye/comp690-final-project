"""Does the same INT8 file give the same predictions on another CPU? (post-test, exploratory)

Found on 2026-11-21: on Longleaf's Intel Xeon Gold 6140 (AVX-512 without VNNI), two INT8
graphs that give bit-identical logits on the M4 (the archived graph and its dense rewrite)
differ by up to 0.012. A likely cause is ONNX Runtime's x86 8-bit kernels: without VNNI they
multiply with VPMADDUBSW, which adds pairs of products in 16 bits and can saturate, while
Arm's dot-product instructions accumulate in 32 bits. If so, INT8 accuracy, and not only
INT8 speed, depends on the chip.

This script runs the archived FP32 and INT8 ONNX graphs of all 459 final models (the same
files, byte for byte, that produced the archived test records on the M4) on session 2 with
ONNX Runtime on this machine, and compares every prediction with the archived one. FP32 is
the control: it should agree almost everywhere. With --reduce-range it also re-quantizes
each FP32 graph with 7-bit weights (ONNX Runtime's documented remedy for the saturation;
calibration on the model's own session-1 trials, as for the archived graphs) and scores
that graph too, against the archived FP32 predictions and labels. Writes
results/supplementary/int8_portability/<label>.json with the CPU, its integer extensions,
per-model agreement and accuracy; refuses to overwrite.
"""

from __future__ import annotations

import argparse
import json
import socket
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import onnxruntime as ort

from mibci import data, deploy, variants
from mibci.artifacts import model_exports, sha256, write_json
from mibci.layout import FINAL_MODELS
from mibci.model import load_bundle
from mibci.paths import RESULTS_DIR
from mibci.timing import cpu

OUT = RESULTS_DIR / "supplementary/int8_portability"


def reduce_range_row(bundle_path, scaler, b, X, ses, archived: dict) -> dict:
    """Re-quantize the archived FP32 graph with 7-bit weights and score it."""
    session1 = data.load_session(int(b["subject"]), 1)
    calibration = scaler.transform(session1["X"][~session1["artifact"]][:, b["channels"]])
    fp32, _ = model_exports(bundle_path)
    with tempfile.TemporaryDirectory() as folder:
        graph = Path(folder) / f"{bundle_path.stem}_int8_rr.onnx"
        variants.quantize(fp32, graph, calibration, reduce_range=True)
        pred = deploy.logits(deploy.session(graph), X).argmax(1)
    clean = ~ses["artifact"]
    reference = np.array(archived["pred"]["fp32"])
    return {
        "agreement_with_archived_fp32": float((pred == reference).mean()),
        "primary_acc": float((pred[clean] == ses["y"][clean]).mean()),
        "archived_fp32_primary_acc": float((reference[clean] == ses["y"][clean]).mean()),
    }


def score(bundle_path, ses, archived: dict, reduce_range: bool = False) -> dict:
    _, scaler, b = load_bundle(bundle_path)
    X = scaler.transform(ses["X"][:, b["channels"]])
    clean = ~ses["artifact"]
    row = {"model": bundle_path.stem}
    if reduce_range:
        row["int8_reduce_range"] = reduce_range_row(bundle_path, scaler, b, X, ses, archived)
    for precision, onnx_path in zip(("fp32", "int8"), model_exports(bundle_path), strict=True):
        pred = deploy.logits(deploy.session(onnx_path), X).argmax(1)
        before = np.array(archived["pred"][precision])
        row[precision] = {
            "onnx_sha256": sha256(onnx_path),
            "agreement_with_archived": float((pred == before).mean()),
            "changed_predictions": int((pred != before).sum()),
            "primary_acc": float((pred[clean] == ses["y"][clean]).mean()),
            "archived_primary_acc": float((before[clean] == ses["y"][clean]).mean()),
        }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default=socket.gethostname().split(".")[0], help="name of the output file")
    parser.add_argument("--reduce-range", action="store_true", help="also score 7-bit-weight INT8 graphs")
    args = parser.parse_args()
    out = OUT / f"{args.label}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; keep it and use another --label")
    rows = []
    for subject in data.SUBJECTS:
        ses = data.load_session(subject, 2, final_evaluation=True)
        for bundle_path in sorted(FINAL_MODELS.glob(f"sub-{subject:02d}_seed-*.pt")):
            archived = json.loads((RESULTS_DIR / "test" / f"{bundle_path.stem}.json").read_text())
            rows.append({"subject": subject, **score(bundle_path, ses, archived, args.reduce_range)})
        changed = sum(r["int8"]["changed_predictions"] for r in rows if r["subject"] == subject)
        print(f"participant {subject}: {changed} INT8 predictions differ from the M4 archive")
    summary = {
        p: {
            "models": len(rows),
            "models_with_any_change": sum(r[p]["changed_predictions"] > 0 for r in rows),
            "mean_agreement": float(np.mean([r[p]["agreement_with_archived"] for r in rows])),
            "min_agreement": float(np.min([r[p]["agreement_with_archived"] for r in rows])),
            "mean_primary_acc_change_pp": float(
                100 * np.mean([r[p]["primary_acc"] - r[p]["archived_primary_acc"] for r in rows])
            ),
        }
        for p in ("fp32", "int8")
    }
    if args.reduce_range:
        rr = [r["int8_reduce_range"] for r in rows]
        summary["int8_reduce_range"] = {
            "models": len(rr),
            "mean_agreement_with_archived_fp32": float(np.mean([r["agreement_with_archived_fp32"] for r in rr])),
            "min_agreement_with_archived_fp32": float(np.min([r["agreement_with_archived_fp32"] for r in rr])),
            "mean_primary_acc_change_vs_fp32_pp": float(
                100 * np.mean([r["primary_acc"] - r["archived_fp32_primary_acc"] for r in rr])
            ),
        }
    write_json(out, {
        "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "status": "supplementary_post_test", "host": socket.gethostname(), "cpu": cpu(),
        "onnxruntime": ort.__version__, "summary": summary, "rows": rows,
        "reference": "archived predictions in results/test, made on the original Apple M4 Mac mini",
    })  # fmt: skip
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
