"""Export trained models to ONNX and quantize them to INT8 (static PTQ).

Calibration always uses the model's own normalized training trials. Validation
models are also scored on their held-out runs (FP32 ONNX and INT8), together with
a check that the FP32 export reproduces the PyTorch logits. Final models are only
exported here; evaluate_test.py scores them on session 2.
"""

import argparse
from pathlib import Path

import numpy as np

from mibci import deploy
from mibci.artifacts import completed, model_exports, require_files
from mibci.experiment import MODELS_DIR, RESULTS_DIR, run_pool, session1, write_json
from mibci.model import load_bundle
from mibci.train import predict, scores


def quant_job(bundle_path: str, record_path: str) -> None:
    path = Path(bundle_path)
    model, scaler, b = load_bundle(path)
    X, y, tr, va = session1(b["subject"], b["split"])
    X = X[:, b["channels"]]
    fp32, int8 = path.with_suffix(".onnx"), path.with_name(f"{path.stem}_int8.onnx")
    deploy.export_onnx(model, fp32)
    check_X = scaler.transform(X[tr][:32])
    check_torch = predict(model, check_X)
    check_onnx = deploy.logits(deploy.session(fp32), check_X)
    if not np.allclose(check_torch, check_onnx, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f"FP32 ONNX export exceeds tolerance: {path}")
    deploy.quantize_int8(fp32, int8, scaler.transform(X[tr]))
    record = {"name": path.stem, "subject": b["subject"], "split": b["split"], "condition": b["condition"],
              "seed": b["seed"], "training_export_max_abs_diff": float(np.abs(check_torch-check_onnx).max()),
              "size": {"fp32": deploy.size_bytes(fp32), "int8": deploy.size_bytes(int8)}}
    if va is not None:
        X_va = scaler.transform(X[va])
        ref = predict(model, X_va)
        out32 = deploy.logits(deploy.session(fp32), X_va)
        out8 = deploy.logits(deploy.session(int8), X_va)
        record.update(
            export_max_abs_diff=float(np.abs(out32 - ref).max()),
            export_pred_agreement=float((out32.argmax(1) == ref.argmax(1)).mean()),
            fp32=scores(out32, y[va]),
            int8=scores(out8, y[va]),
            int8_pred_agreement=float((out8.argmax(1) == out32.argmax(1)).mean()),
        )
    write_json(Path(record_path), record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["validation", "final"], required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    out_dir = RESULTS_DIR / args.phase / "quant"
    jobs = []
    records = sorted((RESULTS_DIR / args.phase / "runs").glob("sub-*.json"))
    if not records:
        raise SystemExit(f"No training records in {RESULTS_DIR / args.phase / 'runs'}")
    bundles = [MODELS_DIR / args.phase / f"{p.stem}.pt" for p in records]
    require_files(bundles, "quantization training bundles")
    for bundle in bundles:
        record = out_dir / f"{bundle.stem}.json"
        if not completed(record, model_exports(bundle), {"name": bundle.stem}):
            jobs.append((bundle.stem, dict(bundle_path=str(bundle), record_path=str(record))))
    run_pool(quant_job, jobs, args.workers, threads=1)


if __name__ == "__main__":
    main()
