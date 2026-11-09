"""Score every final model on session 2, the held-out test day.

Refuses to run until results/frozen_config.json exists. Each model is run through
ONNX Runtime in FP32 and INT8 on the trials without artifact flags (primary analysis)
and on all trials (secondary analysis), normalized with its session-1 statistics.
Per-trial predictions are kept for the analysis and the replay demo.
"""

import argparse
from pathlib import Path

from mibci import data, deploy
from mibci.experiment import MODELS_DIR, RESULTS_DIR, run_pool, write_json
from mibci.model import load_bundle
from mibci.train import predict, scores
from train_final import frozen_config


def eval_job(bundle_path: str, record_path: str) -> None:
    path = Path(bundle_path)
    model, scaler, b = load_bundle(path)
    ses = data.load_session(b["subject"], 2, final_evaluation=True)
    X = scaler.transform(ses["X"][:, b["channels"]])
    y, clean = ses["y"], ~ses["artifact"]
    out = {"fp32": deploy.logits(deploy.session(path.with_suffix(".onnx")), X),
           "int8": deploy.logits(deploy.session(path.with_name(f"{path.stem}_int8.onnx")), X)}
    torch_pred = predict(model, X).argmax(1)
    write_json(Path(record_path), {
        "name": path.stem, "subject": b["subject"], "seed": b["seed"], "condition": b["condition"],
        "channels": [data.EEG_CHANNELS[c] for c in b["channels"]],
        "n_trials": len(y), "n_clean": int(clean.sum()),
        "primary": {p: scores(lg[clean], y[clean]) for p, lg in out.items()},
        "secondary": {p: scores(lg, y) for p, lg in out.items()},
        "export_pred_agreement": float((out["fp32"].argmax(1) == torch_pred).mean()),
        "int8_pred_agreement": float((out["int8"].argmax(1) == out["fp32"].argmax(1)).mean()),
        "pred": {p: lg.argmax(1).tolist() for p, lg in out.items()},
        "y": y.tolist(),
        "artifact": ses["artifact"].astype(int).tolist(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    frozen_config()  # the analysis plan must be frozen before session 2 is read

    jobs = []
    for bundle in sorted((MODELS_DIR / "final").glob("sub-*.pt")):
        record = RESULTS_DIR / "test" / f"{bundle.stem}.json"
        if not record.exists():
            jobs.append((bundle.stem, dict(bundle_path=str(bundle), record_path=str(record))))
    run_pool(eval_job, jobs, args.workers, threads=1)
    print(f"scored {len(jobs)} models; summaries come from analyze.py")


if __name__ == "__main__":
    main()
