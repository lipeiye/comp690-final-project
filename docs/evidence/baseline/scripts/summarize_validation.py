"""Summarize the validation phase and apply the prespecified decision rule.

Accuracy is averaged over the three folds within each subject, then over subjects.
FP32 and INT8 accuracies come from ONNX Runtime; the FP32 export reproduces PyTorch.

Decision rule (proposal, Section 4): among candidates whose mean validation accuracy
is within 2 percentage points of the 22-channel FP32 baseline, prefer fewer
electrodes, then a smaller model (parameter bytes), then higher accuracy. Random
subsets are controls, not candidates.

--pruning-base  picks the channel-reduced condition to prune from the unpruned
                candidates and writes results/validation/pruning_base.json.
--freeze        picks the final configuration from all candidates (pruned ones
                included) and writes results/frozen_config.json, the analysis plan
                that must exist before session 2 may be evaluated.
"""

import argparse
import datetime
import hashlib
import json

import pandas as pd

from mibci.experiment import RESULTS_DIR, write_json
from mibci.prune import FINETUNE_EPOCHS, FINETUNE_LR, FRACTIONS
from mibci.selection import CHANNEL_COUNTS, FULL, METHODS

VAL_DIR = RESULTS_DIR / "validation"
TOLERANCE = 0.02
SEEDS = (0, 1, 2)


def parse_condition(condition: str) -> dict:
    """'proposed-8-prune25' -> method proposed, 8 electrodes, 25% of filters pruned."""
    parts = condition.split("-")
    return {"method": "full" if condition.startswith("full") else parts[0], "electrodes": int(parts[1]),
            "pruned": int(parts[2].removeprefix("prune")) if len(parts) > 2 else 0}


def load_runs() -> pd.DataFrame:
    rows = []
    for path in sorted((VAL_DIR / "quant").glob("sub-*.json")):
        q = json.loads(path.read_text())
        run = json.loads((VAL_DIR / "runs" / path.name).read_text())
        rows.append({"subject": q["subject"], "fold": q["split"], "condition": q["condition"],
                     **parse_condition(q["condition"]), "acc_torch": run["val"]["acc"],
                     "acc_fp32": q["fp32"]["acc"], "acc_int8": q["int8"]["acc"],
                     "f1_fp32": q["fp32"]["macro_f1"], "f1_int8": q["int8"]["macro_f1"],
                     "bytes_fp32": q["size"]["fp32"]["parameter_bytes"],
                     "bytes_int8": q["size"]["int8"]["parameter_bytes"],
                     "export_max_abs_diff": q["export_max_abs_diff"]})
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["condition", "method", "electrodes", "pruned"]
    per_subject = runs.groupby(keys + ["subject"]).mean(numeric_only=True).reset_index()
    summary = per_subject.groupby(keys).agg(
        subjects=("subject", "nunique"),
        acc_fp32=("acc_fp32", "mean"), acc_fp32_sd=("acc_fp32", "std"),
        acc_int8=("acc_int8", "mean"), acc_int8_sd=("acc_int8", "std"),
        f1_fp32=("f1_fp32", "mean"), f1_int8=("f1_int8", "mean"),
        bytes_fp32=("bytes_fp32", "mean"), bytes_int8=("bytes_int8", "mean"),
    ).reset_index()
    summary["int8_minus_fp32"] = summary["acc_int8"] - summary["acc_fp32"]
    summary = summary.sort_values(["electrodes", "pruned", "method"], ascending=[False, True, True])
    return summary, per_subject


def decide(summary: pd.DataFrame, include_full: bool) -> dict:
    baseline = float(summary.loc[summary["condition"] == FULL, "acc_fp32"].iloc[0])
    candidates = []
    for _, row in summary.iterrows():
        if row["method"].startswith("random") or (row["method"] == "full" and not include_full):
            continue
        for precision in ("fp32", "int8"):
            candidates.append({"condition": row["condition"], "precision": precision,
                               "electrodes": int(row["electrodes"]), "parameter_bytes": float(row[f"bytes_{precision}"]),
                               "val_acc": float(row[f"acc_{precision}"])})
    eligible = [c for c in candidates if c["val_acc"] >= baseline - TOLERANCE]
    chosen = (min(eligible, key=lambda c: (c["electrodes"], c["parameter_bytes"], -c["val_acc"])) if eligible
              else max(candidates, key=lambda c: c["val_acc"]))
    return {**chosen, "baseline_full22_fp32_val_acc": baseline, "eligible": bool(eligible),
            "n_eligible": len(eligible), "tolerance": TOLERANCE}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pruning-base", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    runs = load_runs()
    summary, per_subject = summarize(runs)
    summary.to_csv(VAL_DIR / "summary.csv", index=False, float_format="%.4f")
    per_subject.to_csv(VAL_DIR / "per_subject.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summary[["condition", "subjects", "acc_fp32", "acc_fp32_sd", "acc_int8", "int8_minus_fp32",
                       "bytes_fp32", "bytes_int8"]].to_string(index=False, float_format="%.3f"))
    print(f"runs: {len(runs)}, max |ONNX - PyTorch| logit difference: {runs['export_max_abs_diff'].max():.1e}")

    if args.pruning_base:
        choice = decide(summary[summary["pruned"] == 0], include_full=False)
        write_json(VAL_DIR / "pruning_base.json", choice)
        print("pruning base:", choice)

    if args.freeze:
        choice = decide(summary, include_full=True)
        summary_bytes = (VAL_DIR / "summary.csv").read_bytes()
        frozen = {
            "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "note": "Written from validation results only, before any session-2 data was scored.",
            "epochs": json.loads((VAL_DIR / "budget.json").read_text())["epochs"],
            "seeds": list(SEEDS),
            "channel_counts": list(CHANNEL_COUNTS),
            "methods": list(METHODS),
            "pruning": {"bases": [FULL, json.loads((VAL_DIR / "pruning_base.json").read_text())["condition"]],
                        "fractions": list(FRACTIONS), "finetune_epochs": FINETUNE_EPOCHS, "finetune_lr": FINETUNE_LR},
            "final_configuration": choice,
            "primary_analysis": "session-2 trials without expert artifact flags; metrics averaged over seeds "
                                "within each participant, then summarized over the nine participants",
            "secondary_analysis": "the same models scored on all session-2 trials, flagged ones included",
            "comparisons": {
                "redundancy_vs_relevance": "proposed-K minus relevance-K, K in (16, 8), FP32 and INT8",
                "redundancy_vs_weightnorm": "proposed-K minus weightnorm-K, K in (16, 8), FP32 and INT8",
                "redundancy_vs_random": "proposed-K minus the mean of random1-3-K, K in (16, 8), FP32 and INT8",
                "vs_full": "every condition minus full-22 FP32; count participants losing more than 2 points",
                "quantization": "INT8 minus FP32 for every condition",
                "pruning": "pruned minus unpruned base, FP32 and INT8",
            },
            "uncertainty": {"method": "participant-level bootstrap of paired differences, percentile interval",
                            "resamples": 10000, "level": 0.95, "seed": 690},
            "validation_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        }
        write_json(RESULTS_DIR / "frozen_config.json", frozen)
        print("final configuration:", choice)


if __name__ == "__main__":
    main()
