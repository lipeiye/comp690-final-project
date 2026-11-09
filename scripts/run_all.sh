#!/usr/bin/env bash
# The full pipeline, in the order it was run.
# Subjects 1-6 train on the Apple GPU (MPS) while 7-9 train on the CPU
# (see device_for in src/mibci/experiment.py); two_pools runs both groups at once.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv.nosync}"
export MIBCI_DATA_DIR="${MIBCI_DATA_DIR:-$PWD/data.nosync}"
export MIBCI_RESULTS_DIR="${MIBCI_RESULTS_DIR:-$PWD/results}"
if [[ -f "$MIBCI_RESULTS_DIR/frozen_config.json" ]]; then
  echo "Analysis already frozen. Use fresh MIBCI_RESULTS_DIR and MIBCI_DATA_DIR paths; see docs/REPRODUCIBILITY.md." >&2
  exit 1
fi
mkdir -p "$MIBCI_DATA_DIR/logs"

two_pools() {
  uv run python "$@" --subjects 1 2 3 4 5 6 --workers 3 &
  local gpu=$!
  uv run python "$@" --subjects 7 8 9 --workers 6 &
  local cpu=$!
  wait "$gpu"
  wait "$cpu"
}

uv run python scripts/prepare_data.py

# Validation phase: session 1 only.
two_pools scripts/run_validation.py --stage full
uv run python scripts/choose_budget.py
two_pools scripts/run_validation.py --stage subsets
uv run python scripts/quantize_models.py --phase validation
uv run python scripts/summarize_validation.py --pruning-base
two_pools scripts/run_pruning.py --phase validation
uv run python scripts/quantize_models.py --phase validation
uv run python scripts/summarize_validation.py --freeze   # writes results/frozen_config.json

# Final models on all of session 1, then the held-out test day (session 2).
two_pools scripts/train_final.py --stage full
two_pools scripts/train_final.py --stage subsets
two_pools scripts/run_pruning.py --phase final
uv run python scripts/quantize_models.py --phase final
uv run python scripts/evaluate_test.py

uv run python scripts/benchmark.py   # on an otherwise idle machine, on AC power
uv run python scripts/analyze.py
uv run python scripts/replay_demo.py --save results/demo/replay.mp4
