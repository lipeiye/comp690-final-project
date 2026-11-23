"""Where the main (historical) run keeps its records and models, and its frozen plan.

The validation, budget, final-training and test scripts share these names; keeping them
here stops scripts from importing each other.
"""

from __future__ import annotations

import json

from mibci.paths import MODELS_DIR, RESULTS_DIR

VAL_DIR = RESULTS_DIR / "validation"
VAL_MODELS = MODELS_DIR / "validation"
FINAL_DIR = RESULTS_DIR / "final"
FINAL_MODELS = MODELS_DIR / "final"
FROZEN_CONFIG = RESULTS_DIR / "frozen_config.json"

VALIDATION_SPLITS = ["fold-0", "fold-1", "fold-2"]
MAX_EPOCHS = 500  # the 22-channel validation curves run this long; the budget is chosen from them
CHECKPOINT_EVERY = 25


def frozen_config() -> dict:
    """The plan frozen before session 2 was scored; stop if the analysis is not frozen yet."""
    if not FROZEN_CONFIG.exists():
        raise SystemExit("Freeze the analysis plan first: scripts/summarize_validation.py --freeze")
    return json.loads(FROZEN_CONFIG.read_text())
