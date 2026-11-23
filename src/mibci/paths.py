"""Where the project reads and writes: one definition for every script and tool.

`MIBCI_DATA_DIR` and `MIBCI_RESULTS_DIR` redirect data/models and result records; they
are read once, at import, so set them before Python starts. `mibci.data` and
`mibci.experiment` hold their own copies of these lines because three finished studies
pin those files byte for byte; tests/test_paths.py keeps the copies equal.

Reports, slides and release archives always describe the project's own results, so the
generators that build them use `PROJECT_RESULTS_DIR`, which ignores `MIBCI_RESULTS_DIR`.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _from_env(variable: str, default: Path) -> Path:
    return Path(os.environ.get(variable, default)).expanduser().resolve()


DATA_DIR = _from_env("MIBCI_DATA_DIR", PROJECT_ROOT / "data.nosync")
RESULTS_DIR = _from_env("MIBCI_RESULTS_DIR", PROJECT_ROOT / "results")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

PROJECT_RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
PRESENTATION_DIR = PROJECT_ROOT / "presentation"


def display(path: Path) -> str:
    """A path relative to the project root when it lies inside it, else the absolute path."""
    path = Path(path).resolve()
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
