"""Small, dependency-free safeguards for resumable research artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


class ArtifactError(RuntimeError):
    """An existing result cannot safely be reused or replaced."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_files(paths, context: str = "operation") -> None:
    missing = [str(p) for p in map(Path, paths) if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise ArtifactError(f"{context}: missing or empty artifacts:\n  " + "\n  ".join(missing)
                            + "\nRestore original artifacts, or use fresh MIBCI_RESULTS_DIR and "
                            "MIBCI_DATA_DIR directories to reproduce independently. "
                            "Historical results will not be silently replaced.")


def completed(record: Path, artifacts=(), expected: dict | None = None) -> bool:
    """Reuse a result only with its artifacts and matching metadata.

    A missing result can be retried after an interrupted write. An existing result
    with missing artifacts is an error: retraining in place could replace weights
    underlying already published test scores. This guard deliberately does not
    claim that a nonempty model has been deserialized or cryptographically verified.
    """
    record = Path(record)
    if not record.exists():
        return False
    try:
        result = json.loads(record.read_text())
    except (OSError, ValueError) as error:
        raise ArtifactError(f"Invalid result JSON {record}; preserve it before recovery: {error}") from error
    if not isinstance(result, dict):
        raise ArtifactError(f"Expected a JSON object in {record}")
    for key, value in (expected or {}).items():
        if result.get(key) != value:
            raise ArtifactError(f"{record}: expected {key}={value!r}, found {result.get(key)!r}; "
                                "use a fresh results directory for different settings")
    require_files(artifacts, f"resuming {record}")
    return True


def write_json(path: Path, obj) -> None:
    """Write a complete JSON record atomically, including strict finite numbers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=1, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def model_exports(bundle: Path) -> tuple[Path, Path]:
    bundle = Path(bundle)
    return bundle.with_suffix(".onnx"), bundle.with_name(f"{bundle.stem}_int8.onnx")
