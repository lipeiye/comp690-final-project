"""Plans, records and shards for the studies added after the midterm review.

A study keeps its plan in `RESULTS_DIR/supplementary/<name>/protocol.json`, one JSON record
per finished job under that directory, and its models under
`MODELS_DIR/supplementary/<name>/`. The plan pins the SHA-256 of the source files that
decide its numbers; once the study has records, tools/guard.py refuses edits to those
files, and `Study.load` refuses to continue if they changed anyway.

Jobs are dicts with a unique "name". A stage runs the jobs of the plan whose record is
missing, so an interrupted stage is simply started again. `shard(jobs, "i/n")` gives
SLURM array task i of n a fixed slice of the whole plan (not of what is still pending),
so tasks that start at different times never take the same job.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mibci.artifacts import ArtifactError, completed, sha256, write_json
from mibci.env import environment
from mibci.paths import MODELS_DIR, PROJECT_ROOT, RESULTS_DIR

# Every study trains and scores through these modules.
CORE_SOURCES = ("src/mibci/data.py", "src/mibci/model.py", "src/mibci/train.py", "src/mibci/experiment.py")
POST_TEST = {
    "status": "supplementary_post_test",
    "historical_test_results_already_known": True,
}


class Study:
    def __init__(self, name: str, script: str, sources: Iterable[str] = ()):
        self.name = name
        self.out = RESULTS_DIR / "supplementary" / name
        self.models = MODELS_DIR / "supplementary" / name
        self.sources = sorted({*CORE_SOURCES, *sources, script})

    @property
    def protocol_path(self) -> Path:
        return self.out / "protocol.json"

    def create(self, jobs: list[dict], *, purpose: str, claim_boundary: str, **design) -> dict:
        """Write the plan once; with a plan already on disk, verify and return that one instead."""
        if self.protocol_path.exists():
            return self.load()
        names = [job["name"] for job in jobs]
        if len(names) != len(set(names)):
            raise ValueError("job names must be unique")
        frozen = RESULTS_DIR / "frozen_config.json"
        plan = {
            "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            **POST_TEST,
            "purpose": purpose,
            "claim_boundary": claim_boundary,
            **design,
            "historical_frozen_config_sha256": sha256(frozen),
            "source_code_sha256": {p: sha256(PROJECT_ROOT / p) for p in self.sources},
            "jobs": jobs,
        }
        write_json(self.protocol_path, plan)
        return plan

    def load(self) -> dict:
        plan = json.loads(self.protocol_path.read_text())
        if sha256(RESULTS_DIR / "frozen_config.json") != plan["historical_frozen_config_sha256"]:
            raise ArtifactError("Historical frozen configuration changed since this study was planned")
        for relative, digest in plan["source_code_sha256"].items():
            if sha256(PROJECT_ROOT / relative) != digest:
                raise ArtifactError(f"Pinned source changed: {relative}; start a new study directory")
        return plan

    def record(self, kind: str, job: dict) -> Path:
        return self.out / kind / f"{job['name']}.json"

    def model(self, job: dict) -> Path:
        return self.models / f"{job['name']}.pt"

    def pending(
        self,
        jobs: list[dict],
        kind: str,
        artifacts: Callable[[dict], list[Path]] = lambda job: [],
        expected: Callable[[dict], dict] = lambda job: {"name": job["name"]},
    ) -> list[dict]:
        return [j for j in jobs if not completed(self.record(kind, j), artifacts(j), expected(j))]

    def log_environment(self, label: str, **extra) -> None:
        write_json(self.out / "environments" / f"{label}.json", {**environment(), **extra})


def shard(jobs: list[dict], spec: str | None) -> list[dict]:
    """Jobs of array task i out of n ("i/n", 0-based); all jobs when spec is None."""
    if not spec:
        return jobs
    index, count = (int(part) for part in spec.split("/"))
    if not 0 <= index < count:
        raise ValueError(f"bad shard {spec}: need 0 <= i < n")
    return jobs[index::count]
