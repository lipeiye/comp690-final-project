"""Hash and hydrate retained result files, without including mutable supplementary runs."""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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


def inspect(path):
    content = read_bytes_retry(path)
    return str(path.relative_to(ROOT)), {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    paths = [ROOT / "results/frozen_config.json", ROOT / "results/data_summary.csv"]
    for group in ["validation", "final", "test", "analysis", "benchmark", "demo"]:
        paths.extend(p for p in (ROOT / "results" / group).rglob("*") if p.is_file() and p.name != ".DS_Store")
    results = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for number, future in enumerate(as_completed([pool.submit(inspect, p) for p in paths]), 1):
            name, row = future.result()
            results[name] = row
            if number % 100 == 0:
                print(f"Historical files hashed: {number}/{len(paths)}", flush=True)
    target = ROOT / "docs/evidence/historical_results_manifest.json"
    target.write_text(
        json.dumps(
            {
                "files": dict(sorted(results.items())),
                "scope": "All retained files in historical validation/final/test/analysis/benchmark/demo and "
                "frozen config/data summary; excludes new maintenance and supplementary results.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {len(results)} historical hashes", flush=True)


if __name__ == "__main__":
    main()
