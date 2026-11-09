"""Hash and hydrate retained result files, without including mutable supplementary runs."""

from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def inspect(path):
    for attempt in range(3):
        try:
            content=path.read_bytes()
            return str(path.relative_to(ROOT)),{"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()}
        except OSError as error:
            if error.errno not in {11,35,60,89} or attempt==2:raise
            time.sleep(.2*(attempt+1))


def main():
    paths=[ROOT/"results/frozen_config.json",ROOT/"results/data_summary.csv"]
    for group in ["validation","final","test","analysis","benchmark","demo"]:
        paths.extend(p for p in (ROOT/"results"/group).rglob("*") if p.is_file() and p.name!=".DS_Store")
    results={}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for number,future in enumerate(as_completed([pool.submit(inspect,p) for p in paths]),1):
            name,row=future.result();results[name]=row
            if number%100==0:print(f"Historical files hashed: {number}/{len(paths)}",flush=True)
    target=ROOT/"docs/evidence/historical_results_manifest.json"
    target.write_text(json.dumps({"files":dict(sorted(results.items())),
        "scope":"All retained files in historical validation/final/test/analysis/benchmark/demo and frozen config/data summary; excludes new maintenance and supplementary results."},indent=2)+"\n")
    print(f"Wrote {len(results)} historical hashes",flush=True)


if __name__=="__main__":main()
