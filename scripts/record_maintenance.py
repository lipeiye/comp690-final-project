"""Record source changes and verify preservation of retained historical evidence."""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"docs/evidence"


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before=json.loads((EVIDENCE/"before.sha256.json").read_text())
    changed=[];preserved=[];patches=[]
    for relative,old in before.items():
        path=ROOT/relative
        current=digest(path)
        if relative.startswith("results/"):
            if current!=old["sha256"]:raise RuntimeError(f"Historical evidence changed: {relative}")
            preserved.append(relative)
        if current!=old["sha256"]:
            changed.append({"path":relative,"before_sha256":old["sha256"],"after_sha256":current})
            original=EVIDENCE/"baseline"/relative
            if original.exists():
                patches.extend(difflib.unified_diff(original.read_text().splitlines(keepends=True),
                    path.read_text().splitlines(keepends=True),fromfile="a/"+relative,tofile="b/"+relative))
    new=[]
    existing_without_byte_snapshot={"presentation/deck_source/deck.json"}
    for directory in ["scripts","src","tests","docs","presentation"]:
        for path in sorted((ROOT/directory).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or "evidence" in path.parts or path.name==".DS_Store":continue
            relative=str(path.relative_to(ROOT))
            if relative in existing_without_byte_snapshot:
                changed.append({"path":relative,"before_sha256":None,"after_sha256":digest(path),
                    "note":"Preexisting metadata file; initial byte snapshot was not captured. Inserted supplementary matched slide in its order; no byte-exact before diff is claimed."})
                continue
            if relative not in before:
                new.append({"path":relative,"sha256":digest(path),"bytes":path.stat().st_size})
                if path.suffix in [".py",".md"]:
                    patches.extend(difflib.unified_diff([],path.read_text().splitlines(keepends=True),
                        fromfile="/dev/null",tofile="b/"+relative))
    (EVIDENCE/"changes.patch").write_text("".join(patches))
    (EVIDENCE/"change_inventory.json").write_text(json.dumps({"modified":changed,"added":new,
        "historical_evidence_verified_unchanged":preserved,
        "limits":"Baseline hashes cover the initial saved source/evidence set. New runtime logs and large data/model artifacts are inventoried separately in their protocols and release manifests."},indent=2)+"\n")
    print(f"{len(changed)} modified, {len(new)} added, {len(preserved)} historical source files unchanged.")


if __name__=="__main__":main()
