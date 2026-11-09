"""Create a checksummed local release archive without environments or raw EEG data."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,help="Optional existing model directory; contents copied under models/")
    args=parser.parse_args()
    if args.output.exists():raise SystemExit("Release exists; use a new output filename")
    if args.model_dir and not args.model_dir.is_dir():parser.error("model-dir does not exist")
    files={}
    for directory in ["src","scripts","tests","docs","presentation","results"]:
        for p in (ROOT/directory).rglob("*"):
            if p.is_file() and not set(p.parts)&{"__pycache__",".DS_Store"}:
                files[str(p.relative_to(ROOT))]=p
    for name in ["README.md","pyproject.toml","uv.lock",".python-version",".gitignore","COMP690_Project_Proposal.md","peer tutoring transcript.pdf"]:
        p=ROOT/name
        if p.is_file():files[name]=p
    if args.model_dir:
        for p in args.model_dir.rglob("*"):
            if p.is_file() and p.suffix in [".pt",".onnx",".json"]:
                files[str(Path("models")/p.relative_to(args.model_dir))]=p
    manifest={"original_training_hardware":"Apple M4 Mac mini (author clarification)",
              "models_included":bool(args.model_dir),"model_source_directory":str(args.model_dir) if args.model_dir else None,
              "claim":"Source/results delivery archive. Included model origin must be read from its supplementary protocol or original metadata.",
              "excluded":"virtual environments, raw/processed EEG, caches, unrelated files",
              "files":{}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with zipfile.ZipFile(args.output,"x",compression=zipfile.ZIP_DEFLATED) as archive, ThreadPoolExecutor(max_workers=12) as reader:
            entries=sorted(files.items())
            contents=reader.map(lambda entry: entry[1].read_bytes(),entries)
            for (name,p),content in zip(entries,contents):
                manifest["files"][name]={"sha256":hashlib.sha256(content).hexdigest(),"bytes":len(content)}
                archive.writestr(name,content)
            archive.writestr("RELEASE_MANIFEST.json",json.dumps(manifest,indent=2)+"\n")
    except BaseException:
        args.output.unlink(missing_ok=True)
        raise
    print(f"Packaged {len(files)} files into {args.output}")


if __name__=="__main__":main()
