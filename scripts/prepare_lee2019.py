"""Download and cache the Lee et al. (2019) OpenBMI motor-imagery sessions (second dataset).

For every participant: fetch both session files through MOABB (about 600 MB each, 108 files,
about 65 GB in all; run this on Longleaf, not on the Mac), record each file's SHA-256, apply
mibci.lee2019's causal preprocessing and write processed/lee2019/sub-XX_ses-Y.npz plus a
summary CSV. Raw files can be deleted afterwards with --delete-raw (the hashes are kept).
Session 2 is cached but stays locked behind lee2019.load_session.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from mibci import lee2019
from mibci.artifacts import sha256
from mibci.paths import RESULTS_DIR, display


def prepare(subject: int, delete_raw: bool) -> list[dict]:
    rows = []
    for session, path in lee2019.download(subject).items():
        digest = sha256(path)
        arrays = lee2019.preprocess_session(path)
        lee2019.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(lee2019.session_path(subject, session), source_sha256=digest, **arrays)
        counts = np.bincount(arrays["y"], minlength=len(lee2019.CLASSES))
        rows.append({"subject": subject, "session": session, "trials": len(arrays["y"]),
                     **{f"n_{c}": int(n) for c, n in zip(lee2019.CLASSES, counts, strict=True)},
                     "channels": len(arrays["channels"]), "source_bytes": path.stat().st_size,
                     "source_sha256": digest})  # fmt: skip
        if delete_raw:
            path.unlink()
        print(f"sub-{subject:02d} ses-{session}: {len(arrays['y'])} trials, {len(arrays['channels'])} channels")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", type=int, nargs="+", default=lee2019.SUBJECTS)
    parser.add_argument("--delete-raw", action="store_true", help="remove each .mat file once it is cached")
    args = parser.parse_args()
    out = RESULTS_DIR / "supplementary/lee2019_data"
    out.mkdir(parents=True, exist_ok=True)
    for subject in args.subjects:
        target = out / f"sub-{subject:02d}.csv"
        if target.exists() and all(lee2019.session_path(subject, s).exists() for s in (1, 2)):
            continue
        rows = prepare(subject, args.delete_raw)
        with target.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"summaries in {display(out)}")


if __name__ == "__main__":
    main()
