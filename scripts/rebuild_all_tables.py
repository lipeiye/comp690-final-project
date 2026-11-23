"""Rebuild every table, figure and report of the project from its stored records, and compare byte for byte.

This is the course report's push-button artifact. It runs every generator of tools/golden.py that needs
only records (no EEG data and no trained models): the main study's tables, each post-midterm study's
analysis, the computing summary, the technical report, the slides' HTML and the course report. They run
in a sandbox copy of the repository, and every output is compared with the repository's copy; the
repository itself is never written. Exit status 0 means everything was reproduced exactly.

    python3 scripts/rebuild_all_tables.py            # records only (minutes)
    python3 scripts/rebuild_all_tables.py --with-eeg # also the generators that read EEG and models

Rerunning the experiments themselves is described in the course report's Artifacts section and in README.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEEDS_EEG = ("electrode", "redundancy", "audit")  # read cached EEG windows or the trained models


def generators() -> list:
    listed = subprocess.run([sys.executable, str(ROOT / "tools" / "golden.py"), "--list"], capture_output=True,
                            text=True, check=True).stdout.splitlines()  # fmt: skip
    return [line.split()[0] for line in listed if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-eeg", action="store_true", help="also run the generators that read EEG or models")
    parser.add_argument("--keep", action="store_true", help="keep the sandbox for inspection")
    args = parser.parse_args()
    names = [n for n in generators() if args.with_eeg or n not in NEEDS_EEG]
    command = [sys.executable, str(ROOT / "tools" / "golden.py"), *names] + (["--keep"] if args.keep else [])
    print(f"rebuilding {len(names)} generators: {' '.join(names)}", flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
