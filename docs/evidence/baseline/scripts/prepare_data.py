"""Download BCI Competition IV 2a, preprocess both sessions and cache them.

Also writes a manifest of the preprocessing settings and source-file hashes,
and results/data_summary.csv with trial and artifact counts per session.
"""

import argparse
import csv
import hashlib
import importlib.metadata as md
import json

import numpy as np

from mibci import data


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=data.SUBJECTS)
    args = parser.parse_args()
    data.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for subject in args.subjects:
        for session, mat_path in data.download(subject).items():
            arrays = data.preprocess_session(mat_path)
            np.savez(data.session_path(subject, session), source_sha256=sha256(mat_path), **arrays)
            print(
                f"sub-{subject:02d} ses-{session}: {len(arrays['y'])} trials, "
                f"{np.unique(arrays['run']).size} runs, {arrays['artifact'].sum()} flagged, "
                f"first window {arrays['first_window_s']:.2f} s into its run"
            )

    manifest = {
        "dataset": "BCI Competition IV 2a (BNCI 2014-001), downloaded via MOABB",
        "channels": data.EEG_CHANNELS,
        "classes": data.CLASSES,
        "sfreq_hz": data.SFREQ,
        "filter": f"causal Butterworth band-pass, order {data.FILTER_ORDER}, "
        f"{data.BAND_HZ[0]}-{data.BAND_HZ[1]} Hz, zero state at each run start (scipy sosfilt)",
        "window_after_cue_s": data.WINDOW_S,
        "n_times": data.N_TIMES,
        "units": "microvolts, recorded reference",
        "versions": {p: md.version(p) for p in ["numpy", "scipy", "moabb"]},
    }
    (data.PROCESSED_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Rebuild the summary from every cached session, not only the ones processed now.
    rows = []
    for path in sorted(data.PROCESSED_DIR.glob("sub-*_ses-*.npz")):
        with np.load(path) as f:
            subject, session = int(path.stem[4:6]), int(path.stem[-1])
            counts = np.bincount(f["y"], minlength=len(data.CLASSES))
            rows.append({
                "subject": subject,
                "session": session,
                "trials": len(f["y"]),
                "flagged": int(f["artifact"].sum()),
                **{f"n_{c}": int(n) for c, n in zip(data.CLASSES, counts)},
                "first_window_s": round(float(f["first_window_s"]), 2),
                "source_sha256": str(f["source_sha256"]),
            })
    out = data.PROJECT_ROOT / "results" / "data_summary.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out.relative_to(data.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
