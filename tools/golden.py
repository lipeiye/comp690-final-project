"""Golden-output check: rebuild generated files in a sandbox and compare them with the repository.

A refactor of analysis, report or slide code must not change a single published number.
This tool copies the code, docs and results into a scratch directory, runs the chosen
generators there (EEG data and models are read in place through MIBCI_DATA_DIR) and
reports every output file that differs from the copy in the repository.

    python3 tools/golden.py --list
    python3 tools/golden.py analysis report        # selected generators
    python3 tools/golden.py --all
    python3 tools/golden.py --all --keep           # keep the sandbox for inspection

Exit status 0 means every compared file is identical (JSON is compared after dropping
the volatile keys listed per generator). Run it before and after a refactor; a
difference is either a bug in the refactor or must be explained in the change itself.
The repository is never written: the tool checks this before it exits.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv.nosync/bin/python"
# README runs these with a bare python3 (3.9 on the author's Mac), so the rebuild must too.
SYSTEM_PYTHON = {"audit", "manifest", "slides", "slides-midterm"}
COPIED = [
    "src",
    "scripts",
    "docs",
    "presentation",
    "results",
    "tests",
    "README.md",
    "agent.md",
    "AGENTS.md",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
]
IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")

# name: (command relative to the sandbox, outputs compared, JSON keys ignored)
GENERATORS: dict[str, tuple[list[str], list[str], set[str]]] = {
    "analysis": (["scripts/analyze.py"], ["results/analysis"], set()),
    "electrode": (["scripts/analyze_electrode_study.py"], ["results/supplementary/electrode_count"], set()),
    "int8": (["scripts/int8_study.py", "--stage", "analyze"], ["results/supplementary/int8_study"], set()),
    "redundancy": (["scripts/redundancy_check.py"], ["results/supplementary/redundancy_check"], set()),
    "report": (["scripts/build_report.py"], ["docs/REPORT.md", "docs/assets", "results/maintenance"], set()),
    "slides": (["scripts/build_presentation.py"], ["presentation/index.html"], set()),
    "slides-midterm": (
        [
            "scripts/build_presentation.py",
            "--deck",
            "presentation/deck_source/deck_midterm.json",
            "--output",
            "presentation/midterm.html",
        ],
        ["presentation/midterm.html"],
        set(),
    ),
    "manifest": (["scripts/manifest_results.py"], ["docs/evidence/historical_results_manifest.json"], set()),
    "audit": (
        ["scripts/audit_project.py", "--deep"],
        ["results/maintenance/audit.json"],
        # where it ran and which local models exist are facts about the machine, not the code
        {
            "audited_at",
            "root",
            "data_dir",
            "missing_model_artifact_count",
            "missing_model_artifacts",
            "missing_processed_data",
            "replay_and_training_ready",
        },
    ),
}

# Post-midterm studies: each analysis script rebuilds its tables, figures and summary from the study's records
# (no EEG). Compared: the top-level generated files of the study directory (records and plans are inputs).
STUDY_ANALYSES = {
    "alignment": "alignment_study",
    "alignment-mechanism": "alignment_mechanism_study",
    "chdrop": "chdrop_study",
    "electrode-data": "electrode_data_study",
    "fresh-noise": "fresh_noise_study",
    "input-cost": "input_cost_study",
    "input-kind": "input_kind_study",
    "input-penalty": "input_penalty_study",
    "int8-capture": "int8_capture_study",
    "int8-kernel-x86": "int8_kernel_x86_study",
    "int8-layers": "int8_layers_study",
    "int8-paths": "int8_paths_study",
    "long-training": "long_training_study",
    "mcu": "mcu_study",
    "mcu-kernel": "mcu_kernel_study",
    "noise-mechanism": "noise_mechanism_study",
    "ort-kernel": "ort_kernel_study",
    "overfit": "overfit_study",
    "pretrain": "pretrain_study",
    "quant": "quant_study",
    "quant-rule": "quant_rule_study",
    "region": "region_study",
    "replication": "replication_study",
    "replication-riemann": "replication_riemann_study",
    "residual-surrogate": "residual_surrogate_study",
    "scalefree-penalty": "scalefree_penalty_study",
    "selection": "selection_study",
}
GENERATED = (".md", ".csv", ".png")


def study_outputs(study: str) -> list[str]:
    folder = ROOT / "results" / "supplementary" / study
    files = sorted(p for p in folder.glob("*") if p.is_file() and (p.suffix in GENERATED or p.name == "analysis.json"))
    return [str(p.relative_to(ROOT)) for p in files]


for _name, _study in STUDY_ANALYSES.items():
    if study_outputs(_study):  # a study whose analysis has not run yet has nothing to compare
        GENERATORS[f"study-{_name}"] = ([f"scripts/analyze_{_study}.py"], study_outputs(_study), set())
GENERATORS["hw-latency"] = (["scripts/analyze_hw_latency.py"], study_outputs("hw_latency"), set())
GENERATORS["int8-portability"] = (["scripts/analyze_int8_portability.py"], study_outputs("int8_portability"), set())
GENERATORS["compute"] = (
    ["scripts/analyze_compute.py"],
    ["results/supplementary/compute_summary.json", "results/supplementary/compute_summary.md"],
    set(),
)
# last, so that it reads the tables the study generators above have just rebuilt
GENERATORS["course-report"] = (
    ["scripts/build_course_report.py"],
    ["docs/COURSE_REPORT.md", "docs/course_report"],
    set(),
)


def snapshot(paths: list[Path]) -> dict[str, tuple[int, int]]:
    """Size and modification time of every file under `paths`."""
    state = {}
    for base in paths:
        for path in [base] if base.is_file() else base.rglob("*"):
            if path.is_file():
                stat = path.stat()
                state[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return state


def make_sandbox(workdir: Path) -> Path:
    sandbox = workdir / "project"
    sandbox.mkdir(parents=True)
    for name in COPIED:
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, sandbox / name, ignore=IGNORED, symlinks=True)
        elif source.is_file():
            shutil.copy2(source, sandbox / name)
    return sandbox


def strip(obj, volatile: set[str]):
    if isinstance(obj, dict):
        return {k: strip(v, volatile) for k, v in obj.items() if k not in volatile}
    if isinstance(obj, list):
        return [strip(v, volatile) for v in obj]
    return obj


def same(expected: Path, actual: Path, volatile: set[str]) -> bool:
    if expected.suffix == ".json" and volatile:
        return strip(json.loads(expected.read_text()), volatile) == strip(json.loads(actual.read_text()), volatile)
    return filecmp.cmp(expected, actual, shallow=False)


def compare(sandbox: Path, outputs: list[str], volatile: set[str]) -> list[str]:
    problems = []
    for relative in outputs:
        expected_root, actual_root = ROOT / relative, sandbox / relative
        if expected_root.is_file() or actual_root.is_file():
            pairs = [(expected_root, actual_root)]
        else:
            names = {p.relative_to(expected_root) for p in expected_root.rglob("*") if p.is_file()}
            names |= {p.relative_to(actual_root) for p in actual_root.rglob("*") if p.is_file()}
            pairs = [(expected_root / n, actual_root / n) for n in sorted(names) if n.name != ".DS_Store"]
        for expected, actual in pairs:
            shown = expected.relative_to(ROOT)
            if not actual.exists():
                problems.append(f"missing in rebuild: {shown}")
            elif not expected.exists():
                problems.append(f"new file in rebuild: {shown}")
            elif not same(expected, actual, volatile):
                problems.append(f"differs: {shown}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("generators", nargs="*", help="names from --list")
    parser.add_argument("--all", action="store_true", help="run every generator")
    parser.add_argument("--list", action="store_true", help="list generators and exit")
    parser.add_argument("--keep", action="store_true", help="keep the sandbox directory")
    parser.add_argument("--workdir", type=Path, help="parent directory for the sandbox (default: system temp)")
    args = parser.parse_args()
    if args.list:
        for name, (command, outputs, _) in GENERATORS.items():
            print(f"{name:15s} {' '.join(command)}  ->  {', '.join(outputs)}")
        return 0
    names = list(GENERATORS) if args.all else args.generators
    unknown = [n for n in names if n not in GENERATORS]
    if not names or unknown:
        parser.error(f"choose generators from --list (unknown: {unknown})" if unknown else "no generator chosen")

    watched = [ROOT / p for p in ("results", "docs", "presentation")]
    before = snapshot(watched)
    workdir = Path(tempfile.mkdtemp(prefix="mibci-golden-", dir=args.workdir))
    failures = 0
    try:
        sandbox = make_sandbox(workdir)
        env = {
            **os.environ,  # noqa: TID251  the generators inherit the caller's environment
            "PYTHONPATH": str(sandbox / "src"),
            "MIBCI_RESULTS_DIR": str(sandbox / "results"),
            "MIBCI_DATA_DIR": os.environ.get("MIBCI_DATA_DIR", str(ROOT / "data.nosync")),  # noqa: TID251  read in place
            "MPLBACKEND": "Agg",
        }
        for name in names:
            command, outputs, volatile = GENERATORS[name]
            start = time.perf_counter()
            python = "python3" if name in SYSTEM_PYTHON else str(PYTHON)
            run = subprocess.run([python, *command], cwd=sandbox, env=env, capture_output=True, text=True, check=False)
            took = time.perf_counter() - start
            if run.returncode != 0:
                failures += 1
                print(f"FAIL {name}: exit {run.returncode} after {took:.0f} s\n{run.stderr[-3000:]}")
                continue
            problems = compare(sandbox, outputs, volatile)
            failures += bool(problems)
            status = "SAME" if not problems else "DIFF"
            print(f"{status} {name} ({took:.0f} s)" + "".join(f"\n     {p}" for p in problems))
    finally:
        if args.keep:
            print(f"sandbox kept at {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)
    if snapshot(watched) != before:
        print("ERROR: the repository changed while the generators ran; inspect git status")
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
