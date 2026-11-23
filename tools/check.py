"""Run every code-quality layer, and hold the line on old lint debt with a ratchet.

    python3 tools/check.py              # format, lint ratchet, project guard, unit tests
    python3 tools/check.py --fast       # the same without unit tests (pre-commit, agent hooks)
    python3 tools/check.py --files a.py # only these files: format and lint, no guard or tests
    python3 tools/check.py --todo       # the cleanup queue: old violations still allowed
    python3 tools/check.py --tighten    # after a cleanup: lower the allowed counts to today's

Layers, strongest first (docs/GUARDRAILS.md explains each):
  1. format   `ruff format --check`; no exceptions outside the pinned files
  2. lint     `ruff check` against tools/lint_baseline.json. A file may never gain a
              violation of a rule; old violations are allowed until cleaned. Counts only
              go down (--tighten), never up. New files start clean.
  3. guard    tools/guard.py: project invariants a generic linter cannot see
  4. tests    unittest, with the project's environment

Exit status is non-zero when any layer fails. Ruff runs as `uvx ruff@RUFF_VERSION`, so the
training environment in uv.lock is not touched; set RUFF to use another executable.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tools/lint_baseline.json"
RUFF_VERSION = "0.17.6"
TARGETS = ["src", "scripts", "tests", "tools"]
PYTHON = ROOT / ".venv.nosync/bin/python"

sys.path.insert(0, str(ROOT / "tools"))
import guard  # noqa: E402  (tools/ is not a package; guard is its sibling)


def ruff(*args: str) -> subprocess.CompletedProcess:
    command = shlex.split(os.environ.get("RUFF", f"uvx ruff@{RUFF_VERSION}"))  # noqa: TID251  tool setting, not a path
    return subprocess.run([*command, *args], cwd=ROOT, capture_output=True, text=True, check=False)


def lint_counts(paths: list[str]) -> tuple[Counter, dict[tuple[str, str], list[str]]]:
    run = ruff("check", "--output-format", "json", "--exit-zero", *paths)
    if run.returncode != 0:
        raise SystemExit(f"ruff failed:\n{run.stderr}")
    counts: Counter = Counter()
    where: dict[tuple[str, str], list[str]] = {}
    for item in json.loads(run.stdout or "[]"):
        key = (Path(item["filename"]).resolve().relative_to(ROOT).as_posix(), item["code"] or "syntax")
        counts[key] += 1
        where.setdefault(key, []).append(f"{key[0]}:{item['location']['row']}: {key[1]} {item['message']}")
    return counts, where


def load_baseline() -> Counter:
    if not BASELINE.exists():
        return Counter()
    data = json.loads(BASELINE.read_text())
    return Counter({(path, code): n for path, codes in data["allowed"].items() for code, n in codes.items()})


def save_baseline(counts: Counter) -> None:
    allowed: dict[str, dict[str, int]] = {}
    for (path, code), n in sorted(counts.items()):
        if n > 0:
            allowed.setdefault(path, {})[code] = n
    BASELINE.write_text(
        json.dumps(
            {
                "about": "Lint violations that existed when the ratchet was introduced. tools/check.py fails if any "
                "count rises or a new file/rule pair appears; --tighten lowers counts after a cleanup. "
                "Files pinned by a finished study (tools/guard.py) cannot change, so their entries are frozen.",
                "ruff": RUFF_VERSION,
                "allowed": allowed,
            },
            indent=1,
        )
        + "\n"
    )


def check_format(paths: list[str]) -> list[str]:
    run = ruff("format", "--check", *paths)
    if run.returncode == 0:
        return []
    # ruff prints " --> path:line:col" under each file that would change, then a diff
    files = sorted({line.split("-->", 1)[1].strip().split(":")[0] for line in run.stdout.splitlines() if "-->" in line})
    return [f"{f}: not formatted; run `uvx ruff@{RUFF_VERSION} format {f}`" for f in files] or [run.stdout + run.stderr]


def check_lint(paths: list[str]) -> tuple[list[str], Counter]:
    counts, where = lint_counts(paths)
    allowed = load_baseline()
    errors = []
    for key, n in sorted(counts.items()):
        if n > allowed.get(key, 0):
            extra = n - allowed.get(key, 0)
            errors.append(f"{key[0]}: {extra} new {key[1]} violation(s) (allowed {allowed.get(key, 0)}):")
            errors.extend(f"       {line}" for line in where[key])
    return errors, counts


def tighten(counts: Counter) -> int:
    allowed = load_baseline()
    risen = [k for k, n in counts.items() if n > allowed.get(k, 0)]
    if risen:
        print("Refusing to tighten while violations rose; fix these first:", *risen, sep="\n  ")
        return 1
    lowered = Counter({k: min(n, counts.get(k, 0)) for k, n in allowed.items()})
    before, after = sum(allowed.values()), sum(lowered.values())
    save_baseline(lowered)
    print(f"Baseline lowered from {before} to {after} allowed violations.")
    return 0


def todo() -> int:
    allowed = load_baseline()
    pinned = set(guard.pinned_files())
    by_file: dict[str, list[tuple[str, int]]] = {}
    for (path, code), n in allowed.items():
        by_file.setdefault(path, []).append((code, n))
    open_debt = {p: c for p, c in by_file.items() if p not in pinned}
    print(f"{sum(n for c in open_debt.values() for _, n in c)} cleanable violations in {len(open_debt)} files:")
    for path, codes in sorted(open_debt.items(), key=lambda item: -sum(n for _, n in item[1])):
        print(f"  {sum(n for _, n in codes):4d}  {path}: " + ", ".join(f"{c}x{n}" for c, n in sorted(codes)))
    frozen = sum(n for p, c in by_file.items() if p in pinned for _, n in c)
    if frozen:
        print(f"{frozen} more sit in files pinned by finished studies; they stay until a study is closed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true", help="skip unit tests and the slow guard check")
    parser.add_argument("--files", nargs="+", help="check only these files (format and lint)")
    parser.add_argument("--todo", action="store_true", help="list the old violations still allowed")
    parser.add_argument("--tighten", action="store_true", help="lower allowed counts to the current ones")
    parser.add_argument("--init-baseline", action="store_true", help="record today's violations (first run only)")
    args = parser.parse_args()
    if args.todo:
        return todo()
    if args.init_baseline:
        if BASELINE.exists():
            parser.error("a baseline exists; it may only shrink (use --tighten)")
        save_baseline(lint_counts(TARGETS)[0])
        print(f"Wrote {BASELINE.relative_to(ROOT)}")
        return 0

    if args.tighten and args.files:
        parser.error("--tighten needs the whole tree; run it without --files")
    paths = [p for p in (args.files or TARGETS) if Path(p).suffix == ".py" or (ROOT / p).is_dir()]
    if args.files and not paths:
        return 0
    results = {"format": check_format(paths)}
    results["lint"], counts = check_lint(paths)
    if args.tighten:
        return tighten(counts)
    if not args.files:
        trees = {p: guard.ast.parse(p.read_text(), filename=str(p)) for p in guard.python_files()}
        results["guard"] = [
            *guard.check_pinned_source(),
            *guard.check_session2_access(trees),
            *guard.check_script_imports(trees),
            *guard.check_hardcoded_paths(trees),
            *guard.check_script_contract(trees),
            *guard.check_study_labels(),
            *guard.check_doc_links(),
            *guard.check_doc_index(),
            *guard.check_noqa_reasons(),
            *guard.check_py39(trees),
            *([] if args.fast else guard.check_frozen_evidence()),
        ]
    if not args.files and not args.fast:
        python = str(PYTHON) if PYTHON.exists() else sys.executable
        run = subprocess.run(
            [python, "-m", "unittest", "discover", "-s", "tests"], cwd=ROOT, capture_output=True, text=True, check=False
        )
        results["tests"] = [] if run.returncode == 0 else [run.stderr[-4000:]]
    for layer, errors in results.items():
        print(f"{'ok  ' if not errors else 'FAIL'} {layer}" + "".join(f"\n     {e}" for e in errors))
    allowed, current = load_baseline(), counts
    if not args.files and sum(current.values()) < sum(allowed.values()):
        print(
            f"note: {sum(allowed.values()) - sum(current.values())} old violations are gone; "
            "run `python3 tools/check.py --tighten` to lock that in"
        )
    return 1 if any(results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
