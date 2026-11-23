"""Project invariants that a generic linter cannot see. Standard library only.

Each check turns a rule that used to live only in agent.md into an error with a fix:

  pinned-source     files pinned by an executed study's protocol.json keep their SHA-256
  session2-access   only allowlisted files may open the held-out test day
  script-imports    scripts do not import other scripts; shared code lives in src/mibci
  hardcoded-paths   no machine-specific absolute paths in code
  script-contract   every script has a docstring, argparse (so --help is safe) and a main guard
  study-labels      every study protocol says whether test results were already known
  frozen-evidence   historical result files match docs/evidence/historical_results_manifest.json
  doc-links         local links in the Markdown guides point at files that exist
  doc-index         every script and tool is listed in README.md
  noqa-reasons      every lint exception says why, on the same line
  py39-compat       files run with a bare python3 (3.9 here) avoid 3.10+ features

    python3 tools/guard.py            # all checks
    python3 tools/guard.py --fast     # skip frozen-evidence (hashes ~3,000 files)

The allowlists below are the single place to change when a new file legitimately needs
an exception; the reason goes next to the entry.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import re
import sys
import tokenize
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ("src", "scripts", "tests", "tools")

# Files that may read session 2 (the held-out test day). Everything else must leave
# data.load_session at its default, which raises PermissionError for session 2.
SESSION2_READERS = {
    "scripts/evaluate_test.py": "the frozen one-time final evaluation",
    "scripts/replay_demo.py": "replay of recorded EEG, labelled as such",
    "scripts/run_matched_control.py": "post-test matched-budget study (pinned)",
    "scripts/run_electrode_study.py": "post-test electrode-count study (pinned)",
    "scripts/int8_study.py": "post-test scoring of deployment variants",
    "scripts/analyze_electrode_study.py": "post-test diagnostics, labelled in its outputs",
    "scripts/region_study.py": "post-test region study; session 2 only in its evaluate stage",
    "scripts/quant_study.py": "post-test bit-width study; session 2 for reporting only",
    "scripts/quant_rule_study.py": "post-test check of session-1 precision maps; session 2 scores them",
    "scripts/selection_control_study.py": "post-test hardware control for the selection study; session 2 in evaluate",
    "scripts/selection_study.py": "post-test selection study; session 2 only in its evaluate stage",
    "scripts/alignment_study.py": "post-test alignment study; unlabeled session-2 trials align, labels score",
    "scripts/replication_riemann_study.py": "prospective Lee2019 Riemannian-selection test; session 2 in evaluate",
    "scripts/replication_study.py": "prospective Lee2019 replication; session 2 only in its evaluate stage",
    "scripts/int8_portability.py": "re-scores the archived graphs on another CPU; compares with archived predictions",
    "scripts/overfit_study.py": "post-test test of the overfitting explanation; session 2 only in its evaluate stage",
    "scripts/mcu_study.py": "post-test microcontroller study; session-2 windows as emulator inputs and for accuracy",
    "scripts/electrode_data_study.py": "post-test electrode count x training data; session 2 only in evaluate",
    "scripts/ort_kernel_study.py": "post-test INT8 kernel check; session 2 only to compare kernel and runtime outputs",
    "scripts/ort_kernel_swapped_study.py": "post-test second INT8 kernel; session 2 only to compare outputs",
    "scripts/input_cost_study.py": "post-test cost of extra inputs; session 2 only in its evaluate stage",
    "scripts/noise_mechanism_study.py": "post-test mechanism of the input cost; session 2 only in its evaluate stage",
    "scripts/input_penalty_study.py": "post-test input penalty; lam chosen on session 1, session 2 only in evaluate",
    "scripts/input_kind_study.py": "post-test kinds of extra input; session 2 only in its evaluate stage",
    "scripts/residual_surrogate_study.py": "post-test information and cost of dropped electrodes; session 2 evaluate",
    "scripts/fresh_noise_study.py": "post-test noise drawn afresh every epoch; session 2 only in its evaluate stage",
    "scripts/long_training_study.py": "post-test four-times-longer training; session 2 only in its evaluate stage",
    "scripts/scalefree_penalty_study.py": "post-test scale-free input penalty; lam on session 1, session 2 in evaluate",
    "scripts/pretrain_study.py": "post-test pretraining on other people; session 2 only in its evaluate stage",
    "scripts/chdrop_study.py": "post-test channel-dropout mechanism; session 2 only in its evaluate stage",
    "scripts/alignment_mechanism_study.py": "post-test alignment variants; unlabeled session-2 trials align, labels",
    "scripts/mcu_kernel_study.py": "post-test MCU kernel; session-2 windows only as emulator input, bit-exact",
    "scripts/int8_kernel_x86_study.py": "post-test kernels on x86; session 2 only to compare archived predictions",
    "tests/test_pipeline.py": "asserts that the lock raises",
    "tests/test_lee2019.py": "asserts that the second dataset's lock raises",
    "src/mibci/lee2019.py": "defines the second dataset's locked loader",
}
# Script-to-script imports kept only because the importing file is pinned.
SCRIPT_IMPORT_ALLOW = {("scripts/run_electrode_curve.py", "run_electrode_study")}
PATH_PATTERN = re.compile(r"(/Users/|/home/|/private/tmp|/var/folders|C:\\\\)")
DOCS = ("README.md", "agent.md", "AGENTS.md", "docs/GUARDRAILS.md", "docs/REPRODUCIBILITY.md")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def python_files() -> Iterator[Path]:
    for directory in CODE_DIRS:
        for path in sorted((ROOT / directory).rglob("*.py")):
            if "__pycache__" not in path.parts:
                yield path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def executed_protocols() -> Iterator[tuple[Path, dict]]:
    """Protocols of studies that produced records; plans that never ran pin nothing."""
    for path in sorted((ROOT / "results").rglob("protocol.json")):
        if any((path.parent / sub).is_dir() for sub in ("runs", "test")):
            yield path, json.loads(path.read_text())


def pinned_files() -> dict[str, list[tuple[str, str]]]:
    """relative path -> [(expected SHA-256, protocol that pins it)]."""
    pins: dict[str, list[tuple[str, str]]] = {}
    for path, plan in executed_protocols():
        for relative, digest in plan.get("source_code_sha256", {}).items():
            pins.setdefault(relative, []).append((digest, rel(path)))
    return pins


def check_pinned_source() -> list[str]:
    errors = []
    for relative, pins in sorted(pinned_files().items()):
        current = sha256(ROOT / relative) if (ROOT / relative).exists() else None
        for digest, protocol in pins:
            if current != digest:
                errors.append(
                    f"{relative}: changed, but {protocol} pins its SHA-256. That study's runs were made with the "
                    "pinned bytes; restore the file (git checkout) and put new behaviour in a new module or study."
                )
    return errors


def calls(tree: ast.AST, name: str) -> Iterator[ast.Call]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if called == name:
                yield node


def check_session2_access(trees: dict[Path, ast.Module]) -> list[str]:
    errors = []
    for path, tree in trees.items():
        if rel(path) in SESSION2_READERS or rel(path) == "src/mibci/data.py":
            continue
        for call in calls(tree, "load_session"):
            second = call.args[1] if len(call.args) > 1 else None
            unlocks = [k for k in call.keywords if k.arg == "final_evaluation"]
            literal_two = isinstance(second, ast.Constant) and second.value == 2
            opens = any(not (isinstance(k.value, ast.Constant) and k.value.value is False) for k in unlocks)
            if literal_two or opens:
                errors.append(
                    f"{rel(path)}:{call.lineno}: opens the held-out test day (session 2). Selection, training, "
                    "calibration and tuning must use session 1 only. A post-test analysis must be labelled as such "
                    "and added to SESSION2_READERS in tools/guard.py with its reason."
                )
    return errors


def check_script_imports(trees: dict[Path, ast.Module]) -> list[str]:
    scripts = {p.stem for p in (ROOT / "scripts").glob("*.py")}
    errors = []
    for path, tree in trees.items():
        if not rel(path).startswith("scripts/"):
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module.split(".")[0]]
            errors.extend(
                f"{rel(path)}:{node.lineno}: imports the script '{name}'. Move the shared code "
                "into src/mibci and import it from there."
                for name in names
                if name in scripts and (rel(path), name) not in SCRIPT_IMPORT_ALLOW
            )
    return errors


def check_hardcoded_paths(trees: dict[Path, ast.Module]) -> list[str]:
    errors = []
    for path, tree in trees.items():
        if path == Path(__file__).resolve():
            continue
        errors.extend(
            f"{rel(path)}:{node.lineno}: machine-specific absolute path in a string. Build paths "
            "from mibci.paths (PROJECT_ROOT, RESULTS_DIR, DATA_DIR) or take them as arguments."
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and PATH_PATTERN.search(node.value)
        )
    return errors


def check_script_contract(trees: dict[Path, ast.Module]) -> list[str]:
    errors = []
    for path, tree in trees.items():
        if not rel(path).startswith(("scripts/", "tools/")) or path.suffix != ".py":
            continue
        where = rel(path)
        if not ast.get_docstring(tree):
            errors.append(f"{where}: no module docstring. Say what the script reads, writes and refuses to do.")
        if not list(calls(tree, "ArgumentParser")):
            errors.append(f"{where}: no argparse parser, so '--help' would run the script. Parse arguments first.")
        guarded = any(
            isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and getattr(n.test.left, "id", "") == "__name__"
            for n in tree.body
        )
        if not guarded:
            errors.append(f"{where}: no 'if __name__ == \"__main__\":' guard; importing it would run it.")
    return errors


def check_study_labels() -> list[str]:
    errors = []
    for path in sorted((ROOT / "results").rglob("protocol.json")):
        plan = json.loads(path.read_text())
        if not isinstance(plan.get("historical_test_results_already_known"), bool):
            errors.append(f"{rel(path)}: missing boolean 'historical_test_results_already_known'.")
        if not isinstance(plan.get("claim_boundary"), str) or not plan.get("claim_boundary"):
            errors.append(f"{rel(path)}: missing 'claim_boundary' (what this study can and cannot support).")
    return errors


def check_frozen_evidence() -> list[str]:
    manifest = json.loads((ROOT / "docs/evidence/historical_results_manifest.json").read_text())
    errors = []
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"{relative}: historical result missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            errors.append(f"{relative}: historical result changed; restore it, never edit archived evidence")
    return errors


LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def check_doc_links() -> list[str]:
    errors = []
    for name in DOCS:
        doc = ROOT / name
        if not doc.exists():
            continue
        for number, line in enumerate(doc.read_text().splitlines(), 1):
            for target in LINK.findall(line):
                if re.match(r"[a-z]+:", target):
                    continue
                if not (doc.parent / target).exists():
                    errors.append(f"{name}:{number}: link target does not exist: {target}")
    return errors


def check_doc_index() -> list[str]:
    readme = (ROOT / "README.md").read_text()

    def listed(name: str) -> bool:  # as a whole word: "check.py" must not match "redundancy_check.py"
        return re.search(rf"(?<![\w/]){re.escape(name)}", readme) is not None

    missing = [rel(p) for d in ("scripts", "tools") for p in sorted((ROOT / d).glob("*.py")) if not listed(p.name)]
    return [f"{m}: not listed in README.md; add it to the script index with its purpose" for m in missing]


NOQA = re.compile(r"#\s*noqa\b(?::\s*(?P<codes>[A-Z]+[0-9]+(?:\s*,\s*[A-Z]+[0-9]+)*))?(?P<rest>.*)$")


def check_noqa_reasons() -> list[str]:
    """A lint exception must name its rule and give a reason on the same line."""
    pinned = set(pinned_files())
    errors = []
    for path in python_files():
        if rel(path) in pinned:
            continue
        with path.open() as stream:
            comments = [(t.start[0], t.string) for t in tokenize.generate_tokens(stream.readline)
                        if t.type == tokenize.COMMENT]  # fmt: skip
        for number, comment in comments:
            match = NOQA.search(comment)
            if not match:
                continue
            if not match.group("codes"):
                errors.append(f"{rel(path)}:{number}: blanket '# noqa'; name the rule, e.g. '# noqa: E402  reason'")
            elif len(match.group("rest").strip(" ()")) < 3:
                errors.append(f"{rel(path)}:{number}: '# noqa: {match.group('codes')}' gives no reason; add one "
                              "after the rule on the same line")  # fmt: skip
    return errors


def py39_files() -> list[Path]:
    """Files that pyproject.toml lints as Python 3.9 because they run with a bare python3."""
    text = (ROOT / "pyproject.toml").read_text()
    section = re.search(r"\[tool\.ruff\.per-file-target-version\](.*?)(?:\n\[|\Z)", text, re.S)
    patterns = re.findall(r'^"([^"]+)"\s*=\s*"py39"', section.group(1), re.M) if section else []
    return [p for p in python_files() if any(fnmatch.fnmatch(rel(p), pattern) for pattern in patterns)]


def check_py39(trees: dict[Path, ast.Module]) -> list[str]:
    errors = []
    for path in py39_files():
        for node in ast.walk(trees[path]):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "zip":
                if any(k.arg == "strict" for k in node.keywords):
                    errors.append(
                        f"{rel(path)}:{node.lineno}: zip(strict=...) needs Python 3.10, but this file runs with "
                        "a bare python3 (3.9 on the author's Mac); check the lengths explicitly instead"
                    )
            elif type(node).__name__ == "Match":
                errors.append(f"{rel(path)}:{node.lineno}: 'match' needs Python 3.10; this file must run on 3.9")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fast", action="store_true", help="skip the frozen-evidence hash check")
    args = parser.parse_args()
    trees = {p: ast.parse(p.read_text(), filename=str(p)) for p in python_files()}
    checks = {
        "pinned-source": check_pinned_source,
        "session2-access": lambda: check_session2_access(trees),
        "script-imports": lambda: check_script_imports(trees),
        "hardcoded-paths": lambda: check_hardcoded_paths(trees),
        "script-contract": lambda: check_script_contract(trees),
        "study-labels": check_study_labels,
        "doc-links": check_doc_links,
        "doc-index": check_doc_index,
        "noqa-reasons": check_noqa_reasons,
        "py39-compat": lambda: check_py39(trees),
    }
    if not args.fast:
        checks["frozen-evidence"] = check_frozen_evidence
    failed = 0
    for name, check in checks.items():
        errors = check()
        failed += bool(errors)
        print(f"{'ok  ' if not errors else 'FAIL'} {name}" + "".join(f"\n     {e}" for e in errors))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
