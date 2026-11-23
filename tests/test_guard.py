"""Each guard check must fail on the real mistake that made us write it (docs/GUARDRAILS.md, section 5)."""

import ast
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import guard


class GuardCatchesRealMistakes(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for directory in guard.CODE_DIRS:
            (self.root / directory).mkdir()
        (self.root / "results").mkdir()
        self.patch = patch.object(guard, "ROOT", self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def write(self, relative: str, text: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def trees(self):
        return {p: ast.parse(p.read_text()) for p in guard.python_files()}

    def test_an_import_reorder_in_a_pinned_script_is_caught(self):
        original = "import sys\nimport json\n"
        pinned = self.write("scripts/run_matched_control.py", original)
        digest = hashlib.sha256(original.encode()).hexdigest()
        study = self.root / "results/supplementary/matched_budget"
        (study / "runs").mkdir(parents=True)
        pins = {"source_code_sha256": {"scripts/run_matched_control.py": digest}}
        (study / "protocol.json").write_text(json.dumps(pins))
        self.assertEqual(guard.check_pinned_source(), [])
        pinned.write_text("import json\nimport sys\n")  # what `ruff check --fix` did on 2026-11-20
        self.assertIn("pins its SHA-256", guard.check_pinned_source()[0])

    def test_a_plan_that_never_ran_pins_nothing(self):
        self.write("scripts/old.py", "x = 2\n")
        study = self.root / "results/supplementary/unexecuted"
        study.mkdir(parents=True)
        (study / "protocol.json").write_text(json.dumps({"source_code_sha256": {"scripts/old.py": "0" * 64}}))
        self.assertEqual(guard.check_pinned_source(), [])

    def test_a_script_importing_another_script_is_caught(self):
        self.write("scripts/summarize_validation.py", "def parse_condition(c):\n    return c\n")
        self.write("scripts/analyze.py", "from summarize_validation import parse_condition\n")
        errors = guard.check_script_imports(self.trees())
        self.assertEqual(len(errors), 1)
        self.assertIn("summarize_validation", errors[0])

    def test_a_script_without_argparse_is_caught(self):
        self.write("scripts/build_report.py", '"""Build the report."""\n\ndef main():\n    pass\n\n'
                   'if __name__ == "__main__":\n    main()\n')  # fmt: skip
        errors = guard.check_script_contract(self.trees())
        self.assertTrue(any("argparse" in e for e in errors))

    def test_zip_strict_in_a_bare_python3_script_is_caught(self):
        self.write("pyproject.toml", '[tool.ruff.per-file-target-version]\n"scripts/audit_project.py" = "py39"\n')
        self.write("scripts/audit_project.py", "pairs = list(zip([1], [2], strict=True))\n")
        self.write("scripts/other.py", "pairs = list(zip([1], [2], strict=True))\n")
        errors = guard.check_py39(self.trees())
        self.assertEqual(len(errors), 1)
        self.assertIn("audit_project.py", errors[0])

    def test_readme_index_needs_the_whole_name(self):
        self.write("scripts/redundancy_check.py", "")
        self.write("tools/check.py", "")
        self.write("README.md", "| `redundancy_check.py` | ... |\n")
        errors = guard.check_doc_index()
        self.assertEqual(len(errors), 1)
        self.assertIn("tools/check.py", errors[0])

    def test_lint_exceptions_need_a_reason(self):
        self.write("src/a.py", "import os  # noqa: F401\n")
        self.write("src/b.py", "import os  # noqa: F401  re-exported for the scripts\n")
        self.write("src/c.py", 'message = "use # noqa sparingly"\n')
        self.write("src/d.py", "import os  # noqa\n")
        errors = guard.check_noqa_reasons()
        self.assertEqual(sorted(e.split(":")[0] for e in errors), ["src/a.py", "src/d.py"])

    def test_opening_the_test_day_outside_the_allowlist_is_caught(self):
        self.write("scripts/tune.py", "from mibci import data\nses = data.load_session(1, 2, final_evaluation=True)\n")
        self.assertEqual(len(guard.check_session2_access(self.trees())), 1)

    def test_machine_paths_are_caught(self):
        machine_path = "/".join(["", "Users", "someone", "project", "results"])  # assembled: not a literal here
        self.write("scripts/plot.py", f'OUT = "{machine_path}"\n')
        self.assertEqual(len(guard.check_hardcoded_paths(self.trees())), 1)


if __name__ == "__main__":
    unittest.main()
