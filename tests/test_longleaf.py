"""tools/longleaf.py must never put an older Longleaf copy over a file this Mac already has."""

import argparse
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import longleaf


class PullKeepsLocalFiles(unittest.TestCase):
    def commands(self, models: bool) -> list[list[str]]:
        calls = []
        with (
            patch.object(longleaf.subprocess, "run", side_effect=lambda cmd, **kw: calls.append(cmd)),
            patch.object(longleaf.Path, "mkdir"),
        ):
            longleaf.pull(argparse.Namespace(study="noise_mechanism_study", models=models))
        return calls

    def test_pulling_records_never_replaces_a_local_file(self):
        # 2026-11-22: the Longleaf summary.md (02:41) was older than the local one (03:01)
        (command,) = self.commands(models=False)
        self.assertIn("--ignore-existing", command)

    def test_pulling_models_never_replaces_a_local_model(self):
        commands = self.commands(models=True)
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertIn("--ignore-existing", command)


class PushCarriesPinnedSources(unittest.TestCase):
    def test_every_file_a_study_plan_pins_is_pushed(self):
        # 2026-11-22: int8_kernel_x86_study pins kernels/*.c, which push did not copy to Longleaf
        root = Path(__file__).resolve().parents[1]
        for protocol in sorted((root / "results").rglob("protocol.json")):
            for relative in json.loads(protocol.read_text()).get("source_code_sha256", {}):
                with self.subTest(protocol=protocol.parent.name, file=relative):
                    self.assertTrue(relative in longleaf.CODE or relative.split("/")[0] in longleaf.CODE)


if __name__ == "__main__":
    unittest.main()
