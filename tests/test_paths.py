"""mibci.paths must agree with the copies of its definitions inside pinned modules."""

import unittest

from mibci import data, experiment, layout, paths


class PathAgreementTests(unittest.TestCase):
    def test_pinned_modules_resolve_the_same_directories(self):
        self.assertEqual(paths.PROJECT_ROOT, data.PROJECT_ROOT)
        self.assertEqual(paths.DATA_DIR, data.DATA_DIR)
        self.assertEqual(paths.RAW_DIR, data.RAW_DIR)
        self.assertEqual(paths.PROCESSED_DIR, data.PROCESSED_DIR)
        self.assertEqual(paths.RESULTS_DIR, experiment.RESULTS_DIR)
        self.assertEqual(paths.MODELS_DIR, experiment.MODELS_DIR)

    def test_main_run_layout_lives_under_the_configured_directories(self):
        self.assertEqual(layout.VAL_DIR.parent, paths.RESULTS_DIR)
        self.assertEqual(layout.FINAL_MODELS.parent, paths.MODELS_DIR)

    def test_display_never_fails_outside_the_project(self):
        self.assertEqual(paths.display(paths.PROJECT_ROOT / "results"), "results")
        self.assertTrue(paths.display(paths.PROJECT_ROOT.parent).startswith("/"))


if __name__ == "__main__":
    unittest.main()
