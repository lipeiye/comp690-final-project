import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mibci import data

SCRIPT=Path(__file__).resolve().parents[1]/"scripts/run_matched_control.py"
spec=importlib.util.spec_from_file_location("matched_test_module",SCRIPT)
matched=importlib.util.module_from_spec(spec)
spec.loader.exec_module(matched)


class SupplementProtocolTests(unittest.TestCase):
    def fixture(self,root):
        frozen={"epochs":475,"seeds":[0],"pruning":{"bases":["full-22","relevance-16"]}}
        (root/"frozen_config.json").write_text(json.dumps(frozen))
        (root/"final/rankings").mkdir(parents=True)
        (root/"final/runs").mkdir(parents=True)
        subsets={"full-22":list(range(22)),"relevance-16":list(range(16))}
        (root/"final/rankings/sub-01_seed-0.json").write_text(json.dumps({"subsets":subsets}))
        for condition,indices in subsets.items():
            (root/f"final/runs/sub-01_seed-0_{condition}.json").write_text(json.dumps(
                {"channels":[data.EEG_CHANNELS[i] for i in indices]}))

    def test_missing_original_weights_require_explicit_reproduction_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            with patch.object(matched,"RESULTS_DIR",root),patch.object(matched,"OUT",root/"supp"),\
                 patch.object(matched,"MODELS",root/"models"),patch.object(matched,"MODELS_DIR",root/"original_models"),\
                 patch.object(data,"SUBJECTS",[1]):
                with self.assertRaises((RuntimeError,FileNotFoundError)):
                    matched.create_plan(False)
                self.assertFalse((root/"supp/protocol.json").exists())
                matched.create_plan(True)
                plan=matched.load_plan()
                self.assertEqual(plan["base_origin"],"independent_reproduction")
                self.assertTrue(plan["historical_test_results_already_known"])
                self.assertEqual(plan["fractions"],[0.0,.25,.5])
                self.assertEqual(plan["finetune_epochs"],100)

    def test_changed_historical_ranking_cannot_be_silently_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            with patch.object(matched,"RESULTS_DIR",root),patch.object(matched,"OUT",root/"supp"),\
                 patch.object(matched,"MODELS",root/"models"),patch.object(data,"SUBJECTS",[1]):
                matched.create_plan(True)
                original=(root/"supp/protocol.json").read_bytes()
                (root/"final/rankings/sub-01_seed-0.json").write_text("{}")
                with self.assertRaisesRegex(RuntimeError,"ranking changed"):
                    matched.load_plan()
                self.assertEqual((root/"supp/protocol.json").read_bytes(),original)


if __name__=="__main__":unittest.main()
