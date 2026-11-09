import json
from pathlib import Path
import tempfile
import unittest

from mibci.artifacts import ArtifactError, completed, write_json


class ArtifactRecoveryTests(unittest.TestCase):
    def test_orphan_record_never_silently_triggers_retraining(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"record.json"
            write_json(path,{"seed":0})
            before=path.read_bytes()
            with self.assertRaisesRegex(ArtifactError,"Historical results"):
                completed(path,[Path(temp)/"missing.pt"])
            self.assertEqual(path.read_bytes(),before)

    def test_resumes_only_complete_matching_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            record=Path(temp)/"record.json";model=Path(temp)/"model.pt"
            self.assertFalse(completed(record,[model]))
            write_json(record,{"seed":0});model.write_bytes(b"nonempty fixture")
            self.assertTrue(completed(record,[model],{"seed":0}))
            with self.assertRaises(ArtifactError):completed(record,[model],{"seed":1})
            model.write_bytes(b"")
            with self.assertRaises(ArtifactError):completed(record,[model])

    def test_corrupt_json_and_nonfinite_writes_preserve_existing_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"record.json";path.write_text("{broken")
            with self.assertRaises(ArtifactError):completed(path)
            write_json(path,{"valid":True});before=path.read_bytes()
            with self.assertRaises(ValueError):write_json(path,{"bad":float("nan")})
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(json.loads(path.read_text()),{"valid":True})


if __name__=="__main__":unittest.main()
