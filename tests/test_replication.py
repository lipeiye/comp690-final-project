"""Two-class EEGNet bundles and fake quantization (mibci.replication)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mibci import fakequant, replication
from mibci.model import build_eegnet, deploy_copy
from mibci.train import ChannelStandardizer, predict


class TwoClassTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = build_eegnet(6, n_classes=2).eval()
        self.X = np.random.default_rng(0).standard_normal((5, 6, 500)).astype(np.float32)

    def test_bundle_round_trip(self) -> None:
        scaler = ChannelStandardizer().fit(self.X)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "m.pt"
            replication.save(path, self.model, [0, 2, 3, 5, 7, 9], scaler, subject=1)
            model, loaded, bundle = replication.load(path)
        self.assertEqual(bundle["n_classes"], 2)
        np.testing.assert_array_equal(predict(model, self.X), predict(self.model, self.X))
        np.testing.assert_array_equal(loaded.transform(self.X), scaler.transform(self.X))

    def test_folded_matches_the_network(self) -> None:
        folded = replication.folded(self.model)
        np.testing.assert_allclose(fakequant.logits(folded, self.X), predict(self.model, self.X), atol=1e-5)
        self.assertIs(fakequant.deploy_copy, deploy_copy)  # the pinned module is restored

    def test_pinned_copy_assumes_four_classes(self) -> None:
        """The reason this module exists: the pinned deploy copy cannot hold a two-class network."""
        with self.assertRaises(RuntimeError):
            deploy_copy(self.model)


if __name__ == "__main__":
    unittest.main()
