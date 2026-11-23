"""mibci.int8_paths: the shape grid covers EEGNet's layers, and its graphs run as INT8 QLinearConv nodes."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mibci import deploy, int8_paths


class Int8Paths(unittest.TestCase):
    def test_the_grid_holds_eegnets_layers(self):
        names = {s["name"] for s in int8_paths.grid()}
        for layer in ("spatial_g8_m2", "spatial_g1_m16", "temporal_g1_m8", "temporal_g16_m1"):
            self.assertIn(layer, names)
        self.assertTrue(all(s["groups"] * s["per_group"] <= int8_paths.MAX_OUTPUTS for s in int8_paths.grid()))

    def test_macs_count_one_window(self):
        spatial = {"kernel": "spatial", "groups": 8, "per_group": 2}
        self.assertEqual(int8_paths.macs(spatial), 16 * 1 * 500 * 22)
        temporal = {"kernel": "temporal", "groups": 1, "per_group": 8}
        self.assertEqual(int8_paths.macs(temporal), 8 * 22 * 501 * 64)

    def test_the_int8_graph_runs_as_qlinearconv_and_follows_fp32(self):
        shape = {"name": "spatial_g8_m2", "kernel": "spatial", "groups": 8, "per_group": 2}
        X = int8_paths.inputs(shape, 16, seed=1)
        with tempfile.TemporaryDirectory() as d:
            fp32, int8 = Path(d) / "a.onnx", Path(d) / "a_int8.onnx"
            int8_paths.export(shape, fp32, int8, X)
            self.assertIn("QLinearConv", deploy.executed_ops(int8))
            a = deploy.logits(deploy.session(fp32), X[:1], batch=1)  # the graphs are exported for batch 1
            b = deploy.logits(deploy.session(int8), X[:1], batch=1)
        self.assertEqual(a.shape, (1, 16, 1, 500))
        self.assertLess(np.abs(a - b).max(), 0.05 * np.abs(a).max())


if __name__ == "__main__":
    unittest.main()
