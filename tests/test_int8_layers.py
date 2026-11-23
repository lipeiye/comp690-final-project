"""mibci.int8_layers: capturing every int8 tensor, the int16 pair-saturation test and 7-bit layers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import torch

from mibci import deploy, int8_layers
from mibci.model import build_eegnet


class Int8Layers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        model = build_eegnet(6).eval()
        rng = np.random.default_rng(0)
        cls.X = rng.standard_normal((12, 6, 500)).astype(np.float32)
        cls.tmp = tempfile.TemporaryDirectory()
        fp32, int8 = Path(cls.tmp.name) / "m.onnx", Path(cls.tmp.name) / "m_int8.onnx"
        deploy.export_onnx(model, fp32)
        deploy.quantize_int8(fp32, int8, cls.X)
        cls.path = int8
        cls.graph = onnx.load(str(int8))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_capture_returns_every_int8_tensor_and_the_same_logits(self):
        names = int8_layers.quantized_outputs(self.graph)
        self.assertEqual(len(names), 8)  # input, two convolutions, pool, two convolutions, pool, logits
        got = int8_layers.capture(self.graph, self.X[:4])
        expected = deploy.logits(deploy.session(self.path), self.X[:4])
        np.testing.assert_array_equal(got["logits"], expected)
        self.assertTrue(all(got[n].dtype == np.int8 for n in names))

    def test_pair_saturation_finds_sums_beyond_int16(self):
        x = np.full((1, 1, 1, 4), 127, np.int8)  # +128 -> 255 as the x86 kernel takes it
        eight = np.full((1, 1, 1, 4), 127, np.int8)  # 255 * 127 * 2 = 64,770 > 32,767
        seven = np.full((1, 1, 1, 4), 63, np.int8)  # 255 * 63 * 2 = 32,130: fits
        self.assertTrue(int8_layers.pair_saturation(x, 0, eight, 1, (0, 0, 0, 0)).all())
        self.assertFalse(int8_layers.pair_saturation(x, 0, seven, 1, (0, 0, 0, 0)).any())
        mixed = np.array([127, -127, 127, -127], np.int8).reshape(1, 1, 1, 4)  # pairs cancel
        self.assertFalse(int8_layers.pair_saturation(x, 0, mixed, 1, (0, 0, 0, 0)).any())

    def test_seven_bit_layer_keeps_the_graph_and_halves_the_grid(self):
        patched = int8_layers.seven_bit(self.graph, "conv_temporal")
        w = int8_layers.initializer(patched, "conv_temporal.weight_quantized")
        self.assertTrue((w >= -64).all() and (w <= 63).all())
        before = int8_layers.initializer(self.graph, "conv_temporal.weight_scale")
        np.testing.assert_allclose(int8_layers.initializer(patched, "conv_temporal.weight_scale"), 2 * before)
        logits = int8_layers.capture(patched, self.X[:2])["logits"]
        self.assertEqual(logits.shape, (2, 4))

    def test_every_weighted_layer_can_be_made_seven_bit(self):
        for layer in int8_layers.WEIGHTED:
            with self.subTest(layer=layer):
                patched = int8_layers.seven_bit(self.graph, layer)
                prefix = int8_layers.PREFIX.get(layer, layer)
                w = int8_layers.initializer(patched, f"{prefix}.weight_quantized")
                self.assertTrue((w >= -64).all() and (w <= 63).all())
                self.assertEqual(int8_layers.capture(patched, self.X[:2])["logits"].shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
