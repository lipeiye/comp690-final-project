"""mibci.int8_capture: integer tensors read from the optimized graph reproduce a normal session."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import torch

from mibci import deploy, int8_capture, int8_layers
from mibci.model import build_eegnet


class Int8Capture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        model = build_eegnet(6).eval()
        cls.X = np.random.default_rng(1).standard_normal((12, 6, 500)).astype(np.float32)
        cls.tmp = tempfile.TemporaryDirectory()
        fp32, int8 = Path(cls.tmp.name) / "m.onnx", Path(cls.tmp.name) / "m_int8.onnx"
        deploy.export_onnx(model, fp32)
        deploy.quantize_int8(fp32, int8, cls.X)
        cls.path, cls.graph = int8, onnx.load(str(int8))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_capture_equals_a_normal_session(self):
        for g in (self.graph, int8_layers.seven_bit(self.graph, "final_layer")):
            c = int8_capture.capture(g, self.X[:4])
            np.testing.assert_array_equal(c["tensors"]["logits"], c["plain"])
        plain = deploy.logits(deploy.session(self.path), self.X[:4])
        np.testing.assert_array_equal(int8_capture.capture(self.graph, self.X[:4])["plain"], plain)

    def test_every_weighted_layer_has_an_integer_kernel_and_a_label(self):
        opt = int8_capture.optimized(self.graph)
        ints = int8_capture.integer_tensors(opt)
        self.assertTrue(all(k in (onnx.TensorProto.INT8, onnx.TensorProto.UINT8) for k in ints.values()))
        made = {int8_capture.labels(opt)[name] for name in int8_capture.producers(opt)}
        self.assertEqual(made, set(int8_capture.LAYERS.values()))


if __name__ == "__main__":
    unittest.main()
