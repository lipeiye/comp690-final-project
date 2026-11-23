"""kernels/ort_conv_u8.c: the uint8 operators equal the archived int8 operators shifted by 128.

Needs ONNX Runtime's C API headers and a C compiler: set MIBCI_ORT_HEADERS to the headers' directory
(skipped otherwise). The int8 operators (kernels/ort_grouped_conv.c, kernels/ort_spatial_first_conv.c) were
checked against ONNX Runtime's own INT8 on all 459 models; with x_u8 = x_s8 + 128 and zero points shifted the
same way, sum w * (x - zero) is unchanged, so the uint8 output must be the int8 output plus 128, exactly."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from mibci import ortkernel, ortkernel_swapped, ortkernel_u8

HEADERS = os.environ.get("MIBCI_ORT_HEADERS")  # noqa: TID251  an optional test input, not a project setting


def graph(op: str, domain: str, inputs: dict, x_type: int, attrs: dict) -> bytes:
    """A one-node graph: x is the graph input, everything else an initializer."""
    names = ["x", "x_scale", "x_zero", "w", "w_scale", "w_zero", "y_scale", "y_zero", "b"]
    node = helper.make_node(op, [n for n in names if n == "x" or n in inputs], ["y"], domain=domain, **attrs)
    inits = [numpy_helper.from_array(v, k) for k, v in inputs.items()]
    g = helper.make_graph([node], "g", [helper.make_tensor_value_info("x", x_type, None)],
                          [helper.make_tensor_value_info("y", x_type, None)], inits)  # fmt: skip
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid(domain, 1)], ir_version=9)
    return m.SerializeToString()


def run(model: bytes, lib: Path, x: np.ndarray) -> np.ndarray:
    opts = ort.SessionOptions()
    opts.register_custom_ops_library(str(lib))
    return ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]


@unittest.skipUnless(HEADERS, "set MIBCI_ORT_HEADERS to ONNX Runtime's C API headers")
class Uint8Kernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        cls.u8 = ortkernel_u8.build(Path(HEADERS), d / "u8.so")
        cls.grouped = ortkernel.build(Path(HEADERS), d / "grouped.so")
        cls.swapped = ortkernel_swapped.build(Path(HEADERS), d / "swapped.so")
        cls.rng = np.random.default_rng(0)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def params(self, m: int, h: int, bias: bool) -> dict:
        w = self.rng.integers(-127, 128, (m, 1, h, 1), np.int8)
        p = {"x_scale": np.float32(0.05), "x_zero": np.int8(-3), "w": w,
             "w_scale": self.rng.uniform(0.001, 0.01, m).astype(np.float32), "w_zero": np.zeros(m, np.int8),
             "y_scale": np.float32(0.4), "y_zero": np.int8(5)}  # fmt: skip
        if bias:
            p["b"] = self.rng.integers(-5000, 5000, m).astype(np.int32)
        return p

    @staticmethod
    def shifted(p: dict) -> dict:
        q = dict(p)
        for k in ("x_zero", "y_zero"):
            q[k] = np.uint8(int(p[k]) + 128)
        return q

    def check(self, op, domain_s8, lib_s8, p, x, attrs):
        s8 = run(graph(op, domain_s8, p, TensorProto.INT8, attrs), lib_s8, x)
        xu = (x.astype(np.int16) + 128).astype(np.uint8)
        u8 = run(graph(op, ortkernel_u8.DOMAIN, self.shifted(p), TensorProto.UINT8, attrs), self.u8, xu)
        np.testing.assert_array_equal(u8.astype(np.int16), s8.astype(np.int16) + 128)
        self.assertGreater(len(np.unique(s8)), 10)  # not all clamped

    def test_grouped_spatial_convolution(self):
        for g, d in ((8, 2), (4, 3)):  # EEGNet's shape (the int8 file's NEON path) and another (its C loop)
            p = self.params(g * d, 22, bias=True)
            x = self.rng.integers(-128, 128, (3, 22, 37, g), np.int8)
            self.check(ortkernel.OP, ortkernel.DOMAIN, self.grouped, p, x, {"group": g})

    def test_spatial_first_convolution(self):
        for m, bias in ((16, False), (5, True)):
            p = self.params(m, 22, bias=bias)
            x = self.rng.integers(-128, 128, (2, 22, 41, 1), np.int8)
            self.check(ortkernel_swapped.OP, ortkernel_swapped.DOMAIN, self.swapped, p, x, {})


if __name__ == "__main__":
    unittest.main()
