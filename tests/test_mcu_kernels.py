"""mibci.mcu_kernels: the FIR kernel's arithmetic equals the integer reference of mibci.mcu, in both layer orders."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mibci import mcu, mcu_kernels
from mibci.model import build_eegnet
from mibci.train import set_seed


def quantized(order: str, n_chans: int = 6) -> tuple[dict, np.ndarray]:
    """A random EEGNet (random BatchNorm statistics, so every layer has a bias), int8, and int8 input."""
    set_seed(7)
    model = build_eegnet(n_chans)
    gen = torch.Generator().manual_seed(8)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.copy_(torch.randn(m.num_features, generator=gen))
                m.running_var.copy_(torch.rand(m.num_features, generator=gen) + 0.5)
                m.bias.copy_(torch.randn(m.num_features, generator=gen))
    X = np.random.default_rng(3).standard_normal((4, n_chans, 500)).astype(np.float32)
    net = mcu.export(model.eval(), order)
    qnet = mcu.quantize(net, mcu.calibrate(net, X))
    return qnet, mcu.quantize_input(X, qnet)


def reference_layer(q: dict, x: np.ndarray) -> np.ndarray:
    """One weighted layer exactly as mibci.mcu.int_forward computes it."""
    shifted = (x.astype(np.int64) - q["in_zero"]).astype(np.float64)
    w = q["w"].astype(np.float64)
    if q["op"] == "conv":
        acc = mcu._conv(shifted, {"w": w, "pad": q["pad"]})
    else:
        acc = mcu._dw(shifted, {"w": w, "pad": q["pad"], "mult": q["mult"]})
    acc = np.rint(acc).astype(np.int64) + q["b"]
    return np.clip(mcu.requantize(acc, q["multiplier"], q["shift"]) + q["out_zero"], -128, 127).astype(np.int8)


class FirKernel(unittest.TestCase):
    def test_original_order_temporal_layer(self):
        qnet, xq = quantized("original")
        expected = reference_layer(qnet["layers"][0], xq)
        np.testing.assert_array_equal(mcu_kernels.fir_layer(qnet, xq), expected)

    def test_swapped_order_temporal_layer(self):
        qnet, xq = quantized("swapped")
        layer_input = reference_layer(qnet["layers"][0], xq)  # the 1x1 spatial convolution first
        expected = reference_layer(qnet["layers"][1], layer_input)
        np.testing.assert_array_equal(mcu_kernels.fir_layer(qnet, layer_input), expected)

    def test_taps_are_filters_by_taps(self):
        original, swapped = quantized("original")[0], quantized("swapped")[0]
        self.assertEqual(mcu_kernels.fir_taps(original).shape, (8, 64))
        self.assertEqual(mcu_kernels.fir_taps(swapped).shape, (16, 64))
        np.testing.assert_array_equal(mcu_kernels.fir_taps(swapped)[3], swapped["layers"][1]["w"][0, :, 3])

    def test_header_holds_both_widths_and_sums(self):
        qnet, _ = quantized("swapped")
        with tempfile.TemporaryDirectory() as d:
            mcu_kernels.write_header(Path(d), qnet)
            text = (Path(d) / "fir.h").read_text()
        self.assertIn("#define FIR_FILTERS 16", text)
        self.assertIn("static const int16_t fir_w16[1024]", text)
        self.assertIn(f"{int(mcu_kernels.fir_taps(qnet)[0].astype(np.int64).sum())}", text.split("fir_wsum")[1])


if __name__ == "__main__":
    unittest.main()
