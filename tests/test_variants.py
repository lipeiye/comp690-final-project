import platform
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mibci import deploy, timing, variants
from mibci.costs import cost_summary
from mibci.model import build_eegnet
from mibci.prune import prune_pointwise
from mibci.train import set_seed


def random_model(n_chans: int, prune: float = 0.0):
    """EEGNet with random BatchNorm statistics, so the folded convolutions carry nonzero biases."""
    set_seed(3)
    model = build_eegnet(n_chans)
    generator = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.copy_(torch.randn(module.num_features, generator=generator))
                module.running_var.copy_(torch.rand(module.num_features, generator=generator) + 0.5)
                module.weight.copy_(torch.rand(module.num_features, generator=generator) + 0.5)
                module.bias.copy_(torch.randn(module.num_features, generator=generator))
    return (prune_pointwise(model, prune) if prune else model).eval()


def accumulates_in_32_bits() -> bool:
    """Arm dot products and x86 VNNI sum 8-bit products in 32 bits. Without VNNI, ONNX Runtime's x86
    kernels add pairs of products in 16 bits, which can saturate, and then two graphs holding the same
    integers need not give the same logits (results/supplementary/int8_portability)."""
    if platform.machine().lower() in ("arm64", "aarch64"):
        return True
    features = timing.cpu()["features"]
    return bool(features.get("avx512_vnni") or features.get("avx_vnni"))


class DeploymentVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rewrites_keep_the_fp32_function(self):
        for n_chans, prune in ((22, 0.0), (16, 0.5), (8, 0.0)):
            model = random_model(n_chans, prune)
            X = np.random.default_rng(5).normal(size=(6, n_chans, 500)).astype(np.float32)
            with tempfile.TemporaryDirectory() as temp:
                fp32, dense, first = (Path(temp) / f"{n}.onnx" for n in ("fp32", "dense", "first"))
                deploy.export_onnx(model, fp32)
                variants.spatial_dense(fp32, dense)
                variants.spatial_first(fp32, first)
                reference = deploy.logits(deploy.session(fp32), X)
                self.assertGreater(np.abs(reference).max(), 1e-3)
                for path in (dense, first):
                    np.testing.assert_allclose(deploy.logits(deploy.session(path), X), reference, atol=1e-5, rtol=1e-4)

    def test_costs_match_the_cost_model_and_shrink_after_the_swap(self):
        model = random_model(22)
        with tempfile.TemporaryDirectory() as temp:
            fp32, first = Path(temp) / "fp32.onnx", Path(temp) / "first.onnx"
            deploy.export_onnx(model, fp32)
            variants.spatial_first(fp32, first)
            original, swapped = variants.graph_costs(fp32), variants.graph_costs(first)
        expected = cost_summary(model)
        self.assertEqual(original["macs"], expected["macs"])
        self.assertEqual(original["peak_activation_elems"], expected["peak_activation_elems"])
        # 16 spatial filters over 22 electrodes x 500 samples, then 16 temporal filters of 64 taps x 501 outputs
        first_two = 16 * 22 * 500 + 16 * 64 * 501
        later = expected["macs"] - (8 * 64 * 22 * 501 + 16 * 22 * 501)
        self.assertEqual(swapped["macs"], first_two + later)
        self.assertEqual(swapped["peak_activation_elems"], 22 * 500 + 16 * 500)
        self.assertEqual(swapped["params"] - original["params"], 8 * 64 + 8 - 16)

    def test_quantization_variants_run_the_intended_kernels(self):
        model = random_model(8)
        X = np.random.default_rng(6).normal(size=(12, 8, 500)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            fp32 = folder / "fp32.onnx"
            deploy.export_onnx(model, fp32)
            reference_int8 = folder / "int8.onnx"
            deploy.quantize_int8(fp32, reference_int8, X)
            variants.spatial_dense(fp32, folder / "dense.onnx")
            variants.quantize(folder / "dense.onnx", folder / "dense_int8.onnx", X)
            variants.quantize(fp32, folder / "mixed.onnx", X, spatial_fp32=True)
            variants.quantize(fp32, folder / "int16.onnx", X, activation_bits=16)
            # Same integers through a different kernel: the dense rewrite reproduces the INT8 logits,
            # exactly where 8-bit products are summed in 32 bits (see accumulates_in_32_bits).
            dense = deploy.logits(deploy.session(folder / "dense_int8.onnx"), X)
            reference = deploy.logits(deploy.session(reference_int8), X)
            if accumulates_in_32_bits():
                np.testing.assert_array_equal(dense, reference)
            else:
                np.testing.assert_allclose(dense, reference, atol=0.05)
            mixed = deploy.executed_ops(folder / "mixed.onnx")
            self.assertEqual((mixed.get("QLinearConv"), mixed.get("Conv")), (3, 1))
            # ONNX Runtime's CPU provider has no 16-bit convolution kernel: the convolutions stay float.
            int16 = deploy.executed_ops(folder / "int16.onnx")
            self.assertEqual(int16.get("Conv"), 4)
            self.assertNotIn("QLinearConv", int16)
            for name in ("mixed", "int16"):
                out = deploy.logits(deploy.session(folder / f"{name}.onnx"), X)
                self.assertEqual(out.shape, (12, 4))
                self.assertTrue(np.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
