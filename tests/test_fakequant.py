import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mibci import deploy, fakequant
from mibci.model import build_eegnet
from mibci.train import predict, set_seed


def trained_like(n_chans: int = 8):
    """A random EEGNet with non-trivial BatchNorm statistics, as after training."""
    set_seed(3)
    model = build_eegnet(n_chans)
    for bn in (model.bnorm_temporal, model.bnorm_1, model.bnorm_2):
        bn.running_mean.uniform_(-0.5, 0.5)
        bn.running_var.uniform_(0.5, 2.0)
        bn.weight.data.uniform_(0.5, 1.5)
        bn.bias.data.uniform_(-0.2, 0.2)
    return model.eval()


class FakeQuantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model = trained_like()
        cls.X = np.random.default_rng(4).normal(size=(64, 8, 500)).astype(np.float32)

    def test_folding_keeps_the_function(self):
        folded = fakequant.Folded(self.model)
        np.testing.assert_allclose(fakequant.logits(folded, self.X), predict(self.model, self.X), atol=1e-4)

    def test_wide_grids_change_nothing_and_narrow_grids_change_something(self):
        folded = fakequant.Folded(self.model)
        ranges = fakequant.calibrate(folded, self.X)
        reference = fakequant.logits(folded, self.X)
        wide = fakequant.logits(fakequant.quantized(folded, ranges, 16, 16), self.X)
        narrow = fakequant.logits(fakequant.quantized(folded, ranges, 2, 2), self.X)
        np.testing.assert_allclose(wide, reference, atol=2e-2)
        self.assertGreater(np.abs(narrow - reference).mean(), 10 * np.abs(wide - reference).mean())

    def test_eight_bit_simulation_tracks_onnx_runtime_int8(self):
        folded = fakequant.Folded(self.model)
        ranges = fakequant.calibrate(folded, self.X)
        simulated = fakequant.logits(fakequant.quantized(folded, ranges, 8, 8), self.X)
        with tempfile.TemporaryDirectory() as temp:
            fp32, int8 = Path(temp) / "m.onnx", Path(temp) / "m_int8.onnx"
            deploy.export_onnx(self.model, fp32)
            deploy.quantize_int8(fp32, int8, self.X)
            runtime = deploy.logits(deploy.session(int8), self.X)
        reference = predict(self.model, self.X)
        sim_error, ort_error = np.abs(simulated - reference).mean(), np.abs(runtime - reference).mean()
        self.assertLess(np.abs(simulated - runtime).mean(), 2 * max(sim_error, ort_error))
        self.assertGreater((simulated.argmax(1) == runtime.argmax(1)).mean(), 0.9)

    def test_bit_operations_scale_with_bits(self):
        full = fakequant.bit_operations(self.model, None, None)["bops"]
        int8 = fakequant.bit_operations(self.model, 8, 8)["bops"]
        self.assertEqual(full, int8 * 16)

    def test_quantization_aware_finetuning_runs_and_keeps_shapes(self):
        folded = fakequant.Folded(self.model)
        ranges = fakequant.calibrate(folded, self.X)
        q = fakequant.quantized(folded, ranges, 4, 8)
        y = np.random.default_rng(5).integers(0, 4, 64)
        tuned = fakequant.finetune(q, self.X, y, epochs=2, lr=1e-3, seed=0)
        out = fakequant.logits(tuned, self.X)
        self.assertEqual(out.shape, (64, 4))
        self.assertTrue(np.isfinite(out).all())
        self.assertFalse(tuned.track_ranges)


if __name__ == "__main__":
    unittest.main()
