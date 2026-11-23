"""mibci.mcu (microcontroller export and integer reference) and mibci.overfit helpers, without EEG data."""

import unittest

import numpy as np
import torch

from mibci import mcu, overfit
from mibci.model import build_eegnet
from mibci.train import set_seed


def random_model(n_chans: int) -> torch.nn.Module:
    """EEGNet with random BatchNorm statistics, so every folded layer carries a nonzero bias."""
    set_seed(5)
    model = build_eegnet(n_chans)
    generator = torch.Generator().manual_seed(6)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.copy_(torch.randn(module.num_features, generator=generator))
                module.running_var.copy_(torch.rand(module.num_features, generator=generator) + 0.5)
                module.weight.copy_(torch.rand(module.num_features, generator=generator) + 0.5)
                module.bias.copy_(torch.randn(module.num_features, generator=generator))
    return model.eval()


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.model = random_model(6)
        self.X = np.random.default_rng(0).standard_normal((5, 6, 500)).astype(np.float32)
        with torch.no_grad():
            self.reference = self.model(torch.from_numpy(self.X)).double().numpy()

    def test_both_orders_reproduce_the_network(self):
        for order in mcu.ORDERS:
            net = mcu.export(self.model, order)
            self.assertLess(np.abs(mcu.float_forward(net, self.X) - self.reference).max(), 1e-4, order)

    def test_costs_follow_the_layer_order(self):
        rows = {order: mcu.shapes(mcu.export(self.model, order), 500) for order in mcu.ORDERS}
        macs = {order: sum(r["macs"] for r in rows[order]) for order in mcu.ORDERS}
        self.assertEqual(rows["original"][0]["macs"], 8 * 6 * 501 * 64)  # temporal filters on every electrode
        self.assertEqual(rows["swapped"][1]["macs"], 16 * 501 * 64)  # temporal filters on 16 spatial signals
        self.assertLess(macs["swapped"], macs["original"])

    def test_int8_follows_float(self):
        net = mcu.export(self.model, "swapped")
        qnet = mcu.quantize(net, mcu.calibrate(net, self.X))
        logits = mcu.dequantize_logits(mcu.int_forward(qnet, mcu.quantize_input(self.X, qnet)), qnet)
        span = np.ptp(self.reference)
        self.assertLess(np.abs(logits - self.reference).max(), 0.1 * span)


class RequantizeTest(unittest.TestCase):
    def test_matches_exact_rounding(self):
        rng = np.random.default_rng(1)
        acc = rng.integers(-(2**20), 2**20, size=2000)
        for m in (0.000731, 0.0123, 0.4999, 0.75):
            multiplier, shift = mcu.quantize_multiplier(m)
            got = mcu.requantize(acc, np.int64(multiplier), np.int64(shift))
            exact = mcu.round_half_away(acc * (multiplier / 2**31) * 2.0**shift)
            self.assertLessEqual(np.abs(got - exact).max(), 1, m)  # two rounding steps may differ by one

    def test_multiplier_reconstructs_scale(self):
        for m in (1e-5, 0.3, 0.9999999, 1.7):
            q, shift = mcu.quantize_multiplier(m)
            self.assertAlmostEqual(q / 2**31 * 2.0**shift, m, delta=m * 1e-8)
            self.assertTrue(2**30 <= q < 2**31)


class OverfitHelpersTest(unittest.TestCase):
    def test_shuffled_channels_take_another_trial(self):
        X = np.arange(10 * 3 * 4, dtype=np.float32).reshape(10, 3, 4)
        out = overfit.shuffled_channels(X, [1], "bnci2a", 1, 1)
        np.testing.assert_array_equal(out[:, [0, 2]], X[:, [0, 2]])
        sources = [next(j for j in range(10) if np.array_equal(out[i, 1], X[j, 1])) for i in range(10)]
        self.assertTrue(all(s != i for i, s in enumerate(sources)))
        self.assertEqual(sorted(sources), list(range(10)))

    def test_noise_is_fixed_by_participant_and_day(self):
        X = np.zeros((4, 3, 500), np.float32)
        a, b = overfit.noise_channels(X, [2], "lee2019", 7, 1), overfit.noise_channels(X, [2], "lee2019", 7, 1)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, overfit.noise_channels(X, [2], "lee2019", 7, 2)))
        self.assertAlmostEqual(float(a[:, 2].std()), 1.0, delta=0.1)

    def test_split_masks(self):
        runs = np.repeat(np.arange(6), 4)
        train, held = overfit.split_masks(runs, "rfold-1", "bnci2a")
        self.assertEqual(sorted(set(runs[train])), [2, 3])
        self.assertEqual(int(held.sum()), 16)
        train, held = overfit.split_masks(np.repeat([0, 1], 5), "fold-1", "lee2019")
        self.assertEqual(int(train.sum()), 5)


if __name__ == "__main__":
    unittest.main()
