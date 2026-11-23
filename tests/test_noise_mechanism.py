"""mibci.noise_mechanism: noise draws, removing the noise path, re-estimating batch norms, noise shares."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from mibci import noise_mechanism, overfit
from mibci.model import build_eegnet
from mibci.train import set_seed


def network(n_inputs: int = 6, seed: int = 0):
    set_seed(seed)
    return build_eegnet(n_inputs, n_classes=4).eval()


class NoiseDrawTest(unittest.TestCase):
    def test_draw_zero_is_the_input_cost_study_draw(self):
        zeros = np.zeros((5, 16, 500), np.float32)
        expected = overfit.noise_channels(zeros, list(range(16)), "bnci2a", 3, 1)[:, :4]
        got = noise_mechanism.appended_noise(5, 500, 4, 16, "bnci2a", 3, 1, draw=0)
        np.testing.assert_array_equal(got, expected)

    def test_new_draws_are_nested_independent_and_unit_variance(self):
        six = noise_mechanism.appended_noise(40, 500, 6, 16, "bnci2a", 2, 1, draw=1)
        three = noise_mechanism.appended_noise(40, 500, 3, 16, "bnci2a", 2, 1, draw=1)
        np.testing.assert_array_equal(six[:, :3], three)
        other = noise_mechanism.appended_noise(40, 500, 6, 16, "bnci2a", 2, 1, draw=2)
        self.assertFalse(np.allclose(six, other))
        self.assertAlmostEqual(float(six.std()), 1.0, delta=0.05)


class NoisePathTest(unittest.TestCase):
    def setUp(self):
        self.model = network()
        with torch.no_grad():  # move away from the initial weights so the test is not trivial
            self.model.conv_spatial.parametrizations.weight.original.mul_(5.0)
        self.X = np.random.default_rng(0).standard_normal((12, 6, 500)).astype(np.float32)

    def test_removing_the_noise_path_keeps_every_other_weight(self):
        before = self.model.conv_spatial.weight.detach().clone()
        removed = noise_mechanism.without_noise_path(self.model, [4, 5])
        after = removed.conv_spatial.weight.detach()
        self.assertTrue(torch.equal(after[:, :, 4:], torch.zeros_like(after[:, :, 4:])))
        torch.testing.assert_close(after[:, :, :4], before[:, :, :4])
        silent = self.X.copy()
        silent[:, 4:] = 0.0
        with torch.no_grad():
            a = removed(torch.from_numpy(self.X))
            b = self.model(torch.from_numpy(silent))
        torch.testing.assert_close(a, b)

    def test_recalibration_uses_the_training_statistics(self):
        recal = noise_mechanism.recalibrate(self.model, self.X)
        h = noise_mechanism.spatial_input(self.model, self.X)
        with torch.no_grad():
            z = self.model.conv_spatial(h)
        torch.testing.assert_close(recal.bnorm_1.running_mean, z.mean(dim=(0, 2, 3)), rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(self.model.bnorm_temporal.running_var, recal.bnorm_temporal.running_var)

    def test_noise_share_bounds(self):
        share = noise_mechanism.noise_share(self.model, self.X, [4, 5])
        self.assertEqual(share.shape, (16,))
        self.assertTrue(((share > 0) & (share < 1)).all())
        none = noise_mechanism.noise_share(noise_mechanism.without_noise_path(self.model, [4, 5]), self.X, [4, 5])
        np.testing.assert_allclose(none, 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
