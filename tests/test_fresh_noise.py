"""mibci.fresh_noise: fresh noise per epoch, and a training loop that otherwise is mibci.train.fit."""

import unittest

import numpy as np
import torch

from mibci import fresh_noise, overfit
from mibci.model import build_eegnet
from mibci.train import TrainConfig, fit, set_seed


class FreshNoise(unittest.TestCase):
    def test_same_inputs_every_epoch_is_mibci_train_fit(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 6, 500)).astype(np.float32)
        y = rng.integers(0, 4, 20).astype(np.int64)
        cfg = TrainConfig(epochs=3, seed=7)
        set_seed(cfg.seed)
        a = build_eegnet(6)
        ref = fit(a, X, y, cfg)
        set_seed(cfg.seed)
        b = build_eegnet(6)
        got = fresh_noise.fit_epochwise(b, lambda epoch: X, y, cfg)
        self.assertEqual(ref["history"]["train_loss"], got["history"]["train_loss"])
        for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
            self.assertEqual(ka, kb)
            torch.testing.assert_close(va, vb, rtol=0, atol=0)

    def test_noise_differs_between_epochs_and_repeats_with_the_seed(self):
        args = (12, 500, 6, "bnci2a", 3, "fold-1", 4)
        e1, e2 = fresh_noise.epoch_noise(*args, 1), fresh_noise.epoch_noise(*args, 2)
        np.testing.assert_array_equal(e1, fresh_noise.epoch_noise(*args, 1))
        self.assertFalse(np.allclose(e1, e2))
        other_seed = fresh_noise.epoch_noise(12, 500, 6, "bnci2a", 3, "fold-1", 5, 1)
        self.assertFalse(np.allclose(e1, other_seed))
        self.assertAlmostEqual(float(e1.std()), 1.0, delta=0.1)  # unit variance after the band-pass

    def test_draw0_is_the_input_cost_studys_input(self):
        X = np.random.default_rng(1).standard_normal((10, 22, 500)).astype(np.float32)
        channels = [0, 2, 4, 6]
        got = fresh_noise.draw0(X, channels, 6, 16, "bnci2a", 2, 1)
        noise = overfit.noise_channels(np.zeros((10, 16, 500), np.float32), list(range(16)), "bnci2a", 2, 1)
        np.testing.assert_array_equal(got, np.concatenate([X[:, channels], noise[:, :6]], axis=1))

    def test_electrodes_do_not_change_between_epochs(self):
        X = np.random.default_rng(2).standard_normal((8, 22, 500)).astype(np.float32)
        channels = [1, 3, 5]

        def epoch_inputs(epoch):
            noise = fresh_noise.epoch_noise(8, 500, 4, "lee2019", 9, "final", 0, epoch)
            return np.concatenate([X[:, channels], noise], axis=1)

        a, b = epoch_inputs(1), epoch_inputs(2)
        np.testing.assert_array_equal(a[:, :3], b[:, :3])
        self.assertFalse(np.allclose(a[:, 3:], b[:, 3:]))

    def test_unseen_noise_is_not_draw0(self):
        unseen = fresh_noise.unseen_noise(10, 500, 6, 16, "bnci2a", 2)
        noise = overfit.noise_channels(np.zeros((10, 16, 500), np.float32), list(range(16)), "bnci2a", 2, 1)
        self.assertEqual(unseen.shape, (10, 6, 500))
        self.assertFalse(np.allclose(unseen, noise[:, :6]))


if __name__ == "__main__":
    unittest.main()
