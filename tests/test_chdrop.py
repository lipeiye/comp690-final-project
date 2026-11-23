"""mibci.chdrop: batch-norm variance ratios, failed electrodes, failure sets."""

import unittest

import numpy as np
import torch

from mibci import chdrop, noise_mechanism
from mibci.model import build_eegnet
from mibci.overfit import fit_channel_dropout
from mibci.train import TrainConfig, set_seed


def data(n: int = 32, inputs: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    shared = rng.standard_normal((n, 1, 500))
    X = (3 * shared + 0.3 * rng.standard_normal((n, inputs, 500))).astype(np.float32)  # highly correlated inputs
    X = (X - X.mean(axis=(0, 2), keepdims=True)) / X.std(axis=(0, 2), keepdims=True)
    return X.astype(np.float32), rng.integers(0, 4, n).astype(np.int64)


class ChannelDropoutMechanism(unittest.TestCase):
    def test_re_estimated_batch_norms_match_their_inputs(self):
        X, y = data()
        set_seed(0)
        model = build_eegnet(X.shape[1], n_classes=4)
        fit_channel_dropout(model, X, y, TrainConfig(epochs=3, batch_size=8, seed=0), p=0.5)
        fixed = noise_mechanism.recalibrate(model, X)
        ratios = chdrop.variance_ratios(fixed, X)
        for name in ("bnorm_1", "bnorm_2"):
            np.testing.assert_allclose(ratios[name], 1.0, rtol=1e-3, err_msg=name)

    def test_a_failed_electrode_is_set_to_its_training_mean(self):
        X, _ = data()
        out = chdrop.with_broken(X, [1, 4])
        self.assertTrue((out[:, [1, 4]] == 0).all())
        np.testing.assert_array_equal(out[:, [0, 2, 3, 5]], X[:, [0, 2, 3, 5]])
        self.assertFalse((X[:, 1] == 0).all())  # the input itself is unchanged

    def test_failure_sets_are_fixed_and_distinct_within_a_set(self):
        a, b = chdrop.failure_sets(22, 3, 5, 694), chdrop.failure_sets(22, 3, 5, 694)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 5)
        for s in a:
            self.assertEqual(len(set(s)), 3)
            self.assertTrue(all(0 <= i < 22 for i in s))

    def test_ratios_cover_every_batch_norm(self):
        X, _ = data()
        torch.manual_seed(0)
        model = build_eegnet(X.shape[1], n_classes=4)
        ratios = chdrop.variance_ratios(model, X)
        self.assertEqual(set(ratios), set(chdrop.BATCH_NORMS))
        self.assertEqual(len(ratios["bnorm_1"]), 16)


if __name__ == "__main__":
    unittest.main()
