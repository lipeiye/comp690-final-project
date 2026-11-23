"""mibci.residual_surrogate: prediction + residual = the electrode; surrogates move whole residual blocks."""

import unittest

import numpy as np

from mibci import input_kind
from mibci import residual_surrogate as rs


class ResidualSurrogate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        mixing = rng.standard_normal((6, 10))
        cls.X = (np.einsum("ce,net->nct", mixing, rng.standard_normal((40, 10, 200)))).astype(np.float32)
        cls.X[:, 4:] += 0.3 * rng.standard_normal((40, 2, 200)).astype(np.float32)  # unpredictable parts
        cls.kept, cls.dropped = [0, 1, 2, 3], [4, 5]
        cls.train = np.arange(40) < 30
        cls.f = rs.fit(cls.X[cls.train], cls.kept, cls.dropped)

    def test_prediction_plus_residual_is_the_electrode(self):
        total = rs.predictable(self.X, self.f) + rs.residual(self.X, self.f)
        np.testing.assert_allclose(total, self.X[:, self.dropped], rtol=0, atol=1e-5)
        real = rs.extra_inputs(self.X, "real", self.f)
        np.testing.assert_array_equal(real, self.X[:, self.dropped])

    def test_surrogate_with_the_identity_is_the_real_electrode(self):
        identity = np.arange(len(self.X))
        surr = rs.extra_inputs(self.X, "surr", self.f, identity)
        np.testing.assert_allclose(surr, self.X[:, self.dropped], rtol=0, atol=1e-5)

    def test_surrogate_moves_whole_residual_blocks(self):
        perm = rs.derangement(len(self.X), "test|0")
        surr = rs.extra_inputs(self.X, "surr", self.f, perm).astype(np.float64)
        moved = surr - rs.predictable(self.X, self.f)
        np.testing.assert_allclose(moved, rs.residual(self.X, self.f)[perm], rtol=0, atol=1e-4)

    def test_derangement_moves_every_trial_and_is_reproducible(self):
        a, b = rs.derangement(31, "k|0"), rs.derangement(31, "k|0")
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(np.sort(a), np.arange(31))
        self.assertFalse((a == np.arange(31)).any())
        self.assertFalse(np.array_equal(a, rs.derangement(31, "k|1")))
        for n in (2, 3, 7):
            self.assertFalse((rs.derangement(n, f"small|{n}") == np.arange(n)).any())

    def test_fit_uses_the_training_trials_only(self):
        changed = self.X.copy()
        changed[~self.train] = 1000.0
        g = rs.fit(changed[self.train], self.kept, self.dropped)
        self.assertEqual(g.digest(), self.f.digest())

    def test_same_fit_as_the_input_kind_study(self):
        held = ~self.train
        left = rs.residual(self.X[held], self.f)
        ours = left.transpose(0, 2, 1).reshape(-1, 2).var(axis=0) / (
            self.X[held][:, self.dropped].astype(np.float64).transpose(0, 2, 1).reshape(-1, 2).var(axis=0)
        )
        theirs = input_kind.unexplained_fraction(self.X, self.kept, self.dropped, self.train, held)
        np.testing.assert_allclose(ours, theirs, rtol=1e-9)

    def test_model_input_puts_the_extra_inputs_after_the_kept_electrodes(self):
        x = rs.model_input(self.X, self.f, "pred")
        self.assertEqual(x.shape, (40, 6, 200))
        np.testing.assert_array_equal(x[:, :4], self.X[:, self.kept])
        with self.assertRaises(ValueError):
            rs.model_input(self.X, self.f, "surr")


if __name__ == "__main__":
    unittest.main()
