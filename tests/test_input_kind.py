"""The extra inputs of scripts/input_kind_study.py (mibci.input_kind)."""

import unittest

import numpy as np

from mibci import input_kind, noise_mechanism, overfit
from mibci.train import ChannelStandardizer


def eeg(trials: int = 30, electrodes: int = 22, times: int = 500, seed: int = 0) -> np.ndarray:
    """Correlated fake EEG: a few shared sources mixed into every electrode, plus a little own noise."""
    rng = np.random.default_rng(seed)
    sources = rng.standard_normal((trials, 4, times))
    mixing = rng.standard_normal((electrodes, 4))
    own = 0.3 * rng.standard_normal((trials, electrodes, times))
    return (10 * (np.einsum("ek,nkt->net", mixing, sources) + own) + 3).astype(np.float32)


class ExtraInputs(unittest.TestCase):
    def setUp(self):
        self.X = eeg()
        self.channels = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18, 20]  # sorted, as in the plan
        self.order = [13, 17, 1, 11, 7, 2, 3, 0, 4, 10, 18, 9, 12, 20, 8, 6]  # relevance order of the same 16
        self.train = np.arange(len(self.X)) < 20

    def noise(self, added: int, largest: int) -> np.ndarray:
        return noise_mechanism.appended_noise(len(self.X), self.X.shape[2], added, largest, "bnci2a", 3, 1, 0)

    def test_a_equal_one_is_the_input_cost_studys_input(self):
        # scripts/input_cost_study.py: kept electrodes, then the first k of `largest` noise inputs
        added, largest = 6, 16
        zeros = np.zeros((len(self.X), largest, self.X.shape[2]), np.float32)
        noise_cost = overfit.noise_channels(zeros, list(range(largest)), "bnci2a", 3, 1)[:, :added]
        cost_input = np.concatenate([self.X[:, self.channels], noise_cost], axis=1)
        sources = input_kind.copied(self.order, added)
        mean, std = input_kind.source_stats(self.X[self.train], sources)
        ours = input_kind.model_input(self.X, self.channels, "mix100", sources, self.noise(added, largest), mean, std)
        self.assertEqual(ours.dtype, cost_input.dtype)
        np.testing.assert_array_equal(ours, cost_input)

    def test_copies_cycle_through_the_kept_electrodes_from_the_most_relevant(self):
        sources = input_kind.copied(self.order, 46)
        self.assertEqual(sources[:16], self.order)
        self.assertEqual(sources[16:32], self.order)
        self.assertEqual(sources[32:], self.order[:14])
        extra = input_kind.extra_inputs(self.X, "dup", sources[:6], self.noise(6, 16), [0.0] * 6, [1.0] * 6)
        np.testing.assert_array_equal(extra, self.X[:, self.order[:6]])

    def test_shared_inputs_are_one_waveform(self):
        noise = self.noise(6, 16)
        extra = input_kind.extra_inputs(self.X, "shared", self.order[:6], noise, [0.0] * 6, [1.0] * 6)
        for i in range(6):
            np.testing.assert_array_equal(extra[:, i], noise[:, 0])

    def test_a_mix_has_unit_variance_and_the_planned_share_of_its_electrode(self):
        sources = input_kind.copied(self.order, 6)
        mean, std = input_kind.source_stats(self.X[self.train], sources)
        z = ChannelStandardizer().fit(self.X[self.train][:, sources]).transform(self.X[:, sources])
        for kind, a in (("mix25", 0.25), ("mix50", 0.5), ("mix75", 0.75)):
            extra = input_kind.extra_inputs(self.X, kind, sources, self.noise(6, 16), mean, std)[self.train]
            self.assertTrue(np.allclose(extra.var(axis=(0, 2)), 1.0, atol=0.1), kind)
            corr = [np.corrcoef(extra[:, i].ravel(), z[self.train][:, i].ravel())[0, 1] for i in range(6)]
            self.assertTrue(np.allclose(corr, np.sqrt(1 - a), atol=0.05), kind)

    def test_a_dropped_electrode_that_the_kept_ones_explain_has_nothing_left(self):
        X = self.X.copy()
        X[:, 21] = 0.5 * X[:, 0] - 2.0 * X[:, 7]  # an exact combination of kept electrodes
        X[:, 19] = 10 * np.random.default_rng(1).standard_normal(X[:, 19].shape)  # independent of everything
        left = input_kind.unexplained_fraction(X, self.channels, [21, 19], self.train, ~self.train)
        self.assertLess(left[0], 1e-6)
        self.assertGreater(left[1], 0.95)


if __name__ == "__main__":
    unittest.main()
