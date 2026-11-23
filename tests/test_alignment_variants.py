"""mibci.alignment_variants: plain EA unchanged; shrinkage, Ledoit-Wolf and the top-direction whitener."""

import unittest

import numpy as np

from mibci import alignment, alignment_variants


def trials(n: int = 60, channels: int = 8, times: int = 250, seed: int = 0, tiny: float = 0.05) -> np.ndarray:
    """Correlated trials whose last channel is nearly a copy of the first (one direction of tiny variance)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((channels, channels)))
    mix = q * np.linspace(1.0, 2.0, channels)  # eigenvalues of the covariance between 1 and 4
    X = np.einsum("cd,ndt->nct", mix, rng.standard_normal((n, channels, times)))
    X[:, -1] = X[:, 0] + tiny * rng.standard_normal((n, times))
    return X.astype(np.float32)


class AlignmentVariants(unittest.TestCase):
    def test_plain_ea_is_the_original(self):
        X = trials()
        np.testing.assert_array_equal(alignment_variants.aligned(X, "ea"), alignment.aligned(X))
        Y = trials(seed=1)
        np.testing.assert_array_equal(alignment_variants.aligned(X, "ea", reference=Y), alignment.aligned(X, Y))

    def test_shrinkage_keeps_the_trace_and_pulls_toward_the_identity(self):
        R = alignment.mean_covariance(trials().astype(np.float64))
        S = alignment_variants.shrunk(R, 0.1)
        self.assertAlmostEqual(np.trace(S), np.trace(R), places=6)
        np.testing.assert_allclose(alignment_variants.shrunk(R, 1.0), np.trace(R) / 8 * np.eye(8))

    def test_ledoit_wolf_is_small_with_many_trials_and_large_with_few(self):
        many = alignment_variants.ledoit_wolf_trials(trials(n=400))
        few = alignment_variants.ledoit_wolf_trials(trials(n=4))
        self.assertTrue(0 <= many < few <= 1)

    def test_top_whitener_drops_the_tiny_direction(self):
        X = trials()
        W = alignment_variants.whitener(X, "ea-top")
        out = alignment.mean_covariance(alignment.align(X, W).astype(np.float64))
        values = np.sort(np.linalg.eigvalsh(out))
        self.assertLess(values[0], 1e-6)  # the near-copy direction is gone
        np.testing.assert_allclose(values[1:], 1.0, atol=1e-3)  # every other direction whitened
        plain = alignment.mean_covariance(alignment_variants.aligned(X, "ea").astype(np.float64))
        np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(plain)), 1.0, atol=1e-2)  # plain EA blows it up to 1

    def test_floor_whitener_keeps_the_tiny_direction_without_blowing_it_up(self):
        X = trials()
        out = alignment.mean_covariance(alignment_variants.aligned(X, "ea-floor").astype(np.float64))
        values = np.sort(np.linalg.eigvalsh(out))
        R = alignment.mean_covariance(X.astype(np.float64))
        raw = np.sort(np.linalg.eigvalsh(R))
        self.assertAlmostEqual(values[0], raw[0] / (0.01 * raw[-1]), places=3)  # kept, scaled as a 1% direction
        self.assertGreater(values[0], 1e-4)
        np.testing.assert_allclose(values[1:], 1.0, atol=1e-3)

    def test_spectrum_counts_the_tiny_direction(self):
        s = alignment_variants.spectrum(trials())
        self.assertEqual(s["below_1pct_of_largest"], 1)
        self.assertGreater(s["condition_number"], 1e3)
        self.assertEqual(len(s["eigenvalues"]), 8)


if __name__ == "__main__":
    unittest.main()
