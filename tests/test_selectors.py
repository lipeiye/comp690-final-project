import unittest

import numpy as np

from mibci import regions, selectors
from mibci.data import EEG_CHANNELS


def synthetic(n_per_class: int = 24, n_channels: int = 6, informative: int = 2, seed: int = 0):
    """Class c changes the power of channel `informative` (and nothing else)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    t = np.arange(500) / 250
    for c in range(3):
        for _ in range(n_per_class):
            trial = rng.normal(size=(n_channels, 500))
            trial[informative] += (1 + 2 * c) * np.sin(2 * np.pi * 12 * t + rng.uniform(0, 6.3))
            X.append(trial)
            y.append(c)
    runs = np.tile(np.repeat(np.arange(6), n_per_class // 6), 3)
    return np.array(X), np.array(y), runs


class RiemannTests(unittest.TestCase):
    def test_mean_and_distance_basics(self):
        rng = np.random.default_rng(1)
        A = rng.normal(size=(5, 5))
        A = A @ A.T + 5 * np.eye(5)
        B = rng.normal(size=(5, 5))
        B = B @ B.T + 5 * np.eye(5)
        np.testing.assert_allclose(selectors.riemann_mean(np.stack([A, A, A])), A, rtol=1e-7)
        self.assertAlmostEqual(selectors.riemann_distance(A, A), 0.0, places=6)
        self.assertAlmostEqual(selectors.riemann_distance(A, B), selectors.riemann_distance(B, A), places=6)
        W = rng.normal(size=(5, 5)) + 3 * np.eye(5)  # invariance under any invertible congruence
        self.assertAlmostEqual(
            selectors.riemann_distance(W @ A @ W.T, W @ B @ W.T), selectors.riemann_distance(A, B), places=5
        )

    def test_orders_put_the_informative_channel_first(self):
        X, y, runs = synthetic()
        for order in (selectors.riemann_order(X, y), selectors.csp_order(X, y)):
            self.assertEqual(sorted(order), list(range(6)))
            self.assertEqual(order[0], 2)
        order, curve = selectors.wrapper_order(X, y, runs)
        self.assertEqual(sorted(order), list(range(6)))
        self.assertIn(2, order[:2])
        self.assertEqual(len(curve), 5)


class RegionTests(unittest.TestCase):
    def test_partitions_cover_the_montage_and_coalitions_are_complete(self):
        for partition in regions.PARTITIONS.values():
            combos = regions.coalitions(partition)
            self.assertEqual(len(combos), 2 ** len(partition) - 1)
            everything = regions.coalition_channels(partition, combos[-1])
            self.assertEqual(everything, list(range(len(EEG_CHANNELS))))

    def test_grid_neighbours(self):
        self.assertEqual(sorted(regions.neighbours("Cz")), ["C1", "C2", "CPz", "FCz"])
        self.assertEqual(regions.neighbours("POz"), ["Pz"])


if __name__ == "__main__":
    unittest.main()
