"""mibci.pretrain: the project's budget rule, shared electrodes and per-person standardization."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mibci import pretrain
from mibci.paths import PROJECT_RESULTS_DIR


class BudgetRule(unittest.TestCase):
    def test_reproduces_the_frozen_budget_from_the_original_curves(self):
        records = sorted((PROJECT_RESULTS_DIR / "validation" / "full").glob("sub-*.json"))
        curves = np.array([json.loads(p.read_text())["history"]["val_acc"] for p in records])
        frozen = json.loads((PROJECT_RESULTS_DIR / "validation" / "budget.json").read_text())
        out = pretrain.budget_epoch(curves, range(25, curves.shape[1] + 1, 25), window=25, tolerance=0.005)
        self.assertEqual(out["epochs"], frozen["epochs"])  # 475
        self.assertEqual(out["smoothed_max_epoch"], frozen["smoothed_max_epoch"])
        self.assertAlmostEqual(out["smoothed_max"], frozen["smoothed_max_val_acc"], places=12)

    def test_only_saved_epochs_qualify(self):
        curve = np.r_[np.linspace(0.3, 0.7, 10), np.full(10, 0.7)][None]
        out = pretrain.budget_epoch(curve, [5, 15, 20], window=1, tolerance=0.0)
        self.assertEqual(out["epochs"], 15)


class SharedInputs(unittest.TestCase):
    def test_group_electrodes_take_the_best_mean_rank(self):
        orders = [[2, 0, 1, 3], [0, 2, 3, 1], [2, 3, 0, 1]]  # mean positions: 0 -> 1.0, 1 -> 3.0, 2 -> 0.33, 3 -> 1.67
        self.assertEqual(pretrain.group_electrodes(orders, 4, 2), [2, 0])
        self.assertEqual(pretrain.group_electrodes(orders[::-1], 4, 3), [2, 0, 3])
        with self.assertRaises(ValueError):
            pretrain.group_electrodes([[0, 1, 1, 2]], 4, 2)

    def test_ties_go_to_the_lower_index(self):
        self.assertEqual(pretrain.group_electrodes([[1, 0, 2], [0, 1, 2]], 3, 1), [0])

    def test_each_person_is_standardized_with_their_own_statistics(self):
        rng = np.random.default_rng(0)
        a = (5 + 3 * rng.standard_normal((10, 2, 50))).astype(np.float32)
        b = (-2 + 0.5 * rng.standard_normal((6, 2, 50))).astype(np.float32)
        X, y = pretrain.pooled([(a, np.zeros(10)), (b, np.ones(6))])
        self.assertEqual(X.shape, (16, 2, 50))
        self.assertEqual(y.dtype, np.int64)
        for part in (X[:10], X[10:]):
            np.testing.assert_allclose(part.mean(axis=(0, 2)), 0, atol=1e-5)
            np.testing.assert_allclose(part.std(axis=(0, 2)), 1, atol=1e-4)

    def test_lee_groups_of_nine(self):
        self.assertEqual([pretrain.lee_group(s) for s in (1, 9, 10, 18, 19, 54)], [0, 0, 1, 1, 2, 5])

    def test_checkpoints_round_trip(self):
        states = {5: {"w": torch.ones(2)}, 10: {"w": torch.zeros(2)}}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.pt"
            pretrain.save_checkpoints(path, states)
            self.assertTrue(torch.equal(pretrain.load_checkpoint(path, 10)["w"], torch.zeros(2)))


if __name__ == "__main__":
    unittest.main()
