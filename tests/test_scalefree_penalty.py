"""mibci.scalefree_penalty: the frozen training at lam = 0; rescaling a filter cannot lower the penalty."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from mibci.model import build_eegnet
from mibci.scalefree_penalty import fit_scalefree, input_norms, penalty
from mibci.train import TrainConfig, fit, set_seed


def data(n: int = 24, inputs: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    return rng.standard_normal((n, inputs, 500)).astype(np.float32), rng.integers(0, 4, n).astype(np.int64)


def trained(lam: float | None, epochs: int = 3) -> torch.nn.Module:
    X, y = data()
    cfg = TrainConfig(epochs=epochs, batch_size=8, seed=3)
    set_seed(cfg.seed)
    model = build_eegnet(X.shape[1], n_classes=4)
    if lam is None:
        fit(model, X, y, cfg)
    else:
        fit_scalefree(model, X, y, cfg, lam)
    return model


def spatial_weight(inputs: int = 22, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    with torch.no_grad():
        return build_eegnet(inputs, n_classes=4).conv_spatial.weight.detach().clone()


class ScaleFreePenaltyTest(unittest.TestCase):
    def test_zero_penalty_is_the_frozen_training(self):
        a, b = trained(None).state_dict(), trained(0.0).state_dict()
        self.assertEqual(a.keys(), b.keys())
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)

    def test_rescaling_filters_leaves_the_penalty_unchanged(self):
        w = spatial_weight()
        base = float(penalty(w))
        self.assertAlmostEqual(float(penalty(3.7 * w)), base, places=5)  # every weight times one constant
        per_filter = torch.linspace(0.1, 5.0, w.shape[0]).view(-1, 1, 1, 1)
        self.assertAlmostEqual(float(penalty(per_filter * w)), base, places=5)  # each filter its own constant
        g = input_norms(build_eegnet(22, n_classes=4))
        self.assertAlmostEqual(float(g.pow(2).sum()), w.shape[0], places=4)  # sum g_i^2 = number of filters

    def test_shrinking_one_input_lowers_the_penalty(self):
        w = spatial_weight()
        g = input_norms_of(w)
        typical = int(torch.argsort(g)[len(g) // 2])  # the input with the median norm
        shrunk = w.clone()
        shrunk[:, :, typical, :] *= 0.3
        self.assertLess(float(penalty(shrunk)), float(penalty(w)))
        shrunk[:, :, typical, :] = 0.0
        self.assertLess(float(penalty(shrunk)), float(penalty(w)))


def input_norms_of(w: torch.Tensor) -> torch.Tensor:
    return w[:, 0, :, 0].norm(dim=0)


if __name__ == "__main__":
    unittest.main()
