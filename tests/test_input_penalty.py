"""mibci.input_penalty: identical to mibci.train.fit without the penalty; the penalty shrinks inputs."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from mibci.input_penalty import fit_penalized, input_norms
from mibci.model import build_eegnet
from mibci.train import TrainConfig, fit, set_seed


def data(n: int = 24, inputs: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    return rng.standard_normal((n, inputs, 500)).astype(np.float32), rng.integers(0, 4, n).astype(np.int64)


def trained(lam: float | None, epochs: int = 3, lr: float = 1e-3) -> torch.nn.Module:
    X, y = data()
    cfg = TrainConfig(epochs=epochs, batch_size=8, lr=lr, seed=3)
    set_seed(cfg.seed)
    model = build_eegnet(X.shape[1], n_classes=4)
    if lam is None:
        fit(model, X, y, cfg)
    else:
        fit_penalized(model, X, y, cfg, lam)
    return model


class InputPenaltyTest(unittest.TestCase):
    def test_zero_penalty_is_the_frozen_training(self):
        a, b = trained(None).state_dict(), trained(0.0).state_dict()
        self.assertEqual(a.keys(), b.keys())
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]), key)

    def test_penalty_shrinks_the_inputs(self):
        free, penalized = trained(0.0, epochs=6, lr=0.02), trained(0.5, epochs=6, lr=0.02)  # 18 steps
        with torch.no_grad():
            self.assertLess(float(input_norms(penalized).sum()), 0.8 * float(input_norms(free).sum()))

    def test_norms_are_per_input(self):
        model = build_eegnet(7, n_classes=4)
        with torch.no_grad():
            norms = input_norms(model)
            w = model.conv_spatial.weight[:, 0, :, 0]
        self.assertEqual(norms.shape, (7,))
        torch.testing.assert_close(norms, w.norm(dim=0))


if __name__ == "__main__":
    unittest.main()
