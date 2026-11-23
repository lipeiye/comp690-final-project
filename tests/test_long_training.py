"""What scripts/long_training_study.py relies on: a longer run passes through the shorter run's weights.

The study trains 1,900 epochs and keeps the weights at 475, 950 and 1,900. Its 475-epoch checkpoint
stands in for the earlier studies' 475-epoch models, so training longer, keeping checkpoints and
tracking the held-out runs every epoch must not change the first 475 epochs of training.
"""

import unittest

import numpy as np
import torch

from mibci.model import build_eegnet
from mibci.train import TrainConfig, fit, set_seed


def data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6, 250)).astype(np.float32)
    return X, rng.integers(0, 4, n).astype(np.int64)


def trained(epochs: int, **kwargs):
    X, y = data(40, 0)
    set_seed(3)
    model = build_eegnet(6, n_times=250)
    out = fit(model, X, y, TrainConfig(epochs=epochs, batch_size=16, seed=3), **kwargs)
    return model, out


class LongerRunsPassThroughShorterOnes(unittest.TestCase):
    def test_the_checkpoint_equals_the_shorter_run_bit_for_bit(self):
        short, _ = trained(3)
        _, out = trained(6, val=data(20, 1), checkpoints=(3, 6))
        for key, value in short.state_dict().items():
            self.assertTrue(torch.equal(value, out["checkpoints"][3][key]), key)

    def test_tracking_held_out_trials_changes_nothing(self):
        plain, _ = trained(4)
        tracked, out = trained(4, val=data(20, 1))
        self.assertEqual(len(out["history"]["val_acc"]), 4)
        for key, value in plain.state_dict().items():
            self.assertTrue(torch.equal(value, tracked.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()
