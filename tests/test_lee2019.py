import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from mibci import lee2019


def fake_run(rng, n_trials: int, channels: list[str]) -> dict:
    """One run in the file layout MOABB reads: x (samples x channels, microvolts), cue onsets t in
    1000-Hz samples, y_dec (1 right hand, 2 left hand), chan, fs."""
    onsets = 2000 + 4000 * np.arange(n_trials)
    x = rng.normal(size=(onsets[-1] + 6000, len(channels)))
    labels = rng.integers(1, 3, n_trials)
    chan = np.empty((1, len(channels)), dtype=object)
    chan[0, :] = channels
    return {"x": x, "t": onsets[None, :], "y_dec": labels[None, :], "chan": chan, "fs": np.array([[1000]])}


class Lee2019Tests(unittest.TestCase):
    def test_preprocessing_gives_the_eegnet_input_and_maps_labels(self):
        rng = np.random.default_rng(0)
        channels = [f"E{i}" for i in range(62)]
        runs = {key: fake_run(rng, 5, channels) for key in lee2019.RUNS}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sess01_subj01_EEG_MI.mat"
            savemat(path, runs)
            out = lee2019.preprocess_session(path)
        self.assertEqual(out["X"].shape, (10, 62, lee2019.N_TIMES))
        self.assertEqual(lee2019.N_TIMES, 500)
        self.assertEqual(out["X"].dtype, np.float32)
        expected = np.concatenate([np.where(runs[k]["y_dec"][0] == 1, 1, 0) for k in lee2019.RUNS])
        np.testing.assert_array_equal(out["y"], expected)
        np.testing.assert_array_equal(out["run"], np.repeat([0, 1], 5))
        self.assertEqual(list(out["channels"]), channels)
        self.assertFalse(out["artifact"].any())

    def test_held_out_day_is_locked(self):
        with self.assertRaises(PermissionError):
            lee2019.load_session(1, 2)


if __name__ == "__main__":
    unittest.main()
