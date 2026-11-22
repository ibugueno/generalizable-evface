import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

from facial_analysis.events import load_events
from facial_analysis.representations import make_representation


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.events = np.array(
            [
                [0.00, 1, 1, 1],
                [0.02, 1, 1, -1],
                [0.04, 2, 2, 1],
                [0.06, 3, 3, 1],
            ],
            dtype=np.float32,
        )

    def test_all_representations_have_common_shape_and_range(self):
        for name in ("acc", "sae", "tbr", "lads"):
            kwargs = {"time_start": 0.0, "time_end": 0.1}
            if name == "tbr":
                kwargs["bins"] = 4
            if name == "lads":
                kwargs["rate_alpha"] = 1.0
            value = make_representation(name, self.events, 8, 10, **kwargs)
            self.assertEqual(value.shape, (8, 10, 1), name)
            self.assertEqual(value.dtype, np.float32)
            self.assertGreaterEqual(value.min(), -1.0)
            self.assertLessEqual(value.max(), 1.0)

    def test_npz_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.npz"
            np.savez(path, t=self.events[:, 0], x=self.events[:, 1], y=self.events[:, 2], p=self.events[:, 3])
            loaded = load_events(path)
            np.testing.assert_allclose(loaded, self.events)

    def test_h5_loader_slices_time_window(self):
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py is provided by the project container")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.h5"
            with h5py.File(path, "w") as handle:
                group = handle.create_group("events")
                group["t"] = np.arange(10, dtype=np.float64)
                group["x"] = np.arange(10, dtype=np.int16)
                group["y"] = np.arange(10, dtype=np.int16)
                group["p"] = np.ones(10, dtype=np.int8)
            loaded = load_events(path, time_start=2, time_end=5)
            self.assertEqual(len(loaded), 3)
            np.testing.assert_array_equal(loaded[:, 0], [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
