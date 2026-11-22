import csv
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is provided by the project container")
class DatasetTests(unittest.TestCase):
    def test_fac_event_window_sample(self):
        from facial_analysis.dataset import FacEventReconstructionDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb_path = root / "frames.npy"
            event_path = root / "events.npz"
            manifest_path = root / "manifest.csv"
            np.save(rgb_path, np.zeros((4, 24, 32, 3), dtype=np.uint8))
            np.savez(
                event_path,
                t=np.arange(8, dtype=np.float32),
                x=np.arange(8, dtype=np.int32) % 8,
                y=np.arange(8, dtype=np.int32) % 8,
                p=np.ones(8, dtype=np.int8),
            )
            with manifest_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["rgb_path", "event_path", "frame_count", "event_start", "event_end", "event_height", "event_width"],
                )
                writer.writeheader()
                writer.writerow({
                    "rgb_path": str(rgb_path), "event_path": str(event_path), "frame_count": 4,
                    "event_start": 1, "event_end": 5, "event_height": 8, "event_width": 8,
                })
            dataset = FacEventReconstructionDataset(
                manifest_path, image_size=16, sequence_length=4, representation="tbr",
                event_height=8, event_width=8, timestamp_unit="s", tbr_bins=4,
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["rgb"].shape), (4, 3, 16, 16))
            self.assertEqual(tuple(sample["target"].shape), (1, 16, 16))


if __name__ == "__main__":
    unittest.main()
