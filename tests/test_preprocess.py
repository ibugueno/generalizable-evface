import csv
import tempfile
import unittest
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


@unittest.skipIf(h5py is None, "h5py is provided by the project container")
class PreprocessTests(unittest.TestCase):
    def test_fps_aligned_pairs_and_context_manifest(self):
        from scripts import preprocess_facevent

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset" / "0000" / "0000"
            root.mkdir(parents=True)
            video_path = root / "clip_face640.mp4"
            event_path = root / "clip_face640_events.h5"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (32, 32))
            for index in range(4):
                writer.write(np.full((32, 32, 3), index * 40, dtype=np.uint8))
            writer.release()
            with h5py.File(event_path, "w") as handle:
                group = handle.create_group("events")
                group.attrs["height"] = 32
                group.attrs["width"] = 32
                group["t"] = np.array([0, 100_000, 500_000, 600_000], dtype=np.int64)
                group["x"] = np.array([1, 2, 3, 4], dtype=np.int16)
                group["y"] = np.array([1, 2, 3, 4], dtype=np.int16)
                group["p"] = np.array([1, 1, 0, 1], dtype=np.int8)
            output = Path(directory) / "output"
            old_argv = sys.argv
            sys.argv = [
                "preprocess_facevent.py", "--dataset-root", str(Path(directory) / "dataset"),
                "--output-root", str(output), "--representations", "acc", "--sequence-length", "2",
                "--output-size", "16", "--max-frames", "2", "--write-event-representations",
                "--export-rgb-frames", "--export-gray-frames",
            ]
            try:
                preprocess_facevent.main()
            finally:
                sys.argv = old_argv
            with (output / "pairs_acc.csv").open(newline="") as handle:
                pairs = list(csv.DictReader(handle))
            with (output / "reconstruction_manifest_acc.csv").open(newline="") as handle:
                reconstruction = list(csv.DictReader(handle))
            self.assertEqual(len(pairs), 2)
            self.assertEqual(pairs[0]["event_start"], "0.0")
            self.assertEqual(pairs[0]["event_end"], "500000.0")
            self.assertEqual(reconstruction[1]["frame_count"], "2")
            target = np.load(pairs[0]["target_path"])
            self.assertEqual(target.shape, (16, 16, 1))
            self.assertTrue(reconstruction[0]["rgb_path"].endswith("clip_face640.mp4"))
            self.assertTrue(Path(pairs[0]["rgb_path"]).is_file())
            self.assertTrue(Path(pairs[0]["gray_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
