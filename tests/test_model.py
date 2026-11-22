import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

try:
    import torch
except ImportError:  # pragma: no cover - local host may not have the CUDA runtime
    torch = None


@unittest.skipIf(torch is None, "PyTorch is provided by the project container")
class ModelTests(unittest.TestCase):
    def test_tiny_teacher_reconstruction_shape(self):
        from facial_analysis.model import RGBToEventReconstruction, TinyRGBTeacher, infer_teacher_channels

        teacher = TinyRGBTeacher()
        model = RGBToEventReconstruction(teacher, infer_teacher_channels(teacher, 64), 64)
        output = model(torch.rand(2, 4, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 1, 64, 64))
        self.assertTrue(all(not parameter.requires_grad for parameter in teacher.parameters()))


if __name__ == "__main__":
    unittest.main()
