import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from Part1.autoencoder.model import CompactAutoencoder, training_split
from experiment_utils.common import DATA, restore_map
from experiment_utils.prediction import infer_category


class OptionalExtensionTests(unittest.TestCase):
    def test_autoencoder_handles_non_square_non_stride_aligned_images_and_backprop(self):
        torch.manual_seed(111)
        model = CompactAutoencoder()
        image = torch.rand(2, 3, 65, 91)
        reconstruction = model(image)
        self.assertEqual(reconstruction.shape, image.shape)
        self.assertTrue(torch.isfinite(reconstruction).all())
        self.assertGreaterEqual(float(reconstruction.detach().min()), 0)
        self.assertLessEqual(float(reconstruction.detach().max()), 1)
        torch.nn.functional.mse_loss(reconstruction, image).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    @unittest.skipUnless((DATA / 'candle/train/good').exists(), 'local VisA dataset required')
    def test_normal_fit_validation_and_calibration_are_disjoint_and_complete(self):
        for category in ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum'):
            fit, validation, calibration = training_split(category)
            self.assertEqual(training_split(category), (fit, validation, calibration))
            self.assertEqual(len(calibration), 20)
            self.assertFalse(set(fit) & set(validation))
            self.assertFalse((set(fit) | set(validation)) & set(calibration))
            self.assertEqual(set(fit) | set(validation) | set(calibration), {p.name for p in (DATA / category / 'train/good').iterdir()})

    def test_saved_float_maps_are_lossless_and_restore_without_normalizing(self):
        working = np.array([[.12, .125], [.13, .135]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'map.npz'
            np.savez_compressed(path, map=working)
            with np.load(path) as saved:
                restored = restore_map(saved['map'], (8, 6))
                np.testing.assert_array_equal(working, saved['map'])
        self.assertEqual(restored.shape, (6, 8))
        self.assertGreaterEqual(float(restored.min()), .11999)
        self.assertLessEqual(float(restored.max()), .13501)
        self.assertIsNone(infer_category('outside/part.jpg'))
        with self.assertRaises(ValueError):
            infer_category('data/candle/ground_truth/bad/000.png')


if __name__ == '__main__':
    unittest.main()
