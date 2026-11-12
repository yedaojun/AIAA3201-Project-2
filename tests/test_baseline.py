import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from anomaly_baseline.core import TemplateDetector, histogram_auc, image_auc, load_image


class BaselineTests(unittest.TestCase):
    def test_auc_matches_pairwise_definition_with_ties(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        scores = np.array([0, 0, 1, 2, 2, 2])
        positive, negative = scores[labels == 1], scores[labels == 0]
        expected = np.mean([(p > n) + .5 * (p == n) for p in positive for n in negative])
        self.assertAlmostEqual(image_auc(labels, scores), expected)
        self.assertAlmostEqual(histogram_auc(np.bincount(positive, minlength=3), np.bincount(negative, minlength=3)), expected)

    def test_normal_pixels_affect_pixel_auc(self):
        self.assertEqual(histogram_auc([0, 1], [1, 0]), 1)
        self.assertAlmostEqual(histogram_auc([0, 1], [1, 1]), .75)
        with self.assertRaises(ValueError):
            histogram_auc([0, 0], [3, 4])

    def test_identity_and_local_defect(self):
        temporary_root = (Path.cwd() / 'tmp').resolve()
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            root = Path(folder)
            self.assertTrue(root.resolve().is_relative_to(temporary_root))
            image = np.full((100, 160, 3), 100, dtype=np.uint8)
            Image.fromarray(image).save(root / 'normal.png')
            detector = TemplateDetector([root / 'normal.png'])
            score, anomaly, _, _ = detector.predict(root / 'normal.png')
            self.assertEqual(score, 0)
            self.assertEqual(int(anomaly.max()), 0)
            image[35:55, 60:90] = 220
            Image.fromarray(image).save(root / 'defect.png')
            score, anomaly, _, _ = detector.predict(root / 'defect.png')
            self.assertGreater(score, 0)
            self.assertGreater(float(anomaly[38:52, 65:85].mean()), float(anomaly[:20, :20].mean()))
            self.assertEqual(anomaly.shape, (100, 160))
            prepared, _ = load_image(root / 'normal.png')
            self.assertEqual(prepared.shape[0], 448)
            self.assertEqual(prepared.shape[1] % 14, 0)

    def test_manifest_has_no_calibration_leakage(self):
        manifest = Path('configs/reference_manifest.json')
        if not manifest.exists():
            self.skipTest('Run data preparation first')
        data = json.loads(manifest.read_text())
        for selection in data['categories'].values():
            self.assertEqual(len(set(selection['calibration'])), 20)
            for shot, seeds in selection['references'].items():
                for names in seeds.values():
                    self.assertEqual(len(set(names)), int(shot))
                    self.assertFalse(set(names).intersection(selection['calibration']))


if __name__ == '__main__':
    unittest.main()
