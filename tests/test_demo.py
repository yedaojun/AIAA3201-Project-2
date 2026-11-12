import unittest
from pathlib import Path

from anomaly_baseline.demo import category_from_dataset_path


class DemoPathTests(unittest.TestCase):
    def setUp(self):
        self.raw = Path('data/VisA').resolve()
        self.prepared = Path('data/VisA_pytorch/1cls').resolve()
        self.roots = (self.raw, self.prepared)

    def test_category_is_identical_for_normal_and_abnormal_images(self):
        for category in ('candle', 'cashew', 'fryum'):
            for split in ('good', 'bad'):
                path = self.prepared / category / 'test' / split / '001.JPG'
                self.assertEqual(category_from_dataset_path(path, self.roots), category)
            self.assertEqual(category_from_dataset_path(self.raw / category / 'Data/Images/Normal/001.JPG', self.roots), category)

    def test_mask_is_not_a_product_image(self):
        with self.assertRaises(ValueError):
            category_from_dataset_path(self.raw / 'candle/Data/Masks/Anomaly/001.png', self.roots)
        with self.assertRaises(ValueError):
            category_from_dataset_path(self.prepared / 'candle/ground_truth/bad/001.png', self.roots)

    def test_external_image_requires_manual_category(self):
        self.assertIsNone(category_from_dataset_path(Path('tmp/candle/photo.jpg'), self.roots))


if __name__ == '__main__':
    unittest.main()
