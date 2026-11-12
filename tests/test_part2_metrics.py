import unittest
import numpy as np
try:
    from sklearn.metrics import roc_auc_score
except ModuleNotFoundError:
    roc_auc_score = None

from anomaly_dino.metrics import positive_negative_auc


@unittest.skipIf(roc_auc_score is None, 'Run with .venv-part2 for the sklearn reference check')
class ExactFloatAUROCTests(unittest.TestCase):
    def test_matches_sklearn_with_ties_and_tiny_float_differences(self):
        positive = np.array([0., .1, .2, np.nextafter(.2, 1.), .9, 1.])
        negative = np.array([0., 0., .1, .2, .7, .9, 1.])
        expected = roc_auc_score([1] * len(positive) + [0] * len(negative), np.r_[positive, negative])
        self.assertAlmostEqual(positive_negative_auc(positive.copy(), negative.copy()), expected, places=13)

    def test_random_float64_matches_sklearn(self):
        rng = np.random.default_rng(3201)
        positive, negative = rng.normal(.2, 1, 3000), rng.normal(0, 1, 20000)
        expected = roc_auc_score([1] * len(positive) + [0] * len(negative), np.r_[positive, negative])
        self.assertAlmostEqual(positive_negative_auc(positive.copy(), negative.copy()), expected, places=13)

    def test_degenerate_inputs(self):
        self.assertEqual(positive_negative_auc(np.zeros(3), np.zeros(5)), .5)
        with self.assertRaises(ValueError):
            positive_negative_auc(np.zeros(0), np.ones(3))


if __name__ == '__main__':
    unittest.main()
