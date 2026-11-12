import unittest

from tools.compare_prediction_logs import compare_rows, MATCH


def prediction(sample, truth, score, method):
    predicted = score > .5
    row = dict.fromkeys(MATCH, 'shared')
    row.update(dataset='VisA', split='1cls', category='candle', shots='4', seed='0',
               sample_id=sample, ground_truth_label=str(truth), score=str(score), threshold='.5',
               method_id=method, image_file=sample, anomaly_map=method + '/map.png', run_id=method,
               is_correct=str(int(predicted == bool(truth))),
               outcome=('TP' if predicted else 'FN') if truth else ('FP' if predicted else 'TN'))
    return row


class ComparisonTests(unittest.TestCase):
    def test_tracks_fixed_false_negatives_and_regressions(self):
        old = [prediction('a', 1, .1, 'old'), prediction('b', 0, .1, 'old')]
        new = [prediction('a', 1, .9, 'new'), prediction('b', 0, .9, 'new')]
        rows = compare_rows(old, new)
        self.assertEqual([r['transition'] for r in rows], ['FN->TP', 'TN->FP'])
        self.assertEqual([r['change'] for r in rows], ['error_fixed', 'regression'])

    def test_refuses_unpaired_or_different_reference_images(self):
        old = [prediction('a', 1, .1, 'old')]
        with self.assertRaises(ValueError):
            compare_rows(old, [prediction('b', 1, .9, 'new')])
        new = prediction('a', 1, .9, 'new')
        new['reference_set_sha256'] = 'different'
        with self.assertRaises(ValueError):
            compare_rows(old, [new])
        with self.assertRaises(ValueError):
            compare_rows(old, [old[0], old[0]])


if __name__ == '__main__':
    unittest.main()
