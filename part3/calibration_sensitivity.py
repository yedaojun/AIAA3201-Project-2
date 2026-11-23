"""Post-hoc leave-one-out audit of the 20-normal-image threshold rule.

No threshold is selected using test labels. Test outcomes are reported only
to show how the fixed calibration rule reacts to one omitted normal image.
"""

import csv
import json

import numpy as np

from experiment_utils.common import ROOT, write_json


OUTPUT = ROOT / 'results/part3/calibration_sensitivity_4shot_seed0.json'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def audit(method, category):
    root = {'official_448': ROOT / 'results/part2',
            'high_res_672': ROOT / 'results/part3/high_resolution_pilot',
            'multi_res_448_672': ROOT / 'results/part3/multiscale_fusion'}[method]
    group = root / category / 'shot_4/seed_0'
    if not (group / 'metrics.json').exists():
        return None
    score_key = 'fused_score' if method == 'multi_res_448_672' else 'score'
    calibration = np.array([float(row[score_key]) for row in read_csv(group / 'calibration.csv')])
    predictions = read_csv(group / 'predictions.csv')
    if len(calibration) != 20:
        raise ValueError(f'Expected 20 calibration images: {group}')
    scores = np.array([float(row[score_key]) for row in predictions])
    labels = np.array([int(row['label']) for row in predictions])
    thresholds = [float(np.percentile(np.delete(calibration, i), 95)) for i in range(20)]
    outcomes = [((scores[labels == 0] > threshold).sum(),
                 (scores[labels == 1] <= threshold).sum()) for threshold in thresholds]
    metric = json.loads((group / 'metrics.json').read_text())
    return {'category': category, 'method': method,
            'original_threshold': metric['threshold'],
            'leave_one_out_threshold_min': min(thresholds),
            'leave_one_out_threshold_max': max(thresholds),
            'original_false_positives': metric['false_positives'],
            'original_false_negatives': metric['false_negatives'],
            'leave_one_out_false_positives_min': int(min(fp for fp, _ in outcomes)),
            'leave_one_out_false_positives_max': int(max(fp for fp, _ in outcomes)),
            'leave_one_out_false_negatives_min': int(min(fn for _, fn in outcomes)),
            'leave_one_out_false_negatives_max': int(max(fn for _, fn in outcomes))}


def main():
    categories = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')
    rows = [result for method in ('official_448', 'high_res_672', 'multi_res_448_672')
            for category in categories if (result := audit(method, category))]
    write_json(OUTPUT, {'protocol': '4-shot seed 0; threshold leave-one-normal-calibration-out diagnostic',
                        'test_labels': 'evaluation only; never used to choose a threshold',
                        'results': rows})
    for row in rows:
        print(row['method'], row['category'],
              'FP', row['original_false_positives'],
              f"[{row['leave_one_out_false_positives_min']}, {row['leave_one_out_false_positives_max']}]",
              'FN', row['original_false_negatives'],
              f"[{row['leave_one_out_false_negatives_min']}, {row['leave_one_out_false_negatives_max']}]",
              flush=True)


if __name__ == '__main__':
    main()
