"""Fixed equal-weight 448+672 DINO image-score fusion, matched pairs.

Exploratory Direction A hypothesis. Both branches use the same frozen backbone,
the selected reference images, and twenty disjoint normal calibration images. Each
branch is divided by its normal-calibrated 95th percentile before averaging.
The fused threshold is estimated from the paired normal calibration images.
No test labels are used for method setup or threshold choice.
"""

import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from experiment_utils.common import ROOT, write_json
from .multilayer import read_rows, save_csv


OUTPUT = ROOT / 'results/part3/multiscale_fusion'
CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')


def paired_records(first, second, key):
    a = {row[key]: row for row in read_rows(first)}
    b = {row[key]: row for row in read_rows(second)}
    if a.keys() != b.keys():
        raise ValueError(f'Unpaired input files: {first} versus {second}')
    return [(name, a[name], b[name]) for name in sorted(a)]


def evaluate(category, seed, shot=4):
    base = ROOT / 'results/part2' / category / f'shot_{shot}' / f'seed_{seed}'
    fine = ROOT / 'results/part3/high_resolution_pilot' / category / f'shot_{shot}' / f'seed_{seed}'
    if not (fine / 'metrics.json').exists():
        return None
    baseline = json.loads((base / 'metrics.json').read_text())
    high_res = json.loads((fine / 'metrics.json').read_text())
    if baseline['references'] != high_res['references'] or baseline['calibration_files'] != high_res['calibration_files']:
        raise ValueError('Reference or calibration images differ between scales')
    t448, t672 = baseline['threshold'], high_res['threshold']
    if t448 <= 0 or t672 <= 0:
        raise ValueError('Expected positive branch thresholds')
    calibrations = []
    for name, a, b in paired_records(base / 'calibration.csv', fine / 'calibration.csv', 'file'):
        calibrations.append({'file': name, 'score_448': float(a['score']),
                             'score_672': float(b['score']),
                             'fused_score': .5 * (float(a['score']) / t448 + float(b['score']) / t672)})
    if len(calibrations) != 20:
        raise ValueError('Expected 20 normal calibration images')
    threshold = float(np.percentile([r['fused_score'] for r in calibrations], 95))
    predictions = []
    for name, a, b in paired_records(base / 'predictions.csv', fine / 'predictions.csv', 'file'):
        if int(a['label']) != int(b['label']):
            raise ValueError('Test labels differ between scales')
        score448, score672 = float(a['score']), float(b['score'])
        score = .5 * (score448 / t448 + score672 / t672)
        label = int(a['label'])
        predicted = int(score > threshold)
        predictions.append({'category': category, 'shot': shot, 'seed': seed,
                            'file': name, 'label': label, 'score_448': score448,
                            'score_672': score672, 'fused_score': score,
                            'fused_threshold': threshold,
                            'decision': 'NG' if predicted else 'OK',
                            'outcome': ('TP' if predicted else 'FN') if label else
                                       ('FP' if predicted else 'TN')})
    result = {'category': category, 'shot': shot, 'seed': seed,
              'method': 'fixed equal-weight threshold-normalized 448+672 DINO score',
              'image_auroc': float(roc_auc_score([r['label'] for r in predictions],
                                                 [r['fused_score'] for r in predictions])),
              'false_positives': sum(r['outcome'] == 'FP' for r in predictions),
              'false_negatives': sum(r['outcome'] == 'FN' for r in predictions),
              'threshold': threshold,
              'baseline_448_image_auroc': baseline['image_auroc'],
              'high_res_672_image_auroc': high_res['image_auroc'],
              'reference_filenames': baseline['references'],
              'calibration_filenames': baseline['calibration_files']}
    folder = OUTPUT / category / f'shot_{shot}' / f'seed_{seed}'
    save_csv(folder / 'calibration.csv', calibrations)
    save_csv(folder / 'predictions.csv', predictions)
    write_json(folder / 'metrics.json', result)
    print(category, f'{shot}-shot', seed, f"448={100*baseline['image_auroc']:.2f}",
          f"672={100*high_res['image_auroc']:.2f}",
          f"fused={100*result['image_auroc']:.2f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shots', nargs='+', type=int, choices=(1, 2, 4, 8), default=[4])
    parser.add_argument('--seeds', nargs='+', type=int, choices=(0, 1, 2), default=[0])
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / 'protocol.json', {
        'method': 'fixed equal 448+672 branch average after dividing by branch normal 95th percentile',
        'fused_threshold': '95th percentile of twenty paired normal calibration scores',
        'weights': [0.5, 0.5], 'shots_available': [1, 2, 4, 8],
        'test_label_use': 'evaluation only',
        'status': 'post-hoc exploratory candidate; no independent holdout'})
    rows = [result for shot in args.shots for seed in args.seeds for category in CATEGORIES
            if (result := evaluate(category, seed, shot))]
    if rows:
        summary = {'categories': len(set(r['category'] for r in rows)),
                   'shots': sorted(set(r['shot'] for r in rows)),
                   'seeds': sorted(set(r['seed'] for r in rows)),
                   'macro_image_auroc_448': float(np.mean([r['baseline_448_image_auroc'] for r in rows])),
                   'macro_image_auroc_672': float(np.mean([r['high_res_672_image_auroc'] for r in rows])),
                   'macro_image_auroc_fused': float(np.mean([r['image_auroc'] for r in rows])),
                   'fused_false_positives': sum(r['false_positives'] for r in rows),
                   'fused_false_negatives': sum(r['false_negatives'] for r in rows)}
        write_json(OUTPUT / 'summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
