"""Exploratory normal-calibrated two-stage 448/672 detector, four shots.

First pass always uses official 448 DINO. A fixed ±25% band around its
normal-calibrated threshold triggers a 672 second pass. The two scores are
equally averaged after division by their own normal-calibrated thresholds.
Outside the band, the 448 score alone is retained. The final threshold uses
only the same twenty paired normal calibration images. No test labels set
the band, weights, or threshold. This is post-hoc exploratory work.
"""

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from experiment_utils.common import ROOT, write_json
from .multilayer import read_rows, save_csv


BAND = .25
CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')
OUT = ROOT / 'results/part3/selective_multiscale'


def paired(a, b):
    rows_a = {r['file']: r for r in read_rows(a)}
    rows_b = {r['file']: r for r in read_rows(b)}
    if rows_a.keys() != rows_b.keys():
        raise ValueError('Inputs are not paired')
    return [(name, rows_a[name], rows_b[name]) for name in sorted(rows_a)]


def gated_score(score448, score672, t448, t672):
    normalized448 = score448 / t448
    use_high_res = abs(normalized448 - 1) <= BAND
    score = (.5 * (normalized448 + score672 / t672) if use_high_res
             else normalized448)
    return score, use_high_res


def evaluate(category, seed):
    base = ROOT / 'results/part2' / category / 'shot_4' / f'seed_{seed}'
    fine = ROOT / 'results/part3/high_resolution_pilot' / category / 'shot_4' / f'seed_{seed}'
    low_metric = json.loads((base / 'metrics.json').read_text())
    high_metric = json.loads((fine / 'metrics.json').read_text())
    if low_metric['references'] != high_metric['references']:
        raise ValueError('Reference images differ')
    t448, t672 = low_metric['threshold'], high_metric['threshold']
    calibration = []
    for name, low, high in paired(base / 'calibration.csv', fine / 'calibration.csv'):
        score, triggered = gated_score(float(low['score']), float(high['score']), t448, t672)
        calibration.append({'file': name, 'score_448': float(low['score']),
                            'score_672': float(high['score']),
                            'selective_score': score, 'use_672': int(triggered)})
    if len(calibration) != 20:
        raise ValueError('Expected 20 normal calibration images')
    threshold = float(np.percentile([r['selective_score'] for r in calibration], 95))
    predictions = []
    for name, low, high in paired(base / 'predictions.csv', fine / 'predictions.csv'):
        if int(low['label']) != int(high['label']):
            raise ValueError('Labels differ')
        score, triggered = gated_score(float(low['score']), float(high['score']), t448, t672)
        label = int(low['label'])
        predicted = int(score > threshold)
        predictions.append({'category': category, 'shot': 4, 'seed': seed,
                            'file': name, 'label': label,
                            'score_448': float(low['score']),
                            'score_672': float(high['score']) if triggered else '',
                            'selective_score': score, 'threshold': threshold,
                            'use_672': int(triggered),
                            'decision': 'NG' if predicted else 'OK',
                            'outcome': ('TP' if predicted else 'FN') if label else
                                       ('FP' if predicted else 'TN')})
    result = {'category': category, 'shot': 4, 'seed': seed,
              'band': BAND, 'image_auroc': float(roc_auc_score(
                  [r['label'] for r in predictions],
                  [r['selective_score'] for r in predictions])),
              'false_positives': sum(r['outcome'] == 'FP' for r in predictions),
              'false_negatives': sum(r['outcome'] == 'FN' for r in predictions),
              'trigger_rate': float(np.mean([r['use_672'] for r in predictions])),
              'threshold': threshold,
              'baseline_image_auroc': low_metric['image_auroc'],
              'full_fusion_image_auroc': json.loads((ROOT / 'results/part3/multiscale_fusion' /
                    category / 'shot_4' / f'seed_{seed}' / 'metrics.json').read_text())['image_auroc'],
              'references': low_metric['references'],
              'calibration_files': low_metric['calibration_files']}
    folder = OUT / category / 'shot_4' / f'seed_{seed}'
    save_csv(folder / 'calibration.csv', calibration)
    save_csv(folder / 'predictions.csv', predictions)
    write_json(folder / 'metrics.json', result)
    print(category, seed, f"AUROC {100*result['image_auroc']:.2f}%",
          f"trigger {100*result['trigger_rate']:.1f}%", flush=True)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'protocol.json', {
        'status': 'post-hoc exploratory two-stage candidate; same test split as preceding pilots',
        'shots': 4, 'seeds': [0, 1, 2], 'band': BAND,
        'trigger': 'absolute difference between score_448 / normal_threshold_448 and 1 <= band',
        'triggered_score': 'equal average of 448 and 672 scores after normal-threshold scaling',
        'untriggered_score': '448 score after normal-threshold scaling',
        'final_threshold': '95th percentile of twenty same normal calibration images',
        'test_label_use': 'evaluation only'})
    rows = [evaluate(category, seed) for seed in range(3) for category in CATEGORIES]
    per_seed = [{
        'seed': seed,
        'image_auroc': float(np.mean([r['image_auroc'] for r in rows if r['seed'] == seed])),
        'trigger_rate': float(np.mean([r['trigger_rate'] for r in rows if r['seed'] == seed]))}
        for seed in range(3)]
    summary = {'macro_by_seed': per_seed,
               'macro_image_auroc_mean': float(np.mean([r['image_auroc'] for r in per_seed])),
               'macro_image_auroc_sd': float(np.std([r['image_auroc'] for r in per_seed], ddof=1)),
               'trigger_rate': float(np.mean([r['trigger_rate'] for r in rows])),
               'false_positives': sum(r['false_positives'] for r in rows),
               'false_negatives': sum(r['false_negatives'] for r in rows),
               'paired_settings': rows}
    write_json(OUT / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in (
        'macro_by_seed', 'macro_image_auroc_mean', 'macro_image_auroc_sd',
        'trigger_rate', 'false_positives', 'false_negatives')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
