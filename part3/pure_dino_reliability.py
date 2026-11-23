"""Post-hoc unsupervised selective-decision audit for frozen DINO branches.

Neither source model has target-defect training. All decision thresholds come
from disjoint normal training images. Review ranking and risk/coverage numbers
are evaluated retrospectively on the test split, not tuned deployment rules.
"""

import csv
import json

import numpy as np
from sklearn.metrics import roc_curve

from experiment_utils.common import ROOT, write_json
from .multilayer import read_rows, save_csv


OUT = ROOT / 'results/part3/pure_dino_reliability_4shot_3seeds'
CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')


def result_rows(category, seed):
    base = ROOT / 'results/part2' / category / 'shot_4' / f'seed_{seed}'
    fine = ROOT / 'results/part3/high_resolution_pilot' / category / 'shot_4' / f'seed_{seed}'
    fusion = ROOT / 'results/part3/multiscale_fusion' / category / 'shot_4' / f'seed_{seed}'
    t448 = json.loads((base / 'metrics.json').read_text())['threshold']
    t672 = json.loads((fine / 'metrics.json').read_text())['threshold']
    tfused = json.loads((fusion / 'metrics.json').read_text())['threshold']
    a = {r['file']: r for r in read_rows(base / 'predictions.csv')}
    b = {r['file']: r for r in read_rows(fine / 'predictions.csv')}
    c = {r['file']: r for r in read_rows(fusion / 'predictions.csv')}
    if not (a.keys() == b.keys() == c.keys()):
        raise ValueError('Unpaired test files')
    rows = []
    for name in sorted(a):
        label = int(a[name]['label'])
        if label != int(b[name]['label']) or label != int(c[name]['label']):
            raise ValueError('Unpaired labels')
        s448 = float(a[name]['score']) / t448
        s672 = float(b[name]['score']) / t672
        sfused = float(c[name]['fused_score']) / tfused
        d448 = int(s448 > 1)
        d672 = int(s672 > 1)
        dfused = int(sfused > 1)
        consensus = min(abs(s448 - 1), abs(s672 - 1)) if d448 == d672 else 0.0
        rows.append({'category': category, 'seed': seed, 'file': name, 'label': label,
                     's448_normalized': s448, 's672_normalized': s672,
                     'sfused_normalized': sfused,
                     'official_error': int(d448 != label),
                     'fusion_error': int(dfused != label),
                     'official_false_negative': int(label and not d448),
                     'fusion_false_negative': int(label and not dfused),
                     'official_confidence': abs(s448 - 1),
                     'fusion_confidence': abs(sfused - 1),
                     'consensus_confidence': consensus,
                     'branch_disagreement': abs(s448 - s672)})
    return rows


def main():
    all_rows = []
    operating = []
    for category in CATEGORIES:
        for seed in range(3):
            rows = result_rows(category, seed)
            all_rows += rows
            for method, key in (('official', 's448_normalized'), ('fusion', 'sfused_normalized')):
                fpr, tpr, _ = roc_curve([r['label'] for r in rows], [r[key] for r in rows])
                for limit in (.01, .05):
                    operating.append({'category': category, 'seed': seed, 'method': method,
                                      'test_fpr_max': limit, 'test_recall': float(tpr[fpr <= limit].max())})
    save_csv(OUT / 'per_image.csv', all_rows)
    save_csv(OUT / 'operating_points.csv', operating)
    risk = []
    for category in CATEGORIES:
        for seed in range(3):
            rows = [r for r in all_rows if r['category'] == category and r['seed'] == seed]
            for method, rank in (('official', 'official_confidence'),
                                 ('fusion', 'fusion_confidence'),
                                 ('fusion_consensus', 'consensus_confidence')):
                error_key = 'official_error' if method == 'official' else 'fusion_error'
                fn_key = 'official_false_negative' if method == 'official' else 'fusion_false_negative'
                ranked = sorted(rows, key=lambda r: (-r[rank], r['file']))
                for coverage in (.8, .9, .95, 1.0):
                    keep = ranked[:round(coverage * len(ranked))]
                    review = ranked[len(keep):]
                    risk.append({'category': category, 'seed': seed, 'method': method,
                                 'coverage': coverage, 'accepted': len(keep),
                                 'reviewed': len(review),
                                 'accepted_error_rate': sum(r[error_key] for r in keep) / len(keep),
                                 'reviewed_false_negatives': sum(r[fn_key] for r in review),
                                 'total_false_negatives': sum(r[fn_key] for r in rows)})
    save_csv(OUT / 'risk_coverage.csv', risk)
    summary = {'protocol': 'post-hoc, 4-shot, 3 fixed reference groups, six VisA categories',
               'supervision': 'frozen DINOv2 only; 448 and 672 branches share normal references',
               'threshold': 'each branch and fusion independently calibrated with 20 disjoint normal training files',
               'warning': 'test-set evaluation, no independent holdout; REVIEW is unresolved, not corrected',
               'macro_test_recall': {method: {str(int(100*limit)): float(np.mean([
                   r['test_recall'] for r in operating if r['method'] == method
                   and r['test_fpr_max'] == limit])) for limit in (.01, .05)}
                   for method in ('official', 'fusion')},
               'macro_risk': {method: {str(int(100*coverage)): {
                   'accepted_error_rate': float(np.mean([r['accepted_error_rate'] for r in risk
                       if r['method'] == method and r['coverage'] == coverage])),
                   'reviewed_false_negatives': int(sum(r['reviewed_false_negatives'] for r in risk
                       if r['method'] == method and r['coverage'] == coverage))}
                   for coverage in (.8, .9, .95, 1.0)}
                   for method in ('official', 'fusion', 'fusion_consensus')}}
    write_json(OUT / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
