"""Normal-only threshold estimator sensitivity, with test outcomes reported afterward.

Four shots, three reference groups. Candidate thresholds are estimated from
the same twenty disjoint normal training images. Test labels never select an
estimator or tune a parameter. Bootstrap variability is a normal-calibration
diagnostic, not a confidence interval for test performance.
"""

import csv
import json

import numpy as np
from scipy.stats import gumbel_r

from experiment_utils.common import ROOT, write_json


CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')
OUT = ROOT / 'results/part3/calibration_methods_4shot_3seeds'
BOOTSTRAPS = 200


def records(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def threshold(scores, estimator):
    if estimator == 'empirical_95p':
        return float(np.percentile(scores, 95))
    if estimator == 'gaussian_mean_plus_1.645sd':
        return float(np.mean(scores) + 1.645 * np.std(scores, ddof=1))
    if estimator == 'gumbel_95p':
        loc, scale = gumbel_r.fit(scores)
        return float(gumbel_r.ppf(.95, loc, scale))
    raise ValueError(estimator)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3201)
    results = []
    for method, root, score_key in (
            ('official_448', ROOT / 'results/part2', 'score'),
            ('fixed_448_672', ROOT / 'results/part3/multiscale_fusion', 'fused_score')):
        for category in CATEGORIES:
            for seed in range(3):
                folder = root / category / 'shot_4' / f'seed_{seed}'
                calibration = np.array([float(row[score_key]) for row in
                                        records(folder / 'calibration.csv')])
                predictions = records(folder / 'predictions.csv')
                if len(calibration) != 20:
                    raise ValueError(f'Expected 20 normal calibration images: {folder}')
                labels = np.array([int(row['label']) for row in predictions])
                scores = np.array([float(row[score_key]) for row in predictions])
                if (labels == 0).sum() == 0 or (labels == 1).sum() == 0:
                    raise ValueError('Missing class in test split')
                resamples = rng.choice(calibration, size=(BOOTSTRAPS, 20), replace=True)
                for estimator in ('empirical_95p', 'gaussian_mean_plus_1.645sd', 'gumbel_95p'):
                    selected = threshold(calibration, estimator)
                    boot = np.array([threshold(row, estimator) for row in resamples])
                    if not np.isfinite(boot).all():
                        raise ValueError('Nonfinite bootstrap threshold')
                    results.append({
                        'method': method, 'category': category, 'seed': seed,
                        'estimator': estimator, 'normal_calibration_count': 20,
                        'threshold': selected,
                        'bootstrap_threshold_sd': float(np.std(boot, ddof=1)),
                        'bootstrap_threshold_p05': float(np.percentile(boot, 5)),
                        'bootstrap_threshold_p95': float(np.percentile(boot, 95)),
                        'test_false_positives': int(np.sum((labels == 0) & (scores > selected))),
                        'test_false_negatives': int(np.sum((labels == 1) & (scores <= selected)))})
        print(f'{method}: completed', flush=True)
    with (OUT / 'category_seed.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = []
    for method in ('official_448', 'fixed_448_672'):
        for estimator in ('empirical_95p', 'gaussian_mean_plus_1.645sd', 'gumbel_95p'):
            chosen = [r for r in results if r['method'] == method and r['estimator'] == estimator]
            summary.append({'method': method, 'estimator': estimator,
                            'settings': len(chosen),
                            'mean_bootstrap_threshold_sd': float(np.mean([
                                r['bootstrap_threshold_sd'] for r in chosen])),
                            'test_false_positives_total': sum(r['test_false_positives'] for r in chosen),
                            'test_false_negatives_total': sum(r['test_false_negatives'] for r in chosen)})
    with (OUT / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_json(OUT / 'protocol.json', {
        'scope': '4-shot, three fixed reference groups, six categories',
        'calibration_images': 'same twenty disjoint normal training images per setting',
        'bootstrap_resamples': BOOTSTRAPS, 'bootstrap_rng_seed': 3201,
        'estimators': ['empirical 95th percentile', 'mean + 1.645 sample standard deviations',
                       'fitted Gumbel 95th percentile'],
        'test_label_use': 'outcome audit only; no estimator selected from test results',
        'warning': 'Gumbel distribution is an assumption, not established by twenty observations'})
    for item in summary:
        print(item, flush=True)


if __name__ == '__main__':
    main()
