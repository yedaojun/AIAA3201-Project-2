"""Post-hoc image-score audit with a fresh normal-only threshold per top-k rule.

Reads saved official patch maps; no feature model or reference bank is changed.
The rules are diagnostics chosen after inspecting test errors, not a validated
improvement. Each setting uses its own twenty disjoint normal calibrators.
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from experiment_utils.common import ROOT


OUT = ROOT / 'results/part3/calibrated_aggregation_audit'
CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')
FRACTIONS = (0.001, 0.005, 0.01, 0.02)


def records(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def topmean(patch, fraction):
    flat = np.asarray(patch).ravel()
    k = max(1, int(len(flat) * fraction))
    return float(np.partition(flat, len(flat) - k)[-k:].mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setting_rows = []
    cases = []
    for category in CATEGORIES:
        for seed in (0, 1, 2):
            folder = ROOT / 'results/part2' / category / 'shot_4' / f'seed_{seed}'
            calibration = records(folder / 'calibration.csv')
            predictions = records(folder / 'predictions.csv')
            if len(calibration) != 20:
                raise ValueError(f'Need 20 calibrators: {folder}')
            calibration_maps = [folder / 'anomaly_maps' / f'seed={seed}' /
                                category / 'test/good' / f'__calibration__{Path(r["file"]).stem}.npy'
                                for r in calibration]
            for path in calibration_maps:
                if not path.is_file():
                    raise FileNotFoundError(path)
            cal_scores = {f: np.array([topmean(np.load(path), f) for path in calibration_maps])
                          for f in FRACTIONS}
            test_scores = {f: [] for f in FRACTIONS}
            for row in predictions:
                patch = np.load(folder / row['anomaly_map'])
                for f in FRACTIONS:
                    test_scores[f].append(topmean(patch, f))
                if abs(test_scores[0.01][-1] - float(row['score'])) > 2e-5:
                    raise ValueError(f'Official score mismatch: {category} {seed} {row["file"]}')
            labels = np.array([int(r['label']) for r in predictions])
            for f in FRACTIONS:
                threshold = float(np.percentile(cal_scores[f], 95))
                scores = np.array(test_scores[f])
                decisions = scores > threshold
                setting_rows.append({'category': category, 'seed': seed, 'fraction': f,
                                     'threshold': threshold,
                                     'image_auroc': float(roc_auc_score(labels, scores)),
                                     'fp': int(np.sum((labels == 0) & decisions)),
                                     'fn': int(np.sum((labels == 1) & ~decisions))})
                for i, row in enumerate(predictions):
                    if category in ('candle', 'capsules') and row['file'] in (
                            'test/bad/000.JPG', 'test/bad/087.JPG'):
                        cases.append({'category': category, 'seed': seed, 'file': row['file'],
                                      'fraction': f, 'score': float(scores[i]),
                                      'threshold': threshold, 'decision': 'NG' if decisions[i] else 'OK'})
            print(category, seed, flush=True)
    with (OUT / 'category_seed.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(setting_rows[0]))
        writer.writeheader()
        writer.writerows(setting_rows)
    with (OUT / 'cases.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    summary = []
    for f in FRACTIONS:
        chosen = [r for r in setting_rows if r['fraction'] == f]
        per_seed = [np.mean([r['image_auroc'] for r in chosen if r['seed'] == seed])
                    for seed in (0, 1, 2)]
        summary.append({'fraction': f, 'mean_image_auroc': float(np.mean(per_seed)),
                        'sample_sd_image_auroc': float(np.std(per_seed, ddof=1)),
                        'fp_total': sum(r['fp'] for r in chosen),
                        'fn_total': sum(r['fn'] for r in chosen)})
    with (OUT / 'summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (OUT / 'protocol.json').write_text(json.dumps({
        'method': 'official DINO patch maps, four shots, reference groups 0/1/2',
        'calibration': '20 disjoint normal training images per category/group; 95th percentile recalculated for each fraction',
        'scope': 'post-hoc diagnostic from saved maps; fractions selected after error inspection',
        'test_label_use': 'retrospective outcome audit only; no independent validation'
    }, indent=2), encoding='utf-8')
    print(summary, flush=True)


if __name__ == '__main__':
    main()
