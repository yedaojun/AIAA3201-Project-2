"""Retrospective score-aggregation diagnostic from saved patch maps (no model runs)."""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from experiment_utils.common import ROOT

OUT = ROOT / 'results/part3/aggregation_screen'
CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')


def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def topmean(values, k):
    values = np.asarray(values).ravel()
    k = min(len(values), max(1, int(k)))
    return float(np.partition(values, len(values) - k)[-k:].mean())


def scores(patch):
    flat = patch.ravel()
    foreground = flat[flat > 0]
    if not len(foreground):
        foreground = flat
    center = float(np.median(foreground))
    values = {}
    for fraction in (.001, .0025, .005, .01, .02):
        k = max(1, int(len(flat) * fraction))  # floor matches the official top-1% code
        values[f'full_{fraction:g}'] = topmean(flat, k)
    for k in (1, 3, 5):
        values[f'fg_top{k}'] = topmean(foreground, k)
        values[f'fg_centered_top{k}'] = topmean(np.maximum(0, foreground - center), k)
    return values


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for method in ('official', 'two_layer'):
        for seed in (0, 1, 2):
            for category in CATEGORIES:
                folder = (ROOT / 'results/part2' / category / 'shot_4' / f'seed_{seed}'
                          if method == 'official' else
                          ROOT / 'results/part3/multilayer' / category / 'shot_4' / f'seed_{seed}')
                predictions = read_csv(folder / 'predictions.csv')
                rows = []
                for row in predictions:
                    map_path = folder / row['anomaly_map'] if method == 'official' else ROOT / row['anomaly_map']
                    patch = np.load(map_path)
                    variants = scores(patch)
                    if abs(variants['full_0.01'] - float(row['score'])) > 2e-5:
                        raise ValueError(f'Saved score mismatch: {map_path}')
                    rows.append({'category': category, 'method': method, 'seed': seed,
                                 'file': row['file'] if method == 'official' else row['sample_id'],
                                 'label': int(row['label']), **variants})
                keys = [k for k in rows[0] if k.startswith(('full_', 'fg_'))]
                for key in keys:
                    summary.append({'method': method, 'seed': seed, 'category': category,
                                    'variant': key,
                                    'image_auroc': roc_auc_score([r['label'] for r in rows],
                                                                [r[key] for r in rows])})
                print(method, seed, category, flush=True)
    with open(OUT / 'category_seed_auc.csv', 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    grouped = {}
    for row in summary:
        grouped.setdefault((row['method'], row['variant'], row['seed']), []).append(row['image_auroc'])
    macro = [{'method': method, 'variant': variant, 'seed': seed,
              'macro_image_auroc': float(np.mean(values))}
             for (method, variant, seed), values in grouped.items()]
    with open(OUT / 'macro_by_seed.csv', 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(macro[0]))
        writer.writeheader()
        writer.writerows(macro)
    aggregate = {}
    for row in macro:
        aggregate.setdefault((row['method'], row['variant']), []).append(row['macro_image_auroc'])
    result = sorted(({'method': method, 'variant': variant,
                      'mean': float(np.mean(values)), 'sample_sd': float(np.std(values, ddof=1))}
                     for (method, variant), values in aggregate.items()),
                    key=lambda r: (r['method'], -r['mean']))
    with open(OUT / 'macro_mean_sd.csv', 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    (OUT / 'protocol.json').write_text(json.dumps({
        'scope': 'post-hoc score diagnostic from saved 4-shot test patch maps, seeds 0/1/2',
        'no_new_reference_or_model': True,
        'foreground_proxy': 'strictly positive query patch distance; official PCA masked background is zero',
        'calibration': 'not rerun; AUROC only, no thresholded decision claim',
        'selection_warning': 'test-set screen, not an independent validation result'}, indent=2), encoding='utf-8')
    for row in result:
        print(row['method'], row['variant'], f"{100*row['mean']:.2f} ± {100*row['sample_sd']:.2f}", flush=True)


if __name__ == '__main__':
    main()
