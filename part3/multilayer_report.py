"""Summarize Direction A; optionally compute exact original-resolution pixel AUROC."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from anomaly_dino.metrics import positive_negative_auc
from anomaly_dino.official import utils
from experiment_utils.common import DATA, ROOT, write_json
from .multilayer import OUTPUT, read_rows, save_csv


def pixel_auc(category, predictions):
    total, positives = 0, 0
    sizes = {}
    for row in predictions:
        image = ROOT / row['image_file']
        with Image.open(image) as source:
            sizes[row['sample_id']] = (source.height, source.width)
            total += source.height * source.width
        if int(row['label']):
            mask_path = DATA / category / 'ground_truth/bad' / (image.stem + '.png')
            with Image.open(mask_path) as source:
                positives += int(np.count_nonzero(np.asarray(source)))
    p = np.empty(positives, dtype=np.float64)
    n = np.empty(total - positives, dtype=np.float64)
    pi = ni = 0
    for row in predictions:
        image = ROOT / row['image_file']
        shape = sizes[row['sample_id']]
        dense = utils.dists2map(np.load(ROOT / row['anomaly_map']), shape)
        if int(row['label']):
            mask_path = DATA / category / 'ground_truth/bad' / (image.stem + '.png')
            with Image.open(mask_path) as source:
                mask = np.asarray(source) != 0
        else:
            mask = np.zeros(shape, dtype=bool)
        if dense.shape != mask.shape:
            raise ValueError(f'Map-mask shape mismatch for {image}')
        selected = dense[mask]
        unselected = dense[~mask]
        p[pi:pi + len(selected)] = selected
        n[ni:ni + len(unselected)] = unselected
        pi += len(selected)
        ni += len(unselected)
    if pi != positives or ni != total - positives:
        raise ValueError('Pixel counts differ from dataset dimensions')
    return positive_negative_auc(p, n), total, positives


def report(with_pixels):
    files = sorted(OUTPUT.glob('*/shot_*/seed_*/metrics.json'))
    if not files:
        raise ValueError('No completed multilayer settings')
    baseline = {(r['category'], int(r['shots']), int(r['seed'])): r
                for r in read_rows(ROOT / 'results/part2/runs.csv')}
    entries = []
    for file in files:
        metric = json.loads(file.read_text(encoding='utf-8'))
        metric['baseline_pixel_auroc'] = float(baseline[metric['category'], metric['shots'],
                                                       metric['seed']]['pixel_auroc'])
        if with_pixels and 'pixel_auroc' not in metric:
            predictions = read_rows(file.parent / 'predictions.csv')
            metric['pixel_auroc'], metric['pixel_count'], metric['positive_pixels'] = pixel_auc(metric['category'], predictions)
            write_json(file, metric)
            print(f'{metric["category"]} {metric["shots"]}-shot seed {metric["seed"]}: '
                  f'pixel AUROC {metric["pixel_auroc"]:.5f}', flush=True)
        entries.append(metric)
    flat = [{k: v for k, v in entry.items() if not isinstance(v, (dict, list))} for entry in entries]
    fields = sorted(set().union(*(entry.keys() for entry in flat)))
    save_csv(OUTPUT / 'category_results.csv', [{k: e.get(k, '') for k in fields} for e in flat])
    summary = []
    for shot in (1, 2, 4, 8):
        per_shot = [r for r in entries if r['shots'] == shot]
        if len(per_shot) != 18:
            continue
        by_seed = []
        for seed in (0, 1, 2):
            group = [r for r in per_shot if r['seed'] == seed]
            if len(group) != 6:
                raise ValueError('Incomplete six-category seed')
            by_seed.append(np.mean([r['image_auroc'] for r in group]))
        item = {'shot': shot, 'image_auroc_mean': float(np.mean(by_seed)),
                'image_auroc_sample_sd': float(np.std(by_seed, ddof=1)),
                'baseline_image_auroc_mean': float(np.mean([r['baseline_image_auroc'] for r in per_shot])),
                'completed_categories_and_seeds': len(per_shot)}
        if all('pixel_auroc' in r for r in per_shot):
            pixels = [float(np.mean([r['pixel_auroc'] for r in per_shot if r['seed'] == seed]))
                      for seed in (0, 1, 2)]
            item.update({'pixel_auroc_mean': float(np.mean(pixels)),
                         'pixel_auroc_sample_sd': float(np.std(pixels, ddof=1)),
                         'baseline_pixel_auroc_mean': float(np.mean([r['baseline_pixel_auroc']
                                                                      for r in per_shot]))})
        summary.append(item)
    if summary:
        columns = list(dict.fromkeys(key for row in summary for key in row))
        save_csv(OUTPUT / 'shot_summary.csv', [{key: row.get(key, '') for key in columns}
                                               for row in summary])
    write_json(OUTPUT / 'summary.json', {'completed_configurations': len(entries),
                                         'expected_configurations': 72,
                                         'by_shot': summary})
    print(json.dumps({'completed': len(entries), 'full_shot_summaries': summary}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pixels', action='store_true')
    args = parser.parse_args()
    report(args.pixels)
