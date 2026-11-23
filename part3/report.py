"""Summarize paired evaluations and render fixed qualitative cases."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from anomaly_dino.official import utils
from experiment_utils.common import ROOT, write_json
from .run import CONFIGURATIONS, OUTPUT, map_file, read_inputs, read_rows, save_rows

CASES = (
    'candle/test/bad/000.JPG', 'capsules/test/bad/087.JPG', 'cashew/test/bad/063.JPG',
    'chewinggum/test/good/342.JPG', 'fryum/test/bad/031.JPG',
)


def plot_cases(output):
    grouped = read_inputs()
    cases = []
    fig, axes = plt.subplots(len(CASES), 4, figsize=(13, 13), layout='constrained')
    for index, suffix in enumerate(CASES):
        sample = grouped[f'VisA/1cls/{suffix}']
        row = sample[4, 0]
        with Image.open(ROOT / row['image_file']) as source:
            image = np.asarray(source.convert('RGB'))
        if row['ground_truth_mask']:
            with Image.open(ROOT / row['ground_truth_mask']) as source:
                mask = np.asarray(source) > 0
        else:
            mask = np.zeros(image.shape[:2], dtype=bool)
        baseline = utils.dists2map(np.load(ROOT / row['anomaly_map']), image.shape)
        with np.load(map_file(output, row)) as data:
            learned = utils.dists2map(data['anyup_2x'][CONFIGURATIONS.index((4, 0))].astype(np.float64), image.shape)
        y, x = np.where(mask)
        if x.size:
            radius = max(85, int(max(np.ptp(x), np.ptp(y)) * .85))
            cx, cy = (int(x.min()) + int(x.max())) // 2, (int(y.min()) + int(y.max())) // 2
            crop = np.s_[max(0, cy - radius):cy + radius, max(0, cx - radius):cx + radius]
        else:
            crop = np.s_[:, :]
        common_max = max(float(baseline.max()), float(learned.max()), 1e-8)
        for col, (array, caption) in enumerate(zip((image, mask, baseline, learned),
                                                   ('Input', 'Ground truth', 'Official', 'AnyUp'), strict=True)):
            axes[index, col].imshow(array[crop], cmap='gray' if col == 1 else 'inferno',
                                    vmin=0 if col else None, vmax=(common_max if col > 1 else 1),
                                    interpolation='nearest')
            axes[index, col].set_axis_off()
            if index == 0:
                axes[index, col].set_title(caption)
        axes[index, 0].text(.5, -.06, f'{suffix} · {row["outcome"]}', transform=axes[index, 0].transAxes,
                           ha='center', va='top', fontsize=9)
        cases.append({'sample_id': row['sample_id'], 'outcome_both_methods': row['outcome'],
                      'score': float(row['score']), 'threshold': float(row['threshold']),
                      'official_peak_inside_mask': bool(mask.ravel()[baseline.argmax()]),
                      'anyup_peak_inside_mask': bool(mask.ravel()[learned.argmax()]),
                      'original_defect_fraction': float(mask.mean())})
    fig.suptitle('Five fixed difficult cases · 4-shot, group 0', fontsize=15)
    fig.savefig(output / 'difficult_cases.png', dpi=145)
    plt.close(fig)
    write_json(output / 'difficult_cases.json', cases)


def summarize(output):
    pairs = [json.loads(path.read_text(encoding='utf-8')) for path in sorted((output / 'paired').glob('*.json'))]
    if not pairs:
        raise ValueError('Run paired evaluation first')
    groups = defaultdict(list)
    for row in pairs:
        row['anyup_delta_pp'] = 100 * (row['pixel_auroc_anyup_2x'] - row['pixel_auroc_official'])
        groups[row['shot'], row['seed']].append(row)
    keys = ('image_auroc_all_methods', 'pixel_auroc_official', 'pixel_auroc_bilinear_2x',
            'pixel_auroc_anyup_2x', 'anyup_delta_pp')
    seed_rows = [{'shot': shot, 'seed': seed, 'categories': len(rows),
                  **{key: statistics.mean(row[key] for row in rows) for key in keys}}
                 for (shot, seed), rows in sorted(groups.items()) if len(rows) == 6]
    shot_rows = []
    for shot in sorted({row['shot'] for row in seed_rows}):
        rows = [row for row in seed_rows if row['shot'] == shot]
        if len(rows) != 3:
            continue
        shot_rows.append({'shot': shot, **{f'{key}_{suffix}': fn(row[key] for row in rows)
                                          for key in keys for suffix, fn in
                                          (('mean', statistics.mean), ('sample_sd', statistics.stdev))}})
    save_rows(output / 'paired_category_results.csv', pairs)
    if seed_rows:
        save_rows(output / 'paired_seed_results.csv', seed_rows)
    if shot_rows:
        save_rows(output / 'shot_summary.csv', shot_rows)

    diagnostic_rows = [row for path in (output / 'paired').glob('*_localization_diagnostics.csv')
                       for row in read_rows(path)]
    diagnostics = {}
    for method in ('official_anomalydino', 'bilinear_2x', 'anyup_2x'):
        subset = [row for row in diagnostic_rows if row['method'] == method and row['peak_inside_gt']]
        tiny = [row for row in subset if float(row['defect_fraction']) <= .001]
        if subset:
            diagnostics[method] = {'images_with_defect': len(subset),
                                   'peak_hit_rate': statistics.mean(int(row['peak_inside_gt']) for row in subset),
                                   'tiny_defect_images': len(tiny),
                                   'tiny_peak_hit_rate': statistics.mean(int(row['peak_inside_gt']) for row in tiny) if tiny else None}
    summary = {'completed_configurations': len(pairs), 'expected_configurations': 72,
               'by_shot': shot_rows, 'localization_diagnostic_4shot_group0': diagnostics}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--figures', action='store_true')
    args = parser.parse_args()
    summarize(args.output)
    if args.figures:
        plot_cases(args.output)


if __name__ == '__main__':
    main()
