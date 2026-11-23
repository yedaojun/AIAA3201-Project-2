"""Exact original-size pixel metrics for every completed 448+672 setting.

This is a post-hoc same-test-split experiment. Both branches use the frozen
backbone and matched normal reference and calibration files. The existing
four-shot three-seed results are reused without changing their values.
"""

import argparse
import json

import numpy as np

from experiment_utils.common import ROOT, write_json
from .multiscale_fusion import CATEGORIES, evaluate
from .pixel_ap import score


OUT = ROOT / 'results/part3/multiscale_fusion/pixel_metrics_all_shots_3seeds.json'
PREVIOUS = ROOT / 'results/part3/multiscale_fusion/pixel_metrics_4shot_3seeds.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--categories', nargs='+', choices=CATEGORIES, default=CATEGORIES)
    parser.add_argument('--shots', nargs='+', type=int, choices=(1, 2, 4, 8), default=[1, 2, 4, 8])
    args = parser.parse_args()
    result = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {
        'protocol': 'post-hoc 448+672 fusion, exact original-resolution pixel AP/AUROC',
        'normal_masks': 'included as all-zero masks', 'categories': {}}
    prior = json.loads(PREVIOUS.read_text(encoding='utf-8'))
    for category in args.categories:
        for shot in args.shots:
            for seed in range(3):
                key = f'{shot}_shot_seed_{seed}'
                group = result['categories'].setdefault(category, {})
                if key in group:
                    continue
                if shot == 4:
                    entry = prior['categories'][category][str(seed)]
                    ap, auc = entry['fusion_pixel_ap'], entry['fusion_pixel_auroc']
                    pixels, positives = entry['pixels'], entry['positive_pixels']
                else:
                    if evaluate(category, seed, shot) is None:
                        print(f'SKIP unavailable: {category} {shot}-shot seed {seed}', flush=True)
                        continue
                    ap, auc, pixels, positives = score(category, 'multi_res', seed,
                                                       include_auc=True, shot=shot)
                group[key] = {'pixel_ap': ap, 'pixel_auroc': auc,
                              'pixels': pixels, 'positive_pixels': positives}
                write_json(OUT, result)
                print(category, shot, seed, f'AP {100*ap:.2f}%, AUROC {100*auc:.2f}%', flush=True)
    summary = {}
    for shot in (1, 2, 4, 8):
        entries = [(category, seed, result['categories'].get(category, {}).get(f'{shot}_shot_seed_{seed}'))
                   for category in CATEGORIES for seed in range(3)]
        if not all(entry is not None for _, _, entry in entries):
            continue
        by_seed = [{metric: float(np.mean([
            entry[metric] for _, seed_of_entry, entry in entries if seed_of_entry == seed]))
            for metric in ('pixel_ap', 'pixel_auroc')} for seed in range(3)]
        summary[str(shot)] = {metric: {
            'mean': float(np.mean([row[metric] for row in by_seed])),
            'sample_sd': float(np.std([row[metric] for row in by_seed], ddof=1))}
            for metric in ('pixel_ap', 'pixel_auroc')}
    result['macro_mean_sd_by_shot'] = summary
    write_json(OUT, result)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
