"""Exact pixel metrics and paired image outcomes for the 672-resolution pilot."""

import json

import numpy as np

from experiment_utils.common import ROOT, write_json
from .pixel_ap import score


OUTPUT = ROOT / 'results/part3/high_resolution_pilot/pixel_metrics_4shot_seed0.json'
CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')


def main():
    result = (json.loads(OUTPUT.read_text(encoding='utf-8')) if OUTPUT.exists()
              else {'protocol': 'post-hoc 4-shot seed 0; exact original-resolution pixel metrics',
                    'normal_masks': 'all-zero; included', 'categories': {}})
    for category in CATEGORIES:
        if category in result['categories']:
            continue
        group = ROOT / 'results/part3/high_resolution_pilot' / category / 'shot_4/seed_0'
        if not (group / 'metrics.json').exists():
            print(f'{category}: no completed 672 pilot yet', flush=True)
            continue
        ap, auc, pixels, positives = score(category, 'high_res', seed=0, include_auc=True)
        baseline = json.loads((ROOT / 'results/part2' / category /
                               'shot_4/seed_0/metrics.json').read_text())
        result['categories'][category] = {
            'high_res_672_pixel_ap': ap, 'high_res_672_pixel_auroc': auc,
            'baseline_448_pixel_auroc': baseline['pixel_auroc'],
            'baseline_448_pixel_ap': json.loads((ROOT /
                'results/part3/pixel_ap_4shot_seed0.json').read_text())
                ['categories'][category]['official']['pixel_ap'],
            'pixels': pixels, 'positive_pixels': positives}
        write_json(OUTPUT, result)
        print(category, f'AP {100*ap:.2f}%, AUROC {100*auc:.2f}%', flush=True)
    if len(result['categories']) == len(CATEGORIES):
        result['macro'] = {name: float(np.mean([result['categories'][category][name]
                                               for category in CATEGORIES]))
                           for name in ('high_res_672_pixel_ap', 'high_res_672_pixel_auroc',
                                        'baseline_448_pixel_ap', 'baseline_448_pixel_auroc')}
        write_json(OUTPUT, result)
        print(json.dumps(result['macro'], indent=2), flush=True)


if __name__ == '__main__':
    main()
