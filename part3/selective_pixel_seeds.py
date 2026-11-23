"""Exact pixel metrics for the fixed two-stage 448/672 pilot, four shots."""

import json

import numpy as np

from experiment_utils.common import ROOT, write_json
from .pixel_ap import score


OUT = ROOT / 'results/part3/selective_multiscale/pixel_metrics_4shot_3seeds.json'
CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')


def main():
    result = (json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists()
              else {'protocol': '4-shot, three groups; exact original-resolution pixel metrics',
                    'map_rule': '448 normalized map unless fixed threshold band triggers equal 448+672 map',
                    'normal_masks': 'all-zero; included', 'categories': {}})
    for category in CATEGORIES:
        result['categories'].setdefault(category, {})
        for seed in range(3):
            key = str(seed)
            if key in result['categories'][category]:
                continue
            ap, auc, pixels, positives = score(category, 'selective_multi_res', seed=seed,
                                               include_auc=True)
            result['categories'][category][key] = {
                'pixel_ap': ap, 'pixel_auroc': auc,
                'pixels': pixels, 'positive_pixels': positives}
            write_json(OUT, result)
            print(category, seed, f'AP {100*ap:.2f}%, AUROC {100*auc:.2f}%', flush=True)
    result['macro_by_seed'] = [{
        'seed': seed, **{metric: float(np.mean([
            result['categories'][category][str(seed)][metric] for category in CATEGORIES]))
            for metric in ('pixel_ap', 'pixel_auroc')}} for seed in range(3)]
    result['macro_mean_sd'] = {
        metric: {'mean': float(np.mean([row[metric] for row in result['macro_by_seed']])),
                 'sample_sd': float(np.std([row[metric] for row in result['macro_by_seed']], ddof=1))}
        for metric in ('pixel_ap', 'pixel_auroc')}
    write_json(OUT, result)
    print(json.dumps(result['macro_mean_sd'], indent=2), flush=True)


if __name__ == '__main__':
    main()
