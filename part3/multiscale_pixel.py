"""Exact pixel AP/AUROC for fixed 448+672 anomaly-map fusion, 4-shot seed 0."""

import json

import numpy as np

from experiment_utils.common import ROOT, write_json
from .pixel_ap import score


OUTPUT = ROOT / 'results/part3/multiscale_fusion/pixel_metrics_4shot_seed0.json'
CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')


def main():
    result = (json.loads(OUTPUT.read_text(encoding='utf-8')) if OUTPUT.exists()
              else {'protocol': 'post-hoc 4-shot seed 0; equal threshold-normalized 448+672 maps',
                    'normal_masks': 'all-zero; included', 'categories': {}})
    for category in CATEGORIES:
        if category in result['categories']:
            continue
        ap, auc, pixels, positives = score(category, 'multi_res', seed=0, include_auc=True)
        result['categories'][category] = {'pixel_ap': ap, 'pixel_auroc': auc,
                                          'pixels': pixels, 'positive_pixels': positives}
        write_json(OUTPUT, result)
        print(category, f'AP {100*ap:.2f}%, AUROC {100*auc:.2f}%', flush=True)
    result['macro'] = {metric: float(np.mean([result['categories'][category][metric]
                                              for category in CATEGORIES]))
                       for metric in ('pixel_ap', 'pixel_auroc')}
    write_json(OUTPUT, result)
    print(json.dumps(result['macro'], indent=2), flush=True)


if __name__ == '__main__':
    main()
