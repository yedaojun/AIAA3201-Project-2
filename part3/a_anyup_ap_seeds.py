"""Exact three-group four-shot pixel AP for A and A+AnyUp."""

import json

import numpy as np

from experiment_utils.common import ROOT, write_json
from .pixel_ap import score


OUT = ROOT / 'results/part3/a_anyup_pixel_ap_4shot_3seeds.json'
CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')
METHODS = ('two_layer', 'two_layer_anyup')


def main():
    result = (json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists()
              else {'protocol': '4-shot; three fixed reference groups; exact original-resolution pixel AP',
                    'methods': list(METHODS), 'normal_masks': 'all-zero and included',
                    'categories': {}})
    seed0 = json.loads((ROOT / 'results/part3/pixel_ap_4shot_seed0.json').read_text())
    for category in CATEGORIES:
        result['categories'].setdefault(category, {})
        for seed in range(3):
            group = result['categories'][category].setdefault(str(seed), {})
            for method in METHODS:
                if method in group:
                    continue
                if seed == 0:
                    entry = dict(seed0['categories'][category][method])
                    if method == 'two_layer':
                        metric = ROOT / 'results/part3/multilayer' / category / 'shot_4/seed_0/metrics.json'
                        entry['pixel_auroc'] = json.loads(metric.read_text())['pixel_auroc']
                    else:
                        metric = ROOT / 'results/part3/two_layer_anyup/shot_4_seed_0' / category / 'metrics.json'
                        entry['pixel_auroc'] = json.loads(metric.read_text())['pixel_auroc_two_layer_anyup']
                else:
                    ap, auc, pixels, positive = score(category, method, seed, include_auc=True)
                    entry = {'pixel_ap': ap, 'pixel_auroc': auc,
                             'pixels': pixels, 'positive_pixels': positive}
                group[method] = entry
                write_json(OUT, result)
                print(category, seed, method, f'{100*entry["pixel_ap"]:.2f}%', flush=True)
    result['macro_by_seed'] = [{
        'seed': seed,
        **{f'{method}_{metric}': float(np.mean([
            result['categories'][category][str(seed)][method][metric]
            for category in CATEGORIES])) for method in METHODS
            for metric in ('pixel_ap', 'pixel_auroc')}} for seed in range(3)]
    result['macro_mean_sd'] = {
        key: {'mean': float(np.mean([row[key] for row in result['macro_by_seed']])),
              'sample_sd': float(np.std([row[key] for row in result['macro_by_seed']], ddof=1))}
        for key in (f'{method}_{metric}' for method in METHODS
                    for metric in ('pixel_ap', 'pixel_auroc'))}
    write_json(OUT, result)
    print(json.dumps(result['macro_mean_sd'], indent=2), flush=True)


if __name__ == '__main__':
    main()
