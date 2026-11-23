"""Three-group pixel AP and AUROC for fixed 448+672 map fusion, four shots."""

import json

import numpy as np

from experiment_utils.common import ROOT, write_json
from .pixel_ap import score


OUT = ROOT / 'results/part3/multiscale_fusion/pixel_metrics_4shot_3seeds.json'
CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')


def main():
    result = (json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists()
              else {'protocol': '4-shot; three fixed groups; exact original-resolution pixel metrics',
                    'maps': 'equal threshold-normalized 448+672 anomaly maps',
                    'normal_masks': 'all-zero; included', 'categories': {}})
    previous = json.loads((ROOT / 'results/part3/multiscale_fusion/pixel_metrics_4shot_seed0.json').read_text())
    baseline_ap = json.loads((ROOT / 'results/part3/anyup_pixel_ap_4shot_3seeds.json').read_text())
    for category in CATEGORIES:
        result['categories'].setdefault(category, {})
        for seed in range(3):
            key = str(seed)
            if key in result['categories'][category]:
                continue
            if seed == 0:
                entry = previous['categories'][category]
                ap, auc = entry['pixel_ap'], entry['pixel_auroc']
                pixels, positives = entry['pixels'], entry['positive_pixels']
            else:
                ap, auc, pixels, positives = score(category, 'multi_res', seed=seed, include_auc=True)
            official = json.loads((ROOT / 'results/part2' / category / 'shot_4' /
                                   f'seed_{seed}' / 'metrics.json').read_text())
            result['categories'][category][key] = {
                'fusion_pixel_ap': ap, 'fusion_pixel_auroc': auc,
                'official_pixel_ap': baseline_ap['categories'][category][key]['official']['pixel_ap'],
                'official_pixel_auroc': official['pixel_auroc'],
                'pixels': pixels, 'positive_pixels': positives}
            write_json(OUT, result)
            print(category, seed, f'fusion AP {100*ap:.2f}%, AUROC {100*auc:.2f}%', flush=True)
    result['macro_by_seed'] = [{
        'seed': seed, **{name: float(np.mean([
            result['categories'][category][str(seed)][name] for category in CATEGORIES]))
            for name in ('fusion_pixel_ap', 'fusion_pixel_auroc',
                         'official_pixel_ap', 'official_pixel_auroc')}} for seed in range(3)]
    result['macro_mean_sd'] = {
        name: {'mean': float(np.mean([row[name] for row in result['macro_by_seed']])),
               'sample_sd': float(np.std([row[name] for row in result['macro_by_seed']], ddof=1))}
        for name in ('fusion_pixel_ap', 'fusion_pixel_auroc',
                     'official_pixel_ap', 'official_pixel_auroc')}
    write_json(OUT, result)
    print(json.dumps(result['macro_mean_sd'], indent=2), flush=True)


if __name__ == '__main__':
    main()
