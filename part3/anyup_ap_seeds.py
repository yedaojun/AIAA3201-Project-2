"""Three-reference-group pixel AP audit for official DINO and score-map AnyUp.

Uses the fixed 4-shot reference filenames, complete VisA test split, and exact
original-resolution masks. Seed 0 is reused from the prior AP audit.
"""

import json

import numpy as np

from experiment_utils.common import ROOT, write_json
from .pixel_ap import score


CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')
METHODS = ('official', 'official_anyup')
OUTPUT = ROOT / 'results/part3/anyup_pixel_ap_4shot_3seeds.json'


def main():
    if OUTPUT.exists():
        result = json.loads(OUTPUT.read_text(encoding='utf-8'))
    else:
        result = {'protocol': '4-shot, three fixed reference groups, exact original-resolution pixel AP',
                  'normal_masks': 'included as all-zero masks',
                  'anyup': 'pretrained 2x score-map restoration, identical image scores',
                  'categories': {}}
    seed0 = json.loads((ROOT / 'results/part3/pixel_ap_4shot_seed0.json').read_text())
    for category in CATEGORIES:
        result['categories'].setdefault(category, {})
        for seed in range(3):
            key = str(seed)
            result['categories'][category].setdefault(key, {})
            for method in METHODS:
                if method in result['categories'][category][key]:
                    continue
                if seed == 0:
                    entry = seed0['categories'][category][method]
                else:
                    ap, total, positive = score(category, method, seed)
                    entry = {'pixel_ap': ap, 'pixels': total, 'positive_pixels': positive}
                result['categories'][category][key][method] = entry
                write_json(OUTPUT, result)
                print(category, seed, method, f'{100*entry["pixel_ap"]:.2f}%', flush=True)
    result['macro_by_seed'] = {
        str(seed): {method: float(np.mean([
            result['categories'][category][str(seed)][method]['pixel_ap']
            for category in CATEGORIES])) for method in METHODS}
        for seed in range(3)}
    result['macro_mean_sd'] = {
        method: {'mean': float(np.mean([result['macro_by_seed'][str(seed)][method]
                                        for seed in range(3)])),
                 'sample_sd': float(np.std([result['macro_by_seed'][str(seed)][method]
                                            for seed in range(3)], ddof=1))}
        for method in METHODS}
    write_json(OUTPUT, result)
    print(json.dumps(result['macro_mean_sd'], indent=2), flush=True)


if __name__ == '__main__':
    main()
