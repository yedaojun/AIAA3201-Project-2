"""Check report follow-up table values against saved experiment JSON."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def main():
    tex = (ROOT / 'report/main.tex').read_text(encoding='utf-8')
    shot = load('results/part3/multiscale_fusion/all_shots_summary.json')
    official = load('results/part3/anyup_pixel_ap_4shot_3seeds.json')
    composed = load('results/part3/a_anyup_pixel_ap_4shot_3seeds.json')
    if shot['settings'] != 72:
        raise ValueError('Incomplete scale comparison')
    for references in (1, 2, 4, 8):
        row = shot['by_shot'][str(references)]
        def cell(method):
            return (f"{100*row[method]['image_auroc_mean']:.2f}"
                    f"$\\pm${100*row[method]['image_auroc_sample_sd']:.2f}")
        paired = f'{references} & {cell("official")} & {cell("two_layer")}'
        scale = f'{references} & {cell("high_res")} & \\bestscore{{{cell("fusion")}}}'
        for expected in (paired, scale):
            if expected not in tex:
                raise ValueError(f'Report multiscale row differs: {expected}')
    for category in CATEGORIES:
        ap_official = float(np.mean([official['categories'][category][str(seed)]
                                     ['official']['pixel_ap'] for seed in range(3)]))
        ap_a = float(np.mean([composed['categories'][category][str(seed)]
                              ['two_layer']['pixel_ap'] for seed in range(3)]))
        ap_anyup = float(np.mean([composed['categories'][category][str(seed)]
                                  ['two_layer_anyup']['pixel_ap'] for seed in range(3)]))
        expected = (category.replace('_', r'\_') +
                    f' & {100*ap_official:.2f} & {100*ap_a:.2f} & {100*ap_anyup:.2f}')
        if expected not in tex:
            raise ValueError(f'Report pixel AP row differs: {expected}')
    print('Verified all four shot rows and six three-group pixel-AP rows against saved JSON', flush=True)


if __name__ == '__main__':
    main()
