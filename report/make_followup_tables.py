"""Render report tables from saved, complete Part 3 experiment records."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'report/tables'
CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    official = load('results/part3/anyup_pixel_ap_4shot_3seeds.json')
    composed = load('results/part3/a_anyup_pixel_ap_4shot_3seeds.json')
    pixel_rows = []
    values = {'official': [], 'two_layer': [], 'two_layer_anyup': []}
    for category in CATEGORIES:
        means = []
        for method in values:
            by_seed = [official['categories'][category][str(seed)]['official']['pixel_ap']
                       if method == 'official' else
                       composed['categories'][category][str(seed)][method]['pixel_ap']
                       for seed in range(3)]
            if len(by_seed) != 3:
                raise ValueError(f'Incomplete pixel AP: {category} {method}')
            values[method].append(by_seed)
            means.append(float(np.mean(by_seed)))
        display = category.replace('_', r'\_')
        pixel_rows.append(display + ' & ' + ' & '.join(f'{100*x:.2f}' for x in means) + r'\\')
    macro = {method: [float(np.mean([values[method][i][seed]
                                    for i in range(len(CATEGORIES))])) for seed in range(3)]
             for method in values}
    pixel_rows += [r'\midrule',
                   'Macro mean & ' + ' & '.join(f'{100*np.mean(macro[method]):.2f}'
                                               for method in values) + r'\\',
                   'Macro sample SD & ' + ' & '.join(f'{100*np.std(macro[method], ddof=1):.2f}'
                                                    for method in values) + r'\\']
    (TABLES / 'pixel_ap_three_groups_rows.tex').write_text('\n'.join(pixel_rows) + '\n', encoding='utf-8')
    scale_path = ROOT / 'results/part3/multiscale_fusion/all_shots_summary.json'
    if not scale_path.exists():
        print('Wrote three-group pixel AP table; awaiting complete scale sweep', flush=True)
        return
    scales = load('results/part3/multiscale_fusion/all_shots_summary.json')
    if scales['settings'] != 72:
        raise ValueError('Expected all 72 category/shot/group settings')
    shot_rows = []
    for shot in (1, 2, 4, 8):
        entry = scales['by_shot'][str(shot)]
        formatted = [f"{100*entry[name]['image_auroc_mean']:.2f}$\\pm$"
                     f"{100*entry[name]['image_auroc_sample_sd']:.2f}"
                     for name in ('official', 'two_layer', 'high_res', 'fusion')]
        shot_rows.append(str(shot) + ' & ' + ' & '.join(formatted) + r'\\')
    (TABLES / 'multiscale_all_shots_rows.tex').write_text('\n'.join(shot_rows) + '\n', encoding='utf-8')
    print('Wrote two complete report tables:', TABLES, flush=True)


if __name__ == '__main__':
    main()
