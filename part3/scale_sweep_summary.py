"""Verify and summarize all 72 paired 448/672 settings by reference count."""

import json

import numpy as np

from experiment_utils.common import ROOT, write_json
from .multilayer import save_csv


CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')
OUT = ROOT / 'results/part3/multiscale_fusion'


def main():
    records = []
    for shot in (1, 2, 4, 8):
        for seed in range(3):
            for category in CATEGORIES:
                paths = {
                    'official': ROOT / 'results/part2' / category / f'shot_{shot}' / f'seed_{seed}' / 'metrics.json',
                    'two_layer': ROOT / 'results/part3/multilayer' / category / f'shot_{shot}' / f'seed_{seed}' / 'metrics.json',
                    'high_res': ROOT / 'results/part3/high_resolution_pilot' / category / f'shot_{shot}' / f'seed_{seed}' / 'metrics.json',
                    'fusion': OUT / category / f'shot_{shot}' / f'seed_{seed}' / 'metrics.json'}
                if not all(path.exists() for path in paths.values()):
                    raise ValueError(f'Incomplete paired setting: {category} {shot} seed {seed}')
                m = {name: json.loads(path.read_text(encoding='utf-8'))
                     for name, path in paths.items()}
                refs = m['official']['references']
                if not (refs == m['two_layer']['reference_filenames'] == m['high_res']['references']
                        == m['fusion']['reference_filenames']):
                    raise ValueError(f'Reference filenames differ: {category} {shot} seed {seed}')
                record = {'category': category, 'shot': shot, 'seed': seed}
                for name, entry in m.items():
                    record[f'{name}_image_auroc'] = entry['image_auroc']
                    record[f'{name}_FP'] = entry['FP'] if name == 'two_layer' else entry['false_positives']
                    record[f'{name}_FN'] = entry['FN'] if name == 'two_layer' else entry['false_negatives']
                records.append(record)
    save_csv(OUT / 'all_shots_category_seed.csv', records)
    macro = []
    for shot in (1, 2, 4, 8):
        for seed in range(3):
            selected = [r for r in records if r['shot'] == shot and r['seed'] == seed]
            macro.append({'shot': shot, 'seed': seed, **{
                name: float(np.mean([r[f'{name}_image_auroc'] for r in selected]))
                for name in ('official', 'two_layer', 'high_res', 'fusion')}})
    save_csv(OUT / 'all_shots_macro_by_seed.csv', macro)
    summary = {'protocol': 'six categories, 1/2/4/8 references, groups 0/1/2, identical filenames;'
                          ' post-hoc 672 and fusion on the same VisA test split',
               'settings': len(records), 'by_shot': {}}
    for shot in (1, 2, 4, 8):
        chosen = [r for r in macro if r['shot'] == shot]
        settings = [r for r in records if r['shot'] == shot]
        summary['by_shot'][str(shot)] = {name: {
            'image_auroc_mean': float(np.mean([r[name] for r in chosen])),
            'image_auroc_sample_sd': float(np.std([r[name] for r in chosen], ddof=1)),
            'FP': int(sum(r[f'{name}_FP'] for r in settings)),
            'FN': int(sum(r[f'{name}_FN'] for r in settings))}
            for name in ('official', 'two_layer', 'high_res', 'fusion')}
    write_json(OUT / 'all_shots_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
