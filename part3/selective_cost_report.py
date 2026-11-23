"""Paired stage-time estimate for the selective 672 branch on 120 fixed queries."""

import json

import numpy as np

from experiment_utils.common import ROOT, write_json
from .multilayer import read_rows


def main():
    benchmark = json.loads((ROOT / 'results/part3/multiscale_fusion/benchmark_4shot_seed0.json').read_text())
    selected = {}
    for category in ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum'):
        folder = ROOT / 'results/part3/selective_multiscale' / category / 'shot_4/seed_0'
        for row in read_rows(folder / 'predictions.csv'):
            selected[category, row['file']] = bool(int(row['use_672']))
    rows = []
    for sample in benchmark['per_image']:
        category = sample['category']
        file = sample['image'].replace('\\', '/').split('/' + category + '/', 1)[1]
        use_672 = selected[category, file]
        rows.append({'category': category, 'file': file,
                     'use_672': use_672,
                     'ms_448': sample['ms_448'],
                     'ms_672_if_used': sample['ms_672'] if use_672 else 0,
                     'estimated_ms': sample['ms_448'] + (sample['ms_672'] if use_672 else 0)})
    result = {'status': 'paired component-time estimate, not a direct conditional-path benchmark',
              'sample_count': len(rows),
              'trigger_rate': float(np.mean([row['use_672'] for row in rows])),
              'mean_ms_448': float(np.mean([row['ms_448'] for row in rows])),
              'mean_ms_full_448_672': benchmark['mean_ms_fused'],
              'estimated_mean_ms_selective': float(np.mean([row['estimated_ms'] for row in rows])),
              'per_image': rows}
    write_json(ROOT / 'results/part3/selective_multiscale/cost_estimate_120.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'per_image'},
                     indent=2), flush=True)


if __name__ == '__main__':
    main()
