"""Pair every official/fusion prediction and preserve all TP/TN/FP/FN changes."""

import json

from experiment_utils.common import ROOT, write_json
from .multilayer import read_rows, save_csv


OUT = ROOT / 'results/part3/multiscale_fusion'
CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')


def outcome(label, decision):
    return ('TP' if decision == 'NG' else 'FN') if label else ('FP' if decision == 'NG' else 'TN')


def main():
    paired = []
    counts = {}
    for shot in (1, 2, 4, 8):
        shot_counts = counts[str(shot)] = {}
        for seed in range(3):
            for category in CATEGORIES:
                first = ROOT / 'results/part2' / category / f'shot_{shot}' / f'seed_{seed}'
                second = OUT / category / f'shot_{shot}' / f'seed_{seed}'
                a = {r['file']: r for r in read_rows(first / 'predictions.csv')}
                b = {r['file']: r for r in read_rows(second / 'predictions.csv')}
                if a.keys() != b.keys():
                    raise ValueError(f'Unpaired files: {category} {shot} {seed}')
                for file in sorted(a):
                    before, after = a[file], b[file]
                    label = int(before['label'])
                    if label != int(after['label']):
                        raise ValueError(f'Unpaired labels: {category} {shot} {seed} {file}')
                    original = outcome(label, before['decision'])
                    changed = outcome(label, after['decision'])
                    transition = f'{original}_to_{changed}'
                    shot_counts[transition] = shot_counts.get(transition, 0) + 1
                    paired.append({'category': category, 'shot': shot, 'seed': seed,
                                   'file': file, 'label': label,
                                   'official_score': before['score'],
                                   'official_threshold': before['threshold'],
                                   'official_outcome': original,
                                   'fusion_score': after['fused_score'],
                                   'fusion_threshold': after['fused_threshold'],
                                   'fusion_outcome': changed, 'transition': transition})
    if len(paired) != 4 * 3 * 6 * 160:
        raise ValueError('Expected 11,520 paired decisions')
    save_csv(OUT / 'all_shots_per_image_transitions.csv', paired)
    write_json(OUT / 'all_shots_transition_counts.json', {'total_rows': len(paired),
                                                           'by_shot': counts})
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    main()
