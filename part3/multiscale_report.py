"""Paired three-seed audit of the exploratory 448+672 image-score fusion."""

import json

import numpy as np
from sklearn.metrics import roc_curve

from experiment_utils.common import ROOT, write_json
from .multilayer import read_rows, save_csv


CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')
OUT = ROOT / 'results/part3/multiscale_fusion'


def main():
    by_setting = []
    transitions = []
    for seed in range(3):
        for category in CATEGORIES:
            official_folder = ROOT / 'results/part2' / category / 'shot_4' / f'seed_{seed}'
            layer_folder = ROOT / 'results/part3/multilayer' / category / 'shot_4' / f'seed_{seed}'
            fine_folder = ROOT / 'results/part3/high_resolution_pilot' / category / 'shot_4' / f'seed_{seed}'
            fused_folder = OUT / category / 'shot_4' / f'seed_{seed}'
            paths = (official_folder / 'metrics.json', layer_folder / 'metrics.json',
                     fine_folder / 'metrics.json', fused_folder / 'metrics.json')
            if not all(path.exists() for path in paths):
                raise ValueError(f'Incomplete paired setting: {category}, seed {seed}')
            official, layer, fine, fused = [json.loads(path.read_text()) for path in paths]
            setting = {'category': category, 'seed': seed,
                               'official_image_auroc': official['image_auroc'],
                               'two_layer_image_auroc': layer['image_auroc'],
                               'high_res_image_auroc': fine['image_auroc'],
                               'fusion_image_auroc': fused['image_auroc'],
                               'official_FP': official['false_positives'],
                               'official_FN': official['false_negatives'],
                               'two_layer_FP': layer['FP'], 'two_layer_FN': layer['FN'],
                               'high_res_FP': fine['false_positives'],
                               'high_res_FN': fine['false_negatives'],
                               'fusion_FP': fused['false_positives'],
                               'fusion_FN': fused['false_negatives']}
            for method, folder, score_key in (
                    ('official', official_folder, 'score'),
                    ('two_layer', layer_folder, 'score'),
                    ('high_res', fine_folder, 'score'),
                    ('fusion', fused_folder, 'fused_score')):
                candidates = read_rows(folder / 'predictions.csv')
                fpr, tpr, _ = roc_curve([int(row['label']) for row in candidates],
                                        [float(row[score_key]) for row in candidates])
                for limit in (.05, .10):
                    setting[f'{method}_recall_at_test_fpr_{int(limit*100)}pct'] = float(tpr[fpr <= limit].max())
            by_setting.append(setting)
            a = {row['file']: row for row in read_rows(official_folder / 'predictions.csv')}
            b = {row['file']: row for row in read_rows(fused_folder / 'predictions.csv')}
            if a.keys() != b.keys():
                raise ValueError('Unpaired test rows')
            for file in sorted(a):
                before, after = a[file], b[file]
                transitions.append({'category': category, 'seed': seed, 'file': file,
                                    'label': int(before['label']),
                                    'official_score': float(before['score']),
                                    'official_threshold': float(before['threshold']),
                                    'official_outcome': ('TP' if before['decision'] == 'NG' else 'FN')
                                    if int(before['label']) else
                                    ('FP' if before['decision'] == 'NG' else 'TN'),
                                    'fusion_score': float(after['fused_score']),
                                    'fusion_threshold': float(after['fused_threshold']),
                                    'fusion_outcome': after['outcome']})
    save_csv(OUT / 'paired_settings_4shot_3seeds.csv', by_setting)
    save_csv(OUT / 'per_image_transitions_4shot_3seeds.csv', transitions)
    per_seed = []
    for seed in range(3):
        selected = [row for row in by_setting if row['seed'] == seed]
        per_seed.append({'seed': seed, **{
            method + '_image_auroc': float(np.mean([row[method + '_image_auroc']
                                                   for row in selected]))
            for method in ('official', 'two_layer', 'high_res', 'fusion')}})
    methods = ('official', 'two_layer', 'high_res', 'fusion')
    summary = {'protocol': '4-shot, all six categories, three fixed reference groups; post-hoc candidate',
               'macro_by_seed': per_seed,
               'macro_mean_sd': {method: {
                   'mean': float(np.mean([row[method + '_image_auroc'] for row in per_seed])),
                   'sample_sd': float(np.std([row[method + '_image_auroc'] for row in per_seed], ddof=1))}
                   for method in methods},
               'total_FP_FN': {method: {
                   'FP': int(sum(row[method + '_FP'] for row in by_setting)),
                   'FN': int(sum(row[method + '_FN'] for row in by_setting))}
                   for method in methods},
               'retrospective_recall_at_test_fpr': {
                   f'{int(limit*100)}pct': {method: float(np.mean([
                       row[f'{method}_recall_at_test_fpr_{int(limit*100)}pct']
                       for row in by_setting])) for method in methods}
                   for limit in (.05, .10)},
               'transitions_vs_official': {
                   label: sum(row['official_outcome'] == before and row['fusion_outcome'] == after
                              for row in transitions)
                   for label, before, after in (
                       ('FN_to_TP', 'FN', 'TP'), ('TP_to_FN', 'TP', 'FN'),
                       ('FP_to_TN', 'FP', 'TN'), ('TN_to_FP', 'TN', 'FP'))},
               'test_rows': len(transitions)}
    write_json(OUT / 'three_seed_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
