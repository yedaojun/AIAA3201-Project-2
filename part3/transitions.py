"""Paired per-image outcome transitions between unchanged Part 2 and Part 3 branches."""

import csv

from experiment_utils.common import ROOT
from .multilayer import OUTPUT as A_OUTPUT, read_rows, save_csv
from .multilayer_fusion import OUTPUT as AD_OUTPUT
from .fusion_selective import OUTPUT as D_OUTPUT

OUTPUT = ROOT / 'results/part3'


def outcome(label, decision):
    return ('TP' if decision == 'NG' else 'FN') if label else ('FP' if decision == 'NG' else 'TN')


def main():
    baseline = {(r['category'], int(r['shots']), int(r['seed']), r['sample_id']): r
                for r in read_rows(ROOT / 'results/part2/predictions_all.csv')}
    d_rows = {(r['category'], int(r['shots']), int(r['seed']), r['sample_id']): r
              for r in read_rows(D_OUTPUT / 'predictions_all.csv')}
    ad_rows = {(r['category'], int(r['shots']), int(r['seed']), r['sample_id']): r
               for r in read_rows(AD_OUTPUT / 'predictions_all.csv')}
    paired = []
    for file in sorted(A_OUTPUT.glob('*/shot_*/seed_*/predictions.csv')):
        if not (file.parent / 'metrics.json').exists():
            continue
        for a in read_rows(file):
            key = (a['category'], int(a['shots']), int(a['seed']), a['sample_id'])
            b, d, ad = baseline[key], d_rows[key], ad_rows[key]
            label = int(a['label'])
            if label != int(b['ground_truth_label']) or label != int(d['label']) or label != int(ad['label']):
                raise ValueError(f'Paired ground-truth mismatch for {key}')
            paired.append({'category': key[0], 'shots': key[1], 'seed': key[2],
                           'sample_id': key[3], 'image_file': a['image_file'], 'label': label,
                           'reference_images_json': b['reference_images_json'],
                           'official_score': b['score'], 'official_threshold': b['threshold'],
                           'official_outcome': b['outcome'],
                           'A_score': a['score'], 'A_threshold': a['threshold'],
                           'A_outcome': a['outcome'],
                           'D_score': d['fused_score'], 'D_threshold': d['fused_threshold'],
                           'D_outcome': outcome(label, d['fused_decision']),
                           'D_gate_outcome': outcome(label, d['gated_decision']),
                           'A_D_score': ad['fused_score'], 'A_D_threshold': ad['fused_threshold'],
                           'A_D_outcome': outcome(label, ad['fused_decision']),
                           'A_D_F_decision': ad['selective_decision'],
                           'A_D_F_review': int(ad['selective_decision'] == 'REVIEW')})
    if not paired:
        raise ValueError('Run A and A+D before creating transitions')
    save_csv(OUTPUT / 'transitions.csv', paired)
    summaries = []
    for shot in (1, 2, 4, 8):
        for seed in (0, 1, 2):
            group = [r for r in paired if r['shots'] == shot and r['seed'] == seed]
            if len(group) != 960:
                continue
            item = {'shots': shot, 'seed': seed, 'images': len(group)}
            for variant in ('A_outcome', 'D_outcome', 'D_gate_outcome', 'A_D_outcome'):
                item[variant + '_FN_to_TP'] = sum(r['official_outcome'] == 'FN' and r[variant] == 'TP'
                                                  for r in group)
                item[variant + '_FP_to_TN'] = sum(r['official_outcome'] == 'FP' and r[variant] == 'TN'
                                                  for r in group)
                item[variant + '_TP_to_FN'] = sum(r['official_outcome'] == 'TP' and r[variant] == 'FN'
                                                  for r in group)
                item[variant + '_TN_to_FP'] = sum(r['official_outcome'] == 'TN' and r[variant] == 'FP'
                                                  for r in group)
            item['A_D_F_reviews'] = sum(r['A_D_F_review'] for r in group)
            summaries.append(item)
    if summaries:
        save_csv(OUTPUT / 'transition_summary.csv', summaries)
    print(f'Paired predictions: {len(paired)}; complete 960-image runs: {len(summaries)}', flush=True)


if __name__ == '__main__':
    main()
