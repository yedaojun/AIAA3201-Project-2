"""Apply the predeclared D/F calibration rule to completed multilayer runs."""

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from experiment_utils.common import ROOT, write_json
from .fusion_selective import CLIP, rates, rows, save_csv, scaling, transform
from .multilayer import OUTPUT as A_OUTPUT

OUTPUT = ROOT / 'results/part3/multilayer_fusion'


def main():
    clip = {r['sample_id']: r for r in rows(CLIP / 'predictions_all.csv')}
    entries, summaries = [], []
    for metric_file in sorted(A_OUTPUT.glob('*/shot_*/seed_*/metrics.json')):
        metric = json.loads(metric_file.read_text(encoding='utf-8'))
        category, shot, seed = metric['category'], metric['shots'], metric['seed']
        folder = metric_file.parent
        acal = rows(folder / 'calibration.csv')
        ccal = rows(CLIP / category / 'calibration.csv')
        if [r['file'] for r in acal] != [r['file'] for r in ccal]:
            raise ValueError('Normal calibration images differ')
        an = np.array([float(r['score']) for r in acal])
        cn = np.array([float(r['score']) for r in ccal])
        ac, az = scaling(an)
        cc, cz = scaling(cn)
        calibrated = .5 * transform(an, ac, az) + .5 * transform(cn, cc, cz)
        threshold = float(np.percentile(calibrated, 95, method='linear'))
        spread = float(np.percentile(calibrated, 75) - np.percentile(calibrated, 25))
        if spread <= 1e-10:
            raise ValueError('Zero calibrated fusion spread')
        group = []
        for a in rows(folder / 'predictions.csv'):
            c = clip[a['sample_id']]
            if a['label'] != c['label']:
                raise ValueError('Mismatched test label')
            score = float(.5 * transform(float(a['score']), ac, az)
                          + .5 * transform(float(c['score']), cc, cz))
            decision = 'NG' if score > threshold else 'OK'
            review = abs(score - threshold) <= .5 * spread
            entry = {'category': category, 'shots': shot, 'seed': seed,
                     'sample_id': a['sample_id'], 'label': int(a['label']),
                     'multilayer_score': float(a['score']), 'clip_score': float(c['score']),
                     'fused_score': score, 'fused_threshold': threshold,
                     'fused_decision': decision,
                     'selective_decision': 'REVIEW' if review else decision,
                     'review': review,
                     'methods_disagree': int(a['decision'] != c['decision'])}
            entries.append(entry)
            group.append({'label': entry['label'], 'score': score, 'decision': decision,
                          'review': review})
        labels = [r['label'] for r in group]
        fused_auc = float(roc_auc_score(labels, [r['score'] for r in group]))
        original_auc = float(metric['baseline_image_auroc'])
        a_auc = float(metric['image_auroc'])
        summaries.append({'category': category, 'shots': shot, 'seed': seed,
                          'original_dino_image_auroc': original_auc,
                          'multilayer_image_auroc': a_auc, 'multilayer_fusion_image_auroc': fused_auc,
                          'delta_vs_original_pp': 100 * (fused_auc - original_auc),
                          'delta_vs_multilayer_pp': 100 * (fused_auc - a_auc),
                          'review_fraction': float(np.mean([r['review'] for r in group])),
                          'review_count': sum(r['review'] for r in group),
                          'fused_errors_reviewed': sum(r['review'] and
                                                      ((r['decision'] == 'NG') != (r['label'] == 1))
                                                      for r in group),
                          'fused_false_negatives_reviewed': sum(r['review'] and r['label'] == 1 and
                                                               r['decision'] == 'OK' for r in group),
                          **rates(group)})
    if not entries:
        raise ValueError('No completed multilayer predictions')
    save_csv(OUTPUT / 'predictions_all.csv', entries)
    save_csv(OUTPUT / 'category_results.csv', summaries)
    shot_summaries = []
    for shot in (1, 2, 4, 8):
        by_seed = [[r for r in summaries if r['shots'] == shot and r['seed'] == seed]
                   for seed in (0, 1, 2)]
        if not all(len(group) == 6 for group in by_seed):
            continue
        item = {'shot': shot}
        for metric in ('original_dino_image_auroc', 'multilayer_image_auroc',
                       'multilayer_fusion_image_auroc', 'delta_vs_original_pp', 'review_fraction'):
            values = [float(np.mean([r[metric] for r in group])) for group in by_seed]
            item[metric + '_mean'] = float(np.mean(values))
            item[metric + '_sample_sd'] = float(np.std(values, ddof=1))
        shot_summaries.append(item)
    if shot_summaries:
        save_csv(OUTPUT / 'shot_summary.csv', shot_summaries)
    write_json(OUTPUT / 'protocol.json', {
        'method': 'frozen two-layer DINO scores plus fixed equal-weight calibrated CLIP fusion',
        'calibration': 'same 20 normal images; median/IQR/asinh, 95th-percentile fused threshold',
        'review': 'within half an IQR of fused threshold',
        'test_label_use': 'evaluation only',
        'clip_supervision': 'external MVTec AD normal and abnormal examples',
        'clip_geometry': 'square 448, while DINO is aspect-preserving',
        'coverage_note': 'only completed A configurations are included; see category_results.csv'})
    for shot, seed in sorted({(r['shots'], r['seed']) for r in summaries}):
        group = [r for r in summaries if r['shots'] == shot and r['seed'] == seed]
        if len(group) == 6:
            print(f'{shot}-shot seed {seed}: original '
                  f'{np.mean([r["original_dino_image_auroc"] for r in group]):.5f}, '
                  f'A {np.mean([r["multilayer_image_auroc"] for r in group]):.5f}, '
                  f'A+D {np.mean([r["multilayer_fusion_image_auroc"] for r in group]):.5f}', flush=True)
    print(f'Completed A+D/F configs: {len(summaries)}', flush=True)


if __name__ == '__main__':
    main()
