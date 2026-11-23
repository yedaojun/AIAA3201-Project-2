"""Calibrated image-score fusion and selective decisions on saved test predictions.

No test labels are read while defining score normalization, fusion, thresholds,
or uncertainty. They are used only after all decisions have been frozen.
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from experiment_utils.common import ROOT, MANIFEST, sha256, write_json

PART2 = ROOT / 'results/part2'
CLIP = ROOT / 'results/part2_anomalyclip'
OUTPUT = ROOT / 'results/part3/fusion_selective'
SHOTS = (1, 2, 4, 8)
SEEDS = (0, 1, 2)


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save_csv(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)


def scaling(normal):
    normal = np.asarray(normal, dtype=float)
    med = float(np.median(normal))
    iqr = float(np.percentile(normal, 75) - np.percentile(normal, 25))
    if iqr <= 1e-10:
        raise ValueError('Normal-score IQR is too small for a calibrated fusion')
    return med, iqr


def transform(score, center, spread):
    return np.arcsinh((np.asarray(score, dtype=float) - center) / spread)


def gated_fusion(dino_z, clip_z, dino_calibrated_threshold):
    """Use more CLIP near DINO's normal-only decision boundary."""
    weight = .5 * np.exp(-np.abs(dino_z - dino_calibrated_threshold))
    return (1 - weight) * dino_z + weight * clip_z, weight


def rates(entries):
    tp = sum(e['label'] == 1 and e['decision'] == 'NG' for e in entries)
    fn = sum(e['label'] == 1 and e['decision'] == 'OK' for e in entries)
    fp = sum(e['label'] == 0 and e['decision'] == 'NG' for e in entries)
    tn = sum(e['label'] == 0 and e['decision'] == 'OK' for e in entries)
    return {'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn,
            'false_positive_rate': fp / (fp + tn), 'false_negative_rate': fn / (fn + tp),
            'accuracy': (tp + tn) / len(entries)}


def fnr_at_test_fpr(labels, scores, target=.10):
    """Retrospective ROC operating point; never used as a deployed threshold."""
    normals = np.asarray(scores, dtype=float)[np.asarray(labels) == 0]
    boundary = float(np.quantile(normals, 1 - target, method='higher'))
    decisions = np.asarray(scores) > boundary
    y = np.asarray(labels)
    return float(np.mean(~decisions[y == 1])), float(np.mean(decisions[y == 0]))


def evaluate():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    clip_all = rows(CLIP / 'predictions_all.csv')
    clip = {r['sample_id']: r for r in clip_all}
    dino_all = rows(PART2 / 'predictions_all.csv')
    if len(clip) != 960 or len(dino_all) != 960 * len(SHOTS) * len(SEEDS):
        raise ValueError('Expected all 960 matched test images and 12 DINO settings')
    published = {(r['category'], int(r['shots']), int(r['seed'])): float(r['image_auroc'])
                 for r in rows(PART2 / 'runs.csv')}
    by_run = {}
    for r in dino_all:
        key = (r['category'], int(r['shots']), int(r['seed']))
        by_run.setdefault(key, []).append(r)
    predictions, evaluations, selective = [], [], []
    for (category, shot, seed), dr in sorted(by_run.items()):
        if len(dr) != len([r for r in clip_all if r['category'] == category]):
            raise ValueError(f'Mismatched category length {category}')
        dcal_rows = rows(PART2 / category / f'shot_{shot}' / f'seed_{seed}' / 'calibration.csv')
        ccal_rows = rows(CLIP / category / 'calibration.csv')
        expected = manifest['categories'][category]['calibration']
        if [r['file'] for r in dcal_rows] != expected or [r['file'] for r in ccal_rows] != expected:
            raise ValueError(f'Calibration identity mismatch for {category}')
        dcal = np.array([float(r['score']) for r in dcal_rows])
        ccal = np.array([float(r['score']) for r in ccal_rows])
        dc, ds = scaling(dcal)
        cc, cs = scaling(ccal)
        # Both terms use their own normal-score IQR; the gate uses normal scores only.
        dz_cal, cz_cal = transform(dcal, dc, ds), transform(ccal, cc, cs)
        fcal = .5 * dz_cal + .5 * cz_cal
        fthreshold = float(np.percentile(fcal, 95, method='linear'))
        dthreshold = float(np.percentile(dcal, 95, method='linear'))
        dz_threshold = float(transform(dthreshold, dc, ds))
        gated_cal, _ = gated_fusion(dz_cal, cz_cal, dz_threshold)
        gated_threshold = float(np.percentile(gated_cal, 95, method='linear'))
        gated_spread = float(np.percentile(gated_cal, 75) - np.percentile(gated_cal, 25))
        if gated_spread <= 1e-10:
            raise ValueError('Gated calibration spread is zero')
        # The uncertainty band uses only the spread of normal calibration scores.
        fspread = float(np.percentile(fcal, 75) - np.percentile(fcal, 25))
        if fspread <= 1e-10:
            raise ValueError('Fused calibration spread is zero')
        band = .5 * fspread
        group = []
        for d in dr:
            c = clip[d['sample_id']]
            if d['image_sha256'] != c['image_sha256'] or d['ground_truth_label'] != c['label']:
                raise ValueError(f'Test image mismatch {d["sample_id"]}')
            dscore, cscore = float(d['score']), float(c['score'])
            dz, cz = float(transform(dscore, dc, ds)), float(transform(cscore, cc, cs))
            fused = .5 * dz + .5 * cz
            gated, clip_weight = gated_fusion(dz, cz, dz_threshold)
            gated, clip_weight = float(gated), float(clip_weight)
            label = int(d['ground_truth_label'])
            decision = 'NG' if fused > fthreshold else 'OK'
            gated_decision = 'NG' if gated > gated_threshold else 'OK'
            margin = float(abs(fused - fthreshold) / fspread)
            uncertain = abs(fused - fthreshold) <= band
            gated_margin = float(abs(gated - gated_threshold) / gated_spread)
            record = {'category': category, 'shots': shot, 'seed': seed,
                      'sample_id': d['sample_id'], 'image_file': d['image_file'], 'label': label,
                      'dino_score': dscore, 'clip_score': cscore, 'fused_score': fused,
                      'fused_threshold': fthreshold, 'fused_decision': decision,
                      'gated_score': gated, 'gated_threshold': gated_threshold,
                      'gated_decision': gated_decision, 'clip_gate_weight': clip_weight,
                      'gated_selective_decision': ('REVIEW' if gated_margin <= .5 else gated_decision),
                      'gated_uncertainty_margin': gated_margin,
                      'selective_decision': 'REVIEW' if uncertain else decision,
                      'uncertainty_margin': margin,
                      'methods_disagree': int(d['predicted_decision'] != c['decision']),
                      'dino_decision': d['predicted_decision'], 'clip_decision': c['decision'],
                      'dino_outcome': d['outcome'], 'clip_outcome': c['outcome']}
            predictions.append(record)
            group.append({'label': label, 'score': fused, 'decision': decision,
                          'review': uncertain, 'margin': margin,
                          'disagree': d['predicted_decision'] != c['decision'],
                          'dino_score': dscore,
                          'gated_score': gated, 'gated_decision': gated_decision,
                          'gated_margin': gated_margin,
                          'dino_margin': abs(dscore - dthreshold) / ds,
                          'dino_decision': d['predicted_decision']})
        labels = [g['label'] for g in group]
        fused_scores = [g['score'] for g in group]
        gated_scores = [g['gated_score'] for g in group]
        dino_scores = [float(r['score']) for r in dr]
        clip_scores = [float(clip[r['sample_id']]['score']) for r in dr]
        if abs(roc_auc_score(labels, dino_scores) - published[category, shot, seed]) > 1e-12:
            raise ValueError(f'Official image AUROC mismatch for {category} {shot} {seed}')
        counts = rates(group)
        gated_counts = rates([{**g, 'decision': g['gated_decision']} for g in group])
        dfnr, dfpr = fnr_at_test_fpr(labels, dino_scores)
        ffnr, ffpr = fnr_at_test_fpr(labels, fused_scores)
        gfnr, gfpr = fnr_at_test_fpr(labels, gated_scores)
        evaluations.append({'category': category, 'shots': shot, 'seed': seed,
                            'test_images': len(group), 'image_auroc_dino': roc_auc_score(labels, dino_scores),
                            'image_auroc_clip': roc_auc_score(labels, clip_scores),
                            'image_auroc_fusion': roc_auc_score(labels, fused_scores),
                            'image_auroc_gated': roc_auc_score(labels, gated_scores),
                            'fusion_delta_pp': 100 * (roc_auc_score(labels, fused_scores) - roc_auc_score(labels, dino_scores)),
                            'gated_delta_pp': 100 * (roc_auc_score(labels, gated_scores) - roc_auc_score(labels, dino_scores)),
                            'calibration_images': len(dcal), 'fusion_threshold': fthreshold,
                            'gated_threshold': gated_threshold,
                            'uncertain_band_half_width': band,
                            'dino_fnr_at_test_fpr_le_10pct': dfnr,
                            'dino_realized_test_fpr': dfpr,
                            'fusion_fnr_at_test_fpr_le_10pct': ffnr,
                            'fusion_realized_test_fpr': ffpr,
                            'gated_fnr_at_test_fpr_le_10pct': gfnr,
                            'gated_realized_test_fpr': gfpr,
                            **{'gated_' + key: value for key, value in gated_counts.items()},
                            **counts})
        for policy in ('dino_margin', 'margin', 'disagreement_then_margin', 'gated_margin'):
            # The second policy ranks disagreements as less certain first.
            ranked = sorted(group, key=lambda e: ((not e['disagree']), e['margin'])
                            if policy == 'disagreement_then_margin' else
                            (e['dino_margin'] if policy == 'dino_margin' else
                             e['gated_margin'] if policy == 'gated_margin' else e['margin']))
            for coverage in (1., .9, .8, .7):
                keep = ranked[len(ranked) - round(coverage * len(ranked)):]
                if policy == 'dino_margin':
                    keep = [{**e, 'decision': e['dino_decision']} for e in keep]
                elif policy == 'gated_margin':
                    keep = [{**e, 'decision': e['gated_decision']} for e in keep]
                outcomes = rates(keep)
                selected_scores = [e['dino_score'] if policy == 'dino_margin' else
                                   e['gated_score'] if policy == 'gated_margin' else e['score'] for e in keep]
                selected_fnr, selected_fpr = fnr_at_test_fpr([e['label'] for e in keep], selected_scores)
                selective.append({'category': category, 'shots': shot, 'seed': seed,
                                  'policy': policy, 'target_coverage': coverage,
                                  'actual_coverage': len(keep) / len(group),
                                  'review_count': len(group) - len(keep),
                                  'retained_error_rate': 1 - outcomes['accuracy'],
                                  'eval_fnr_at_test_fpr_le_10pct': selected_fnr,
                                  'eval_realized_test_fpr': selected_fpr,
                                  **outcomes})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_csv(OUTPUT / 'predictions_all.csv', predictions)
    save_csv(OUTPUT / 'category_results.csv', evaluations)
    save_csv(OUTPUT / 'risk_coverage.csv', selective)
    summary = []
    for shot in SHOTS:
        per_seed = [r for r in evaluations if r['shots'] == shot]
        for key in ('image_auroc_dino', 'image_auroc_clip', 'image_auroc_fusion',
                    'image_auroc_gated', 'fusion_delta_pp', 'gated_delta_pp',
                    'false_positive_rate', 'false_negative_rate'):
            values = [np.mean([r[key] for r in per_seed if r['seed'] == seed]) for seed in SEEDS]
            summary.append({'shot': shot, 'metric': key, 'seed0': values[0], 'seed1': values[1],
                            'seed2': values[2], 'mean': float(np.mean(values)),
                            'sample_sd': float(np.std(values, ddof=1))})
    save_csv(OUTPUT / 'summary_by_shot.csv', summary)
    write_json(OUTPUT / 'protocol.json', {
        'direction': 'D image-level score fusion plus F selective prediction pilot',
        'dino_csv_sha256': sha256(PART2 / 'predictions_all.csv'),
        'clip_csv_sha256': sha256(CLIP / 'predictions_all.csv'),
        'clip_run_config_sha256': sha256(CLIP / 'run_config.json'),
        'manifest_sha256': sha256(MANIFEST),
        'weights': {'equal_dino': .5, 'equal_clip': .5,
                    'gated_clip': '0.5 * exp(-abs(normalized_DINO - normalized_DINO_calibration_q95))'},
        'normalization': 'arcsinh((score - median(normal calibration)) / IQR(normal calibration))',
        'threshold': '95th percentile of 20 paired normal calibration fused scores',
        'review_band': 'absolute fused distance from calibrated threshold <= 0.5 normal fused-score IQR',
        'test_label_use': 'evaluation only', 'clip_supervision': 'external MVTec AD anomalies and normal images',
        'clip_geometry': 'square 448, unlike DINO aspect-preserving 448',
        'seed_note': 'CLIP is deterministic and reused for all DINO reference groups, not 3 CLIP trials',
        'scope': 'image-level only; no pixel-map fusion claimed; normal-only gate is exploratory after equal-fusion pilot',
        'fixed_fpr_metric': 'retrospective test ROC FNR at realized test FPR <=10%; not a deployment threshold'})
    print(json.dumps({'runs': len(evaluations), 'predictions': len(predictions),
                      'four_shot': [r for r in summary if r['shot'] == 4 and r['metric'] in
                                    ('image_auroc_dino', 'image_auroc_fusion', 'fusion_delta_pp')]}, indent=2))


if __name__ == '__main__':
    evaluate()
