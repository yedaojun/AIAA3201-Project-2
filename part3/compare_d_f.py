"""Audit completed A/D/F sweeps and write paired, seed-aware comparison tables."""

import csv
import json

import numpy as np

from experiment_utils.common import ROOT


RESULTS = ROOT / 'results/part3'
OUTPUT = RESULTS / 'df_comparison'
SHOTS = (1, 2, 4, 8)
SEEDS = (0, 1, 2)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sd_by_seed(rows, key):
    values = [float(np.mean([float(row[key]) for row in rows if int(row['seed']) == seed]))
              for seed in SEEDS]
    return float(np.mean(values)), float(np.std(values, ddof=1)), values


def main():
    a = read_csv(RESULTS / 'multilayer/category_results.csv')
    d = read_csv(RESULTS / 'fusion_selective/category_results.csv')
    f = read_csv(RESULTS / 'fusion_selective/risk_coverage.csv')
    predictions = read_csv(RESULTS / 'fusion_selective/predictions_all.csv')
    keys = lambda rows: {(row['category'], int(row['shots']), int(row['seed'])) for row in rows}
    if not (len(a) == len(d) == 72 and keys(a) == keys(d) and len(keys(a)) == 72):
        raise ValueError('A and D must have the same 72 category/shot/seed settings')
    if len(predictions) != 11520 or len(f) != 72 * 4 * 4:
        raise ValueError('F must cover all test images, policies and coverage targets')
    categories = sorted({row['category'] for row in a})
    if len(categories) != 6:
        raise ValueError('Expected six categories')

    a_by_key = {(row['category'], int(row['shots']), int(row['seed'])): row for row in a}
    paired = []
    for row in d:
        key = row['category'], int(row['shots']), int(row['seed'])
        ar = a_by_key[key]
        base = float(row['image_auroc_dino'])
        if abs(base - float(ar['baseline_image_auroc'])) > 1e-10:
            raise ValueError(f'Baseline mismatch: {key}')
        item = {'category': key[0], 'shots': key[1], 'seed': key[2],
                'part2_image_auroc': base, 'a_image_auroc': float(ar['image_auroc']),
                'd_equal_image_auroc': float(row['image_auroc_fusion']),
                'd_gated_image_auroc': float(row['image_auroc_gated'])}
        for candidate in ('a', 'd_equal', 'd_gated'):
            item[candidate + '_gain_pp'] = 100 * (item[candidate + '_image_auroc'] - base)
        paired.append(item)
    paired.sort(key=lambda row: (row['shots'], row['seed'], row['category']))
    summary = []
    for shot in SHOTS:
        group = [row for row in paired if row['shots'] == shot]
        if len(group) != 18:
            raise ValueError(f'Incomplete {shot}-shot comparison')
        item = {'shots': shot, 'categories': 6, 'seeds': 3}
        for metric in ('part2_image_auroc', 'a_image_auroc', 'd_equal_image_auroc',
                       'd_gated_image_auroc', 'a_gain_pp', 'd_equal_gain_pp', 'd_gated_gain_pp'):
            mean, sd, _ = mean_sd_by_seed(group, metric)
            item[metric + '_mean'] = mean
            item[metric + '_sample_sd'] = sd
        item['a_improved_runs'] = sum(row['a_gain_pp'] > 0 for row in group)
        item['d_equal_improved_runs'] = sum(row['d_equal_gain_pp'] > 0 for row in group)
        item['d_gated_improved_runs'] = sum(row['d_gated_gain_pp'] > 0 for row in group)
        item['d_equal_better_than_a_runs'] = sum(row['d_equal_image_auroc'] > row['a_image_auroc'] for row in group)
        item['d_gated_better_than_a_runs'] = sum(row['d_gated_image_auroc'] > row['a_image_auroc'] for row in group)
        summary.append(item)

    coverage = []
    for shot in SHOTS:
        for policy in ('dino_margin', 'margin', 'disagreement_then_margin', 'gated_margin'):
            for target in (1., .9, .8, .7):
                group = [row for row in f if int(row['shots']) == shot and row['policy'] == policy
                         and float(row['target_coverage']) == target]
                if len(group) != 18:
                    raise ValueError(f'Incomplete F comparison: {shot}, {policy}, {target}')
                item = {'shots': shot, 'policy': policy, 'target_coverage': target}
                for metric in ('actual_coverage', 'retained_error_rate',
                               'eval_fnr_at_test_fpr_le_10pct', 'eval_realized_test_fpr'):
                    mean, sd, _ = mean_sd_by_seed(group, metric)
                    item[metric + '_mean'] = mean
                    item[metric + '_sample_sd'] = sd
                coverage.append(item)

    fixed = []
    for shot in SHOTS:
        group = [row for row in predictions if int(row['shots']) == shot]
        if len(group) != 2880:
            raise ValueError(f'Incomplete fixed-band predictions at {shot} shots')
        for rule, review_col, decision_col in (
            ('equal_fusion', 'selective_decision', 'fused_decision'),
            ('exploratory_gate', 'gated_selective_decision', 'gated_decision'),
        ):
            review = [row for row in group if row[review_col] == 'REVIEW']
            auto = [row for row in group if row[review_col] != 'REVIEW']
            errors = lambda rows: sum((row[decision_col] == 'NG') != (int(row['label']) == 1)
                                      for row in rows)
            false_negatives = lambda rows: sum(int(row['label']) == 1 and row[decision_col] == 'OK'
                                               for row in rows)
            fixed.append({'shots': shot, 'rule': rule, 'total_images': len(group),
                          'review_images': len(review), 'review_fraction': len(review) / len(group),
                          'automatic_images': len(auto), 'automatic_error_count': errors(auto),
                          'automatic_error_rate': errors(auto) / len(auto),
                          'all_decision_errors': errors(group), 'errors_sent_to_review': errors(review),
                          'all_false_negatives': false_negatives(group),
                          'false_negatives_sent_to_review': false_negatives(review)})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / 'paired_category_runs.csv', paired)
    write_csv(OUTPUT / 'image_auroc_by_shot.csv', summary)
    per_category = []
    for category in categories:
        group = [row for row in paired if row['shots'] == 4 and row['category'] == category]
        item = {'category': category, 'shots': 4}
        for metric in ('part2_image_auroc', 'a_image_auroc', 'd_equal_image_auroc',
                       'd_gated_image_auroc'):
            item[metric + '_mean'] = float(np.mean([row[metric] for row in group]))
        per_category.append(item)
    write_csv(OUTPUT / 'category_4shot.csv', per_category)
    write_csv(OUTPUT / 'risk_coverage_by_shot.csv', coverage)
    write_csv(OUTPUT / 'fixed_review_band_by_shot.csv', fixed)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5), layout='constrained')
    for metric, label, color in (
        ('part2_image_auroc', 'Part 2 DINO', '#555d65'),
        ('a_image_auroc', 'A: two DINO layers', '#287e6d'),
        ('d_equal_image_auroc', 'D: equal fusion', '#137c8b'),
        ('d_gated_image_auroc', 'D: exploratory gate', '#79539a'),
    ):
        ax.errorbar(SHOTS, [100 * row[metric + '_mean'] for row in summary],
                    yerr=[100 * row[metric + '_sample_sd'] for row in summary],
                    marker='o', lw=2, capsize=3, label=label, color=color)
    ax.set(xlabel='Normal reference images', ylabel='Image AUROC (%)',
           title='Six-category macro AUROC; error bars: sample SD across three DINO reference groups',
           xticks=SHOTS)
    ax.grid(alpha=.25)
    ax.legend(loc='lower right')
    fig.savefig(OUTPUT / 'image_auroc_by_shot.png', dpi=180)
    plt.close(fig)
    manifest = {'categories': categories, 'shots': SHOTS, 'seeds': SEEDS,
                'paired_category_runs': len(paired), 'test_predictions': len(predictions),
                'fairness_note': 'A preserves Part 2 frozen DINO geometry and references. '
                                 'D additionally uses externally anomaly-supervised CLIP with square input; '
                                 'its three rows per shot are DINO reference groups, not CLIP retraining.',
                'selection_note': 'Gated D followed inspection of equal D results; exploratory.',
                'f_note': '90% coverage is a retrospective fixed workload comparison on test rankings. '
                          'The normal-calibrated review band is the deployable rule, with variable coverage. '
                          'Neither is a calibrated risk guarantee.'}
    (OUTPUT / 'audit.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'paired_runs': len(paired), 'predictions': len(predictions),
                      'image_summary': [{k: round(row[k], 6) for k in
                                         ('shots', 'part2_image_auroc_mean', 'a_image_auroc_mean',
                                          'd_equal_image_auroc_mean', 'd_gated_image_auroc_mean')}
                                        for row in summary],
                      'fixed_band': fixed}, indent=2))


if __name__ == '__main__':
    main()
