"""Strict paired comparison of two standardized method prediction CSVs."""
import argparse
import csv
import json
from pathlib import Path

KEY = ('dataset', 'split', 'category', 'shots', 'seed', 'sample_id')
MATCH = ('dataset_split_sha256', 'reference_set_sha256', 'calibration_set_sha256',
         'image_sha256', 'ground_truth_label', 'ground_truth_mask_sha256', 'input_short_edge', 'patch_multiple')


def index_rows(rows):
    indexed = {}
    for row in rows:
        key = tuple(str(row[field]) for field in KEY)
        if key in indexed:
            raise ValueError(f'Duplicate sample/configuration: {key}')
        indexed[key] = row
    return indexed


def compare_rows(before, after):
    before, after = index_rows(before), index_rows(after)
    if before.keys() != after.keys():
        raise ValueError(f'Incomplete pairing: before-only={len(before.keys()-after.keys())}, after-only={len(after.keys()-before.keys())}')
    paired = []
    for key in sorted(before):
        old, new = before[key], after[key]
        for field in MATCH:
            if str(old[field]) != str(new[field]):
                raise ValueError(f'Unfair comparison; {field} differs for {key}')
        for row in (old, new):
            actual = int(row['ground_truth_label'])
            predicted = int(float(row['score']) > float(row['threshold']))
            expected = ('TP' if predicted else 'FN') if actual else ('FP' if predicted else 'TN')
            if row['outcome'] != expected or int(row['is_correct']) != int(actual == predicted):
                raise ValueError(f'Inconsistent predicted outcome for {key}')
        was_correct, now_correct = int(old['is_correct']), int(new['is_correct'])
        change = ('both_correct' if now_correct else 'both_wrong') if was_correct == now_correct else ('error_fixed' if now_correct else 'regression')
        paired.append({**{field: old[field] for field in KEY},
                       'image_file': old['image_file'], 'ground_truth_label': old['ground_truth_label'],
                       'before_method': old['method_id'], 'after_method': new['method_id'],
                       'before_outcome': old['outcome'], 'after_outcome': new['outcome'],
                       'transition': old['outcome'] + '->' + new['outcome'], 'change': change,
                       'before_score': old['score'], 'after_score': new['score'],
                       'before_threshold': old['threshold'], 'after_threshold': new['threshold'],
                       'before_anomaly_map': old['anomaly_map'], 'after_anomaly_map': new['anomaly_map'],
                       'reference_set_sha256': old['reference_set_sha256'],
                       'before_run_id': old['run_id'], 'after_run_id': new['run_id']})
    return paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', required=True, type=Path)
    parser.add_argument('--after', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    def read(path):
        with path.open(encoding='utf-8-sig', newline='') as file:
            return list(csv.DictReader(file))
    paired = compare_rows(read(args.before), read(args.after))
    if not paired:
        raise ValueError('No samples to compare')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    summary = {'paired_predictions': len(paired), 'before': str(args.before), 'after': str(args.after),
               'transition_counts': {}, 'change_counts': {},
               'note': 'Scores from different methods need not have the same scale. Decisions use each method own normal-only calibrated threshold.'}
    for row in paired:
        for source, target in [('transition', 'transition_counts'), ('change', 'change_counts')]:
            value = row[source]
            summary[target][value] = summary[target].get(value, 0) + 1
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
