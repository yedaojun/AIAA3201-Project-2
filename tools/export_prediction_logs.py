"""Export immutable predictions into portable per-image and per-run CSV logs."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def write_csv(path, rows, fieldnames=None):
    if not rows and fieldnames is None:
        raise ValueError(f'No rows to export: {path}')
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha_json(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def portable(path):
    return path.resolve().relative_to(PROJECT).as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=PROJECT / 'results/part1')
    parser.add_argument('--data-root', type=Path, default=PROJECT / 'data/VisA_pytorch/1cls')
    parser.add_argument('--manifest', type=Path, default=PROJECT / 'configs/reference_manifest.json')
    parser.add_argument('--method-id', default='baseline_rgb_template')
    args = parser.parse_args()
    results = args.results.resolve()
    config_path = results / 'run_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != config['manifest_sha256']:
        raise ValueError('Run references were generated using a different manifest')
    all_metrics = json.loads((results / 'metrics_all.json').read_text())
    images, runs, inputs = [], [], []
    checksums = {}
    seen = set()

    def file_sha(path):
        key = str(path.resolve())
        if key not in checksums:
            checksums[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        return checksums[key]

    for metric in all_metrics:
        category, shot, seed = metric['category'], metric['shot'], metric['seed']
        group = results / category / f'shot_{shot}' / f'seed_{seed}'
        stored = json.loads((group / 'metrics.json').read_text())
        if metric != stored:
            raise ValueError(f'Aggregate/per-run metrics differ: {group}')
        selection = manifest['categories'][category]
        if metric['references'] != selection['references'][str(shot)][str(seed)]:
            raise ValueError(f'Reference selection differs: {group}')
        if metric['calibration_files'] != selection['calibration']:
            raise ValueError(f'Calibration selection differs: {group}')
        reference_paths = [args.data_root / category / 'train/good' / name for name in metric['references']]
        calibration_paths = [args.data_root / category / 'train/good' / name for name in metric['calibration_files']]
        reference_identity = [{'file': portable(p), 'sha256': file_sha(p)} for p in reference_paths]
        calibration_identity = [{'file': portable(p), 'sha256': file_sha(p)} for p in calibration_paths]
        reference_sha, calibration_sha = sha_json(reference_identity), sha_json(calibration_identity)
        run_id = f'{args.method_id}/{category}/{shot}shot/seed{seed}/{metric["run_signature"][:12]}'
        common = {'method_id': args.method_id, 'method_name': config['config']['method'],
                  'dataset': 'VisA', 'split': '1cls', 'category': category,
                  'shots': shot, 'seed': seed, 'run_id': run_id,
                  'run_signature': metric['run_signature'],
                  'dataset_split_sha256': manifest['split_sha256'],
                  'manifest_sha256': manifest_sha, 'reference_set_sha256': reference_sha,
                  'calibration_set_sha256': calibration_sha,
                  'input_short_edge': config['config']['short_edge'],
                  'patch_multiple': config['config']['patch_multiple']}
        common.update(model_name=config['config'].get('model_name', ''),
                      weights_sha256=config.get('weights_sha256', ''),
                      upstream_commit=config.get('anomalydino_commit', ''))
        for role, identities in [('reference', reference_identity), ('calibration', calibration_identity)]:
            for position, item in enumerate(identities):
                inputs.append({**common, 'role': role, 'position': position,
                               'input_file': item['file'], 'file_sha256': item['sha256']})
        with (group / 'predictions.csv').open(encoding='utf-8-sig', newline='') as file:
            predictions = list(csv.DictReader(file))
        counts = dict.fromkeys(('TP', 'TN', 'FP', 'FN'), 0)
        for row in predictions:
            actual, score, threshold = int(row['label']), float(row['score']), float(row['threshold'])
            predicted = int(score > threshold)
            if row['decision'] != ('NG' if predicted else 'OK') or threshold != metric['threshold']:
                raise ValueError(f'Inconsistent score/decision: {group}/{row["file"]}')
            outcome = ('TP' if predicted else 'FN') if actual else ('FP' if predicted else 'TN')
            counts[outcome] += 1
            key = ('VisA', '1cls', category, shot, seed, row['file'])
            if key in seen:
                raise ValueError(f'Duplicate prediction: {key}')
            seen.add(key)
            image_path = args.data_root / category / row['file']
            mask_path = args.data_root / category / 'ground_truth/bad' / (image_path.stem + '.png') if actual else None
            images.append({**common, 'sample_id': f'VisA/1cls/{category}/{row["file"]}',
                           'image_file': portable(image_path), 'image_sha256': file_sha(image_path),
                           'ground_truth_label': actual, 'ground_truth_name': 'bad' if actual else 'good',
                           'ground_truth_mask': portable(mask_path) if mask_path else '',
                           'ground_truth_mask_sha256': file_sha(mask_path) if mask_path else '',
                           'normal_mask_definition': '' if actual else 'all-zero mask, same dimensions as image',
                           'predicted_label': predicted, 'predicted_decision': row['decision'],
                           'score': score, 'threshold': threshold,
                           'outcome': outcome, 'is_correct': int(actual == predicted),
                           'outcome_zh': {'TP': '异常正确检出', 'TN': '正常正确通过', 'FP': '正常误报为异常', 'FN': '异常漏报为正常'}[outcome],
                           'reference_images_json': json.dumps([x['file'] for x in reference_identity]),
                           'closest_template': portable(args.data_root / category / 'train/good' / row['closest_template']) if row['closest_template'] else '',
                           'anomaly_map': portable(group / row['anomaly_map']) if row['anomaly_map'] else '',
                           'anomaly_map_format': config['config'].get('map_storage', 'full-resolution uint8 PNG'),
                           'inference_ms': float(row['inference_seconds']) * 1000,
                           'metrics_file': portable(group / 'metrics.json')})
        assert counts['FP'] == metric['false_positives'] and counts['FN'] == metric['false_negatives']
        assert counts['TP'] + counts['FN'] == metric['anomalous_test_images']
        assert counts['TN'] + counts['FP'] == metric['normal_test_images']
        tp, tn, fp, fn = (counts[key] for key in ('TP', 'TN', 'FP', 'FN'))
        runs.append({**common, 'test_images': len(predictions), **counts,
                     'accuracy': (tp + tn) / len(predictions),
                     'precision': tp / (tp + fp) if tp + fp else '',
                     'recall': tp / (tp + fn), 'specificity': tn / (tn + fp),
                     'f1': 2 * tp / (2 * tp + fp + fn),
                     'image_auroc': metric['image_auroc'], 'pixel_auroc': metric['pixel_auroc'],
                     'threshold': metric['threshold'],
                     'inference_ms_mean': metric['inference_ms_mean'], 'gpu_peak_memory_mb': metric['gpu_peak_memory_mb'],
                     'device': metric['device'], 'calibration_image_count': len(calibration_paths),
                     'reference_images_json': json.dumps([x['file'] for x in reference_identity]),
                     'calibration_images_json': json.dumps([x['file'] for x in calibration_identity]),
                     'method_config_json': json.dumps(config['config'], ensure_ascii=False),
                     'metrics_file': portable(group / 'metrics.json'),
                     'predictions_file': portable(group / 'predictions.csv')})
    images.sort(key=lambda r: (r['category'], r['shots'], r['seed'], r['sample_id']))
    write_csv(results / 'predictions_all.csv', images)
    write_csv(results / 'errors_only.csv', [row for row in images if not row['is_correct']], list(images[0]))
    write_csv(results / 'runs.csv', runs)
    write_csv(results / 'input_images.csv', inputs)
    metadata = {'schema_version': 1, 'method_id': args.method_id,
                'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
                'run_count': len(runs), 'test_prediction_count': len(images),
                'incorrect_prediction_count': sum(not row['is_correct'] for row in images),
                'input_manifest_rows': len(inputs), 'path_base': str(PROJECT),
                'path_format': 'project-relative POSIX paths', 'csv_encoding': 'utf-8-sig',
                'file_checksums': {name: hashlib.sha256((results / name).read_bytes()).hexdigest()
                    for name in ['predictions_all.csv', 'errors_only.csv', 'runs.csv', 'input_images.csv']},
                'notes': 'Labels and masks are used only after prediction for evaluation. Thresholds are unchanged from original calibration.'}
    (results / 'csv_export_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
