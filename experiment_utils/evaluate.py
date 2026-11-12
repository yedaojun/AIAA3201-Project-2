"""Full official test-split evaluation and auditable per-image optional-method logs."""
import csv
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from anomaly_baseline.run import get_mask, save_csv
from anomaly_dino.metrics import positive_negative_auc
from .common import DATA, ROOT, load_manifest, outcome, restore_map, sha256, write_json


def exact_pixel_auc(rows, category, folder):
    total, positive_count, shapes = 0, 0, {}
    for row in rows:
        path = DATA / category / row['file']
        with Image.open(path) as source:
            shapes[row['file']] = source.size
            total += source.width * source.height
        if row['label']:
            with Image.open(DATA / category / 'ground_truth/bad' / (path.stem + '.png')) as source:
                positive_count += int(np.count_nonzero(np.array(source)))
    positive = np.empty(positive_count, dtype=np.float32)
    negative = np.empty(total - positive_count, dtype=np.float32)
    pi, ni = 0, 0
    for row in rows:
        with np.load(folder / row['anomaly_map']) as source:
            dense = restore_map(source['map'], shapes[row['file']])
        mask = get_mask(DATA, category, 'bad' if row['label'] else 'good', Path(row['file']), dense.shape)
        p, n = dense[mask], dense[~mask]
        positive[pi:pi + len(p)], negative[ni:ni + len(n)] = p, n
        pi, ni = pi + len(p), ni + len(n)
    assert pi == positive_count and pi + ni == total
    started = perf_counter()
    score = positive_negative_auc(positive, negative)
    print(f'  {category}: exact pixel AUROC on {total:,} original-size pixels ({perf_counter()-started:.1f}s)', flush=True)
    return score, total, positive_count


def evaluate_category(predictor, category, output, settings, run_signature, training=None):
    folder = Path(output) / category
    folder.mkdir(parents=True, exist_ok=True)
    metric_file = folder / 'metrics.json'
    if metric_file.exists():
        previous = json.loads(metric_file.read_text())
        if previous['run_signature'] != run_signature:
            raise ValueError('Completed optional evaluation signature changed.')
        return previous
    selection = load_manifest()['categories'][category]
    calibration, rows = [], []
    raw_file = folder / 'raw_predictions.json'
    if raw_file.exists():
        raw = json.loads(raw_file.read_text())
        if raw['signature'] != run_signature:
            raise ValueError('Unfinished optional evaluation signature changed.')
        calibration, rows = raw['calibration'], raw['rows']
        peak = raw['gpu_peak_memory_mb']
    else:
        for name in selection['calibration']:
            score, *_ = predictor.predict(DATA / category / 'train/good' / name)
            calibration.append({'file': name, 'score': score})
        threshold = float(np.percentile([r['score'] for r in calibration], 95, method='linear'))
        torch.cuda.reset_peak_memory_stats()
        for label in ('good', 'bad'):
            for path in sorted((DATA / category / 'test' / label).iterdir()):
                score, working, dense, elapsed, reconstruction = predictor.predict(path)
                if not np.isfinite(working).all() or not np.isfinite(score):
                    raise ValueError('Nonfinite prediction')
                relative_map = f'maps/{label}/{path.stem}.npz'
                target = folder / relative_map
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target, map=working.astype(np.float32))
                if reconstruction is not None:
                    target = folder / f'reconstructions/{label}/{path.stem}.png'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.rint(np.clip(reconstruction, 0, 1) * 255).astype(np.uint8)).save(target)
                actual, predicted = int(label == 'bad'), int(score > threshold)
                rows.append({'category': category, 'file': f'test/{label}/{path.name}',
                             'label': actual, 'score': score, 'threshold': threshold,
                             'decision': 'NG' if predicted else 'OK', 'outcome': outcome(actual, predicted),
                             'is_correct': int(actual == predicted), 'inference_seconds': elapsed,
                             'anomaly_map': relative_map})
                if len(rows) % 40 == 0:
                    print(f'{settings["method_id"]} {category}: {len(rows)} test images', flush=True)
        peak = torch.cuda.max_memory_allocated() / 1024**2
        write_json(raw_file, {'signature': run_signature, 'calibration': calibration, 'rows': rows,
                             'gpu_peak_memory_mb': peak})
    threshold = float(np.percentile([r['score'] for r in calibration], 95, method='linear'))
    save_csv(folder / 'calibration.csv', calibration)
    save_csv(folder / 'predictions.csv', rows)
    px_auc, total, positives = exact_pixel_auc(rows, category, folder)
    labels, scores = [r['label'] for r in rows], [r['score'] for r in rows]
    counts = {k: sum(r['outcome'] == k for r in rows) for k in ('TP', 'TN', 'FP', 'FN')}
    metrics = {'method_id': settings['method_id'], 'category': category, 'run_signature': run_signature,
               'image_auroc': float(roc_auc_score(labels, scores)), 'image_ap': float(average_precision_score(labels, scores)),
               'pixel_auroc': px_auc, 'pixel_count': total, 'anomalous_pixels': positives,
               'threshold': threshold, 'test_images': len(rows), **counts,
               'accuracy': (counts['TP'] + counts['TN']) / len(rows),
               'recall': counts['TP'] / (counts['TP'] + counts['FN']),
               'false_positive_rate': counts['FP'] / (counts['FP'] + counts['TN']),
               'inference_ms_mean': float(np.mean([r['inference_seconds'] for r in rows]) * 1000),
               'gpu_peak_memory_mb': peak, 'device': torch.cuda.get_device_name(0),
               'calibration_files': selection['calibration'], 'training': training}
    write_json(metric_file, metrics)
    print(f'DONE {settings["method_id"]} {category}: Image={metrics["image_auroc"]:.4f}, Pixel={px_auc:.4f}', flush=True)
    return metrics


def summarize(output, metrics, settings):
    output = Path(output)
    write_json(output / 'metrics_all.json', metrics)
    flat = [{k: v for k, v in m.items() if k not in ('training', 'calibration_files')} for m in metrics]
    save_csv(output / 'summary_by_category.csv', flat)
    summary = {'method_id': settings['method_id'], 'category_count': len(metrics),
               'seed_count': 1, 'seed_note': 'one prespecified run; not a few-shot seed-stability experiment'}
    for key in ('image_auroc', 'pixel_auroc', 'image_ap', 'accuracy', 'inference_ms_mean'):
        summary[key + '_mean'] = float(np.mean([m[key] for m in metrics]))
    summary['gpu_peak_memory_mb'] = max(m['gpu_peak_memory_mb'] for m in metrics)
    save_csv(output / 'summary_macro.csv', [summary])
    images, inputs = [], []
    for metric in metrics:
        category, folder = metric['category'], output / metric['category']
        training_file = folder / 'training_images.csv'
        if training_file.exists():
            with training_file.open(encoding='utf-8-sig', newline='') as file:
                inputs.extend(list(csv.DictReader(file)))
        else:
            inputs.extend({'category': category, 'role': 'calibration',
                           'file': (DATA / category / 'train/good' / n).relative_to(ROOT).as_posix(),
                           'sha256': sha256(DATA / category / 'train/good' / n)} for n in metric['calibration_files'])
        with (folder / 'predictions.csv').open(encoding='utf-8-sig', newline='') as file:
            for row in csv.DictReader(file):
                source = DATA / category / row['file']
                mask = DATA / category / 'ground_truth/bad' / (source.stem + '.png') if int(row['label']) else None
                images.append({'method_id': settings['method_id'], 'run_id': metric['run_signature'],
                               'sample_id': f'VisA/1cls/{category}/{row["file"]}',
                               'image_file': source.relative_to(ROOT).as_posix(), 'image_sha256': sha256(source),
                               'ground_truth_mask': mask.relative_to(ROOT).as_posix() if mask else '',
                               'ground_truth_mask_sha256': sha256(mask) if mask else '', **row,
                               'anomaly_map': (folder / row['anomaly_map']).relative_to(ROOT).as_posix(),
                               'fit_image_count': metric['training']['fit_images'] if metric['training'] else 0,
                               'target_normal_reference_count': '', 'protocol': settings['protocol']})
    save_csv(output / 'predictions_all.csv', images)
    # save_csv infers fields from the first row, so write even an empty error table explicitly.
    errors = [r for r in images if int(r['is_correct']) == 0]
    with (output / 'errors_only.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(images[0]))
        writer.writeheader()
        writer.writerows(errors)
    save_csv(output / 'input_images.csv', inputs)
    write_json(output / 'summary.json', summary)
    return summary
