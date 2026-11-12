"""Evaluate template difference on the complete official test split."""
import argparse
import csv
import hashlib
import html
import json
import platform
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .core import CATEGORIES, CONFIG, SEEDS, SHOTS, TemplateDetector, histogram_auc, image_auc, write_panel


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def get_mask(root, category, label, path, shape):
    if label == 'good':
        return np.zeros(shape, dtype=bool)
    with Image.open(root / category / 'ground_truth/bad' / (path.stem + '.png')) as source:
        mask = np.asarray(source) > 0
    if mask.shape != shape:
        raise ValueError(f'Incorrect mask dimensions: {path}')
    return mask


def evaluate(root, output, category, shot, seed, selections, signature, save_maps):
    folder = output / category / f'shot_{shot}' / f'seed_{seed}'
    metric_path = folder / 'metrics.json'
    if metric_path.exists():
        previous = json.loads(metric_path.read_text())
        if previous['run_signature'] != signature:
            raise ValueError(f'Existing run differs: use a new output directory: {folder}')
        print(f'Resume: {category}, {shot}-shot, seed {seed}', flush=True)
        return previous
    folder.mkdir(parents=True, exist_ok=True)
    names = selections['references'][str(shot)][str(seed)]
    assert len(names) == shot and not set(names).intersection(selections['calibration'])
    detector = TemplateDetector([root / category / 'train/good' / name for name in names])
    calibration = []
    for name in selections['calibration']:
        score, _, template, _ = detector.predict(root / category / 'train/good' / name)
        calibration.append({'file': name, 'score': score, 'closest_template': template})
    threshold = float(np.percentile([row['score'] for row in calibration], 95, method='linear'))
    save_csv(folder / 'calibration.csv', calibration)
    positive, negative = np.zeros(256, dtype=np.int64), np.zeros(256, dtype=np.int64)
    rows = []
    # Calibration warms reference shape caches before test-time inference timing.
    for label in ('good', 'bad'):
        for path in sorted((root / category / 'test' / label).glob('*')):
            if not path.is_file():
                continue
            score, anomaly, template, elapsed = detector.predict(path)
            # Ground truth is first accessed after prediction and fixed thresholding.
            decision = 'NG' if score > threshold else 'OK'
            mask = get_mask(root, category, label, path, anomaly.shape)
            positive += np.bincount(anomaly[mask], minlength=256)
            negative += np.bincount(anomaly[~mask], minlength=256)
            relative_map = f'maps/{label}/{path.stem}.png'
            if save_maps:
                destination = folder / relative_map
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(destination), anomaly):
                    raise OSError(f'Could not save anomaly map: {destination}')
            rows.append({'category': category, 'shot': shot, 'seed': seed,
                         'file': f'test/{label}/{path.name}', 'label': int(label == 'bad'),
                         'score': score, 'threshold': threshold, 'decision': decision,
                         'closest_template': template, 'inference_seconds': elapsed,
                         'anomaly_map': relative_map if save_maps else ''})
    save_csv(folder / 'predictions.csv', rows)
    normal = [row for row in rows if row['label'] == 0]
    anomaly_rows = [row for row in rows if row['label'] == 1]
    metrics = {'category': category, 'shot': shot, 'seed': seed,
               'image_auroc': image_auc([r['label'] for r in rows], [r['score'] for r in rows]),
               'pixel_auroc': histogram_auc(positive, negative),
               'inference_ms_mean': float(np.mean([row['inference_seconds'] for row in rows]) * 1000),
               'threshold': threshold, 'normal_test_images': len(normal),
               'anomalous_test_images': len(anomaly_rows),
               'false_positives': sum(row['decision'] == 'NG' for row in normal),
               'false_negatives': sum(row['decision'] == 'OK' for row in anomaly_rows),
               'pixel_count': int(positive.sum() + negative.sum()),
               'anomalous_pixels': int(positive.sum()), 'gpu_peak_memory_mb': 0,
               'device': 'CPU (GPU not used by this baseline)', 'references': names,
               'calibration_files': selections['calibration'], 'run_signature': signature}
    np.savez_compressed(folder / 'pixel_score_histograms.npz', positive=positive, negative=negative)
    metric_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f'{category:12s} {shot}-shot seed {seed}: image AUROC={metrics["image_auroc"]:.4f}, '
          f'pixel AUROC={metrics["pixel_auroc"]:.4f}, {metrics["inference_ms_mean"]:.1f} ms/image', flush=True)
    return metrics


def summaries(metrics, output):
    per_category, macro = [], []
    categories = sorted({row['category'] for row in metrics})
    for shot in sorted({row['shot'] for row in metrics}):
        seeds = sorted({row['seed'] for row in metrics if row['shot'] == shot})
        for category in categories:
            selected = [row for row in metrics if row['shot'] == shot and row['category'] == category]
            row = {'category': category, 'shot': shot, 'seed_count': len(selected)}
            for key in ('image_auroc', 'pixel_auroc', 'inference_ms_mean'):
                values = [result[key] for result in selected]
                row[key + '_mean'] = float(np.mean(values))
                row[key + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            per_category.append(row)
        row = {'shot': shot, 'category_count': len(categories), 'seed_count': len(seeds)}
        for key in ('image_auroc', 'pixel_auroc', 'inference_ms_mean'):
            values = [np.mean([m[key] for m in metrics if m['shot'] == shot and m['seed'] == seed]) for seed in seeds]
            row[key + '_mean'] = float(np.mean(values))
            row[key + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        macro.append(row)
    save_csv(output / 'summary_by_category.csv', per_category)
    save_csv(output / 'summary_macro.csv', macro)
    return macro


def examples(root, output, metrics):
    # Prefer 4-shot seed 0 for a consistent illustrative setting; no method tuning.
    selected = [m for m in metrics if m['shot'] == 4 and m['seed'] == 0]
    if not selected:
        selected = metrics[:1]
    candidates = []
    for metric in selected:
        folder = output / metric['category'] / f'shot_{metric["shot"]}' / f'seed_{metric["seed"]}'
        with (folder / 'predictions.csv').open(encoding='utf-8-sig', newline='') as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            row['score'], row['threshold'], row['label'] = float(row['score']), float(row['threshold']), int(row['label'])
            row['error'] = (row['decision'] == 'NG') != bool(row['label'])
        rows.sort(key=lambda r: (not r['error'], -abs(r['score'] - r['threshold']) if r['error'] else abs(r['score'] - r['threshold'])))
        candidates.append((metric, rows[0]))
    gallery = []
    for index, (metric, row) in enumerate(candidates[:5], 1):
        category = metric['category']
        path = root / category / row['file']
        detector = TemplateDetector([root / category / 'train/good' / name for name in metric['references']])
        score, anomaly, template, _ = detector.predict(path)
        mask = get_mask(root, category, 'bad' if row['label'] else 'good', path, anomaly.shape)
        kind = ('FN' if row['label'] else 'FP') if row['error'] else 'difficult correct'
        caption = f'{category} | {metric["shot"]}-shot seed {metric["seed"]} | {kind} | score={score:.4f} threshold={row["threshold"]:.4f}'
        relative = f'examples/{index:02d}_{category}.png'
        write_panel(path, mask, anomaly, caption, output / relative)
        gallery.append({'caption': caption, 'path': relative, 'file': row['file'], 'closest_template': template})
    (output / 'examples.json').write_text(json.dumps(gallery, indent=2), encoding='utf-8')
    return gallery


def report(output, macro, gallery):
    rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in row.values()) + '</tr>' for row in macro)
    headers = ''.join(f'<th>{key}</th>' for key in macro[0])
    figures = ''.join(f'<figure><img src="{g["path"]}"><figcaption>{html.escape(g["caption"])}</figcaption></figure>' for g in gallery)
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Part 1 results</title>
<style>body{{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#172332}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ddd;padding:8px}}img{{width:100%}}figure{{margin:30px 0}}pre{{white-space:pre-wrap;background:#f4f6f8;padding:18px}}</style>
<h1>Part 1: RGB Template Difference</h1><p>Complete test split. Category macro averages are computed separately for each seed, then summarized across seeds; standard deviations use ddof=1. AUROC values range from 0 to 1.</p>
<table><tr>{headers}</tr>{rows}</table><h2>Representative difficult examples</h2>
<p>One automatically selected difficult example per category (up to five), preferentially an actual error. This selection is for analysis only. Color scale is fixed across images.</p>{figures}
<h2>Method and evaluation settings</h2><pre>{html.escape(json.dumps(CONFIG, indent=2))}</pre>
<p>Failure mechanisms to inspect: pose and instance-layout changes, unstable backgrounds and illumination, small low-contrast defects, and normal-reference coverage. The examples illustrate actual predictions; no settings were chosen using test scores.</p></html>'''
    (output / 'report.html').write_text(content, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('data/VisA_pytorch/1cls'))
    parser.add_argument('--manifest', type=Path, default=Path('configs/reference_manifest.json'))
    parser.add_argument('--output', type=Path, default=Path('results/part1'))
    parser.add_argument('--categories', nargs='+', choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument('--shots', nargs='+', type=int, choices=SHOTS, default=list(SHOTS))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--no-save-maps', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    # Include implementation fingerprint to prevent reuse after changing the method.
    source_hash = hashlib.sha256((Path(__file__).read_bytes() + Path(__file__).with_name('core.py').read_bytes())).hexdigest()
    settings = {'config': CONFIG, 'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                'implementation_sha256': source_hash, 'save_maps': not args.no_save_maps,
                'python': platform.python_version(), 'platform': platform.platform(),
                'numpy': np.__version__, 'opencv': cv2.__version__, 'opencv_threads': 1}
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'run_config.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    metrics = []
    for category in args.categories:
        for shot in args.shots:
            for seed in args.seeds:
                metric = evaluate(args.data_root, args.output, category, shot, seed,
                                  manifest['categories'][category], signature, not args.no_save_maps)
                metrics.append(metric)
                (args.output / 'metrics_all.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    macro = summaries(metrics, args.output)
    gallery = examples(args.data_root, args.output, metrics)
    report(args.output, macro, gallery)
    print('DONE: ' + str((args.output / 'report.html').resolve()), flush=True)


if __name__ == '__main__':
    main()
