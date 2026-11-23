"""Direction C pilot: frozen DINOv2 with a CutPaste-trained linear patch head."""

import argparse
import json
from time import perf_counter

import faiss
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from anomaly_dino.official import load_model, post_eval
from experiment_utils.common import DATA, ROOT, configure, load_manifest, write_json
from .multilayer import read_rows, save_csv


OUTPUT = ROOT / 'results/part3/synthetic_cutpaste_pilot'
SYNTHETICS_PER_REFERENCE = 8
SYNTHETIC_SEED = 20261122


def cutpaste(tensor, rng):
    """Paste a patch from one location of the resized normal image into another."""
    result = tensor.clone()
    _, height, width = result.shape
    rect_h = int(rng.integers(22, 65))
    rect_w = int(rng.integers(22, 65))
    src_y = int(rng.integers(0, height - rect_h))
    src_x = int(rng.integers(0, width - rect_w))
    for _ in range(100):
        dst_y = int(rng.integers(0, height - rect_h))
        dst_x = int(rng.integers(0, width - rect_w))
        if abs(dst_y - src_y) >= rect_h or abs(dst_x - src_x) >= rect_w:
            break
    else:
        raise ValueError('Could not place a non-overlapping CutPaste patch')
    result[:, dst_y:dst_y + rect_h, dst_x:dst_x + rect_w] = \
        tensor[:, src_y:src_y + rect_h, src_x:src_x + rect_w]
    mask = np.zeros((height, width), dtype=np.float32)
    mask[dst_y:dst_y + rect_h, dst_x:dst_x + rect_w] = 1
    return result, mask


def make_head(model, references):
    rng = np.random.default_rng(SYNTHETIC_SEED)
    normal, synthetic = [], []
    for image in references:
        tensor, grid = model.prepare_image(str(image))
        normal.append(model.extract_features(tensor).astype(np.float32))
        for _ in range(SYNTHETICS_PER_REFERENCE):
            altered, mask = cutpaste(tensor, rng)
            features = model.extract_features(altered).astype(np.float32)
            overlap = mask.reshape(grid[0], 14, grid[1], 14).mean(axis=(1, 3)).ravel()
            synthetic.append(features[overlap >= .10])
    negatives = np.concatenate(normal)
    positives = np.concatenate(synthetic)
    if len(positives) < 100:
        raise ValueError('Too few synthetic anomaly patches')
    x = np.concatenate((negatives, positives))
    y = np.concatenate((np.zeros(len(negatives)), np.ones(len(positives))))
    head = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight='balanced',
                                                              max_iter=300, random_state=0))
    head.fit(x, y)
    return head, {'normal_training_images': len(references),
                  'synthetic_training_images': len(references) * SYNTHETICS_PER_REFERENCE,
                  'normal_training_patches': len(negatives),
                  'synthetic_positive_patches': len(positives)}


def predict(model, head, image):
    started = perf_counter()
    tensor, grid = model.prepare_image(str(image))
    features = model.extract_features(tensor)
    foreground = model.compute_background_mask(features, grid, threshold=10, masking_type=True)
    probabilities = head.predict_proba(features.astype(np.float32))[:, 1]
    patch = np.zeros(len(foreground), dtype=np.float64)
    patch[foreground] = probabilities[foreground]
    score = float(post_eval.mean_top1p(patch))
    torch.cuda.synchronize()
    return score, patch.reshape(grid), 1000 * (perf_counter() - started)


def run(categories):
    configure()
    faiss.omp_set_num_threads(4)
    threadpool_limits(4)
    model = load_model()
    manifest = load_manifest()
    baseline = {(row['category'], int(row['shots']), int(row['seed'])): row
                for row in read_rows(ROOT / 'results/part2/runs.csv')}
    summary = []
    for category in categories:
        entry = manifest['categories'][category]
        folder = OUTPUT / category / 'shot_4' / 'seed_0'
        metric_file = folder / 'metrics.json'
        if metric_file.exists():
            summary.append(json.loads(metric_file.read_text(encoding='utf-8')))
            print(f'Skipped completed {category}', flush=True)
            continue
        references = entry['references']['4']['0']
        if set(references) & set(entry['calibration']):
            raise ValueError('Reference/calibration overlap')
        head, training = make_head(model, [DATA / category / 'train/good' / name for name in references])
        calibration = []
        for filename in entry['calibration']:
            score, _, _ = predict(model, head, DATA / category / 'train/good' / filename)
            calibration.append({'file': filename, 'score': score})
        threshold = float(np.percentile([row['score'] for row in calibration], 95, method='linear'))
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for kind in ('good', 'bad'):
            for image in sorted((DATA / category / 'test' / kind).glob('*.JPG')):
                score, patch, elapsed = predict(model, head, image)
                label = int(kind == 'bad')
                decision = 'NG' if score > threshold else 'OK'
                map_path = folder / 'maps' / kind / f'{image.stem}.npy'
                map_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(map_path, patch.astype(np.float32))
                rows.append({'category': category, 'shots': 4, 'seed': 0,
                             'sample_id': f'VisA/1cls/{category}/test/{kind}/{image.name}',
                             'image_file': image.relative_to(ROOT).as_posix(), 'label': label,
                             'score': score, 'threshold': threshold, 'decision': decision,
                             'outcome': ('TP' if decision == 'NG' else 'FN') if label else
                                        ('FP' if decision == 'NG' else 'TN'),
                             'inference_ms': elapsed, 'anomaly_map': map_path.relative_to(ROOT).as_posix()})
        labels = [row['label'] for row in rows]
        auc = float(roc_auc_score(labels, [row['score'] for row in rows]))
        metric = {'category': category, 'shots': 4, 'seed': 0,
                  'image_auroc': auc,
                  'part2_image_auroc': float(baseline[category, 4, 0]['image_auroc']),
                  'reference_filenames': references, 'calibration_filenames': entry['calibration'],
                  'training': training, 'threshold': threshold, 'test_images': len(rows),
                  'inference_ms_mean': float(np.mean([row['inference_ms'] for row in rows])),
                  'peak_gpu_mib': torch.cuda.max_memory_allocated() / 2**20,
                  **{key: sum(row['outcome'] == key for row in rows)
                     for key in ('TP', 'TN', 'FP', 'FN')}}
        save_csv(folder / 'calibration.csv', calibration)
        save_csv(folder / 'predictions.csv', rows)
        write_json(metric_file, metric)
        summary.append(metric)
        print(f'{category}: C={auc:.4f}, Part 2={metric["part2_image_auroc"]:.4f}', flush=True)
    if set(categories) == set(manifest['categories']) and len(summary) == 6:
        write_json(OUTPUT / 'pilot_summary.json', summary)
        print(f'Completed {len(summary)} categories; macro C '
              f'{np.mean([row["image_auroc"] for row in summary]):.4f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--categories', nargs='+', default=['candle', 'capsules', 'cashew',
                                                               'chewinggum', 'fryum', 'pipe_fryum'])
    args = parser.parse_args()
    run(args.categories)
