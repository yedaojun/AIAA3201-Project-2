"""Exploratory DINOv2 short-edge 672 pilot with the original Part 2 protocol.

This is a post-hoc hypothesis check, not an independently selected final model.
The 448 Part 2 reproduction remains unchanged. Reference images and the 20
normal calibration images come from the fixed manifest.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import faiss
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from anomaly_dino.official import backbones
from anomaly_dino.predict import OfficialPredictor
from experiment_utils.common import DATA, ROOT, MANIFEST, configure, load_manifest, sha256, write_json
from .multilayer import read_rows, save_csv


OUTPUT = ROOT / 'results/part3/high_resolution_pilot'


def load_high_resolution_model():
    real_load = torch.hub.load
    with patch.object(torch.hub, 'load', lambda repo, name: real_load(
            str(ROOT / 'third_party/dinov2'), name, source='local')):
        model = backbones.get_model('dinov2_vits14', 'cuda', smaller_edge_size=672)
    model.model.eval().requires_grad_(False)
    return model


def run_category(model, category, seed, manifest):
    folder = OUTPUT / category / 'shot_4' / f'seed_{seed}'
    if (folder / 'metrics.json').exists():
        print(f'{category} seed {seed}: existing pilot found', flush=True)
        return
    base = DATA / category
    selection = manifest['categories'][category]
    references = selection['references']['4'][str(seed)]
    if set(references) & set(selection['calibration']):
        raise ValueError('Calibration image appears in reference bank')
    detector = OfficialPredictor(model, [base / 'train/good' / name for name in references])
    calibration = []
    for name in selection['calibration']:
        score, _, _, _ = detector.predict(base / 'train/good' / name)
        calibration.append({'file': name, 'score': score})
    threshold = float(np.percentile([row['score'] for row in calibration], 95))
    predictions = []
    for kind in ('good', 'bad'):
        for index, image in enumerate(sorted((base / 'test' / kind).glob('*.JPG')), 1):
            score, _, patch_map, elapsed = detector.predict(image)
            relative = f'test/{kind}/{image.name}'
            map_file = folder / 'anomaly_maps' / kind / (image.stem + '.npy')
            map_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(map_file, patch_map.astype(np.float32))
            label = int(kind == 'bad')
            predicted = int(score > threshold)
            predictions.append({'category': category, 'shot': 4, 'seed': seed,
                                'file': relative, 'label': label, 'score': score,
                                'threshold': threshold, 'decision': 'NG' if predicted else 'OK',
                                'outcome': ('TP' if predicted else 'FN') if label else
                                           ('FP' if predicted else 'TN'),
                                'inference_seconds': elapsed,
                                'anomaly_map': str(map_file.relative_to(ROOT))})
            if index % 50 == 0:
                print(f'{category} seed {seed} {kind}: {index}', flush=True)
    baseline = json.loads((ROOT / 'results/part2' / category / 'shot_4' / f'seed_{seed}' / 'metrics.json').read_text())
    result = {'category': category, 'shot': 4, 'seed': seed, 'short_edge': 672,
              'references': references, 'calibration_files': selection['calibration'],
              'threshold': threshold,
              'image_auroc': float(roc_auc_score([r['label'] for r in predictions],
                                                 [r['score'] for r in predictions])),
              'false_positives': sum(r['outcome'] == 'FP' for r in predictions),
              'false_negatives': sum(r['outcome'] == 'FN' for r in predictions),
              'mean_inference_ms': float(np.mean([r['inference_seconds'] for r in predictions]) * 1000),
              'baseline_448_image_auroc': baseline['image_auroc'],
              'baseline_448_false_positives': baseline['false_positives'],
              'baseline_448_false_negatives': baseline['false_negatives'],
              'bank_vectors': int(detector.index.ntotal)}
    save_csv(folder / 'calibration.csv', calibration)
    save_csv(folder / 'predictions.csv', predictions)
    write_json(folder / 'metrics.json', result)
    print(f'{category} seed {seed}: 672 AUROC {result["image_auroc"]:.4%}; '
          f'448 AUROC {baseline["image_auroc"]:.4%}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--categories', nargs='+', default=['candle', 'capsules', 'cashew',
                                                              'chewinggum', 'fryum', 'pipe_fryum'])
    parser.add_argument('--seeds', nargs='+', type=int, choices=(0, 1, 2), default=[0])
    args = parser.parse_args()
    configure()
    faiss.omp_set_num_threads(4)
    threadpool_limits(4)
    manifest = load_manifest()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / 'protocol.json', {
        'status': 'post-hoc exploratory pilot; test results do not validate parameter selection',
        'model': 'same frozen official dinov2_vits14 as Part 2',
        'only_change': 'short-edge image resolution from 448 to 672',
        'shot': 4, 'reference_groups': args.seeds,
        'reference_manifest_sha256': sha256(MANIFEST),
        'calibration': 'same disjoint 20 normal training images, 95th percentile',
        'image_score': 'official mean_top1p',
        'test_images': 'same official VisA split as Part 2',
        'test_mask_use': 'none during model setup, scoring, and calibration',
        'selected_categories': args.categories,
    })
    started = perf_counter()
    model = load_high_resolution_model()
    for seed in args.seeds:
        for category in args.categories:
            run_category(model, category, seed, manifest)
    print(f'Pilot elapsed seconds: {perf_counter() - started:.1f}', flush=True)


if __name__ == '__main__':
    main()
