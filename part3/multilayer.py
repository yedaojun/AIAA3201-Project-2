"""Direction A pilot: fixed two-layer DINOv2 features before nearest-neighbor scoring."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from anomaly_dino.official import load_model
from anomaly_dino.predict import OfficialPredictor
from experiment_utils.common import DATA, ROOT, MANIFEST, configure, sha256, write_json

OUTPUT = ROOT / 'results/part3/multilayer'
LAYERS = (8, 11)  # zero-based transformer blocks, selected before test inspection


class TwoLayerDINO:
    """Use equal unit-length layer contributions and original final-layer PCA mask."""

    def __init__(self, original, layers=LAYERS, alpha=0.5):
        if len(layers) != 2 or layers[1] != 11 or not 0 <= alpha <= 1:
            raise ValueError('Expected intermediate/final layer pair and 0 <= alpha <= 1')
        self.original = original
        self.layers = tuple(layers)
        self.alpha = alpha
        self.last_final = None
        self.prepare_image = original.prepare_image

    def extract_features(self, tensor):
        with torch.inference_mode():
            vectors = self.original.model.get_intermediate_layers(tensor.unsqueeze(0).cuda(), n=self.layers)
            early, final = [v.squeeze(0).cpu().numpy().astype(np.float32) for v in vectors]
        self.last_final = final
        early = early / np.maximum(np.linalg.norm(early, axis=1, keepdims=True), 1e-12)
        final_unit = final / np.maximum(np.linalg.norm(final, axis=1, keepdims=True), 1e-12)
        return np.concatenate((np.sqrt(self.alpha) * early,
                               np.sqrt(1 - self.alpha) * final_unit), axis=1)

    def compute_background_mask(self, features, grid, **kwargs):
        if self.last_final is None or len(self.last_final) != len(features):
            raise ValueError('The final-layer mask features do not match the query')
        return self.original.compute_background_mask(self.last_final, grid, **kwargs)


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save_csv(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)


def outcome(label, decision):
    return ('TP' if decision == 'NG' else 'FN') if label else ('FP' if decision == 'NG' else 'TN')


def run(categories, shots, seeds):
    configure()
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    baseline = {(r['category'], int(r['shots']), int(r['seed'])): r
                for r in read_rows(ROOT / 'results/part2/runs.csv')}
    settings = {'direction': 'A multilayer DINOv2 features before nearest-neighbor distance',
                'layers_zero_based': list(LAYERS), 'feature_normalization': 'per-layer unit length then concatenate',
                'mask': 'unaltered final-layer DINO PCA query mask',
                'memory': 'official rotations, same reference filenames, unmasked bank',
                'resolution': 'short edge 448, aspect-preserving, 14-pixel patch',
                'score': 'official cosine NN then top 1 percent patch mean',
                'threshold': '95th percentile of same 20 normal calibration images',
                'source_manifest_sha256': sha256(MANIFEST),
                'part2_predictions_sha256': sha256(ROOT / 'results/part2/predictions_all.csv'),
                'test_label_use': 'evaluation only', 'feature_weights_trained_on_visa': False}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = OUTPUT / 'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text(encoding='utf-8')) != settings:
        raise ValueError('Saved multilayer experiment settings differ')
    write_json(protocol, settings)
    original = load_model()
    model = TwoLayerDINO(original)
    sample = DATA / 'candle/train/good' / manifest['categories']['candle']['calibration'][0]
    tensor, grid = original.prepare_image(str(sample))
    expected = original.extract_features(tensor)
    feature = model.extract_features(tensor)
    if feature.shape != (grid[0] * grid[1], expected.shape[1] * 2):
        raise ValueError('Unexpected multilayer feature shape')
    if not np.allclose(model.last_final, expected, atol=2e-6, rtol=2e-6):
        raise ValueError('Final DINO layer differs from the Part 2 extractor')
    a = original.compute_background_mask(expected, grid, threshold=10, masking_type=True)
    b = model.compute_background_mask(feature, grid, threshold=10, masking_type=True)
    if not np.array_equal(a, b):
        raise ValueError('The original foreground mask changed')
    print('Verified final-layer features and foreground mask against Part 2', flush=True)
    for category in categories:
        selection = manifest['categories'][category]
        test = [(label, p) for label in ('good', 'bad')
                for p in sorted((DATA / category / 'test' / label).glob('*.JPG'))]
        for shot in shots:
            for seed in seeds:
                folder = OUTPUT / category / f'shot_{shot}' / f'seed_{seed}'
                metric_file = folder / 'metrics.json'
                if metric_file.exists():
                    continue
                references = selection['references'][str(shot)][str(seed)]
                if set(references) & set(selection['calibration']):
                    raise ValueError('Calibration image leaked into reference bank')
                paths = [DATA / category / 'train/good' / name for name in references]
                detector = OfficialPredictor(model, paths, seed=seed)
                calibration = []
                for name in selection['calibration']:
                    score, _, _, _ = detector.predict(DATA / category / 'train/good' / name)
                    calibration.append({'file': name, 'score': score})
                threshold = float(np.percentile([r['score'] for r in calibration], 95, method='linear'))
                torch.cuda.reset_peak_memory_stats()
                predictions = []
                for index, (kind, path) in enumerate(test, 1):
                    score, _, patch, elapsed = detector.predict(path)
                    label = int(kind == 'bad')
                    decision = 'NG' if score > threshold else 'OK'
                    map_path = folder / 'maps' / kind / f'{path.stem}.npy'
                    map_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(map_path, patch)
                    predictions.append({'category': category, 'shots': shot, 'seed': seed,
                                        'sample_id': f'VisA/1cls/{category}/test/{kind}/{path.name}',
                                        'image_file': (path.relative_to(ROOT)).as_posix(),
                                        'label': label, 'score': score, 'threshold': threshold,
                                        'decision': decision, 'outcome': outcome(label, decision),
                                        'inference_ms': 1000 * elapsed,
                                        'anomaly_map': map_path.relative_to(ROOT).as_posix()})
                    if index % 40 == 0:
                        print(f'{category} {shot}-shot seed {seed}: {index}/{len(test)}', flush=True)
                save_csv(folder / 'calibration.csv', calibration)
                save_csv(folder / 'predictions.csv', predictions)
                labels = [r['label'] for r in predictions]
                scores = [r['score'] for r in predictions]
                counts = {k: sum(r['outcome'] == k for r in predictions) for k in ('TP', 'TN', 'FP', 'FN')}
                metric = {'category': category, 'shots': shot, 'seed': seed,
                          'image_auroc': float(roc_auc_score(labels, scores)),
                          'baseline_image_auroc': float(baseline[category, shot, seed]['image_auroc']),
                          'threshold': threshold, 'reference_filenames': references,
                          'calibration_filenames': selection['calibration'],
                          'inference_ms_mean': float(np.mean([r['inference_ms'] for r in predictions])),
                          'gpu_peak_mib': torch.cuda.max_memory_allocated() / 2**20,
                          'test_images': len(predictions), **counts}
                write_json(metric_file, metric)
                print(f'DONE {category} {shot}-shot seed {seed}: image AUROC '
                      f'{metric["image_auroc"]:.5f} vs {metric["baseline_image_auroc"]:.5f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--categories', nargs='+', default=['candle', 'capsules', 'cashew',
                                                               'chewinggum', 'fryum', 'pipe_fryum'])
    parser.add_argument('--shots', nargs='+', type=int, choices=(1, 2, 4, 8), default=[4])
    parser.add_argument('--seeds', nargs='+', type=int, choices=(0, 1, 2), default=[0])
    args = parser.parse_args()
    run(args.categories, args.shots, args.seeds)
