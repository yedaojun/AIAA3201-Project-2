"""Prespecified 4-shot pilot for B foreground bank and E stratified coreset."""

import argparse
import json
from time import perf_counter

import cv2
import faiss
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from anomaly_dino.official import FeatureArchive, load_model, post_eval, utils
from anomaly_dino.predict import OfficialPredictor
from experiment_utils.common import DATA, ROOT, configure, load_manifest, write_json
from .multilayer import read_rows, save_csv


OUTPUT = ROOT / 'results/part3/background_memory_pilot'


def index_for(features):
    bank = np.asarray(features, dtype=np.float32).copy()
    faiss.normalize_L2(bank)
    index = faiss.IndexFlatL2(bank.shape[1])
    index.add(bank)
    return index


def banks(model, paths, baseline_index):
    foreground, selected = [], []
    rng = np.random.default_rng(20261122)
    all_count = foreground_count = 0
    for path in paths:
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        for rotated in utils.augment_image(rgb):
            tensor, grid = model.prepare_image(rotated)
            features = model.extract_features(tensor)
            mask = model.compute_background_mask(features, grid, threshold=10, masking_type=True)
            if mask.sum() == 0:
                raise ValueError(f'Empty foreground mask: {path}')
            foreground.append(features[mask])
            # One in four patches per reference rotation; fixed before test evaluation.
            count = max(1, round(len(features) / 4))
            selected.append(all_count + np.sort(rng.choice(len(features), count, replace=False)))
            all_count += len(features)
            foreground_count += int(mask.sum())
    if all_count != baseline_index.ntotal:
        raise ValueError('Reference bank and rotated feature segments differ')
    original = baseline_index.reconstruct_n(0, baseline_index.ntotal)
    subset = original[np.concatenate(selected)]
    return {'part2_fresh': baseline_index, 'b_foreground_bank': index_for(np.concatenate(foreground)),
            'e_stratified_25pct': index_for(subset)}, {
                'original_bank_patches': all_count, 'b_bank_patches': foreground_count,
                'e_bank_patches': len(subset),
                'b_fraction': foreground_count / all_count,
                'e_fraction': len(subset) / all_count}


def query(model, indices, path):
    started = perf_counter()
    tensor, grid = model.prepare_image(str(path))
    features = model.extract_features(tensor)
    mask = model.compute_background_mask(features, grid, threshold=10, masking_type=True)
    selected = np.asarray(features[mask], dtype=np.float32).copy()
    faiss.normalize_L2(selected)
    outputs = {}
    for name, index in indices.items():
        distances, _ = index.search(selected, k=1)
        patch = np.zeros(len(mask), dtype=np.float64)
        patch[mask] = (distances / 2).squeeze()
        outputs[name] = (float(post_eval.mean_top1p(patch)), patch.reshape(grid))
    torch.cuda.synchronize()
    return outputs, 1000 * (perf_counter() - started)


def run(categories, max_test_images):
    configure()
    faiss.omp_set_num_threads(4)
    threadpool_limits(4)
    model = load_model()
    FeatureArchive(model)
    manifest = load_manifest()
    baseline = {(row['category'], row['image_file']): float(row['score'])
                for row in read_rows(ROOT / 'results/part2/predictions_all.csv')
                if row['shots'] == '4' and row['seed'] == '0'}
    summary = []
    for category in categories:
        entry = manifest['categories'][category]
        folder = (OUTPUT / (f'debug_first_{max_test_images}' if max_test_images else '') /
                  category / 'shot_4' / 'seed_0')
        if (folder / 'metrics.json').exists():
            summary.append(json.loads((folder / 'metrics.json').read_text(encoding='utf-8')))
            print(f'Skipped completed {category}', flush=True)
            continue
        paths = [DATA / category / 'train/good' / name for name in entry['references']['4']['0']]
        detector = OfficialPredictor(model, paths, seed=0)
        indices, sizes = banks(model, paths, detector.index)
        calibration = {name: [] for name in indices}
        for filename in entry['calibration']:
            outputs, _ = query(model, indices, DATA / category / 'train/good' / filename)
            for name in indices:
                calibration[name].append(outputs[name][0])
        threshold = {name: float(np.percentile(scores, 95, method='linear'))
                     for name, scores in calibration.items()}
        torch.cuda.reset_peak_memory_stats()
        predictions = []
        for kind in ('good', 'bad'):
            images = sorted((DATA / category / 'test' / kind).glob('*.JPG'))
            if max_test_images:
                images = images[:max_test_images]
            for image in images:
                outputs, elapsed = query(model, indices, image)
                image_file = image.relative_to(ROOT).as_posix()
                label = int(kind == 'bad')
                for method, (score, patch) in outputs.items():
                    map_path = folder / 'maps' / method / kind / f'{image.stem}.npy'
                    map_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(map_path, patch.astype(np.float32))
                    decision = 'NG' if score > threshold[method] else 'OK'
                    predictions.append({'category': category, 'shots': 4, 'seed': 0,
                                        'method': method, 'image_file': image_file,
                                        'label': label, 'score': score,
                                        'threshold': threshold[method], 'decision': decision,
                                        'outcome': ('TP' if decision == 'NG' else 'FN') if label else
                                                   ('FP' if decision == 'NG' else 'TN'),
                                        'anomaly_map': map_path.relative_to(ROOT).as_posix(),
                                        'shared_query_ms': elapsed})
        metrics = {'category': category, 'shots': 4, 'seed': 0,
                   'reference_filenames': entry['references']['4']['0'],
                   'calibration_filenames': entry['calibration'], 'bank': sizes,
                   'peak_gpu_mib': torch.cuda.max_memory_allocated() / 2**20,
                   'test_images': len(predictions) // len(indices), 'methods': {}}
        for method in indices:
            group = [row for row in predictions if row['method'] == method]
            labels = [row['label'] for row in group]
            scores = [row['score'] for row in group]
            metrics['methods'][method] = {'image_auroc': float(roc_auc_score(labels, scores)),
                                          'threshold': threshold[method],
                                          **{outcome: sum(row['outcome'] == outcome for row in group)
                                             for outcome in ('TP', 'TN', 'FP', 'FN')}}
        if max_test_images is None:
            errors = [abs(row['score'] - baseline[category, row['image_file']])
                      for row in predictions if row['method'] == 'part2_fresh']
            metrics['max_abs_baseline_score_difference'] = max(errors)
            if max(errors) > 1e-5:
                raise ValueError(f'Fresh baseline did not match published Part 2: {category}, {max(errors)}')
        save_csv(folder / 'predictions.csv', predictions)
        write_json(folder / 'metrics.json', metrics)
        summary.append(metrics)
        print(f'{category}: ' + ', '.join(f'{name}={value["image_auroc"]:.4f}'
                                        for name, value in metrics['methods'].items()), flush=True)
    if max_test_images is None and set(categories) == set(manifest['categories']) and len(summary) == 6:
        write_json(OUTPUT / 'pilot_summary.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--categories', nargs='+', default=['candle', 'capsules', 'cashew',
                                                               'chewinggum', 'fryum', 'pipe_fryum'])
    parser.add_argument('--max-test-images', type=int, default=None,
                        help='Debug only: take the first N normal and anomalous test images')
    args = parser.parse_args()
    run(args.categories, args.max_test_images)
