"""Finish the 672-short-edge DINO sweep with cached, unchanged query features.

The fixed four-shot pilot used OfficialPredictor directly. This sweep reuses
each query's 672 DINO features and PCA mask across shots/reference groups;
only the normal bank and nearest-neighbor retrieval change. A saved four-shot
setting is checked before each category. Timing here covers retrieval and
scoring only; use the separate paired benchmark for end-to-end time.
"""

import argparse
import gc
import json
from time import perf_counter

import faiss
import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from anomaly_dino.official import post_eval
from anomaly_dino.predict import OfficialPredictor
from experiment_utils.common import DATA, ROOT, MANIFEST, configure, load_manifest, sha256, write_json
from .high_resolution_pilot import load_high_resolution_model
from .multilayer import read_rows, save_csv


CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')
OUT = ROOT / 'results/part3/high_resolution_pilot'


def query_features(model, path):
    tensor, grid = model.prepare_image(str(path))
    patch_features = model.extract_features(tensor)
    foreground = model.compute_background_mask(patch_features, grid, threshold=10, masking_type=True)
    features = patch_features[foreground].astype(np.float32, copy=True)
    if len(features):
        faiss.normalize_L2(features)
    return features, foreground, grid


def cached_score(index, item):
    features, foreground, grid = item
    started = perf_counter()
    if len(features):
        distances, _ = index.search(features, k=1)
        values = (distances / 2).squeeze()
    else:
        values = np.empty(0, dtype=np.float32)
    output = np.zeros_like(foreground, dtype=float)
    output[foreground] = values
    score = float(post_eval.mean_top1p(output))
    elapsed = perf_counter() - started
    return score, output.reshape(grid), elapsed


def category_paths(category, selection):
    base = DATA / category
    calibration = [base / 'train/good' / name for name in selection['calibration']]
    good = sorted((base / 'test/good').glob('*.JPG'))
    bad = sorted((base / 'test/bad').glob('*.JPG'))
    if len(calibration) != 20 or not good or not bad:
        raise ValueError(f'Incomplete input split: {category}')
    return calibration, good, bad


def verify_cache(category, selection, cache):
    base = DATA / category
    references = selection['references']['4']['0']
    detector = OfficialPredictor(MODEL, [base / 'train/good' / name for name in references])
    saved = {row['file']: row for row in read_rows(
        ROOT / 'results/part3/high_resolution_pilot' / category /
        'shot_4/seed_0/predictions.csv')}
    names = list(saved)[:3] + list(saved)[-3:]
    for name in names:
        score, _, _ = cached_score(detector.index, cache[base / name])
        if abs(score - float(saved[name]['score'])) > 1e-6:
            raise ValueError(f'Cached query path differs from saved 4-shot pilot: {category}/{name}')
    print(f'{category}: cached path matches six saved four-shot scores', flush=True)


def run_setting(category, selection, cache, shot, seed):
    folder = OUT / category / f'shot_{shot}' / f'seed_{seed}'
    if (folder / 'metrics.json').exists():
        return
    base = DATA / category
    references = selection['references'][str(shot)][str(seed)]
    if len(references) != shot or set(references) & set(selection['calibration']):
        raise ValueError('Reference/calibration protocol mismatch')
    detector = OfficialPredictor(MODEL, [base / 'train/good' / name for name in references], seed)
    calibration = []
    for name in selection['calibration']:
        score, _, _ = cached_score(detector.index, cache[base / 'train/good' / name])
        calibration.append({'file': name, 'score': score})
    threshold = float(np.percentile([r['score'] for r in calibration], 95))
    predictions = []
    for kind in ('good', 'bad'):
        for image in sorted((base / 'test' / kind).glob('*.JPG')):
            score, patch, elapsed = cached_score(detector.index, cache[image])
            label = int(kind == 'bad')
            predicted = int(score > threshold)
            map_file = folder / 'anomaly_maps' / kind / (image.stem + '.npy')
            map_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(map_file, patch.astype(np.float32))
            predictions.append({'category': category, 'shot': shot, 'seed': seed,
                                'file': f'test/{kind}/{image.name}', 'label': label,
                                'score': score, 'threshold': threshold,
                                'decision': 'NG' if predicted else 'OK',
                                'outcome': ('TP' if predicted else 'FN') if label else
                                           ('FP' if predicted else 'TN'),
                                'retrieval_scoring_seconds': elapsed,
                                'anomaly_map': str(map_file.relative_to(ROOT))})
    baseline = json.loads((ROOT / 'results/part2' / category / f'shot_{shot}' /
                           f'seed_{seed}' / 'metrics.json').read_text())
    if baseline['references'] != references:
        raise ValueError('Baseline and 672 reference filenames differ')
    metric = {'category': category, 'shot': shot, 'seed': seed, 'short_edge': 672,
              'references': references, 'calibration_files': selection['calibration'],
              'threshold': threshold,
              'image_auroc': float(roc_auc_score([r['label'] for r in predictions],
                                                 [r['score'] for r in predictions])),
              'false_positives': sum(r['outcome'] == 'FP' for r in predictions),
              'false_negatives': sum(r['outcome'] == 'FN' for r in predictions),
              'retrieval_scoring_ms_mean': float(np.mean([
                  r['retrieval_scoring_seconds'] for r in predictions]) * 1000),
              'timing_boundary': 'cached query features: CPU FAISS retrieval and scoring only',
              'baseline_448_image_auroc': baseline['image_auroc'],
              'baseline_448_false_positives': baseline['false_positives'],
              'baseline_448_false_negatives': baseline['false_negatives'],
              'bank_vectors': int(detector.index.ntotal)}
    save_csv(folder / 'calibration.csv', calibration)
    save_csv(folder / 'predictions.csv', predictions)
    write_json(folder / 'metrics.json', metric)
    print(f'{category} {shot}-shot seed {seed}: 672 {100*metric["image_auroc"]:.2f}%, '
          f'448 {100*baseline["image_auroc"]:.2f}%', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shots', nargs='+', type=int, choices=(1, 2, 8), default=[1, 2, 8])
    parser.add_argument('--seeds', nargs='+', type=int, choices=(0, 1, 2), default=[0, 1, 2])
    parser.add_argument('--categories', nargs='+', choices=CATEGORIES, default=CATEGORIES)
    args = parser.parse_args()
    configure()
    faiss.omp_set_num_threads(4)
    threadpool_limits(4)
    manifest = load_manifest()
    write_json(OUT / 'sweep_protocol.json', {
        'model': 'frozen dinov2_vits14, official agnostic preprocessing with short edge 672',
        'references': str(MANIFEST.relative_to(ROOT)),
        'manifest_sha256': sha256(MANIFEST),
        'shots_requested': args.shots, 'groups_requested': args.seeds,
        'categories_requested': args.categories,
        'query_cache': 'exact 672 feature and PCA-mask reuse across settings; six saved pilot scores checked per category',
        'calibration': 'same disjoint 20 normal training files; 95th percentile',
        'test_label_use': 'evaluation only',
        'status': 'post-hoc image-scale candidate; same VisA test split as initial pilot'})
    global MODEL
    MODEL = load_high_resolution_model()
    for category in args.categories:
        selection = manifest['categories'][category]
        pending = [(shot, seed) for shot in args.shots for seed in args.seeds
                   if not (OUT / category / f'shot_{shot}' / f'seed_{seed}' / 'metrics.json').exists()]
        if not pending:
            print(f'{category}: all requested settings already complete', flush=True)
            continue
        calibration, good, bad = category_paths(category, selection)
        cache = {}
        for index, path in enumerate(calibration + good + bad, 1):
            cache[path] = query_features(MODEL, path)
            if index % 50 == 0 or index == len(calibration) + len(good) + len(bad):
                print(f'{category}: cached {index}/{len(calibration)+len(good)+len(bad)} query features', flush=True)
        verify_cache(category, selection, cache)
        for shot, seed in pending:
            run_setting(category, selection, cache, shot, seed)
        del cache
        gc.collect()
    print('Completed requested 672 sweep settings', flush=True)


if __name__ == '__main__':
    main()
