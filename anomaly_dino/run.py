import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from functools import partial
from time import perf_counter

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import cv2
import faiss
import numpy as np
from PIL import Image
import torch
from threadpoolctl import threadpool_limits

from anomaly_baseline.core import CATEGORIES, SHOTS, SEEDS, image_auc
from anomaly_baseline.run import save_csv, summaries
from .official import ROOT, load_model, prepare_view, FeatureArchive, detection, dense_map, post_eval
from .metrics import positive_negative_auc

CONFIG = {
    'method': 'Official AnomalyDINO (DINOv2 ViT-S/14)', 'short_edge': 448, 'patch_multiple': 14,
    'model_name': 'dinov2_vits14', 'preprocess': 'agnostic', 'rotation': True,
    'test_pca_mask': True, 'mask_ref_images': False, 'knn_metric': 'L2_normalized',
    'k_neighbors': 1, 'faiss_on_cpu': True, 'precision': 'FP32 backbone; original float64 output patch maps',
    'image_score': 'official mean_top1p: floor(1% of all patch scores), masked patches set to zero',
    'dense_map': 'official dists2map: bilinear original-size resize and scipy Gaussian sigma=4',
    'pixel_auroc': 'exact AUROC over original-resolution float64 dense maps, including all normal test images',
    'threshold_percentile': 95, 'calibration_count': 20,
    'query_feature_cache': 'features archived; query extraction always fresh for measured inference',
    'reference_feature_cache': 'exact tensor-keyed disk cache; only bank construction reuses features',
    'map_storage': 'lossless official .npy patch maps; dense maps reconstructed with official function',
    'timing': 'official CUDA-synchronized per-query timer, includes decode/preprocess/extraction/PCA/kNN; excludes scoring, dense-map restoration and output I/O',
    'timing_comparison_note': 'Part 1 original timer also includes score and map restoration; report boundaries explicitly',
    'test_time_refinements': 'official agnostic masking/rotation and official sigma=4 only',
}


def git_commit(folder):
    executable = Path('C:/Users/Ziqing/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe')
    return subprocess.check_output([str(executable), '-C', str(folder), 'rev-parse', 'HEAD'], text=True).strip()


def pixel_auc(rows, root, category, group):
    total, positive_count = 0, 0
    shapes = {}
    for row in rows:
        source = root / category / row['file']
        with Image.open(source) as image:
            shapes[row['file']] = (image.height, image.width, 3)
            total += image.height * image.width
        if row['label']:
            with Image.open(root / category / 'ground_truth/bad' / (source.stem + '.png')) as image:
                positive_count += int(np.count_nonzero(np.array(image)))
    positive = np.empty(positive_count, dtype=np.float64)
    negative = np.empty(total - positive_count, dtype=np.float64)
    pi, ni = 0, 0
    for row in rows:
        score_map = dense_map(group / row['anomaly_map'], shapes[row['file']])
        if not np.isfinite(score_map).all():
            raise ValueError('Nonfinite anomaly map')
        if row['label']:
            with Image.open(root / category / 'ground_truth/bad' / (Path(row['file']).stem + '.png')) as image:
                mask = np.array(image) > 0
            selected = score_map[mask]
            positive[pi:pi + len(selected)] = selected
            pi += len(selected)
            selected = score_map[~mask]
        else:
            selected = score_map.ravel()
        negative[ni:ni + len(selected)] = selected
        ni += len(selected)
    assert pi == positive_count and pi + ni == total
    started = perf_counter()
    value = positive_negative_auc(positive, negative)
    print(f'  exact pixel AUROC: {total:,} pixels, sort/rank {perf_counter()-started:.1f}s', flush=True)
    return value, total, positive_count


def run_one(model, view, root, output, manifest, category, shot, seed, signature):
    group = output / category / f'shot_{shot}' / f'seed_{seed}'
    metric_file = group / 'metrics.json'
    if metric_file.exists():
        existing = json.loads(metric_file.read_text())
        if existing['run_signature'] != signature:
            raise ValueError(f'Different completed configuration: {group}')
        print(f'Resume {category} {shot}-shot seed {seed}', flush=True)
        return existing
    group.mkdir(parents=True, exist_ok=True)
    raw_file = group / 'official_scores.json'
    if raw_file.exists():
        raw = json.loads(raw_file.read_text())
        if raw['signature'] != signature:
            raise ValueError('Unfinished official run signature mismatch')
    else:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        scores, bank_time, times = detection.run_anomaly_detection(
            model, category, str(view).replace('\\', '/'), shot,
            {category: ['bad']}, plots_dir=str(group).replace('\\', '/'),
            save_examples=False, knn_metric='L2_normalized', knn_neighbors=1,
            faiss_on_cpu=True, masking=True, mask_ref_images=False,
            rotation=True, seed=seed, save_patch_dists=True, save_tiffs=False)
        raw = {'scores': scores, 'bank_time': bank_time, 'times': times,
               'peak_gpu_mb': torch.cuda.max_memory_allocated() / 1024**2,
               'official_wall_seconds': perf_counter() - started, 'signature': signature}
        raw_file.write_text(json.dumps(raw, indent=2), encoding='utf-8')
    selection = manifest['categories'][category]
    calibration = []
    for name in selection['calibration']:
        key = 'good/__calibration__' + name
        calibration.append({'file': name, 'score': raw['scores'][key], 'closest_template': ''})
    threshold = float(np.percentile([r['score'] for r in calibration], 95))
    save_csv(group / 'calibration.csv', calibration)
    rows = []
    for label in ('good', 'bad'):
        for image in sorted((root / category / 'test' / label).glob('*')):
            if not image.is_file():
                continue
            key = label + '/' + image.name
            relative = f'anomaly_maps/seed={seed}/{category}/test/{label}/{image.stem}.npy'
            assert (group / relative).is_file()
            patch = np.load(group / relative)
            assert abs(float(post_eval.mean_top1p(patch.ravel())) - raw['scores'][key]) < 1e-12
            rows.append({'category': category, 'shot': shot, 'seed': seed,
                         'file': f'test/{label}/{image.name}', 'label': int(label == 'bad'),
                         'score': raw['scores'][key], 'threshold': threshold,
                         'decision': 'NG' if raw['scores'][key] > threshold else 'OK',
                         'closest_template': '', 'inference_seconds': raw['times'][key], 'anomaly_map': relative})
    save_csv(group / 'predictions.csv', rows)
    print(f'{category} {shot}-shot seed {seed}: official predictions complete; evaluating pixels...', flush=True)
    px_auc, pixel_count, anomalous_pixels = pixel_auc(rows, root, category, group)
    normals = [r for r in rows if not r['label']]
    abnormal = [r for r in rows if r['label']]
    metrics = {'category': category, 'shot': shot, 'seed': seed,
               'image_auroc': image_auc([r['label'] for r in rows], [r['score'] for r in rows]),
               'pixel_auroc': px_auc, 'threshold': threshold,
               'normal_test_images': len(normals), 'anomalous_test_images': len(abnormal),
               'false_positives': sum(r['decision'] == 'NG' for r in normals),
               'false_negatives': sum(r['decision'] == 'OK' for r in abnormal),
               'inference_ms_mean': float(np.mean([r['inference_seconds'] for r in rows]) * 1000),
               'pixel_count': pixel_count, 'anomalous_pixels': anomalous_pixels,
               'gpu_peak_memory_mb': raw['peak_gpu_mb'], 'device': torch.cuda.get_device_name(0) + ' + CPU FAISS',
               'references': selection['references'][str(shot)][str(seed)],
               'calibration_files': selection['calibration'], 'run_signature': signature,
               'memory_bank_seconds': raw['bank_time'], 'official_wall_seconds': raw['official_wall_seconds']}
    metric_file.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f'DONE {category} {shot}-shot seed {seed}: image={metrics["image_auroc"]:.4f}, pixel={px_auc:.4f}, '
          f'{metrics["inference_ms_mean"]:.1f} ms/image, GPU peak {metrics["gpu_peak_memory_mb"]:.0f} MB', flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--categories', nargs='+', choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument('--shots', nargs='+', type=int, choices=SHOTS, default=list(SHOTS))
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--output', type=Path, default=ROOT / 'results/part2')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    threadpool_limits(4)
    detection.tqdm = partial(detection.tqdm, disable=True)
    manifest_file = ROOT / 'configs/reference_manifest.json'
    manifest = json.loads(manifest_file.read_text())
    root = ROOT / 'data/VisA_pytorch/1cls'
    view = prepare_view(manifest, root)
    settings = {'config': CONFIG, 'manifest_sha256': hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
                'anomalydino_commit': git_commit(ROOT / 'third_party/AnomalyDINO'),
                'dinov2_commit': git_commit(ROOT / 'third_party/dinov2'),
                'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name('official.py').read_bytes() + Path(__file__).with_name('metrics.py').read_bytes()).hexdigest(),
                'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                'opencv': cv2.__version__, 'faiss': faiss.__version__,
                'weights_sha256': hashlib.sha256((ROOT / 'models/torch/hub/checkpoints/dinov2_vits14_pretrain.pth').read_bytes()).hexdigest()}
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'run_config.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    model = load_model()
    archive = FeatureArchive(model)
    # Warm up GPU without including calibration or test labels in model setup.
    path = root / args.categories[0] / 'train/good' / manifest['categories'][args.categories[0]]['references']['1']['0'][0]
    tensor, _ = model.prepare_image(str(path))
    for _ in range(3):
        model.extract_features(tensor)
    all_metrics = []
    for category in args.categories:
        for shot in args.shots:
            for seed in args.seeds:
                all_metrics.append(run_one(model, view, root, args.output, manifest, category, shot, seed, signature))
                (args.output / 'metrics_all.json').write_text(json.dumps(all_metrics, indent=2), encoding='utf-8')
    summaries(all_metrics, args.output)
    (args.output / 'feature_archive_stats.json').write_text(json.dumps(archive.stats, indent=2), encoding='utf-8')
    print('PART 2 COMPLETE:', args.output, flush=True)


if __name__ == '__main__':
    main()
