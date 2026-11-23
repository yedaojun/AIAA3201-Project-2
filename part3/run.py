"""Evaluate pretrained upsampling on the completed, paired Part 2 runs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from PIL import Image

from anomaly_dino.metrics import positive_negative_auc
from anomaly_dino.official import load_model, utils
from experiment_utils.common import ROOT, MANIFEST, configure, sha256, write_json
from .upsampling import (
    CHECKPOINT_SHA256, CHECKPOINT_URL, REVISION, SCALE, load_upsampler, restore_scores,
    intermediate_bilinear,
)

SOURCE = ROOT / 'results/part2'
OUTPUT = ROOT / 'results/part3/anyup_score_maps'
CONFIGURATIONS = [(shot, seed) for shot in (1, 2, 4, 8) for seed in range(3)]
METHODS = ('bilinear_2x', 'anyup_2x')


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save_rows(path, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_inputs():
    grouped = defaultdict(dict)
    for row in read_rows(SOURCE / 'predictions_all.csv'):
        key = (int(row['shots']), int(row['seed']))
        sample = grouped[row['sample_id']]
        if key in sample:
            raise ValueError(f'Duplicate configuration for {row["sample_id"]}')
        sample[key] = row
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if len(grouped) != 960:
        raise ValueError('Expected the full 960-image VisA test split')
    for sample in grouped.values():
        if set(sample) != set(CONFIGURATIONS):
            raise ValueError('Incomplete reference group or shot count')
        for (shot, seed), row in sample.items():
            references = [Path(p).name for p in json.loads(row['reference_images_json'])]
            if references != manifest['categories'][row['category']]['references'][str(shot)][str(seed)]:
                raise ValueError('Reference filenames differ from the fixed Part 2 manifest')
    return grouped


def map_file(output, row):
    image = Path(row['image_file'])
    return output / 'maps' / row['category'] / image.parent.name / f'{image.stem}.npz'


def saved_maps(sample):
    return np.stack([np.load(ROOT / sample[config]['anomaly_map']) for config in CONFIGURATIONS])


def generate(grouped, output):
    configure()
    model = load_model()
    upsampler = load_upsampler()
    for index, sample in enumerate(grouped.values(), 1):
        row = sample[CONFIGURATIONS[0]]
        path = map_file(output, row)
        if path.exists():
            continue
        coarse = saved_maps(sample)
        image, grid = model.prepare_image(str(ROOT / row['image_file']))
        if tuple(grid) != coarse.shape[-2:]:
            raise ValueError(f'Saved map does not match input preprocessing: {path}')
        features = model.extract_features(image)
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        learned = restore_scores(upsampler, image, features, coarse)
        runtime = 1000 * (perf_counter() - started)
        bilinear = intermediate_bilinear(coarse)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream, anyup_2x=learned, bilinear_2x=bilinear,
                     upsampling_ms_all_12=runtime, gpu_peak_mib=torch.cuda.max_memory_allocated() / 2**20)
        temporary.replace(path)
        if index == 1 or index % 25 == 0 or index == len(grouped):
            print(f'Maps {index}/{len(grouped)}; {row["category"]}; learned stage {runtime:.1f} ms', flush=True)


def evaluate_config(samples, output, config, method):
    """Exact rank statistic on all annotated and normal pixels at original size."""
    mode = config == (4, 0)
    index = CONFIGURATIONS.index(config)
    counts = []
    for sample in samples:
        row = sample[config]
        with Image.open(ROOT / row['image_file']) as image:
            shape = (image.height, image.width)
        pixels = 0
        if row['ground_truth_mask']:
            with Image.open(ROOT / row['ground_truth_mask']) as mask:
                pixels = int(np.count_nonzero(np.asarray(mask)))
        counts.append((shape, pixels))
    positive_count = sum(pixels for _, pixels in counts)
    pixel_count = sum(h * w for (h, w), _ in counts)
    # Saved learned maps are float32. Preserve their values and produce a
    # float64 dense map with the same resize and Gaussian used in Part 2.
    positive = np.empty(positive_count, dtype=np.float64)
    negative = np.empty(pixel_count - positive_count, dtype=np.float64)
    pi = ni = 0
    diagnostics = []
    for sample, (shape, pixels) in zip(samples, counts, strict=True):
        row = sample[config]
        if method == 'official_anomalydino':
            coarse = np.load(ROOT / row['anomaly_map'])
        else:
            with np.load(map_file(output, row)) as data:
                coarse = data[method][index]
        dense = utils.dists2map(coarse.astype(np.float64, copy=False), shape)
        if not np.isfinite(dense).all():
            raise ValueError(f'Nonfinite anomaly map for {row["sample_id"]}')
        hit = None
        if pixels:
            with Image.open(ROOT / row['ground_truth_mask']) as source:
                mask = np.asarray(source) > 0
            positive[pi:pi + pixels] = dense[mask]
            pi += pixels
            remainder = dense[~mask]
            if mode:
                hit = int(mask.ravel()[dense.argmax()])
        else:
            remainder = dense.ravel()
        negative[ni:ni + len(remainder)] = remainder
        ni += len(remainder)
        if mode:
            diagnostics.append({'sample_id': row['sample_id'], 'method': method,
                                'defect_fraction': pixels / dense.size, 'peak_inside_gt': hit,
                                'peak_heatmap_score': float(dense.max()), 'image_score': row['score'],
                                'threshold': row['threshold'], 'outcome': row['outcome']})
    if pi != len(positive) or ni != len(negative):
        raise ValueError('Pixel counts differ from masks')
    return positive_negative_auc(positive, negative), diagnostics


def evaluate(grouped, output, shots, seeds, categories):
    by_category = defaultdict(list)
    for sample in grouped.values():
        by_category[sample[CONFIGURATIONS[0]]['category']].append(sample)
    results = {(row['category'], int(row['shots']), int(row['seed'])): row
               for row in read_rows(SOURCE / 'runs.csv')}
    result_dir = output / 'paired'
    result_dir.mkdir(exist_ok=True)
    all_diagnostics = []
    for category, samples in by_category.items():
        if categories and category not in categories:
            continue
        if (not all(map_file(output, s[CONFIGURATIONS[0]]).is_file() for s in samples)):
            raise ValueError(f'Learned maps are incomplete for {category}')
        for config in CONFIGURATIONS:
            shot, seed = config
            if shot not in shots or seed not in seeds:
                continue
            path = result_dir / f'{category}_{shot}shot_seed{seed}.json'
            if path.exists():
                continue
            started = perf_counter()
            row = results[category, shot, seed]
            value = {'category': category, 'shot': shot, 'seed': seed,
                     'image_auroc_all_methods': float(row['image_auroc']),
                     'pixel_auroc_official': float(row['pixel_auroc']),
                     **{key: int(row[key]) for key in ('FP', 'FN', 'TP', 'TN')}}
            methods = (*METHODS, 'official_anomalydino') if config == (4, 0) else METHODS
            for method in methods:
                auc, diagnostics = evaluate_config(samples, output, config, method)
                if method == 'official_anomalydino':
                    if not np.isclose(auc, value['pixel_auroc_official'], rtol=0, atol=1e-12):
                        raise ValueError('Original AUROC does not reproduce saved Part 2 result')
                else:
                    value[f'pixel_auroc_{method}'] = auc
                all_diagnostics.extend(diagnostics)
            if config == (4, 0):
                save_rows(result_dir / f'{category}_localization_diagnostics.csv', all_diagnostics)
                all_diagnostics.clear()
            write_json(path, value)
            print(f"{category} {shot}-shot seed {seed}: official {value['pixel_auroc_official']:.5f}, "
                  f"bilinear 2x {value['pixel_auroc_bilinear_2x']:.5f}, "
                  f"AnyUp 2x {value['pixel_auroc_anyup_2x']:.5f}, {perf_counter() - started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('generate', 'evaluate'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--shots', type=int, nargs='+', choices=(1, 2, 4, 8), default=(1, 2, 4, 8))
    parser.add_argument('--seeds', type=int, nargs='+', choices=(0, 1, 2), default=(0, 1, 2))
    parser.add_argument('--categories', nargs='+')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    grouped = read_inputs()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = {'method': 'AnyUp guidance applied to patch scores instead of feature values',
                'checkpoint_url': CHECKPOINT_URL, 'checkpoint_sha256': CHECKPOINT_SHA256,
                'upstream_revision': REVISION, 'scale': SCALE, 'input_short_edge': 448,
                'configurations': [list(config) for config in CONFIGURATIONS],
                'controls': ['direct official bilinear+Gaussian', '2x bilinear+official bilinear+Gaussian'],
                'unchanged': 'normal reference groups, scores, thresholds, decisions, frozen DINOv2 weights',
                'predictions_sha256': sha256(SOURCE / 'predictions_all.csv'), 'manifest_sha256': sha256(MANIFEST)}
    metadata = args.output / 'protocol.json'
    if metadata.exists() and json.loads(metadata.read_text(encoding='utf-8')) != settings:
        raise ValueError('Output contains a different protocol; choose a new --output')
    write_json(metadata, settings)
    if args.stage == 'generate':
        generate(grouped, args.output)
    else:
        evaluate(grouped, args.output, args.shots, args.seeds, args.categories)


if __name__ == '__main__':
    main()
