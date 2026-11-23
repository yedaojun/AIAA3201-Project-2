"""Diagnostic stage timing of official and two-layer query paths; four shots/seed 0."""

import json
import statistics
from time import perf_counter

import cv2
import faiss
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from anomaly_dino.official import load_model, post_eval, utils
from anomaly_dino.predict import OfficialPredictor
from experiment_utils.common import DATA, ROOT, configure, load_manifest, write_json
from .multilayer import TwoLayerDINO, read_rows, save_csv


def query(detector, path):
    t0 = perf_counter()
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    tensor, grid = detector.model.prepare_image(str(path))
    t1 = perf_counter()
    patch_features = detector.model.extract_features(tensor)
    torch.cuda.synchronize()
    t2 = perf_counter()
    mask = detector.model.compute_background_mask(patch_features, grid, threshold=10, masking_type=True)
    features = patch_features[mask].copy()
    faiss.normalize_L2(features)
    t3 = perf_counter()
    distances, _ = detector.index.search(features, k=1)
    t4 = perf_counter()
    output = np.zeros_like(mask, dtype=float)
    output[mask] = (distances / 2).squeeze()
    score = float(post_eval.mean_top1p(output))
    utils.dists2map(output.reshape(grid), image.shape)
    torch.cuda.synchronize()
    t5 = perf_counter()
    return score, {'decode_prepare_ms': 1000 * (t1-t0),
                   'extract_ms': 1000 * (t2-t1), 'mask_normalize_ms': 1000 * (t3-t2),
                   'faiss_ms': 1000 * (t4-t3), 'score_restore_ms': 1000 * (t5-t4),
                   'total_ms': 1000 * (t5-t0)}


def main():
    configure()
    faiss.omp_set_num_threads(4)
    threadpool_limits(4)
    original = load_model()
    improved = TwoLayerDINO(original)
    records = []
    for category, selection in load_manifest()['categories'].items():
        base = DATA / category
        references = [base / 'train/good' / n for n in selection['references']['4']['0']]
        detectors = {'official': OfficialPredictor(original, references),
                     'two_layer': OfficialPredictor(improved, references)}
        expected = {
            'official': {r['file']: float(r['score']) for r in read_rows(
                ROOT / 'results/part2' / category / 'shot_4/seed_0/predictions.csv')},
            'two_layer': {r['sample_id']: float(r['score']) for r in read_rows(
                ROOT / 'results/part3/multilayer' / category / 'shot_4/seed_0/predictions.csv')},
        }
        images = [p for kind in ('good', 'bad') for p in sorted((base / 'test' / kind).glob('*.JPG'))[:10]]
        for detector in detectors.values():
            query(detector, images[0])
        for i, path in enumerate(images):
            order = ('official', 'two_layer') if i % 2 == 0 else ('two_layer', 'official')
            for method in order:
                score, stages = query(detectors[method], path)
                suffix = 'test/' + path.parent.name + '/' + path.name
                key = suffix if method == 'official' else f'VisA/1cls/{category}/{suffix}'
                if abs(score - expected[method][key]) > 1e-5:
                    raise ValueError(f'Score mismatch: {method} {key}')
                records.append({'category': category, 'file': suffix, 'method': method, **stages})
        print('Timed stages:', category, flush=True)
    destination = ROOT / 'results/part3/stage_benchmark'
    save_csv(destination / 'query_stages.csv', records)
    summary = {'device': torch.cuda.get_device_name(0), 'shots': 4, 'seed': 0,
               'images_per_method': 120, 'warmup': 'one query per detector and category',
               'measurement': 'serial paired, same images, CUDA synchronized at feature stage and end',
               'excluded': 'model and reference-bank setup, output I/O and metrics'}
    for method in ('official', 'two_layer'):
        part = [r for r in records if r['method'] == method]
        summary[method] = {k: statistics.mean(r[k] for r in part)
                           for k in ('decode_prepare_ms', 'extract_ms', 'mask_normalize_ms',
                                     'faiss_ms', 'score_restore_ms', 'total_ms')}
    write_json(destination / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
