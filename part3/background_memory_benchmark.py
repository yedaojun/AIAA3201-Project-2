"""Matched retrieval and bank-size benchmark for the B/E four-shot pilot."""

import json
import statistics
from time import perf_counter

import faiss
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from anomaly_dino.official import FeatureArchive, load_model
from anomaly_dino.predict import OfficialPredictor
from experiment_utils.common import DATA, configure, load_manifest
from .background_memory_pilot import OUTPUT, banks
from .multilayer import save_csv


def main():
    configure()
    faiss.omp_set_num_threads(4)
    threadpool_limits(4)
    model = load_model()
    FeatureArchive(model)
    manifest = load_manifest()
    measurements, bank_sizes = [], []
    for category, entry in manifest['categories'].items():
        base = DATA / category
        references = [base / 'train/good' / name for name in entry['references']['4']['0']]
        original = OfficialPredictor(model, references, seed=0)
        indices, sizes = banks(model, references, original.index)
        for method, index in indices.items():
            bank_sizes.append({'category': category, 'method': method,
                               'patches': index.ntotal, 'feature_bytes': index.ntotal * index.d * 4})
        images = [image for label in ('good', 'bad')
                  for image in sorted((base / 'test' / label).glob('*.JPG'))[:10]]
        if len(images) != 20:
            raise ValueError(f'Expected 20 benchmark images: {category}')
        for image in images:
            tensor, grid = model.prepare_image(str(image))
            features = model.extract_features(tensor)
            mask = model.compute_background_mask(features, grid, threshold=10, masking_type=True)
            query = np.asarray(features[mask], dtype=np.float32).copy()
            faiss.normalize_L2(query)
            for repeat in range(3):
                methods = list(indices)
                methods = methods[repeat:] + methods[:repeat]
                for method in methods:
                    start = perf_counter()
                    indices[method].search(query, k=1)
                    elapsed = (perf_counter() - start) * 1000
                    measurements.append({'category': category, 'image_file': image.name,
                                         'repeat': repeat, 'method': method,
                                         'retrieval_ms': elapsed})
        print(f'Timed {category}: 20 queries × 3 repeats', flush=True)
    save_csv(OUTPUT / 'retrieval_timing.csv', measurements)
    save_csv(OUTPUT / 'bank_sizes.csv', bank_sizes)
    summary = {'device': torch.cuda.get_device_name(0), 'shots': 4, 'seed': 0,
               'images': 120, 'repeats': 3, 'faiss_threads': 4,
               'boundary': 'CPU FAISS nearest-neighbor retrieval only, identical freshly extracted DINO queries',
               'excluded': 'image decode, DINO feature extraction, PCA query mask, map restoration and model load',
               'methods': {}}
    for method in ('part2_fresh', 'b_foreground_bank', 'e_stratified_25pct'):
        timed = [row['retrieval_ms'] for row in measurements if row['method'] == method]
        sizes = [row['feature_bytes'] for row in bank_sizes if row['method'] == method]
        summary['methods'][method] = {'retrieval_ms_mean': statistics.mean(timed),
                                      'bank_feature_mib_mean': statistics.mean(sizes) / 2**20,
                                      'bank_patches_mean': statistics.mean(row['patches'] for row in bank_sizes
                                                                          if row['method'] == method)}
    (OUTPUT / 'retrieval_timing.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
