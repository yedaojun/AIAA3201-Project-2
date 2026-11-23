"""Paired warmed 448/672 query-time benchmark for fixed 4-shot seed 0."""

import json

import faiss
import numpy as np
from threadpoolctl import threadpool_limits

from anomaly_dino.official import load_model
from anomaly_dino.predict import OfficialPredictor
from experiment_utils.common import DATA, ROOT, configure, load_manifest, write_json
from .high_resolution_pilot import load_high_resolution_model


CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')
OUTPUT = ROOT / 'results/part3/multiscale_fusion/benchmark_4shot_seed0.json'


def main():
    configure()
    faiss.omp_set_num_threads(4)
    threadpool_limits(4)
    manifest = load_manifest()
    model448 = load_model()
    model672 = load_high_resolution_model()
    rows = []
    for category in CATEGORIES:
        base = DATA / category
        references = [base / 'train/good' / name for name in
                      manifest['categories'][category]['references']['4']['0']]
        detector448 = OfficialPredictor(model448, references)
        detector672 = OfficialPredictor(model672, references)
        images = (sorted((base / 'test/good').glob('*.JPG'))[:10] +
                  sorted((base / 'test/bad').glob('*.JPG'))[:10])
        if len(images) != 20:
            raise ValueError('Expected ten normal and ten anomalous queries')
        for detector in (detector448, detector672):
            detector.predict(images[0])
        for image in images:
            _, _, _, time448 = detector448.predict(image)
            _, _, _, time672 = detector672.predict(image)
            rows.append({'category': category, 'image': str(image.relative_to(ROOT)),
                         'ms_448': time448 * 1000, 'ms_672': time672 * 1000,
                         'ms_fused': (time448 + time672) * 1000})
        print(category, f"448 {np.mean([r['ms_448'] for r in rows if r['category'] == category]):.1f} ms",
              f"672 {np.mean([r['ms_672'] for r in rows if r['category'] == category]):.1f} ms",
              flush=True)
    result = {'device': 'local RTX 4060 Laptop GPU; CPU FAISS, four CPU threads',
              'protocol': 'same 120 fixed queries, four shots, seed 0; two frozen models resident',
              'boundary': 'includes image decode, fresh extraction, NN search, score, map restoration; excludes model/bank setup and file I/O',
              'queries': len(rows),
              'mean_ms_448': float(np.mean([r['ms_448'] for r in rows])),
              'mean_ms_672': float(np.mean([r['ms_672'] for r in rows])),
              'mean_ms_fused': float(np.mean([r['ms_fused'] for r in rows])),
              'per_image': rows}
    write_json(OUTPUT, result)
    print(json.dumps({k: result[k] for k in ('queries', 'mean_ms_448',
                                             'mean_ms_672', 'mean_ms_fused')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
