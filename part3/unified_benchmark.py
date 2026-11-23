"""One warmed 120-query timing boundary for all report comparison paths."""

from time import perf_counter
import json

import faiss
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from anomaly_dino.official import load_model, utils
from anomaly_dino.predict import OfficialPredictor
from experiment_utils.common import DATA, ROOT, configure, load_manifest, write_json
from .high_resolution_pilot import load_high_resolution_model
from .multilayer import TwoLayerDINO, read_rows, save_csv
from .upsampling import load_upsampler, restore_scores


OUT = ROOT / 'results/part3/unified_benchmark_4shot_seed0'
METHODS = ('official_448', 'two_layer_a', 'a_plus_anyup', 'official_672', 'fixed_448_672')


def main():
    configure()
    faiss.omp_set_num_threads(4)
    threadpool_limits(4)
    model448 = load_model()
    model672 = load_high_resolution_model()
    layer = TwoLayerDINO(model448)
    anyup = load_upsampler()
    rows = []
    for category, selection in load_manifest()['categories'].items():
        base = DATA / category
        refs = [base / 'train/good' / name for name in selection['references']['4']['0']]
        detector448 = OfficialPredictor(model448, refs)
        detector_a = OfficialPredictor(layer, refs)
        detector672 = OfficialPredictor(model672, refs)
        expected448 = {r['file']: float(r['score']) for r in read_rows(
            ROOT / 'results/part2' / category / 'shot_4/seed_0/predictions.csv')}
        expected_a = {r['sample_id'].split(category + '/', 1)[1]: float(r['score']) for r in read_rows(
            ROOT / 'results/part3/multilayer' / category / 'shot_4/seed_0/predictions.csv')}
        expected672 = {r['file']: float(r['score']) for r in read_rows(
            ROOT / 'results/part3/high_resolution_pilot' / category / 'shot_4/seed_0/predictions.csv')}
        images = [p for kind in ('good', 'bad') for p in sorted((base / 'test' / kind).glob('*.JPG'))[:10]]
        if len(images) != 20:
            raise ValueError('Incomplete query list')

        def learned_map(tensor, _features, patch, shape):
            coarse = restore_scores(anyup, tensor, layer.last_final, patch[None])[0]
            return utils.dists2map(coarse.astype(np.float64), shape)

        # Warm every path before timing; all weights and banks stay resident.
        for detector, restore in ((detector448, None), (detector_a, None),
                                  (detector_a, learned_map), (detector672, None)):
            detector.predict(images[0], restore_map=restore)
        for index, image in enumerate(images):
            file = f'test/{image.parent.name}/{image.name}'
            conditions = list(METHODS)
            if index % 2:
                conditions.reverse()
            for name in conditions:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = perf_counter()
                if name == 'fixed_448_672':
                    score448 = detector448.predict(image)[0]
                    score672 = detector672.predict(image)[0]
                    score = (score448, score672)
                else:
                    detector, restore = {
                        'official_448': (detector448, None),
                        'two_layer_a': (detector_a, None),
                        'a_plus_anyup': (detector_a, learned_map),
                        'official_672': (detector672, None),
                    }[name]
                    score = detector.predict(image, restore_map=restore)[0]
                torch.cuda.synchronize()
                elapsed = (perf_counter() - start) * 1000
                expected = {'official_448': expected448[file],
                            'two_layer_a': expected_a[file],
                            'a_plus_anyup': expected_a[file],
                            'official_672': expected672[file]}
                if name == 'fixed_448_672':
                    if abs(score[0] - expected448[file]) > 1e-5 or abs(score[1] - expected672[file]) > 1e-5:
                        raise ValueError(f'Saved score mismatch: {category} {file} {name}')
                elif abs(score - expected[name]) > 1e-5:
                    raise ValueError(f'Saved score mismatch: {category} {file} {name}')
                rows.append({'category': category, 'file': file, 'method': name,
                             'ms': elapsed, 'peak_allocated_gpu_mib': torch.cuda.max_memory_allocated() / 2**20})
        print(f'{category}: 20 paired queries × five paths', flush=True)
    save_csv(OUT / 'per_query.csv', rows)
    summary = {'device': torch.cuda.get_device_name(0), 'shots': 4, 'reference_group': 0,
               'queries_per_method': 120,
               'timing_boundary': 'same process and 120 fixed queries, all models/banks resident; each path warmed; alternating path order; synchronized GPU; decode, extraction, retrieval, map restoration included; setup/output/metrics excluded',
               'memory_boundary': 'max PyTorch allocated GPU tensor MiB during query, including resident weights; not total GPU device memory'}
    for name in METHODS:
        values = [r for r in rows if r['method'] == name]
        summary[name] = {'mean_ms': float(np.mean([r['ms'] for r in values])),
                         'peak_allocated_gpu_mib': float(max(r['peak_allocated_gpu_mib'] for r in values))}
    write_json(OUT / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
