"""Fair warmed end-to-end timing of official DINO, A and A+AnyUp on fixed queries."""

import statistics

import faiss
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from anomaly_dino.official import load_model, utils
from anomaly_dino.predict import OfficialPredictor
from experiment_utils.common import DATA, ROOT, configure, load_manifest, write_json
from .composed import OUTPUT
from .multilayer import TwoLayerDINO, read_rows, save_csv
from .upsampling import load_upsampler, restore_scores


def main():
    configure()
    faiss.omp_set_num_threads(4)
    threadpool_limits(4)
    original = load_model()
    two_layer = TwoLayerDINO(original)
    upsampler = load_upsampler()
    rows = []
    for category, selection in load_manifest()['categories'].items():
        base = DATA / category
        paths = [base / 'train/good' / name for name in selection['references']['4']['0']]
        baseline = OfficialPredictor(original, paths)
        improved = OfficialPredictor(two_layer, paths)
        expected_baseline = {r['file']: float(r['score'])
                             for r in read_rows(ROOT / 'results/part2' / category /
                                                'shot_4/seed_0/predictions.csv')}
        expected_improved = {r['sample_id']: float(r['score'])
                             for r in read_rows(ROOT / 'results/part3/multilayer' / category /
                                                'shot_4/seed_0/predictions.csv')}

        def learned_map(tensor, _features, patch, shape):
            return utils.dists2map(restore_scores(upsampler, tensor, two_layer.last_final,
                                                  patch[None])[0].astype(np.float64), shape)

        images = [p for kind in ('good', 'bad') for p in sorted((base / 'test' / kind).glob('*.JPG'))[:10]]
        if len(images) != 20:
            raise ValueError(f'Expected 20 benchmark images per category: {category}')
        baseline.predict(images[0])
        improved.predict(images[0])
        improved.predict(images[0], restore_map=learned_map)
        for index, image in enumerate(images):
            conditions = [('official', baseline, None), ('two_layer', improved, None),
                          ('two_layer_anyup', improved, learned_map)]
            if index % 2:
                conditions.reverse()
            for name, detector, restoration in conditions:
                torch.cuda.reset_peak_memory_stats()
                score, _, _, seconds = detector.predict(image, restore_map=restoration)
                relative = 'test/' + image.parent.name + '/' + image.name
                sample_id = 'VisA/1cls/' + category + '/' + relative
                expected = (expected_baseline[relative] if name == 'official' else
                            expected_improved[sample_id])
                if abs(score - expected) > 1e-5:
                    raise ValueError(f'Saved score mismatch for {name} {image}')
                rows.append({'category': category, 'sample_id': sample_id, 'method': name,
                             'ms': 1000 * seconds,
                             'peak_gpu_mib': torch.cuda.max_memory_allocated() / 2**20})
        print(f'Timed {category}: 20 images × 3 methods', flush=True)
    save_csv(OUTPUT / 'timing.csv', rows)
    summary = {'device': torch.cuda.get_device_name(0), 'shots': 4, 'seed': 0,
               'input_short_edge': 448,
               'included': 'fresh decode, preprocessing, feature extraction, retrieval, scoring, map restoration',
               'excluded': 'reference-bank/model setup, output I/O, metric evaluation',
               'memory_note': 'peak allocated GPU tensor memory; both frozen models resident for each timing'}
    for name in ('official', 'two_layer', 'two_layer_anyup'):
        part = [r for r in rows if r['method'] == name]
        summary[name] = {'images': len(part), 'mean_ms': statistics.mean(r['ms'] for r in part),
                         'peak_gpu_mib': max(r['peak_gpu_mib'] for r in part)}
    write_json(OUTPUT / 'timing.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
