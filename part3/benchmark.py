"""Time official versus learned localization, with fresh query extraction."""

import statistics

import faiss
import torch
from threadpoolctl import threadpool_limits

from anomaly_dino.official import FeatureArchive, load_model, utils
from anomaly_dino.predict import OfficialPredictor
from experiment_utils.common import DATA, ROOT, configure, load_manifest, write_json
from .run import OUTPUT, SOURCE, read_rows, save_rows
from .upsampling import load_upsampler, restore_scores


def main():
    configure()
    faiss.omp_set_num_threads(4)
    threadpool_limits(4)
    model = load_model()
    FeatureArchive(model)
    upsampler = load_upsampler()
    manifest = load_manifest()
    reference_scores = {row['image_file']: float(row['score'])
                        for row in read_rows(SOURCE / 'predictions_all.csv')
                        if row['shots'] == '4' and row['seed'] == '0'}

    def learned_map(tensor, features, patch, shape):
        coarse = restore_scores(upsampler, tensor, features, patch[None])[0]
        return utils.dists2map(coarse.astype('float64'), shape)

    rows = []
    for category, entry in manifest['categories'].items():
        base = DATA / category
        detector = OfficialPredictor(model, [base / 'train/good' / name for name in entry['references']['4']['0']])
        images = [image for label in ('good', 'bad') for image in sorted((base / 'test' / label).glob('*.JPG'))[:10]]
        if len(images) != 20:
            raise ValueError(f'Expected 20 benchmark queries in {category}')
        detector.predict(images[0])
        detector.predict(images[0], restore_map=learned_map)
        for index, image in enumerate(images):
            path = image.relative_to(ROOT).as_posix()
            methods = [('official_anomalydino', None), ('anyup_2x', learned_map)]
            if index % 2:
                methods.reverse()
            for method, replacement in methods:
                torch.cuda.reset_peak_memory_stats()
                score, _, _, seconds = detector.predict(image, restore_map=replacement)
                if abs(score - reference_scores[path]) > 1e-5:
                    raise ValueError(f'Benchmark did not reproduce the Part 2 image score for {path}')
                rows.append({'category': category, 'image_file': path, 'method': method,
                             'ms': seconds * 1000, 'peak_gpu_mib': torch.cuda.max_memory_allocated() / 2**20})
        print(f'Timed {category} (20 paired images)', flush=True)

    save_rows(OUTPUT / 'timing.csv', rows)
    summary = {'device': torch.cuda.get_device_name(0), 'shots': 4, 'seed': 0, 'input_short_edge': 448,
               'timing': 'serial, warmed; decode, preprocessing, fresh features, matching, scoring and map restoration',
               'excluded': 'model and reference setup, output-file I/O, metrics',
               'gpu_memory': 'peak allocated GPU tensor memory; both models resident for both timings'}
    for method in ('official_anomalydino', 'anyup_2x'):
        subset = [row for row in rows if row['method'] == method]
        summary[method] = {'images': len(subset), 'ms_mean': statistics.mean(r['ms'] for r in subset),
                           'peak_gpu_mib': max(r['peak_gpu_mib'] for r in subset)}
    write_json(OUTPUT / 'timing.json', summary)


if __name__ == '__main__':
    main()
