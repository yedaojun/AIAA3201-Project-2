"""Serial fresh-query, same-input benchmark for required and optional methods."""
import csv
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import faiss
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from anomaly_baseline.core import CATEGORIES, TemplateDetector
from anomaly_baseline.run import save_csv
from experiment_utils.common import DATA, configure, load_manifest, write_json


def benchmark_method(method):
    manifest = load_manifest()
    model = shared = archive = None
    if method == 'official_anomalydino':
        from anomaly_dino.official import load_model, FeatureArchive
        from anomaly_dino.predict import OfficialPredictor
        model = load_model()
        archive = FeatureArchive(model)
    elif method == 'official_anomalyclip':
        from Part2.anomalyclip.model import AnomalyCLIPPredictor
        shared = AnomalyCLIPPredictor()
    rows = []
    for category in CATEGORIES:
        references = [DATA / category / 'train/good' / name for name in manifest['categories'][category]['references']['4']['0']]
        if method == 'baseline_rgb_template':
            detector = TemplateDetector(references)
        elif method == 'official_anomalydino':
            detector = OfficialPredictor(model, references, 0)
        elif method == 'compact_autoencoder':
            from Part1.autoencoder.model import AutoencoderPredictor
            detector = AutoencoderPredictor(ROOT / 'results/part1_autoencoder' / category / 'best.pt')
        else:
            detector = shared
        images = sorted((DATA / category / 'test/good').glob('*.JPG'))[:10] + sorted((DATA / category / 'test/bad').glob('*.JPG'))[:10]
        assert len(images) == 20
        detector.predict(DATA / category / 'train/good' / manifest['categories'][category]['calibration'][0])
        torch.cuda.reset_peak_memory_stats()
        for image in images:
            prediction = detector.predict(image)
            elapsed = prediction[3] if method in ('compact_autoencoder', 'official_anomalyclip') else prediction[-1]
            rows.append({'method_id': method, 'category': category,
                         'sample_id': f'VisA/1cls/{category}/test/{image.parent.name}/{image.name}',
                         'image_file': image.relative_to(ROOT).as_posix(),
                         'inference_ms': elapsed * 1000, 'score': prediction[0],
                         'gpu_peak_memory_mb': 0 if method == 'baseline_rgb_template' else torch.cuda.max_memory_allocated() / 1024**2,
                         'protocol': '4-shot seed0' if method in ('baseline_rgb_template', 'official_anomalydino') else 'independent optional protocol',
                         'timer': 'decode + preprocess + representation + score + full-size map; no bank construction/output IO; CUDA synchronized'})
        print('Benchmark:', method, category, flush=True)
        del detector
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def main():
    configure()
    faiss.omp_set_num_threads(4)
    threadpool_limits(4)
    output = ROOT / 'results/extensions'
    output.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []
    for method in ['baseline_rgb_template', 'official_anomalydino', 'compact_autoencoder', 'official_anomalyclip']:
        selected = benchmark_method(method)
        rows.extend(selected)
        times = [r['inference_ms'] for r in selected]
        summary.append({'method_id': method, 'images': len(selected), 'mean_ms': float(np.mean(times)),
                        'median_ms': float(np.median(times)), 'p95_ms': float(np.percentile(times, 95)),
                        'gpu_peak_memory_mb': max(r['gpu_peak_memory_mb'] for r in selected)})
        save_csv(output / 'timing_same_inputs.csv', rows)
        save_csv(output / 'timing_summary.csv', summary)
        gc.collect()
        torch.cuda.empty_cache()
    for method in {r['method_id'] for r in rows}:
        assert {r['sample_id'] for r in rows if r['method_id'] == method} == {r['sample_id'] for r in rows[:120]}
    write_json(output / 'timing_environment.json', {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
               'threads': 4, 'method_order': [r['method_id'] for r in summary], 'serial_queries': True,
               'input_selection': 'first 10 sorted good and bad test files per category; 120 unique inputs per method',
               'notes': 'Methods retain their declared preprocessing and supervision; this is a shared-hardware latency comparison, not matched accuracy training.'})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
