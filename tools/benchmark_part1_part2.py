"""Fresh end-to-end timing on identical fixed 4-shot/seed-0 inputs for both methods."""
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import cv2
import faiss
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from anomaly_baseline.core import CATEGORIES, TemplateDetector
from anomaly_dino.official import load_model, FeatureArchive
from anomaly_dino.predict import OfficialPredictor

cv2.setNumThreads(1)
torch.set_num_threads(4)
faiss.omp_set_num_threads(4)
threadpool_limits(4)
data = ROOT / 'data/VisA_pytorch/1cls'
manifest = json.loads((ROOT / 'configs/reference_manifest.json').read_text())
model = load_model()
archive = FeatureArchive(model)
rows = []
for category in CATEGORIES:
    references = [data / category / 'train/good' / name for name in manifest['categories'][category]['references']['4']['0']]
    baseline = TemplateDetector(references)
    dino = OfficialPredictor(model, references, 0)
    for detector in (baseline, dino):
        detector.predict(references[0])
    images = sorted((data / category / 'test/good').glob('*.JPG'))[:10] + sorted((data / category / 'test/bad').glob('*.JPG'))[:10]
    assert len(images) == 20
    torch.cuda.reset_peak_memory_stats()
    for image in images:
        for method, detector in [('baseline_rgb_template', baseline), ('official_anomalydino', dino)]:
            prediction = detector.predict(image)
            rows.append({'method_id': method, 'category': category, 'shots': 4, 'seed': 0,
                         'image_file': image.relative_to(ROOT).as_posix(),
                         'inference_ms_end_to_end': prediction[-1] * 1000,
                         'score': prediction[0],
                         'timing': 'decode + geometry + representation + matching + image score + original-size map; excludes bank build and output I/O'})
    print('Benchmark done:', category, flush=True)
destination = ROOT / 'results/comparisons'
destination.mkdir(exist_ok=True)
with (destination / 'timing_same_inputs.csv').open('w', encoding='utf-8-sig', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
summary = {}
for method in ('baseline_rgb_template', 'official_anomalydino'):
    times = [r['inference_ms_end_to_end'] for r in rows if r['method_id'] == method]
    summary[method] = {'images': len(times), 'mean_ms': float(np.mean(times)), 'median_ms': float(np.median(times))}
summary['note'] = 'Measured sequentially after full evaluations, same 120 inputs per method, 4-shot seed 0, fresh query features. Original run CSVs retain official timing boundaries.'
summary['environment'] = {'python': sys.version, 'numpy': np.__version__, 'opencv': cv2.__version__,
                          'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
                          'cpu_threads': 4, 'opencv_threads': 1}
(destination / 'timing_same_inputs.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2), flush=True)
