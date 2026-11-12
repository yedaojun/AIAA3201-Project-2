"""Method-independent prediction export. Model implementations live under each Part."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from anomaly_baseline.core import CATEGORIES
from .common import ROOT, configure, write_json


def infer_category(path):
    parts = Path(path).parts
    if 'ground_truth' in parts or 'Masks' in parts:
        raise ValueError('Choose a product image, not a ground-truth mask.')
    found = [category for category in CATEGORIES if category in parts]
    return found[-1] if found else None


def save_prediction(method, category, image, predictor, results, output):
    metric = json.loads((Path(results) / category / 'metrics.json').read_text())
    score, working, dense, elapsed, reconstruction = predictor.predict(image)
    threshold = metric['threshold']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'anomaly_map.npz', map=working.astype(np.float32))
    if reconstruction is not None:
        Image.fromarray(np.rint(np.clip(reconstruction, 0, 1) * 255).astype(np.uint8)).save(output / 'reconstruction.png')
    result = {'method': method, 'category': category, 'image': str(Path(image).resolve()),
              'score': score, 'threshold': threshold, 'decision': 'NG' if score > threshold else 'OK',
              'inference_ms': elapsed * 1000, 'anomaly_map': str((output / 'anomaly_map.npz').resolve())}
    write_json(output / 'prediction.json', result)
    return result, dense, reconstruction


def run_cli(method, loader, results, default_output):
    parser = argparse.ArgumentParser(description=method + ' single-image inspection')
    parser.add_argument('--category', choices=CATEGORIES)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=default_output)
    args = parser.parse_args()
    category = infer_category(args.image) or args.category
    if category is None:
        parser.error('Specify --category for external images.')
    configure()
    result, _, _ = save_prediction(method, category, args.image, loader(category), results, args.output)
    print(json.dumps(result, indent=2), flush=True)
