"""Train/evaluate only the optional Part 1 convolutional autoencoder."""
import argparse
import json
from pathlib import Path

import torch

from anomaly_baseline.core import CATEGORIES
from experiment_utils.common import ROOT, MANIFEST, configure, sha256, signature, write_json
from experiment_utils.evaluate import evaluate_category, summarize
from .model import AutoencoderPredictor, TRAIN_CONFIG, train_category


def main():
    parser = argparse.ArgumentParser(description='Part 1 optional convolutional autoencoder')
    parser.add_argument('--categories', nargs='+', choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument('--output', type=Path, default=ROOT / 'results/part1_autoencoder')
    parser.add_argument('--train-only', action='store_true')
    args = parser.parse_args()
    configure()
    settings = {'method_id': 'compact_autoencoder', 'protocol': 'additional normal-only trained reconstruction baseline; not few-shot',
                'config': TRAIN_CONFIG, 'map_storage': 'float32 working-size NPZ; original-size bilinear restore',
                'score': 'top ceil(1%) mean of RGB absolute reconstruction residual, Gaussian sigma=2',
                'map_range': [0, 1], 'manifest_sha256': sha256(MANIFEST),
                'evaluation_sha256': sha256(ROOT / 'experiment_utils/evaluate.py'),
                'common_sha256': sha256(ROOT / 'experiment_utils/common.py'), 'torch': torch.__version__}
    write_json(args.output / 'run_config.json', settings)
    metrics = []
    for category in args.categories:
        checkpoint = train_category(category, args.output)
        if args.train_only:
            continue
        training = json.loads((checkpoint.parent / 'training_complete.json').read_text())
        run_signature = signature({'settings': settings, 'category': category, 'checkpoint': sha256(checkpoint)})
        target = args.output / category / 'metrics.json'
        if target.exists():
            metric = json.loads(target.read_text())
            if metric['run_signature'] != run_signature:
                raise ValueError('Saved autoencoder evaluation differs; use a new output directory.')
        else:
            predictor = AutoencoderPredictor(checkpoint)
            metric = evaluate_category(predictor, category, args.output, settings, run_signature, training)
            del predictor
            torch.cuda.empty_cache()
        metrics.append(metric)
    if metrics:
        summarize(args.output, metrics, settings)
    print('Part 1 autoencoder finished:', args.output, flush=True)


if __name__ == '__main__':
    main()
