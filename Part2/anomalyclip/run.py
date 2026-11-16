"""Evaluate only the optional Part 2 AnomalyCLIP zero-shot method."""
import argparse
import json
from pathlib import Path

import torch

from anomaly_baseline.core import CATEGORIES
from experiment_utils.common import ROOT, configure, sha256, signature, write_json
from experiment_utils.evaluate import evaluate_category, summarize
from .model import AnomalyCLIPPredictor, clip_settings


def main():
    parser = argparse.ArgumentParser(description='Part 2 optional AnomalyCLIP')
    parser.add_argument('--categories', nargs='+', choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=111)
    args = parser.parse_args()
    args.output = args.output or (ROOT / 'results/part2_anomalyclip' if args.seed == 111 else
                                  ROOT / 'results/part2_anomalyclip_multiseed' / f'seed_{args.seed}')
    configure()
    settings = clip_settings(args.seed)
    settings['evaluation_sha256'] = sha256(ROOT / 'experiment_utils/evaluate.py')
    settings['common_sha256'] = sha256(ROOT / 'experiment_utils/common.py')
    settings['torch'] = torch.__version__
    config_file = args.output / 'run_config.json'
    if config_file.exists() and json.loads(config_file.read_text()) != settings:
        raise ValueError('Saved run settings differ. Choose a new --output directory to preserve the original experiment.')
    write_json(config_file, settings)
    metrics, predictor = [], None
    for category in args.categories:
        run_signature = signature({'settings': settings, 'category': category, 'checkpoint': settings['prompt_sha256']})
        target = args.output / category / 'metrics.json'
        if target.exists():
            metric = json.loads(target.read_text())
            if metric['run_signature'] != run_signature:
                raise ValueError('Saved AnomalyCLIP evaluation differs; use a new output directory.')
        else:
            if predictor is None:
                predictor = AnomalyCLIPPredictor(args.seed)
            metric = evaluate_category(predictor, category, args.output, settings, run_signature)
        metrics.append(metric)
    summarize(args.output, metrics, settings)
    print('Part 2 AnomalyCLIP finished:', args.output, flush=True)


if __name__ == '__main__':
    main()
