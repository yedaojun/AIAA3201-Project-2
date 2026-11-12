"""Single-image runnable demo, using a previously calibrated configuration."""
import argparse
import json
from pathlib import Path

import cv2

from .core import CATEGORIES, TemplateDetector, write_panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--category', required=True, choices=CATEGORIES)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-root', type=Path, default=Path('data/VisA_pytorch/1cls'))
    parser.add_argument('--results', type=Path, default=Path('results/part1'))
    parser.add_argument('--output', type=Path, default=Path('results/demo_part1'))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    settings = json.loads((args.results / 'run_config.json').read_text())['config']
    metrics = json.loads((args.results / args.category / f'shot_{args.shot}' / f'seed_{args.seed}' / 'metrics.json').read_text())
    detector = TemplateDetector([args.data_root / args.category / 'train/good' / name for name in metrics['references']], settings)
    score, anomaly, template, elapsed = detector.predict(args.image)
    result = {'image': str(args.image.resolve()), 'category': args.category,
              'shot': args.shot, 'seed': args.seed, 'score': score,
              'threshold': metrics['threshold'], 'decision': 'NG' if score > metrics['threshold'] else 'OK',
              'closest_template': template, 'inference_ms': elapsed * 1000}
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output / 'heatmap_raw.png'), anomaly)
    write_panel(args.image, None, anomaly,
                f'{args.category} | {result["decision"]} | score={score:.4f} threshold={metrics["threshold"]:.4f}',
                args.output / 'prediction.png')
    (args.output / 'prediction.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
