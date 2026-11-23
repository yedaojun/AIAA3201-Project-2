"""Fixed difficult-case visuals for official and two-layer DINO at 4-shot group 0."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from anomaly_dino.official import utils
from experiment_utils.common import DATA, ROOT
from .multilayer import OUTPUT

CASES = (
    ('candle', 'bad/000.JPG'),
    ('capsules', 'bad/087.JPG'),
    ('cashew', 'bad/063.JPG'),
    ('chewinggum', 'good/342.JPG'),
    ('fryum', 'bad/031.JPG'),
    ('pipe_fryum', 'bad/000.JPG'),
)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    fig, axes = plt.subplots(len(CASES), 4, figsize=(13, 3.1 * len(CASES)), layout='constrained')
    records = []
    for index, (category, suffix) in enumerate(CASES):
        folder = OUTPUT / category / 'shot_4/seed_0'
        p2 = ROOT / 'results/part2' / category / 'shot_4/seed_0'
        path = DATA / category / 'test' / suffix
        image = np.asarray(Image.open(path).convert('RGB'))
        shape = image.shape[:2]
        if suffix.startswith('bad/'):
            mask = np.asarray(Image.open(DATA / category / 'ground_truth/bad' / (path.stem + '.png'))) != 0
        else:
            mask = np.zeros(shape, dtype=bool)
        original = next(r for r in read_csv(p2 / 'predictions.csv') if r['file'] == 'test/' + suffix)
        improved = next(r for r in read_csv(folder / 'predictions.csv')
                        if r['sample_id'].endswith('/test/' + suffix))
        original_map = utils.dists2map(np.load(p2 / original['anomaly_map']), image.shape)
        improved_map = utils.dists2map(np.load(ROOT / improved['anomaly_map']), image.shape)
        vmax = float(max(original_map.max(), improved_map.max()))
        for ax in axes[index]:
            ax.set_axis_off()
        axes[index, 0].imshow(image)
        axes[index, 0].set_title(f'{category} · {suffix}')
        axes[index, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[index, 1].set_title('Ground truth')
        axes[index, 2].imshow(original_map, cmap='inferno', vmin=0, vmax=vmax)
        axes[index, 2].set_title(f'Official {original["decision"]} · {float(original["score"]):.3f}')
        axes[index, 3].imshow(improved_map, cmap='inferno', vmin=0, vmax=vmax)
        axes[index, 3].set_title(f'Two layers {improved["decision"]} · {float(improved["score"]):.3f}')
        records.append({'category': category, 'file': suffix,
                        'official_score': float(original['score']),
                        'official_threshold': float(original['threshold']),
                        'official_decision': original['decision'],
                        'two_layer_score': float(improved['score']),
                        'two_layer_threshold': float(improved['threshold']),
                        'two_layer_decision': improved['decision'],
                        'official_peak_in_mask': bool(mask.ravel()[original_map.argmax()]) if mask.any() else None,
                        'two_layer_peak_in_mask': bool(mask.ravel()[improved_map.argmax()]) if mask.any() else None})
    fig.suptitle('Fixed difficult cases · official versus two-layer DINO', fontsize=16)
    fig.savefig(OUTPUT / 'difficult_cases_4shot_seed0.png', dpi=150)
    plt.close(fig)
    from experiment_utils.common import write_json
    write_json(OUTPUT / 'difficult_cases_4shot_seed0.json', records)
    print(OUTPUT / 'difficult_cases_4shot_seed0.png')


if __name__ == '__main__':
    main()
