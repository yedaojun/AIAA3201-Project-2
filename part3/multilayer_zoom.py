"""Ground-truth-centered crops for display only, never for scoring."""

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from anomaly_dino.official import utils
from experiment_utils.common import DATA, ROOT
from .multilayer import OUTPUT
from .multilayer_cases import read_csv
from .composed import map_path

CASES = (('candle', '000.JPG'), ('capsules', '087.JPG'),
         ('cashew', '063.JPG'), ('fryum', '031.JPG'))


def main(composed=False):
    columns = 5 if composed else 4
    fig, axes = plt.subplots(len(CASES), columns, figsize=(16 if composed else 13, 3.1 * len(CASES)),
                             layout='constrained')
    for i, (category, filename) in enumerate(CASES):
        image_path = DATA / category / 'test/bad' / filename
        image = np.asarray(Image.open(image_path).convert('RGB'))
        mask = np.asarray(Image.open(DATA / category / 'ground_truth/bad' / (image_path.stem + '.png'))) != 0
        y, x = np.where(mask)
        cy, cx = int(np.median(y)), int(np.median(x))
        radius = min(220, max(70, 3 * max(int(np.ptp(x)) + 1, int(np.ptp(y)) + 1)))
        y0, y1 = max(0, cy - radius), min(image.shape[0], cy + radius)
        x0, x1 = max(0, cx - radius), min(image.shape[1], cx + radius)
        official_folder = ROOT / 'results/part2' / category / 'shot_4/seed_0'
        improved_folder = OUTPUT / category / 'shot_4/seed_0'
        official_row = next(r for r in read_csv(official_folder / 'predictions.csv')
                            if r['file'] == 'test/bad/' + filename)
        improved_row = next(r for r in read_csv(improved_folder / 'predictions.csv')
                            if r['sample_id'].endswith('/test/bad/' + filename))
        original = utils.dists2map(np.load(official_folder / official_row['anomaly_map']), image.shape)
        improved = utils.dists2map(np.load(ROOT / improved_row['anomaly_map']), image.shape)
        learned = (utils.dists2map(np.load(map_path(category, 4, 0, 'bad', filename)), image.shape)
                   if composed else None)
        scale = max(float(original.max()), float(improved.max()),
                    float(learned.max()) if composed else 0)
        views = [image[y0:y1, x0:x1], mask[y0:y1, x0:x1],
                 original[y0:y1, x0:x1], improved[y0:y1, x0:x1]]
        titles = [f'{category} · input zoom', 'Mask', 'Official heatmap', 'Two-layer heatmap']
        if composed:
            views.append(learned[y0:y1, x0:x1])
            titles.append('Two-layer + AnyUp')
        for j, (view, title) in enumerate(zip(views, titles)):
            ax = axes[i, j]
            ax.set_axis_off()
            ax.set_title(title)
            if j == 0:
                ax.imshow(view)
            elif j == 1:
                ax.imshow(view, cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(view, cmap='inferno', vmin=0, vmax=scale)
    fig.suptitle('Fixed 4-shot/group-0 small-defect cases · zoom used for display only', fontsize=15)
    fig.savefig(OUTPUT / ('difficult_cases_zoom_composed.png' if composed else 'difficult_cases_zoom.png'), dpi=170)
    plt.close(fig)
    print(OUTPUT / ('difficult_cases_zoom_composed.png' if composed else 'difficult_cases_zoom.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--composed', action='store_true')
    main(parser.parse_args().composed)
