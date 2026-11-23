"""Build report figures from saved experiment records; no experiment tuning."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from PIL import Image

from anomaly_dino.official import utils
from experiment_utils.common import DATA, ROOT


OUT = Path(__file__).parent / 'figures'
OUT.mkdir(parents=True, exist_ok=True)


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.25))
    ax.set(xlim=(0, 12), ylim=(-.15, 3.65))
    ax.axis('off')
    palette = {'input': '#edf3f8', 'template': '#f8e8cf', 'dino': '#dcefe8',
               'improved': '#d8e4f5', 'output': '#eee8f4'}

    def box(x, y, width, height, label, color):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                                   boxstyle='round,pad=0.04,rounding_size=0.11',
                                   facecolor=palette[color], edgecolor='#56616c', linewidth=1))
        ax.text(x + width / 2, y + height / 2, label, ha='center', va='center',
                fontsize=8.2, linespacing=1.15)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                    mutation_scale=10, linewidth=1.15, color='#59636d'))

    box(.05, 1.2, 1.55, .88, 'Same VisA query\nand normal refs', 'input')
    box(2.05, 2.22, 1.55, .68, 'RGB template\n448 short edge', 'template')
    box(4.02, 2.22, 1.73, .68, 'Closest template\nRGB difference', 'template')
    box(6.18, 2.22, 1.55, .68, 'Smoothed\ndifference map', 'template')
    box(2.05, .42, 1.55, .9, 'Frozen DINOv2\nViT-S/14', 'dino')
    box(4.02, .42, 1.73, .9, 'P2: block 11 (384-d)\nA: blocks 8+11 (768-d)', 'improved')
    box(6.18, .42, 1.55, .9, 'Query PCA mask\n+ normal patch bank', 'improved')
    box(8.13, .42, 1.48, .9, 'Nearest-neighbor\npatch heatmap', 'improved')
    box(10.06, 1.1, 1.86, 1.03, 'Top 1% image score\nnormal calibration\nOK / NG', 'output')
    for source, target in [((1.61, 1.88), (2.04, 2.54)), ((3.61, 2.56), (4.01, 2.56)),
                           ((5.76, 2.56), (6.17, 2.56)), ((7.75, 2.56), (10.05, 1.91)),
                           ((1.61, 1.39), (2.04, .87)), ((3.61, .87), (4.01, .87)),
                           ((5.76, .87), (6.17, .87)), ((7.75, .87), (8.12, .87)),
                           ((9.62, .87), (10.05, 1.31))]:
        arrow(*source, *target)
    box(8.13, 3.02, 1.48, .45, '20 normal\ncalibration images', 'input')
    ax.add_patch(FancyArrowPatch((9.64, 3.20), (10.55, 2.15), arrowstyle='-|>',
                                 mutation_scale=10, linewidth=1.1, linestyle='--', color='#59636d'))
    ax.text(.09, 2.92, 'Part 1 · pixel baseline', fontsize=9.5, fontweight='bold', color='#8a5a17')
    ax.text(.09, .13, 'Part 2 and Part 3 A · frozen feature retrieval', fontsize=9.5,
            fontweight='bold', color='#246e66')
    fig.savefig(OUT / 'pipeline.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'pipeline.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def shot_curves():
    part1 = {int(row['shot']): row for row in rows(ROOT / 'results/part1/summary_macro.csv')}
    part2 = {int(row['shot']): row for row in rows(ROOT / 'results/part2/summary_macro.csv')}
    a = {int(row['shot']): row for row in rows(ROOT / 'results/part3/multilayer/shot_summary.csv')}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), layout='constrained')
    shots = (1, 2, 4, 8)
    for ax, metric, ylim in zip(axes, ('image', 'pixel'), ((45, 101), (73, 101))):
        for name, source, mean_key, sd_key, color in (
            ('RGB template', part1, f'{metric}_auroc_mean', f'{metric}_auroc_std', '#a66b2e'),
            ('Official DINO', part2, f'{metric}_auroc_mean', f'{metric}_auroc_std', '#58616b'),
            ('Two-layer DINO (A)', a, f'{metric}_auroc_mean', f'{metric}_auroc_sample_sd', '#187c72'),
        ):
            ax.errorbar(shots, [100 * float(source[s][mean_key]) for s in shots],
                        yerr=[100 * float(source[s][sd_key]) for s in shots],
                        marker='o', lw=1.8, capsize=2.5, color=color, label=name)
        ax.set(xlabel='Normal references', ylabel=f'{metric.title()} AUROC (%)',
               xticks=shots, ylim=ylim)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8, loc='center left')
    zoom = axes[0].inset_axes([.55, .59, .38, .29])
    for source, key, color in ((part2, 'image_auroc_mean', '#58616b'),
                               (a, 'image_auroc_mean', '#187c72')):
        zoom.plot(shots, [100 * float(source[s][key]) for s in shots], marker='o',
                  lw=1.35, ms=2.7, color=color)
    zoom.set(xlim=(.7, 8.3), ylim=(93.8, 97.4), xticks=shots, yticks=(94, 96, 97))
    zoom.tick_params(labelsize=6)
    zoom.grid(alpha=.2)
    # Place both insets clear of the main y-axis labels.
    pixel_zoom = axes[1].inset_axes([.32, .28, .48, .38])
    for source, key, color in ((part2, 'pixel_auroc_mean', '#58616b'),
                               (a, 'pixel_auroc_mean', '#187c72')):
        pixel_zoom.plot(shots, [100 * float(source[s][key]) for s in shots], marker='o',
                        lw=1.35, ms=2.7, color=color)
    pixel_zoom.set(xlim=(.7, 8.3), ylim=(97.5, 98.5), xticks=shots, yticks=(97.5, 98, 98.5))
    pixel_zoom.tick_params(labelsize=6)
    pixel_zoom.grid(alpha=.2)
    fig.savefig(OUT / 'shot_curves.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'shot_curves.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


CASES = (
    ('candle', 'bad/000.JPG'),
    ('capsules', 'bad/087.JPG'),
    ('cashew', 'bad/063.JPG'),
    ('chewinggum', 'good/342.JPG'),
    ('fryum', 'bad/031.JPG'),
    ('capsules', 'bad/033.JPG'),
)


def crop_box(shape, center, side):
    height, width = shape
    side = min(side, height, width)
    y0 = max(0, min(height - side, int(center[0] - side / 2)))
    x0 = max(0, min(width - side, int(center[1] - side / 2)))
    return y0, x0, y0 + side, x0 + side


def case_figures():
    all_cases = CASES + (('candle', 'good/0333.JPG'),)
    for page in range(2):
        page_cases = all_cases[:3] if page == 0 else all_cases[3:]
        fig, axes = plt.subplots(len(page_cases), 4, figsize=(10, 1.75 * len(page_cases)), layout='constrained')
        for row_index, (category, suffix) in enumerate(page_cases):
            path = DATA / category / 'test' / suffix
            image = np.asarray(Image.open(path).convert('RGB'))
            shape = image.shape[:2]
            mask = (np.asarray(Image.open(DATA / category / 'ground_truth/bad' /
                                          (path.stem + '.png'))) != 0 if suffix.startswith('bad/')
                    else np.zeros(shape, dtype=bool))
            official = next(row for row in rows(ROOT / 'results/part2' / category /
                                                'shot_4/seed_0/predictions.csv')
                            if row['file'] == 'test/' + suffix)
            improved = next(row for row in rows(ROOT / 'results/part3/multilayer' / category /
                                                'shot_4/seed_0/predictions.csv')
                            if row['sample_id'].endswith('/test/' + suffix))
            old_map = utils.dists2map(np.load(ROOT / 'results/part2' / category /
                                               'shot_4/seed_0' / official['anomaly_map']), image.shape)
            new_map = utils.dists2map(np.load(ROOT / improved['anomaly_map']), image.shape)
            if mask.any():
                center = np.argwhere(mask).mean(axis=0)
            else:
                center = np.unravel_index(np.argmax(old_map), shape)
            y0, x0, y1, x1 = crop_box(shape, center, int(min(shape) * .32))
            vmax = max(float(old_map.max()), float(new_map.max()))
            ax = axes[row_index]
            ax[0].imshow(image)
            ax[0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                      fill=False, edgecolor='#e6323f', linewidth=1.8))
            ax[0].set_ylabel(f'{category}\n{suffix}\n'
                             f'{official["decision"]} → {improved["decision"]}',
                             fontsize=8, rotation=0, labelpad=34, va='center')
            ax[1].imshow(mask[y0:y1, x0:x1], cmap='gray', vmin=0, vmax=1)
            ax[2].imshow(old_map[y0:y1, x0:x1], cmap='inferno', vmin=0, vmax=vmax)
            ax[3].imshow(new_map[y0:y1, x0:x1], cmap='inferno', vmin=0, vmax=vmax)
            for cell in ax:
                cell.set_xticks([])
                cell.set_yticks([])
            ax[2].set_xlabel(f's={float(official["score"]):.3f}  τ={float(official["threshold"]):.3f}', fontsize=8)
            ax[3].set_xlabel(f's={float(improved["score"]):.3f}  τ={float(improved["threshold"]):.3f}', fontsize=8)
        for ax, title in zip(axes[0], ('Full input + crop', 'GT crop', 'Official crop', 'A crop')):
            ax.set_title(title, fontsize=10)
        fig.savefig(OUT / f'difficult_cases_{page + 1}.png', dpi=240, bbox_inches='tight')
        plt.close(fig)


def template_failure():
    category = 'capsules'
    suffix = 'good/016.JPG'
    baseline = next(row for row in rows(ROOT / 'results/part1/capsules/shot_4/seed_0/predictions.csv')
                    if row['file'] == 'test/' + suffix)
    official = next(row for row in rows(ROOT / 'results/part2/capsules/shot_4/seed_0/predictions.csv')
                    if row['file'] == 'test/' + suffix)
    image = np.asarray(Image.open(DATA / category / 'test' / suffix).convert('RGB'))
    reference = np.asarray(Image.open(DATA / category / 'train/good' /
                                      baseline['closest_template']).convert('RGB'))
    template_map = np.asarray(Image.open(ROOT / 'results/part1/capsules/shot_4/seed_0' /
                                          baseline['anomaly_map']).convert('L'))
    dino_map = utils.dists2map(np.load(ROOT / 'results/part2/capsules/shot_4/seed_0' /
                                       official['anomaly_map']), image.shape)
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.2), layout='constrained')
    for ax, data, title, cmap in zip(axes, (image, reference, template_map, dino_map),
                                     ('Normal query', 'Closest RGB template', 'RGB difference',
                                      'DINO distance'), (None, None, 'inferno', 'inferno')):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[2].set_xlabel(f's={float(baseline["score"]):.4f} > τ={float(baseline["threshold"]):.4f} (FP)', fontsize=8)
    axes[3].set_xlabel(f's={float(official["score"]):.4f} < τ={float(official["threshold"]):.4f} (TN)', fontsize=8)
    fig.savefig(OUT / 'template_false_positive.png', dpi=230, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    pipeline()
    shot_curves()
    case_figures()
    template_failure()
    print('Created report figures in', OUT)
