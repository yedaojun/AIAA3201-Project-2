"""Compact comparison figure for the exploratory four-shot Part 3 search."""

import json

import matplotlib.pyplot as plt
import numpy as np

from experiment_utils.common import ROOT


def main():
    paired = json.loads((ROOT / 'results/part3/multiscale_fusion/three_seed_summary.json').read_text())
    selective = json.loads((ROOT / 'results/part3/selective_multiscale/summary.json').read_text())
    pixel = json.loads((ROOT / 'results/part3/multiscale_fusion/pixel_metrics_4shot_3seeds.json').read_text())
    selective_pixel = json.loads((ROOT / 'results/part3/selective_multiscale/pixel_metrics_4shot_3seeds.json').read_text())
    anyup = json.loads((ROOT / 'results/part3/anyup_pixel_ap_4shot_3seeds.json').read_text())
    labels = ['Official DINO', 'Two-layer A', 'Full 448+672', 'Selective 672', 'DINO+AnyUp']
    colors = ['#6b7280', '#4f8bc9', '#2469b3', '#63a3ce', '#e89a42']
    image_means = [paired['macro_mean_sd'][key]['mean'] * 100
                   for key in ('official', 'two_layer', 'fusion')]
    image_sds = [paired['macro_mean_sd'][key]['sample_sd'] * 100
                 for key in ('official', 'two_layer', 'fusion')]
    image_means += [selective['macro_image_auroc_mean'] * 100,
                    paired['macro_mean_sd']['official']['mean'] * 100]
    image_sds += [selective['macro_image_auroc_sd'] * 100,
                  paired['macro_mean_sd']['official']['sample_sd'] * 100]
    pixel_means = [anyup['macro_mean_sd']['official']['mean'] * 100,
                   None,
                   pixel['macro_mean_sd']['fusion_pixel_ap']['mean'] * 100,
                   selective_pixel['macro_mean_sd']['pixel_ap']['mean'] * 100,
                   anyup['macro_mean_sd']['official_anyup']['mean'] * 100]
    pixel_sds = [anyup['macro_mean_sd']['official']['sample_sd'] * 100,
                 None,
                 pixel['macro_mean_sd']['fusion_pixel_ap']['sample_sd'] * 100,
                 selective_pixel['macro_mean_sd']['pixel_ap']['sample_sd'] * 100,
                 anyup['macro_mean_sd']['official_anyup']['sample_sd'] * 100]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, title, means, sds, limits in (
            (axes[0], 'Image AUROC', image_means, image_sds, (93, 98)),
            (axes[1], 'Pixel average precision', pixel_means, pixel_sds, (43, 61))):
        y = np.arange(len(labels))
        for index, (mean, sd) in enumerate(zip(means, sds)):
            if mean is None:
                continue
            ax.barh(index, mean - limits[0], left=limits[0], color=colors[index], height=.58)
            ax.errorbar(mean, index, xerr=sd, color='#15202b', capsize=3, fmt='none', lw=1.1)
            ax.text(mean - (.9 if ax is axes[0] else 1.3), index, f'{mean:.2f}',
                    va='center', ha='right', fontsize=9, color='white', weight='bold')
        ax.set_xlim(*limits)
        ax.set_ylim(-.7, len(labels) - .3)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, weight='bold')
        ax.set_xlabel('Percent')
        ax.set_yticks(y, labels if ax is axes[0] else [''] * len(labels))
        ax.grid(axis='x', alpha=.18)
        ax.set_axisbelow(True)
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)
    axes[1].text(45, 1, 'A: three-seed AP unavailable', va='center', fontsize=8,
                 color='#687588')
    fig.subplots_adjust(left=.19, right=.97, bottom=.20, top=.79, wspace=.16)
    fig.suptitle('Part 3 exploratory comparison: 4 shots, 3 reference groups, 6 categories',
                 y=.95, fontsize=13, weight='bold')
    fig.text(.19, .035, 'Bars: macro mean; whiskers: SD across three reference groups. '
             'Multi-scale variants are post-hoc pilots on the same test images.',
             fontsize=8, color='#536273')
    output = ROOT / 'docs/figures/part3_experiment_search.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor='white')
    print(output, flush=True)


if __name__ == '__main__':
    main()
