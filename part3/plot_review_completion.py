"""Two-panel, source-backed follow-up figure for the review completion note."""

import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment_utils.common import ROOT


def main():
    summary = json.loads((ROOT / 'results/part3/multiscale_fusion/all_shots_summary.json')
                         .read_text(encoding='utf-8'))
    with (ROOT / 'results/part3/defect_size_audit/pixel_ap_by_size_4shot.csv').open(
            encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    colors = {'official': '#657a90', 'two_layer': '#4f95a7',
              'high_res': '#ef9f56', 'fusion': '#cf5d55'}
    labels = {'official': 'Official 448', 'two_layer': 'Two-layer A',
              'high_res': '672 only', 'fusion': 'Fixed 448+672'}
    shots = (1, 2, 4, 8)
    for method in labels:
        y = [100 * summary['by_shot'][str(s)][method]['image_auroc_mean'] for s in shots]
        sd = [100 * summary['by_shot'][str(s)][method]['image_auroc_sample_sd'] for s in shots]
        axes[0].errorbar(shots, y, yerr=sd, marker='o', linewidth=2.2,
                         capsize=3, label=labels[method], color=colors[method])
    axes[0].set(xlabel='Normal references', ylabel='Macro image AUROC (%)',
                title='Detection across 1/2/4/8 references', xticks=shots,
                ylim=(93, 98.5))
    axes[0].grid(alpha=.22)
    axes[0].legend(loc='lower right', fontsize=8)
    grouped = {}
    for row in rows:
        grouped.setdefault((row['method'], row['size']), []).append(float(row['pixel_ap']))
    bins = ('tiny', 'medium', 'large')
    x = np.arange(len(bins))
    for offset, method in enumerate(('official', 'two_layer', 'multi_res')):
        values = [100 * np.mean(grouped[method, size]) for size in bins]
        axes[1].bar(x + (offset - 1) * .24, values, .23,
                    color={'official': colors['official'], 'two_layer': colors['two_layer'],
                           'multi_res': colors['fusion']}[method],
                    label={'official': 'Official 448', 'two_layer': 'Two-layer A',
                           'multi_res': 'Fixed 448+672'}[method])
    axes[1].set(xlabel='Defect area / image area', ylabel='Pixel AP (%)',
                title='Four-shot localization by defect size',
                xticks=x, xticklabels=('<0.5%\n439 images', '0.5–2%\n114 images',
                                      '>2%\n47 images'), ylim=(0, 90))
    axes[1].legend(loc='upper left', fontsize=8)
    axes[1].grid(axis='y', alpha=.22)
    fig.text(.5, -.01,
             'Six VisA categories, three matched reference groups. 672 and fusion are post-hoc on the same test split. '
             'Pixel AP includes all normal images in each bucket; compare methods within buckets. Large defects occur in four categories.',
             ha='center', va='top', fontsize=8)
    path = ROOT / 'docs/figures/part3_followup_allshots.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(path, flush=True)


if __name__ == '__main__':
    main()
