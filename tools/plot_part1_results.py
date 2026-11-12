import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1] / 'results/part1'
with (root / 'summary_macro.csv').open(encoding='utf-8-sig', newline='') as file:
    rows = list(csv.DictReader(file))
shots = [int(row['shot']) for row in rows]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout='constrained')
for ax, metric, label, color in zip(axes, ('image_auroc', 'pixel_auroc'),
                                     ('Image-level detection', 'Pixel-level localization'), ('#2864a0', '#13856d')):
    means = [float(row[metric + '_mean']) for row in rows]
    stds = [float(row[metric + '_std']) for row in rows]
    ax.errorbar(shots, means, yerr=stds, color=color, marker='o', capsize=5, linewidth=2)
    ax.set(xscale='log', xticks=shots, xticklabels=list(map(str, shots)), ylim=(0, 1),
           xlabel='Normal reference images (shots)', ylabel='AUROC', title=label)
    ax.minorticks_off()
    ax.grid(axis='y', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    if metric == 'image_auroc':
        ax.axhline(.5, color='#9aa7b3', linestyle='--', linewidth=1)
fig.suptitle('Part 1: RGB template difference\nSix-category macro average; mean ± sample SD across 3 reference groups', fontsize=12)
destination = root / 'figures'
destination.mkdir(exist_ok=True)
for extension in ('png', 'svg'):
    fig.savefig(destination / f'shot_comparison.{extension}', dpi=180)
plt.close(fig)
report = root / 'report.html'
text = report.read_text(encoding='utf-8')
marker = '<h2>Representative difficult examples</h2>'
figure = '<h2>Few-shot stability</h2><img src="figures/shot_comparison.png" alt="Image and pixel AUROC across shots with sample standard deviations">'
if figure not in text:
    report.write_text(text.replace(marker, figure + marker), encoding='utf-8')
print(destination / 'shot_comparison.png')
