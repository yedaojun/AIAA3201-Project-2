"""Generate a report-ready diagram of the implemented baseline."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

destination = Path(__file__).resolve().parents[1] / 'docs/figures'
destination.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(15, 6), facecolor='#f8fafc')
ax.set(xlim=(0, 15), ylim=(0, 6))
ax.axis('off')
ax.text(.25, 5.65, 'PART 1  /  NORMAL-ONLY TEMPLATE INSPECTION', fontsize=18, weight='bold', color='#142c43')
ax.text(.25, 5.24, 'Fixed method settings; no defect labels used for reference selection or threshold calibration.', fontsize=11, color='#526577')


def box(x, y, w, h, title, body, color='#e7eff8'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.10,rounding_size=.12',
                               facecolor=color, edgecolor='#b7c8d8', linewidth=1.2))
    ax.text(x + .16, y + h - .28, title, fontsize=11, weight='bold', color='#16334f', va='top')
    ax.text(x + .16, y + h - .68, body, fontsize=9.5, color='#334e68', va='top', linespacing=1.5)


def arrow(start, end, color='#52748f'):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15,
                                linewidth=1.7, color=color))


box(.3, 3.25, 2.5, 1.35, 'NORMAL REFERENCES', '1 / 2 / 4 / 8 images\nShared filename manifest')
box(.3, 1.0, 2.5, 1.35, 'QUERY PRODUCT', 'One unseen image\nNo label provided to detector')
box(3.3, 2.1, 2.35, 1.65, 'GEOMETRY', 'Short edge = 448\nBicubic resize\nCrop to multiple of 14')
box(6.15, 2.1, 2.35, 1.65, 'TEMPLATE MATCH', 'Minimum whole-image\nmean absolute RGB error\nChoose one full template')
box(9.0, 2.1, 2.35, 1.65, 'ANOMALY MAP', 'Channel-mean RGB difference\nRound to uint8\nGaussian smoothing: sigma 2')
box(12.0, 3.25, 2.5, 1.35, 'LOCALIZATION', 'Restore original image size\nFixed-scale heatmap / overlay', '#dff2ee')
box(12.0, 1.0, 2.5, 1.35, 'IMAGE DECISION', 'Mean of top 1% scores\nCompare with threshold: OK/NG', '#dff2ee')
box(6.15, .15, 5.2, 1.2, 'INDEPENDENT NORMAL CALIBRATION', 'First 20 sorted normal training images; excluded from references.\nThreshold = 95th percentile of calibration image scores.', '#fff0d9')
arrow((2.9, 3.9), (3.2, 3.2))
arrow((2.9, 1.7), (3.2, 2.6))
arrow((5.75, 2.9), (6.05, 2.9))
arrow((8.6, 2.9), (8.9, 2.9))
arrow((11.45, 3.2), (11.9, 3.9))
arrow((11.45, 2.6), (11.9, 1.7))
arrow((11.45, .78), (13.15, .9), '#ae7b36')
fig.tight_layout(pad=.5)
for extension in ('png', 'svg'):
    fig.savefig(destination / f'part1_pipeline.{extension}', dpi=180, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close(fig)
print(destination / 'part1_pipeline.png')
