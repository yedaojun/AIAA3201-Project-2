from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

root = Path(__file__).resolve().parents[1]
fig, ax = plt.subplots(figsize=(15, 6), facecolor='#f8fafc')
ax.set(xlim=(0, 15), ylim=(0, 6))
ax.axis('off')
ax.text(.2, 5.65, 'PART 2  /  OFFICIAL ANOMALYDINO', fontsize=19, weight='bold', color='#142c43')
ax.text(.2, 5.2, 'Frozen DINOv2 ViT-S/14 · short edge 448 · shared few-shot references · official agnostic preprocessing', fontsize=11, color='#526577')

def box(x, y, w, h, title, body, color='#e7eff8'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.10,rounding_size=.12', facecolor=color, edgecolor='#b7c8d8'))
    ax.text(x+.13, y+h-.2, title, fontsize=11, weight='bold', va='top', color='#16334f')
    ax.text(x+.13, y+h-.62, body, fontsize=9.5, va='top', linespacing=1.4, color='#334e68')

def arrow(a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,linewidth=1.6,color='#52748f'))

box(.3,3.1,2.3,1.5,'NORMAL REFERENCES','1 / 2 / 4 / 8 images\nOfficial 8-angle rotation\nSame filenames as Part 1')
box(.3,.9,2.3,1.4,'QUERY IMAGE','Unseen product image\nNo defect label supplied')
box(3.15,2,2.3,2,'FROZEN DINOv2','ViT-S/14 patch features\nNo fine-tuning\nFP32 GPU extraction')
box(6,3.1,2.3,1.5,'NORMAL MEMORY','Normalized patch bank\nOfficial default:\nreference mask disabled')
box(6,.9,2.3,1.4,'QUERY PATCHES','Official PCA foreground\nmask; background = zero')
box(8.9,2,2.3,2,'NEAREST NORMAL','CPU FAISS exact 1-NN\nDistance = 1 - cosine\nOne score per query patch')
box(11.85,3.1,2.7,1.5,'PIXEL LOCALIZATION','Official resize + Gaussian\nOriginal-size float scores\nExact Pixel AUROC', '#dff2ee')
box(11.85,.9,2.7,1.4,'IMAGE DECISION','Official top-1% aggregation\nNormal-only calibration\nScore + OK / NG', '#dff2ee')
arrow((2.7,3.8),(3.05,3.2));arrow((2.7,1.6),(3.05,2.5))
arrow((5.55,3.2),(5.9,3.8));arrow((5.55,2.5),(5.9,1.6))
arrow((8.4,3.8),(8.8,3.2));arrow((8.4,1.6),(8.8,2.5))
arrow((11.3,3.2),(11.75,3.8));arrow((11.3,2.5),(11.75,1.6))
ax.text(.3,.25,'Calibration: 20 independent normal training images per category; excluded from all reference banks. Threshold = 95th percentile.',fontsize=11,color='#7f5a21')
destination=root/'docs/figures';destination.mkdir(exist_ok=True)
fig.tight_layout(pad=.5)
for extension in ('png','svg'):
    fig.savefig(destination/f'part2_pipeline.{extension}',dpi=180,facecolor=fig.get_facecolor(),bbox_inches='tight')
plt.close(fig)
print(destination/'part2_pipeline.png')
