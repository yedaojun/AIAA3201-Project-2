"""Summarize the prespecified six-category B/C/E four-shot pilot."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment_utils.common import ROOT
from .background_memory_pilot import OUTPUT as BE_OUTPUT
from .multilayer import save_csv
from .synthetic_cutpaste_pilot import OUTPUT as C_OUTPUT


OUTPUT = ROOT / 'results/part3/bce_comparison'
CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')


def main():
    rows = []
    for category in CATEGORIES:
        be = json.loads((BE_OUTPUT / category / 'shot_4/seed_0/metrics.json').read_text(encoding='utf-8'))
        c = json.loads((C_OUTPUT / category / 'shot_4/seed_0/metrics.json').read_text(encoding='utf-8'))
        if be['test_images'] != c['test_images']:
            raise ValueError(f'Test count mismatch: {category}')
        base = be['methods']['part2_fresh']['image_auroc']
        if abs(base - c['part2_image_auroc']) > 1e-10:
            raise ValueError(f'Baseline mismatch: {category}')
        rows.append({'category': category, 'test_images': be['test_images'],
                     'part2_image_auroc': base,
                     'b_image_auroc': be['methods']['b_foreground_bank']['image_auroc'],
                     'c_image_auroc': c['image_auroc'],
                     'e_image_auroc': be['methods']['e_stratified_25pct']['image_auroc'],
                     'b_bank_fraction': be['bank']['b_fraction'],
                     'e_bank_fraction': be['bank']['e_fraction']})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_csv(OUTPUT / 'category_4shot_seed0.csv', rows)
    macro = {'shot': 4, 'seed': 0, 'category_count': len(rows),
             'unique_test_images': sum(row['test_images'] for row in rows),
             'scope': 'prespecified pilot; no three-group or full-shot claim'}
    for key in ('part2_image_auroc', 'b_image_auroc', 'c_image_auroc', 'e_image_auroc',
                'b_bank_fraction', 'e_bank_fraction'):
        macro[key] = float(np.mean([row[key] for row in rows]))
    (OUTPUT / 'summary.json').write_text(json.dumps(macro, indent=2), encoding='utf-8')
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
    for offset, key, label, color in (
        (-1.5, 'part2_image_auroc', 'Part 2 DINO', '#555d65'),
        (-.5, 'b_image_auroc', 'B: foreground bank', '#d07828'),
        (.5, 'e_image_auroc', 'E: 25% coreset', '#137c8b'),
        (1.5, 'c_image_auroc', 'C: CutPaste head', '#79539a'),
    ):
        ax.bar(x + offset * .19, [100 * row[key] for row in rows], width=.19,
               label=label, color=color)
    ax.set(xlabel='VisA category', ylabel='Image AUROC (%)',
           title='B/C/E pilot versus unchanged Part 2: 4 normal references, group 0',
           xticks=x, xticklabels=CATEGORIES)
    ax.set_ylim(65, 102)
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=.25)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', ncol=4)
    fig.savefig(OUTPUT / 'category_4shot_seed0.png', dpi=180)
    plt.close(fig)
    print(json.dumps(macro, indent=2))


if __name__ == '__main__':
    main()
