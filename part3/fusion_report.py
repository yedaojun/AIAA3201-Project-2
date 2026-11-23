"""Create a readable four-shot pilot figure from frozen D/F evaluation tables."""

from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .fusion_selective import OUTPUT, rows


def main():
    table = rows(OUTPUT / 'risk_coverage.csv')
    curves = defaultdict(list)
    for row in table:
        if int(row['shots']) != 4:
            continue
        curves[(row['policy'], float(row['actual_coverage']))].append(float(row['retained_error_rate']))
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    for policy, label, color in (
        ('dino_margin', 'Original DINO and margin', '#555d65'),
        ('margin', 'Calibrated fusion and margin', '#137c8b'),
        ('disagreement_then_margin', 'Fusion and disagreement first', '#d07828'),
        ('gated_margin', 'Gated fusion and margin', '#79539a'),
    ):
        points = sorted((coverage, np.mean(values)) for (name, coverage), values in curves.items()
                        if name == policy)
        ax.plot([100 * x for x, _ in points], [100 * y for _, y in points],
                marker='o', lw=2.3, label=label, color=color)
    ax.set(xlabel='Automatically decided test images (%)',
           ylabel='Error rate among automatically decided images (%)',
           title='Four-shot selective decisions, six VisA categories and three reference groups')
    ax.set_xlim(68, 102)
    ax.grid(alpha=.25)
    ax.legend(loc='upper left')
    fig.savefig(OUTPUT / 'risk_coverage_4shot.png', dpi=180)
    plt.close(fig)
    print(OUTPUT / 'risk_coverage_4shot.png')

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for shot, panel in zip((1, 2, 4, 8), axes.flat):
        shot_curves = defaultdict(list)
        for row in table:
            if int(row['shots']) == shot:
                shot_curves[(row['policy'], float(row['actual_coverage']))].append(
                    float(row['retained_error_rate']))
        for policy, label, color in (
            ('dino_margin', 'Part 2 DINO', '#555d65'),
            ('margin', 'D: equal fusion', '#137c8b'),
            ('disagreement_then_margin', 'D: fusion and disagreement', '#d07828'),
            ('gated_margin', 'D: exploratory gate', '#79539a'),
        ):
            points = sorted((coverage, np.mean(values))
                            for (name, coverage), values in shot_curves.items() if name == policy)
            panel.plot([100 * x for x, _ in points], [100 * y for _, y in points],
                       marker='o', lw=2, label=label, color=color)
        panel.set_title(f'{shot} normal reference image' + ('s' if shot > 1 else ''))
        panel.set_xlim(68, 102)
        panel.grid(alpha=.25)
    fig.supxlabel('Automatically decided test images (%)', y=.105)
    fig.supylabel('Error rate among automatically decided images (%)')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center',
               bbox_to_anchor=(.5, .015), ncol=2)
    fig.tight_layout(rect=(.035, .14, 1, 1))
    fig.savefig(OUTPUT / 'risk_coverage_all_shots.png', dpi=180)
    plt.close(fig)
    print(OUTPUT / 'risk_coverage_all_shots.png')


if __name__ == '__main__':
    main()
