"""Post-hoc defect-area audit using saved decisions and official VisA masks."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from experiment_utils.common import DATA, ROOT

OUT = ROOT / 'results/part3/defect_size_audit'
CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')


def rows(path):
    with open(path, encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def bucket(fraction):
    return 'tiny_<0.5pct' if fraction < .005 else ('medium_0.5-2pct' if fraction <= .02 else 'large_>2pct')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    area = {}
    for category in CATEGORIES:
        for image in sorted((DATA / category / 'test/bad').glob('*.JPG')):
            mask = DATA / category / 'ground_truth/bad' / (image.stem + '.png')
            with Image.open(mask) as im:
                array = np.asarray(im)
            area[category, image.name] = float(np.count_nonzero(array) / array.size)
    entries = []
    methods = ('official', 'two_layer', 'high_res_672', 'multi_res_448_672')
    for method in methods:
        for seed in (0, 1, 2):
            for category in CATEGORIES:
                base = {'official': ROOT / 'results/part2',
                        'two_layer': ROOT / 'results/part3/multilayer',
                        'high_res_672': ROOT / 'results/part3/high_resolution_pilot',
                        'multi_res_448_672': ROOT / 'results/part3/multiscale_fusion'}[method]
                folder = base / category / 'shot_4' / f'seed_{seed}'
                if not (folder / 'predictions.csv').exists():
                    continue
                records = rows(folder / 'predictions.csv')
                normal = [r for r in records if int(r['label']) == 0]
                score_key = 'fused_score' if method == 'multi_res_448_672' else 'score'
                for bin_name in ('tiny_<0.5pct', 'medium_0.5-2pct', 'large_>2pct'):
                    anomalous = []
                    for r in records:
                        if not int(r['label']):
                            continue
                        file = (r['sample_id'] if method == 'two_layer' else r['file']).split('/')[-1]
                        if bucket(area[category, file]) == bin_name:
                            anomalous.append(r)
                    if not anomalous:
                        continue
                    auc = roc_auc_score([0] * len(normal) + [1] * len(anomalous),
                                        [float(r[score_key]) for r in normal + anomalous])
                    recall = sum(r['decision'] == 'NG' for r in anomalous) / len(anomalous)
                    entries.append({'method': method, 'seed': seed, 'category': category,
                                    'size_bin': bin_name, 'normal_count': len(normal),
                                    'anomaly_count': len(anomalous), 'image_auroc': auc,
                                    'recall_normal_calibrated': recall})
    with open(OUT / 'category_seed.csv', 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)
    summaries = []
    for method in methods:
        for bin_name in ('tiny_<0.5pct', 'medium_0.5-2pct', 'large_>2pct'):
            selected = [r for r in entries if r['method'] == method and r['size_bin'] == bin_name]
            summaries.append({'method': method, 'size_bin': bin_name,
                              'categories': len(set(r['category'] for r in selected)),
                              'distinct_anomalies': sum(r['anomaly_count'] for r in selected if r['seed'] == 0),
                              'macro_image_auroc': float(np.mean([r['image_auroc'] for r in selected])),
                              'macro_recall': float(np.mean([r['recall_normal_calibrated'] for r in selected]))})
    with open(OUT / 'summary.csv', 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for row in summaries:
        print(row, flush=True)


if __name__ == '__main__':
    main()
