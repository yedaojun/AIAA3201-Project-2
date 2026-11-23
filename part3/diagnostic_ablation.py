"""Post-hoc layer/weight diagnostics; never selects the preregistered A configuration."""

import argparse
import json

import faiss
import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from anomaly_dino.official import load_model
from anomaly_dino.predict import OfficialPredictor
from experiment_utils.common import DATA, ROOT, configure, load_manifest, write_json
from .multilayer import TwoLayerDINO, read_rows, save_csv


SETTINGS = ((5, 11, .5), (10, 11, .5), (8, 11, .25), (8, 11, .75), (8, 11, 1.0))


def main(indices):
    configure()
    faiss.omp_set_num_threads(4)
    threadpool_limits(4)
    manifest = load_manifest()
    original = load_model()
    output = ROOT / 'results/part3/diagnostic_ablation'
    records = []
    for index in indices:
        early, final, alpha = SETTINGS[index]
        name = f'block{early}+{final}_alpha{alpha:.2f}'
        model = TwoLayerDINO(original, layers=(early, final), alpha=alpha)
        for category, selection in manifest['categories'].items():
            target = output / name / f'{category}.csv'
            if target.exists():
                category_rows = read_rows(target)
            else:
                base = DATA / category
                references = [base / 'train/good' / n for n in selection['references']['4']['0']]
                detector = OfficialPredictor(model, references)
                category_rows = []
                for kind in ('good', 'bad'):
                    for path in sorted((base / 'test' / kind).glob('*.JPG')):
                        score, _, _, _ = detector.predict(path)
                        category_rows.append({'method': name, 'category': category,
                                              'image': 'test/' + kind + '/' + path.name,
                                              'label': int(kind == 'bad'), 'score': score})
                save_csv(target, category_rows)
            auc = 100 * roc_auc_score([int(r['label']) for r in category_rows],
                                      [float(r['score']) for r in category_rows])
            records.append({'method': name, 'category': category, 'image_auroc': auc})
            print(name, category, f'{auc:.2f}', flush=True)
        all_records = []
        for result_file in output.glob('block*/*.csv'):
            saved = read_rows(result_file)
            all_records.append({'method': result_file.parent.name, 'category': result_file.stem,
                                'image_auroc': 100 * roc_auc_score(
                                    [int(r['label']) for r in saved], [float(r['score']) for r in saved])})
        save_csv(output / 'category_results.csv', all_records)
        summary = {method: float(np.mean([r['image_auroc'] for r in all_records if r['method'] == method]))
                   for method in dict.fromkeys(r['method'] for r in all_records)}
        write_json(output / 'summary.json', {'protocol': 'post-hoc, 4-shot, seed 0 only',
                                              'matched_reference_manifest': str(ROOT / 'configs/reference_manifest.json'),
                                              'frozen_backbone': 'dinov2_vits14',
                                              'same_query_PCA_mask': True,
                                              'alpha': 'weight on intermediate unit-normalized descriptor',
                                              'macro_image_auroc_percent': summary,
                                              'warning': 'Diagnostic only; no selection of main A from these results'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices', nargs='+', type=int, choices=range(len(SETTINGS)),
                        default=list(range(len(SETTINGS))))
    main(parser.parse_args().indices)
