"""Validate optional-method provenance, split isolation and saved numerical results."""
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

from anomaly_baseline.core import CATEGORIES
from experiment_utils.common import DATA, load_manifest, sha256, signature, write_json


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def main():
    checks = []
    manifest = load_manifest()
    for name in ['part1_autoencoder', 'part2_anomalyclip']:
        folder = ROOT / 'results' / name
        config = json.loads((folder / 'run_config.json').read_text())
        assert config['evaluation_sha256'] == sha256(ROOT / 'experiment_utils/evaluate.py')
        assert config['common_sha256'] == sha256(ROOT / 'experiment_utils/common.py')
        rows = read_csv(folder / 'predictions_all.csv')
        assert len(rows) == 960 and len({r['sample_id'] for r in rows}) == 960
        for category in CATEGORIES:
            selected = [r for r in rows if r['category'] == category]
            metric = json.loads((folder / category / 'metrics.json').read_text())
            checkpoint_hash = sha256(folder / category / 'best.pt') if name == 'part1_autoencoder' else config['prompt_sha256']
            assert metric['run_signature'] == signature({'settings': config, 'category': category, 'checkpoint': checkpoint_hash})
            expected = {f'VisA/1cls/{category}/test/{label}/{p.name}' for label in ('good', 'bad')
                        for p in (DATA / category / 'test' / label).iterdir()}
            assert {r['sample_id'] for r in selected} == expected
            assert abs(roc_auc_score([int(r['label']) for r in selected], [float(r['score']) for r in selected]) - metric['image_auroc']) < 1e-12
            calibration = read_csv(folder / category / 'calibration.csv')
            assert [r['file'] for r in calibration] == manifest['categories'][category]['calibration']
            threshold = np.percentile([float(r['score']) for r in calibration], 95)
            assert abs(threshold - metric['threshold']) < 1e-12
            total, positive = 0, 0
            test_hashes = set()
            for row in selected:
                path = ROOT / row['image_file']
                assert sha256(path) == row['image_sha256']
                test_hashes.add(row['image_sha256'])
                predicted = int(float(row['score']) > threshold)
                assert row['decision'] == ('NG' if predicted else 'OK')
                assert int(row['is_correct']) == int(predicted == int(row['label']))
                with Image.open(path) as im:
                    total += im.width * im.height
                if row['ground_truth_mask']:
                    with Image.open(ROOT / row['ground_truth_mask']) as im:
                        positive += int(np.count_nonzero(np.array(im)))
                with np.load(ROOT / row['anomaly_map']) as saved:
                    array = saved['map']
                assert array.dtype == np.float32 and array.ndim == 2 and np.isfinite(array).all()
                assert array.min() >= 0 and array.max() <= (4 if name == 'part2_anomalyclip' else 1)
            assert total == metric['pixel_count'] and positive == metric['anomalous_pixels']
            cal_hashes = {sha256(DATA / category / 'train/good' / r['file']) for r in calibration}
            assert not cal_hashes & test_hashes
            if name == 'part1_autoencoder':
                inputs = read_csv(folder / category / 'training_images.csv')
                for item in inputs:
                    assert sha256(ROOT / item['file']) == item['sha256']
                fit = {r['sha256'] for r in inputs if r['role'] == 'fit'}
                validation = {r['sha256'] for r in inputs if r['role'] == 'validation'}
                assert not fit & validation and not (fit | validation) & (cal_hashes | test_hashes)
                history = read_csv(folder / category / 'training_history.csv')
                assert len(history) == 20 and [int(r['epoch']) for r in history] == list(range(1, 21))
                import torch
                checkpoint = torch.load(folder / category / 'best.pt', map_location='cpu', weights_only=True)
                training_config = json.loads((folder / category / 'training_config.json').read_text(encoding='utf-8'))
                assert checkpoint['signature'] == training_config['signature']
                assert training_config['source_sha256'] == sha256(ROOT / 'Part1/autoencoder/model.py')
                best_epoch = int(min(history, key=lambda r: float(r['validation_mse']))['epoch'])
                assert checkpoint['epoch'] == best_epoch
                assert abs(checkpoint['validation_mse'] - min(float(r['validation_mse']) for r in history)) < 1e-12
                checks.append({'method': name, 'category': category, 'fit': len(fit), 'validation': len(validation),
                               'calibration': len(cal_hashes), 'tests': len(selected), 'pixel_count': total,
                               'best_validation_mse': min(float(r['validation_mse']) for r in history),
                               'checkpoint_sha256': checkpoint_hash, 'best_epoch': best_epoch})
            else:
                checks.append({'method': name, 'category': category, 'target_fit_images': 0,
                               'calibration': len(cal_hashes), 'tests': len(selected), 'pixel_count': total})
    from Part2.anomalyclip.model import clip_settings
    provenance = clip_settings()
    vendor = ROOT / 'third_party/AnomalyCLIP'
    python_diff = subprocess.check_output(['git', '-C', str(vendor), 'diff', '--name-only', 'HEAD', '--', '*.py', '*.sh'], text=True)
    assert not python_diff.strip(), python_diff
    result = {'passed': True, 'checks': checks, 'anomalyclip_provenance': provenance,
              'notes': ['Full official 960-image test split for each optional method.',
                        'Zero masks for every normal test image included in pixel counts.',
                        'AUROC from raw scores; no per-image min-max normalization.',
                        'No overlap by file content between AE fit/validation/calibration/test.',
                        'Optional methods have different supervision protocols; not matched few-shot comparisons.']}
    write_json(ROOT / 'results/extensions/audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
