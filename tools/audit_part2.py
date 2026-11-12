"""Check complete run coverage, scores, calibration, source provenance and CSV pairing."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from anomaly_dino.official import post_eval, prepare_view

output = ROOT / 'results/part2'
manifest = json.loads((ROOT / 'configs/reference_manifest.json').read_text())
settings = json.loads((output / 'run_config.json').read_text())
metrics = json.loads((output / 'metrics_all.json').read_text())
assert len(metrics) == len({(m['category'], m['shot'], m['seed']) for m in metrics}) == 72
assert len({m['run_signature'] for m in metrics}) == 1
weights = ROOT / 'models/torch/hub/checkpoints/dinov2_vits14_pretrain.pth'
assert hashlib.sha256(weights.read_bytes()).hexdigest() == settings['weights_sha256']
prepare_view(manifest, ROOT / 'data/VisA_pytorch/1cls')
total_rows, checked_patches, size_cache = 0, 0, {}
for metric in metrics:
    category, shot, seed = metric['category'], metric['shot'], metric['seed']
    group = output / category / f'shot_{shot}' / f'seed_{seed}'
    assert metric == json.loads((group / 'metrics.json').read_text())
    selection = manifest['categories'][category]
    assert metric['references'] == selection['references'][str(shot)][str(seed)]
    assert metric['calibration_files'] == selection['calibration']
    assert not set(metric['references']).intersection(selection['calibration'])
    with (group / 'predictions.csv').open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    with (group / 'calibration.csv').open(encoding='utf-8-sig', newline='') as file:
        cal = list(csv.DictReader(file))
    assert len(cal) == 20 and {r['file'] for r in cal} == set(selection['calibration'])
    assert abs(float(np.percentile([float(r['score']) for r in cal], 95)) - metric['threshold']) < 1e-12
    assert sum(int(r['label']) for r in rows) == 100
    assert len(rows) == sum(manifest['counts'][category][key] for key in ('test/good', 'test/bad'))
    assert abs(roc_auc_score([int(r['label']) for r in rows], [float(r['score']) for r in rows]) - metric['image_auroc']) < 1e-12
    pixels = 0
    for row in rows:
        path = ROOT / 'data/VisA_pytorch/1cls' / category / row['file']
        if path not in size_cache:
            with Image.open(path) as image:
                size_cache[path] = image.width * image.height
        pixels += size_cache[path]
        patch = np.load(group / row['anomaly_map'])
        assert patch.dtype == np.float64 and np.isfinite(patch).all()
        assert abs(float(post_eval.mean_top1p(patch.ravel())) - float(row['score'])) < 1e-12
        assert row['decision'] == ('NG' if float(row['score']) > float(row['threshold']) else 'OK')
        checked_patches += 1
    assert pixels == metric['pixel_count']
    total_rows += len(rows)
assert total_rows == 11520
baseline_metrics = json.loads((ROOT / 'results/part1/metrics_all.json').read_text())
for metric in baseline_metrics:
    histogram_path = ROOT / 'results/part1' / metric['category'] / f'shot_{metric["shot"]}' / f'seed_{metric["seed"]}' / 'pixel_score_histograms.npz'
    hist = np.load(histogram_path)
    reference = roc_auc_score(np.r_[np.ones(256), np.zeros(256)], np.r_[np.arange(256), np.arange(256)],
                              sample_weight=np.r_[hist['positive'], hist['negative']])
    assert abs(reference - metric['pixel_auroc']) < 1e-12
git = shutil.which('git') or 'C:/Users/Ziqing/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
for folder in ('AnomalyDINO', 'dinov2'):
    subprocess.run([git, '-C', str(ROOT / 'third_party' / folder), 'diff', '--exit-code', '--quiet'], check=True)
report = {'passed': True, 'configurations': len(metrics), 'test_predictions': total_rows,
          'raw_patch_scores_verified': checked_patches, 'sklearn_image_auc_parity': True,
          'all_normal_test_pixels_included': True, 'reference_and_calibration_manifest_parity': True,
          'official_tracked_sources_unchanged': True, 'checkpoint_sha256_verified': True,
          'baseline_pixel_auc_sklearn_weighted_parity_all_72': True,
          'float_pixel_auc_validation': 'Unit tests compare exact float64 rank algorithm against sklearn; full-resolution area count checked for every configuration.'}
(output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
