"""Read-only checks of saved references and post-hoc defect geometry; no inference."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with (ROOT / path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    key = lambda r: (r['category'], r['shots'], r['seed'])
    before = {key(r): r for r in read('results/part1/runs.csv')}
    after = {key(r): r for r in read('results/part2/runs.csv')}
    manifest = json.loads((ROOT / 'configs/reference_manifest.json').read_text())
    assert len(before) == len(after) == 72 and before.keys() == after.keys()
    for identity, row in before.items():
        selection = manifest['categories'][row['category']]
        refs = json.loads(row['reference_images_json'])
        calibration = json.loads(row['calibration_images_json'])
        assert refs == json.loads(after[identity]['reference_images_json'])
        assert calibration == json.loads(after[identity]['calibration_images_json'])
        assert len(calibration) == 20 and not set(refs).intersection(calibration)
        assert [Path(p).name for p in refs] == selection['references'][row['shots']][row['seed']]
        training = sorted(p.name for p in (ROOT / 'data/VisA_pytorch/1cls' / row['category'] / 'train/good').iterdir() if p.is_file())
        assert [Path(p).name for p in calibration] == training[:20]

    row = next(r for r in read('results/part2/predictions_all.csv') if
               key(r) == ('candle', '4', '0') and r['sample_id'].endswith('/bad/000.JPG'))
    with Image.open(ROOT / row['image_file']) as image:
        width, height = image.size
    with Image.open(ROOT / row['ground_truth_mask']) as image:
        mask = np.array(image) > 0
    resized = (448, int(height * 448 / width)) if width <= height else (int(width * 448 / height), 448)
    rw, rh = resized[0] - resized[0] % 14, resized[1] - resized[1] % 14
    # Diagnostic annotation resizing only. The model never receives this mask.
    small = np.array(Image.fromarray(mask.astype('uint8')).resize(resized, Image.Resampling.NEAREST))[:rh, :rw]
    ys, xs = np.where(small)
    occupancy = small.reshape(rh // 14, 14, rw // 14, 14).sum((1, 3))
    patch = np.load(ROOT / row['anomaly_map'])
    assert patch.shape == (rh // 14, rw // 14)
    result = {
        'matched_configs': len(before),
        'all_references_identical': True,
        'calibration_disjoint_all_configs': True,
        'calibration_is_first_20_sorted_training_files': True,
        'candle_case': {
            'sample_id': row['sample_id'], 'original_width': width, 'original_height': height,
            'resized_width': rw, 'resized_height': rh, 'patch_grid': list(patch.shape),
            'original_defect_percent': float(100 * mask.mean()),
            'resized_defect_bbox': [int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
            'resized_defect_pixels': int(small.sum()),
            'max_patch_defect_occupancy': float(occupancy.max() / 196),
            'touched_patches': int((occupancy > 0).sum()),
            'score': float(row['score']), 'threshold': float(row['threshold']),
            'raw_map_shape': list(patch.shape),
        },
        'diagnostic_scope': 'Saved predictions and nearest-neighbor-resized annotation only. No inference, tuning, or causal ablation.',
    }
    (ROOT / 'results/requirements_learning_audit.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print('PASS: 72 paired configurations, independent first-20 calibration, saved 32x35 map and post-hoc defect geometry.')


if __name__ == '__main__':
    main()
