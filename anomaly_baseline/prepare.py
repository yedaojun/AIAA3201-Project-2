"""Prepare only six categories using the supplied official 1cls split."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .core import CATEGORIES, SEEDS, SHOTS


def prepare(raw_root, target_root, manifest_path):
    raw_root, target_root, manifest_path = map(Path, (raw_root, target_root, manifest_path))
    split_file = raw_root / 'split_csv/1cls.csv'
    with split_file.open(encoding='utf-8-sig', newline='') as file:
        rows = [row for row in csv.DictReader(file) if row['object'] in CATEGORIES]
    seen = set()
    counts = {category: {'train/good': 0, 'test/good': 0, 'test/bad': 0} for category in CATEGORIES}
    for row in rows:
        category, split = row['object'], row['split']
        label = 'good' if row['label'] == 'normal' else 'bad'
        if split == 'train' and label != 'good':
            raise ValueError('1cls training set must contain normal images only')
        if row['image'] in seen:
            raise ValueError(f'Duplicate split entry: {row["image"]}')
        seen.add(row['image'])
        source = raw_root / row['image']
        destination = target_root / category / split / label / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise ValueError(f'Existing prepared image differs: {destination}')
        else:
            shutil.copyfile(source, destination)
        counts[category][f'{split}/{label}'] += 1
        if label == 'bad':
            mask_destination = target_root / category / 'ground_truth/bad' / Path(row['mask']).name
            mask_destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(raw_root / row['mask']) as image:
                binary = (np.array(image) != 0).astype(np.uint8) * 255
            if not mask_destination.exists():
                Image.fromarray(binary).save(mask_destination)
            else:
                with Image.open(mask_destination) as old:
                    if not np.array_equal(binary, np.array(old)):
                        raise ValueError(f'Existing prepared mask differs: {mask_destination}')
    manifest = {'version': 1, 'dataset': 'VisA', 'categories': {},
                'split_sha256': hashlib.sha256(split_file.read_bytes()).hexdigest(),
                'reference_selection': 'sorted normal train pool excluding calibration; pool[seed*shots:(seed+1)*shots]',
                'calibration_selection': 'first 20 sorted official normal training filenames',
                'counts': counts}
    for category in CATEGORIES:
        files = sorted(path.name for path in (target_root / category / 'train/good').glob('*') if path.is_file())
        if len(files) != counts[category]['train/good']:
            raise ValueError(f'Unexpected extra files in prepared training set: {category}')
        calibration, pool = files[:20], files[20:]
        if len(pool) < max(SHOTS) * len(SEEDS):
            raise ValueError(f'Not enough reference candidates: {category}')
        references = {str(shot): {str(seed): pool[seed * shot:(seed + 1) * shot]
                                  for seed in SEEDS} for shot in SHOTS}
        assert not set(calibration).intersection(name for seeds in references.values() for names in seeds.values() for name in names)
        manifest['categories'][category] = {'calibration': calibration, 'references': references}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Existing manifest differs; refusing to silently change reference selections')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'prepared_root': str(target_root.resolve()), 'manifest': str(manifest_path.resolve()), 'counts': counts}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', default='data/VisA')
    parser.add_argument('--data-root', default='data/VisA_pytorch/1cls')
    parser.add_argument('--manifest', default='configs/reference_manifest.json')
    args = parser.parse_args()
    prepare(args.raw_root, args.data_root, args.manifest)


if __name__ == '__main__':
    main()
