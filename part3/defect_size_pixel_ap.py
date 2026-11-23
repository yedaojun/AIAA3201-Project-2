"""Exact original-size pixel AP by test-defect area, diagnostic only.

Each bucket contains all normal test images (zero masks) and only anomalous
images from that area bucket. GT masks define buckets after all scores are
frozen. Thus AP prevalence changes by bucket and raw values are not a single
balanced leaderboard.
"""

import argparse
import csv
import json

import numpy as np
from PIL import Image

from experiment_utils.common import DATA, ROOT, write_json
from .defect_size_audit import CATEGORIES, OUT, bucket
from .pixel_ap import score


METHODS = ('official', 'two_layer', 'multi_res')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, choices=(0, 1, 2), default=[0, 1, 2])
    parser.add_argument('--categories', nargs='+', choices=CATEGORIES, default=CATEGORIES)
    args = parser.parse_args()
    path = OUT / 'pixel_ap_by_size_4shot.json'
    result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'protocol': 'post-hoc defect-size buckets, four shots, exact original-resolution pixel AP',
        'bucket_definitions': {'tiny': '<0.5%', 'medium': '0.5%-2%', 'large': '>2%'},
        'normal_images': 'all category normal test images included in every bucket',
        'warning': 'bucket AP prevalence differs; same test set used to develop multiscale candidate',
        'categories': {}}
    for category in args.categories:
        files = {'tiny': set(), 'medium': set(), 'large': set()}
        for image in sorted((DATA / category / 'test/bad').glob('*.JPG')):
            with Image.open(DATA / category / 'ground_truth/bad' / (image.stem + '.png')) as im:
                mask = np.asarray(im)
            label = bucket(float(np.count_nonzero(mask) / mask.size))
            files[{'tiny_<0.5pct': 'tiny', 'medium_0.5-2pct': 'medium',
                   'large_>2pct': 'large'}[label]].add(image.name)
        for seed in args.seeds:
            group = result['categories'].setdefault(category, {}).setdefault(str(seed), {})
            for method in METHODS:
                for size, names in files.items():
                    key = f'{method}_{size}'
                    if key in group or not names:
                        continue
                    ap, pixels, positives = score(category, method, seed, bad_files=names)
                    group[key] = {'pixel_ap': ap, 'images_anomalous': len(names),
                                  'pixels': pixels, 'positive_pixels': positives}
                    write_json(path, result)
                    print(category, seed, method, size, f'{100*ap:.2f}%', flush=True)
    entries = []
    for category, seeds in result['categories'].items():
        for seed, values in seeds.items():
            for key, metrics in values.items():
                method, size = key.rsplit('_', 1)
                entries.append({'category': category, 'seed': seed, 'method': method,
                                'size': size, **metrics})
    if entries:
        with (OUT / 'pixel_ap_by_size_4shot.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(entries[0]))
            writer.writeheader()
            writer.writerows(entries)
    for size in ('tiny', 'medium', 'large'):
        for method in METHODS:
            values = [r['pixel_ap'] for r in entries if r['size'] == size and r['method'] == method]
            if values:
                print(method, size, f'{100*np.mean(values):.2f}% across {len(values)} category/group cells', flush=True)


if __name__ == '__main__':
    main()
