"""Exact original-resolution pixel AP, including zero masks for normal test images."""

import json
import numpy as np
from PIL import Image

from anomaly_dino.metrics import positive_negative_auc
from anomaly_dino.official import utils
from experiment_utils.common import DATA, ROOT, write_json
from .multilayer import read_rows


def average_precision(positive, negative):
    positive.sort()
    negative.sort()
    values, counts = np.unique(positive, return_counts=True)
    tp = len(positive) - np.searchsorted(positive, values, side='left')
    fp = len(negative) - np.searchsorted(negative, values, side='left')
    return float(np.sum(counts * tp / (tp + fp)) / len(positive))


def score(category, method, seed=0, include_auc=False, shot=4, bad_files=None):
    if method in ('official', 'official_anyup'):
        group = ROOT / 'results/part2' / category / f'shot_{shot}' / f'seed_{seed}'
    elif method == 'high_res':
        group = ROOT / 'results/part3/high_resolution_pilot' / category / f'shot_{shot}' / f'seed_{seed}'
    elif method in ('multi_res', 'selective_multi_res'):
        base = ('multiscale_fusion' if method == 'multi_res' else 'selective_multiscale')
        group = ROOT / 'results/part3' / base / category / f'shot_{shot}' / f'seed_{seed}'
    else:
        group = ROOT / 'results/part3/multilayer' / category / f'shot_{shot}' / f'seed_{seed}'
    records = read_rows(group / 'predictions.csv')
    if bad_files is not None:
        bad_files = set(bad_files)
        def keep(row):
            if not int(row['label']):
                return True
            file = (row['file'] if method in
                    ('official', 'official_anyup', 'high_res', 'multi_res', 'selective_multi_res')
                    else row['image_file'].split(category + '/', 1)[1])
            return file.split('/')[-1] in bad_files
        records = [row for row in records if keep(row)]
    total = 0
    positives = 0
    shapes = {}
    for row in records:
        file = row['file'] if method in ('official', 'official_anyup', 'high_res', 'multi_res', 'selective_multi_res') else row['image_file'].split(category + '/', 1)[1]
        image = DATA / category / file
        with Image.open(image) as im:
            shapes[file] = (im.height, im.width, 3)
            total += im.height * im.width
        if int(row['label']):
            mask = DATA / category / 'ground_truth/bad' / (image.stem + '.png')
            with Image.open(mask) as im:
                positives += np.count_nonzero(np.asarray(im))
    pos = np.empty(positives, dtype=np.float64)
    neg = np.empty(total - positives, dtype=np.float64)
    pi = ni = 0
    if method in ('multi_res', 'selective_multi_res'):
        low_group = ROOT / 'results/part2' / category / f'shot_{shot}' / f'seed_{seed}'
        fine_group = ROOT / 'results/part3/high_resolution_pilot' / category / f'shot_{shot}' / f'seed_{seed}'
        low_rows = {row['file']: row for row in read_rows(low_group / 'predictions.csv')}
        fine_rows = {row['file']: row for row in read_rows(fine_group / 'predictions.csv')}
        low_threshold = json.loads((low_group / 'metrics.json').read_text())['threshold']
        fine_threshold = json.loads((fine_group / 'metrics.json').read_text())['threshold']
    for row in records:
        file = row['file'] if method in ('official', 'official_anyup', 'high_res', 'multi_res', 'selective_multi_res') else row['image_file'].split(category + '/', 1)[1]
        if method in ('multi_res', 'selective_multi_res'):
            low_map = np.load(low_group / low_rows[file]['anomaly_map'])
            low_heatmap = utils.dists2map(low_map, shapes[file]) / low_threshold
            if method == 'multi_res' or int(row['use_672']):
                fine_map = np.load(ROOT / fine_rows[file]['anomaly_map'])
                fine_heatmap = utils.dists2map(fine_map, shapes[file]) / fine_threshold
                heatmap = .5 * (low_heatmap + fine_heatmap)
            else:
                heatmap = low_heatmap
        else:
            if method == 'official':
                map_file = group / row['anomaly_map']
            elif method == 'high_res':
                map_file = ROOT / row['anomaly_map']
            elif method == 'official_anyup':
                map_file = ROOT / 'results/part3/anyup_score_maps/maps' / category / file.removeprefix('test/')
                map_file = map_file.with_suffix('.npz')
            elif method == 'two_layer_anyup':
                map_file = ROOT / 'results/part3/two_layer_anyup' / f'shot_{shot}_seed_{seed}' / category / file.removeprefix('test/')
                map_file = map_file.with_suffix('.npy')
            else:
                map_file = ROOT / row['anomaly_map']
            if method == 'official_anyup':
                with np.load(map_file) as maps:
                    coarse = maps['anyup_2x'][(1, 2, 4, 8).index(shot) * 3 + seed]
            else:
                coarse = np.load(map_file)
            heatmap = utils.dists2map(coarse.astype(np.float64, copy=False), shapes[file])
        if int(row['label']):
            with Image.open(DATA / category / 'ground_truth/bad' / (file.split('/')[-1].split('.')[0] + '.png')) as im:
                mask = np.asarray(im) > 0
            selected = heatmap[mask]
            pos[pi:pi + len(selected)] = selected
            pi += len(selected)
            selected = heatmap[~mask]
        else:
            selected = heatmap.ravel()
        neg[ni:ni + len(selected)] = selected
        ni += len(selected)
    if pi != positives or ni != total - positives:
        raise ValueError('Pixel count mismatch')
    ap = average_precision(pos, neg)
    if include_auc:
        return ap, positive_negative_auc(pos, neg), total, positives
    return ap, total, positives


def main():
    destination = ROOT / 'results/part3/pixel_ap_4shot_seed0.json'
    result = (json.loads(destination.read_text(encoding='utf-8')) if destination.exists() else
              {'protocol': 'exact original-resolution pixel AP, 4-shot/seed 0',
               'normal_masks': 'included as all-zero masks', 'categories': {}})
    for category in ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum'):
        result['categories'].setdefault(category, {})
        for method in ('official', 'official_anyup', 'two_layer', 'two_layer_anyup'):
            if method in result['categories'][category]:
                continue
            ap, total, positives = score(category, method)
            result['categories'][category][method] = {'pixel_ap': ap, 'pixels': total,
                                                        'positive_pixels': positives}
            print(category, method, f'{100*ap:.2f}%', flush=True)
            write_json(destination, result)
    result['macro'] = {method: float(np.mean([result['categories'][c][method]['pixel_ap']
                                             for c in result['categories']]))
                       for method in ('official', 'official_anyup', 'two_layer', 'two_layer_anyup')}
    write_json(destination, result)
    print(json.dumps(result['macro'], indent=2), flush=True)


if __name__ == '__main__':
    main()
