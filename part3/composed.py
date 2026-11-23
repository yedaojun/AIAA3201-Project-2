"""Exploratory composition: two-layer retrieval followed by pretrained AnyUp map restoration."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from anomaly_dino.metrics import positive_negative_auc
from anomaly_dino.official import load_model, utils
from experiment_utils.common import DATA, ROOT, MANIFEST, configure, sha256, write_json
from .multilayer import OUTPUT as A_OUTPUT, read_rows
from .upsampling import CHECKPOINT_SHA256, load_upsampler, restore_scores

OUTPUT = ROOT / 'results/part3/two_layer_anyup'


def map_path(category, shot, seed, kind, name):
    return OUTPUT / f'shot_{shot}_seed_{seed}' / category / kind / (Path(name).stem + '.npy')


def input_rows(category, shot, seed):
    folder = A_OUTPUT / category / f'shot_{shot}' / f'seed_{seed}'
    if not (folder / 'metrics.json').exists():
        raise ValueError(f'Complete Direction A first: {folder}')
    return read_rows(folder / 'predictions.csv')


def generate(categories, shot, seed):
    configure()
    original = load_model()
    learned = load_upsampler()
    for category in categories:
        rows = input_rows(category, shot, seed)
        for index, row in enumerate(rows, 1):
            image_path = ROOT / row['image_file']
            kind = 'bad' if int(row['label']) else 'good'
            target = map_path(category, shot, seed, kind, image_path.name)
            if target.exists():
                continue
            patch = np.load(ROOT / row['anomaly_map'])
            tensor, grid = original.prepare_image(str(image_path))
            if tuple(grid) != patch.shape:
                raise ValueError('A map and original DINO guidance grid mismatch')
            features = original.extract_features(tensor)
            restored = restore_scores(learned, tensor, features, patch[None])[0]
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.tmp.npy')
            np.save(temporary, restored)
            temporary.replace(target)
            if index == 1 or index % 40 == 0 or index == len(rows):
                print(f'{category} {shot}-shot seed {seed}: learned {index} / '
                      f'{len(rows)}', flush=True)


def evaluate(categories, shot, seed):
    details = []
    for category in categories:
        rows = input_rows(category, shot, seed)
        total = positives = 0
        for row in rows:
            image = ROOT / row['image_file']
            with Image.open(image) as source:
                total += source.width * source.height
            if int(row['label']):
                with Image.open(DATA / category / 'ground_truth/bad' / (image.stem + '.png')) as source:
                    positives += int(np.count_nonzero(np.asarray(source)))
        p = np.empty(positives, dtype=np.float64)
        n = np.empty(total - positives, dtype=np.float64)
        pi = ni = 0
        for row in rows:
            image = ROOT / row['image_file']
            kind = 'bad' if int(row['label']) else 'good'
            target = map_path(category, shot, seed, kind, image.name)
            if not target.exists():
                raise ValueError(f'Incomplete composed maps: {target}')
            with Image.open(image) as source:
                shape = (source.height, source.width)
            dense = utils.dists2map(np.load(target), shape)
            if int(row['label']):
                with Image.open(DATA / category / 'ground_truth/bad' / (image.stem + '.png')) as source:
                    mask = np.asarray(source) != 0
            else:
                mask = np.zeros(shape, dtype=bool)
            if mask.shape != dense.shape:
                raise ValueError(f'Ground-truth shape mismatch: {image}')
            pp, nn = dense[mask], dense[~mask]
            p[pi:pi + len(pp)] = pp
            n[ni:ni + len(nn)] = nn
            pi += len(pp)
            ni += len(nn)
        if pi != positives or ni != total - positives:
            raise ValueError('Original-resolution pixel count changed')
        result = {'category': category, 'shots': shot, 'seed': seed,
                  'image_auroc_two_layer': json.loads((A_OUTPUT / category / f'shot_{shot}' /
                                                       f'seed_{seed}' / 'metrics.json').read_text())['image_auroc'],
                  'pixel_auroc_two_layer_anyup': positive_negative_auc(p, n),
                  'pixel_count': total, 'positive_pixels': positives}
        details.append(result)
        write_json(OUTPUT / f'shot_{shot}_seed_{seed}' / category / 'metrics.json', result)
        print(f'{category}: A+AnyUp pixel AUROC {result["pixel_auroc_two_layer_anyup"]:.5f}', flush=True)
    if len(details) == 6:
        print(f'Six-category mean: {np.mean([r["pixel_auroc_two_layer_anyup"] for r in details]):.5f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('generate', 'evaluate'))
    parser.add_argument('--shot', type=int, default=4, choices=(1, 2, 4, 8))
    parser.add_argument('--seed', type=int, default=0, choices=(0, 1, 2))
    parser.add_argument('--categories', nargs='+', default=['candle', 'capsules', 'cashew',
                                                              'chewinggum', 'fryum', 'pipe_fryum'])
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = OUTPUT / 'protocol.json'
    settings = {'method': 'fixed two-layer DINO retrieval then pretrained AnyUp score restoration',
                'score_map_scale': 2, 'final_restoration': 'official bilinear original-size and Gaussian sigma4',
                'anyup_checkpoint_sha256': CHECKPOINT_SHA256,
                'manifest_sha256': sha256(MANIFEST),
                'test_label_use': 'evaluation and display only',
                'experiment_status': 'exploratory composition after individual branch pilot results'}
    if protocol.exists() and json.loads(protocol.read_text(encoding='utf-8')) != settings:
        raise ValueError('Existing composed experiment protocol differs')
    write_json(protocol, settings)
    if args.stage == 'generate':
        generate(args.categories, args.shot, args.seed)
    else:
        evaluate(args.categories, args.shot, args.seed)


if __name__ == '__main__':
    main()
