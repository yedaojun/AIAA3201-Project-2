"""Single-image official-component predictor for demos and end-to-end benchmarks."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import faiss
import numpy as np
import torch

from .official import ROOT, load_model, FeatureArchive, utils, post_eval


class OfficialPredictor:
    def __init__(self, model, references, seed=0):
        np.random.seed(seed)
        self.model = model
        bank = []
        for path in references:
            image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            for rotated in utils.augment_image(image):
                tensor, grid = model.prepare_image(rotated)
                features = model.extract_features(tensor)
                mask = model.compute_background_mask(features, grid, threshold=10, masking_type=False)
                bank.append(features[mask])
        bank = np.concatenate(bank).astype(np.float32)
        faiss.normalize_L2(bank)
        self.index = faiss.IndexFlatL2(bank.shape[1])
        self.index.add(bank)

    def predict(self, path):
        started = perf_counter()
        image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        tensor, grid = self.model.prepare_image(str(path))
        features = self.model.extract_features(tensor)
        mask = self.model.compute_background_mask(features, grid, threshold=10, masking_type=True)
        features = features[mask]
        faiss.normalize_L2(features)
        distances, _ = self.index.search(features, k=1)
        output = np.zeros_like(mask, dtype=float)
        output[mask] = (distances / 2).squeeze()
        patch = output.reshape(grid)
        score = float(post_eval.mean_top1p(output))
        dense = utils.dists2map(patch, image.shape)
        torch.cuda.synchronize()
        return score, dense, patch, perf_counter() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--category', required=True)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--results', type=Path, default=ROOT / 'results/part2')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/demo_part2')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    from threadpoolctl import threadpool_limits
    threadpool_limits(4)
    metric = json.loads((args.results / args.category / f'shot_{args.shot}' / f'seed_{args.seed}' / 'metrics.json').read_text())
    model = load_model()
    archive = FeatureArchive(model)
    detector = OfficialPredictor(model, [ROOT / 'data/VisA_pytorch/1cls' / args.category / 'train/good' / n for n in metric['references']], args.seed)
    score, dense, patch, elapsed = detector.predict(args.image)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / 'patch_distances.npy', patch)
    import tifffile
    tifffile.imwrite(args.output / 'anomaly_map.tiff', dense)
    result = {'method': 'official_anomalydino', 'category': args.category, 'shots': args.shot, 'seed': args.seed,
              'image': str(args.image.resolve()), 'score': score, 'threshold': metric['threshold'],
              'decision': 'NG' if score > metric['threshold'] else 'OK', 'inference_ms_end_to_end': elapsed * 1000}
    (args.output / 'prediction.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    from anomaly_baseline.core import write_panel
    write_panel(args.image, None, np.clip(dense * 255, 0, 255).astype(np.uint8),
                f'AnomalyDINO | {args.category} | {result["decision"]} | score={score:.4f}, T={metric["threshold"]:.4f}', args.output / 'prediction.png')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
