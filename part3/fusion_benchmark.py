"""Paired warmed 4-shot benchmark for DINO, CLIP and D/F image decisions."""

import json
import statistics
from time import perf_counter

import faiss
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from anomaly_dino.official import FeatureArchive, load_model
from anomaly_dino.predict import OfficialPredictor
from experiment_utils.common import DATA, ROOT, configure, load_manifest
from Part2.anomalyclip.model import AnomalyCLIPPredictor
from .fusion_selective import CLIP, OUTPUT, PART2, rows, save_csv, scaling, transform


def main():
    configure()
    faiss.omp_set_num_threads(4)
    threadpool_limits(4)
    dino = load_model()
    FeatureArchive(dino)
    clip = AnomalyCLIPPredictor()
    manifest = load_manifest()
    saved_dino = {row['image_file']: float(row['score']) for row in rows(PART2 / 'predictions_all.csv')
                  if row['shots'] == '4' and row['seed'] == '0'}
    saved_clip = {row['image_file']: float(row['score']) for row in rows(CLIP / 'predictions_all.csv')}
    records = []
    for category, entry in manifest['categories'].items():
        base = DATA / category
        references = [base / 'train/good' / name for name in entry['references']['4']['0']]
        detector = OfficialPredictor(dino, references)
        images = [image for label in ('good', 'bad')
                  for image in sorted((base / 'test' / label).glob('*.JPG'))[:10]]
        if len(images) != 20:
            raise ValueError(f'Expected 20 queries for {category}')
        dcal = np.array([float(row['score']) for row in rows(PART2 / category / 'shot_4/seed_0/calibration.csv')])
        ccal = np.array([float(row['score']) for row in rows(CLIP / category / 'calibration.csv')])
        dc, ds = scaling(dcal)
        cc, cs = scaling(ccal)
        calibrated = .5 * transform(dcal, dc, ds) + .5 * transform(ccal, cc, cs)
        threshold = float(np.percentile(calibrated, 95, method='linear'))
        spread = float(np.percentile(calibrated, 75) - np.percentile(calibrated, 25))
        detector.predict(images[0])
        clip.predict(images[0])
        for image in images:
            torch.cuda.reset_peak_memory_stats()
            started = perf_counter()
            dscore, _, _, dino_seconds = detector.predict(image)
            cscore, _, _, clip_seconds, _ = clip.predict(image)
            fused = float(.5 * transform(dscore, dc, ds) + .5 * transform(cscore, cc, cs))
            decision = 'REVIEW' if abs(fused - threshold) <= .5 * spread else ('NG' if fused > threshold else 'OK')
            torch.cuda.synchronize()
            total_ms = (perf_counter() - started) * 1000
            image_file = image.relative_to(ROOT).as_posix()
            if abs(dscore - saved_dino[image_file]) > 1e-5 or abs(cscore - saved_clip[image_file]) > 1e-5:
                raise ValueError(f'Saved scores do not match fresh inference: {image_file}')
            records.append({'category': category, 'image_file': image_file,
                            'dino_ms': dino_seconds * 1000, 'clip_ms': clip_seconds * 1000,
                            'combined_df_ms': total_ms, 'peak_gpu_mib': torch.cuda.max_memory_allocated() / 2**20,
                            'decision': decision})
        print(f'Timed {category}: 20 paired images', flush=True)
    save_csv(OUTPUT / 'fusion_timing.csv', records)
    summary = {'device': torch.cuda.get_device_name(0), 'shots': 4, 'seed': 0,
               'images': len(records), 'mean_dino_ms': statistics.mean(row['dino_ms'] for row in records),
               'mean_clip_ms': statistics.mean(row['clip_ms'] for row in records),
               'mean_combined_df_ms': statistics.mean(row['combined_df_ms'] for row in records),
               'peak_gpu_mib': max(row['peak_gpu_mib'] for row in records),
               'review_count': sum(row['decision'] == 'REVIEW' for row in records),
               'included': 'fresh query decode, both frozen feature passes, nearest-neighbor retrieval, '
                           'both map restorations, score fusion and selective decision',
               'excluded': 'model load, reference-bank setup, calibration, output I/O, metric calculation',
               'memory_note': 'peak allocated GPU tensor memory with both pretrained models resident; '
                              'not the total process/device allocation',
               'sample_note': 'same 120 images as the existing Part 3 A and AnyUp timing benchmark'}
    (OUTPUT / 'fusion_timing.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
