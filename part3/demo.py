"""Inspect one image with Direction A, optional D fusion and F review."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from anomaly_dino.official import load_model, utils
from anomaly_dino.predict import OfficialPredictor
from experiment_utils.common import DATA, ROOT, MANIFEST, configure, write_json
from Part2.anomalyclip.model import AnomalyCLIPPredictor
from .fusion_selective import CLIP, scaling, transform, rows
from .multilayer import OUTPUT as A_OUTPUT, TwoLayerDINO
from .upsampling import load_upsampler, restore_scores


def panel(image_path, heatmap, label, source_label, destination):
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    # A fixed display range keeps panels comparable; the score itself is untouched.
    heat = cv2.cvtColor(cv2.applyColorMap(np.rint(np.clip(heatmap / .4, 0, 1) * 255).astype(np.uint8),
                                         cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    heat = Image.fromarray(heat)
    overlay = Image.blend(image, heat, .45)
    canvas = Image.new('RGB', (1080, 365), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 8), label, fill='black')
    for index, (title, view) in enumerate(zip(('Input', source_label + ' (0 to 0.4)', 'Overlay'),
                                              (image, heat, overlay))):
        view.thumbnail((344, 322))
        x = index * 360 + (360 - view.width) // 2
        y = 42 + (320 - view.height) // 2
        canvas.paste(view, (x, y))
        draw.text((index * 360 + 8, 26), title, fill='black')
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--category', required=True)
    parser.add_argument('--shots', type=int, default=4, choices=(1, 2, 4, 8))
    parser.add_argument('--seed', type=int, default=0, choices=(0, 1, 2))
    parser.add_argument('--mode', choices=('multilayer', 'fusion_review'), default='fusion_review')
    parser.add_argument('--map-restoration', choices=('official', 'anyup'), default='official')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/part3/demo')
    args = parser.parse_args()
    configure()
    selection = json.loads(MANIFEST.read_text(encoding='utf-8'))['categories'][args.category]
    folder = A_OUTPUT / args.category / f'shot_{args.shots}' / f'seed_{args.seed}'
    if not (folder / 'metrics.json').exists():
        raise ValueError('Run the selected multilayer configuration before this demo')
    references = selection['references'][str(args.shots)][str(args.seed)]
    model = TwoLayerDINO(load_model())
    detector = OfficialPredictor(model, [DATA / args.category / 'train/good' / name for name in references],
                                 seed=args.seed)
    if args.map_restoration == 'anyup':
        upsampler = load_upsampler()

        def learned_map(tensor, _features, patch, shape):
            restored = restore_scores(upsampler, tensor, model.last_final, patch[None])[0]
            return utils.dists2map(restored.astype(np.float64), shape)
    else:
        learned_map = None
    a_score, heat, _, elapsed = detector.predict(args.image, restore_map=learned_map)
    a_cal = rows(folder / 'calibration.csv')
    a_threshold = float(np.percentile([float(r['score']) for r in a_cal], 95, method='linear'))
    result = {'method': 'two_layer_dino' if args.mode == 'multilayer' else 'two_layer_dino_plus_clip_review',
              'image': str(args.image.resolve()), 'category': args.category,
              'shots': args.shots, 'seed': args.seed, 'references': references,
              'dino_score': a_score, 'dino_threshold': a_threshold,
              'dino_decision': 'NG' if a_score > a_threshold else 'OK',
              'heatmap_source': 'two_layer_dino_' + args.map_restoration,
              'dino_inference_ms': elapsed * 1000}
    if args.mode == 'fusion_review':
        c_cal = rows(CLIP / args.category / 'calibration.csv')
        if [r['file'] for r in a_cal] != [r['file'] for r in c_cal]:
            raise ValueError('Calibration files are not paired')
        an = np.array([float(r['score']) for r in a_cal])
        cn = np.array([float(r['score']) for r in c_cal])
        ac, az = scaling(an)
        cc, cz = scaling(cn)
        calibrated = .5 * transform(an, ac, az) + .5 * transform(cn, cc, cz)
        threshold = float(np.percentile(calibrated, 95, method='linear'))
        spread = float(np.percentile(calibrated, 75) - np.percentile(calibrated, 25))
        clip_score, _, _, clip_elapsed, _ = AnomalyCLIPPredictor().predict(args.image)
        fused_score = float(.5 * transform(a_score, ac, az) + .5 * transform(clip_score, cc, cz))
        decision = 'NG' if fused_score > threshold else 'OK'
        selective = 'REVIEW' if abs(fused_score - threshold) <= .5 * spread else decision
        result.update({'clip_score': clip_score, 'fused_score': fused_score,
                       'fused_threshold': threshold, 'fused_decision': decision,
                       'selective_decision': selective,
                       'uncertainty_margin': float(abs(fused_score - threshold) / spread),
                       'clip_inference_ms': clip_elapsed * 1000,
                       'clip_source_training': 'MVTec AD anomalies and normal images',
                       'localization_note': 'heatmap comes from two-layer DINO with selected restoration; CLIP is image-score fusion only'})
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'prediction.json', result)
    np.save(args.output / 'anomaly_map.npy', heat)
    title = (f'{result["selective_decision"]}  |  fused score {result["fused_score"]:.3f}, '
             f'threshold {result["fused_threshold"]:.3f}' if args.mode == 'fusion_review' else
             f'{result["dino_decision"]}  |  DINO score {a_score:.3f}, threshold {a_threshold:.3f}')
    panel(args.image, heat, title,
          'Two-layer + AnyUp heatmap' if args.map_restoration == 'anyup' else 'Two-layer DINO heatmap',
          args.output / 'prediction.png')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
