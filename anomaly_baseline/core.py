from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

CATEGORIES = ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'pipe_fryum')
SHOTS = (1, 2, 4, 8)
SEEDS = (0, 1, 2)
CONFIG = {
    'method': 'RGB template difference', 'short_edge': 448, 'patch_multiple': 14,
    'gaussian_sigma_working_pixels': 2.0, 'top_fraction': 0.01,
    'map_dtype': 'uint8', 'map_levels': 256, 'calibration_count': 20,
    'threshold_percentile': 95, 'alignment': 'none',
    'template_choice': 'minimum whole-image mean absolute RGB difference',
    'geometry': 'PIL bicubic short-edge resize; bottom/right crop to multiple of 14',
    'template_shape_mismatch': 'bilinear resize prepared reference to prepared query shape',
    'postprocess': 'rounded channel-mean RGB difference; uint8 Gaussian blur; uint8 bilinear restore to original size',
    'image_score': 'mean of highest ceil(1% * working_pixels) smoothed scores / 255',
    'pixel_auroc': 'exact 256-bin histogram of actual uint8 output, all normal and anomalous test images',
    'test_time_refinements': 'none beyond the declared fixed smoothing and resize',
    'reference_rotation': False, 'foreground_mask': False,
    'timing': 'CPU wall time including query decode, preprocessing, matching, smoothing and map resize; excludes output I/O and metrics',
}


def load_image(path, short_edge=448):
    with Image.open(path) as source:
        source = source.convert('RGB')
        width, height = source.size
        if width <= height:
            resized = (short_edge, int(short_edge * height / width))
        else:
            resized = (int(short_edge * width / height), short_edge)
        image = np.array(source.resize(resized, Image.Resampling.BICUBIC))
    crop_h, crop_w = image.shape[0] // 14 * 14, image.shape[1] // 14 * 14
    return image[:crop_h, :crop_w].copy(), (width, height)


def histogram_auc(positive, negative):
    """Exact P(score_positive > score_negative) + .5 P(tie)."""
    positive, negative = np.asarray(positive, dtype=np.float64), np.asarray(negative, dtype=np.float64)
    denominator = positive.sum() * negative.sum()
    if denominator == 0:
        raise ValueError('AUROC requires both positive and negative observations')
    return float(np.sum(positive * (np.cumsum(negative) - negative * .5)) / denominator)


def image_auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    _, indices = np.unique(np.asarray(scores), return_inverse=True)
    size = int(indices.max()) + 1
    return histogram_auc(np.bincount(indices[labels], minlength=size),
                         np.bincount(indices[~labels], minlength=size))


class TemplateDetector:
    def __init__(self, reference_paths, config=None):
        self.config = dict(CONFIG if config is None else config)
        self.paths = list(map(Path, reference_paths))
        self.references = [load_image(p, self.config['short_edge'])[0] for p in self.paths]
        if not self.references:
            raise ValueError('At least one reference is required')
        self.resized_banks = {}

    def predict(self, image_path):
        started = perf_counter()
        query, original_size = load_image(image_path, self.config['short_edge'])
        shape = query.shape[:2]
        if shape not in self.resized_banks:
            self.resized_banks[shape] = [
                (reference if reference.shape[:2] == shape else
                 cv2.resize(reference, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)).astype(np.int16)
                for reference in self.references]
        query = query.astype(np.int16)
        best_error, difference, best_index = float('inf'), None, None
        for index, reference in enumerate(self.resized_banks[shape]):
            candidate = np.abs(query - reference)
            error = float(candidate.mean())
            if error < best_error:
                best_error, difference, best_index = error, candidate, index
        anomaly = np.rint(difference.mean(axis=2)).astype(np.uint8)
        sigma = self.config['gaussian_sigma_working_pixels']
        anomaly = cv2.GaussianBlur(anomaly, (0, 0), sigmaX=sigma, sigmaY=sigma)
        flat = anomaly.ravel()
        count = max(1, int(np.ceil(flat.size * self.config['top_fraction'])))
        score = float(np.partition(flat, flat.size - count)[-count:].mean() / 255)
        restored = cv2.resize(anomaly, original_size, interpolation=cv2.INTER_LINEAR)
        return score, restored, self.paths[best_index].name, perf_counter() - started


def write_panel(image_path, mask, anomaly, title, destination):
    from PIL import ImageDraw
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    heat = cv2.cvtColor(cv2.applyColorMap(anomaly, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    overlay = Image.blend(image, Image.fromarray(heat), .45)
    if mask is None:
        mask = np.zeros(anomaly.shape, dtype=np.uint8)
    else:
        mask = (mask > 0).astype(np.uint8) * 255
    panels = (image, Image.fromarray(mask).convert('RGB'), Image.fromarray(heat), overlay)
    canvas = Image.new('RGB', (4 * 270, 320), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 7), title, fill='black')
    for index, (label, panel) in enumerate(zip(('Input', 'Ground truth (blank if unavailable)', 'Heatmap (fixed 0..255)', 'Overlay'), panels)):
        panel.thumbnail((258, 258))
        x = index * 270 + (270 - panel.width) // 2
        canvas.paste(panel, (x, 54 + (258 - panel.height) // 2))
        draw.text((index * 270 + 6, 32), label, fill='black')
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
