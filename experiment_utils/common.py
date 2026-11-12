import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image
import torch

from anomaly_baseline.core import load_image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/VisA_pytorch/1cls'
MANIFEST = ROOT / 'configs/reference_manifest.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for part in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def configure():
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('These local optional experiments require the configured CUDA environment.')


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


def image_tensor(array):
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255)


def start_timer():
    torch.cuda.synchronize()
    return perf_counter()


def finish_timer(started):
    torch.cuda.synchronize()
    return perf_counter() - started


def restore_map(working, size):
    return cv2.resize(working.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)


def top_fraction_score(working, fraction=.01):
    flat = np.asarray(working).ravel()
    count = max(1, int(np.ceil(len(flat) * fraction)))
    return float(np.partition(flat, len(flat) - count)[-count:].mean())


def outcome(label, predicted):
    return ('TP' if predicted else 'FN') if label else ('FP' if predicted else 'TN')
