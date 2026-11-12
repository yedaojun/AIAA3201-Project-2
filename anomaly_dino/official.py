import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'third_party/AnomalyDINO'
sys.path.insert(0, str(VENDOR))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'models/torch'))

import numpy as np
import torch
from src import backbones, detection, post_eval, utils


def load_model():
    real_load = torch.hub.load
    with patch.object(torch.hub, 'load', lambda repo, name: real_load(str(ROOT / 'third_party/dinov2'), name, source='local')):
        model = backbones.get_model('dinov2_vits14', 'cuda', smaller_edge_size=448)
    model.model.eval().requires_grad_(False)
    return model


def prepare_view(manifest, source):
    """Official driver chooses the same slices; calibration queries have a unique prefix."""
    destination = ROOT / 'data/VisA_part2_official/1cls'
    def link(src, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                os.link(src, target)
            except OSError:
                shutil.copyfile(src, target)
        elif target.stat().st_size != src.stat().st_size:
            raise ValueError(f'Existing official view differs: {target}')
    for category, selection in manifest['categories'].items():
        refs = sorted(set(name for seeds in selection['references'].values() for names in seeds.values() for name in names))
        assert not set(refs).intersection(selection['calibration'])
        for name in refs:
            link(source / category / 'train/good' / name, destination / category / 'train/good' / name)
        for name in selection['calibration']:
            link(source / category / 'train/good' / name, destination / category / 'test/good' / ('__calibration__' + name))
        for label in ('good', 'bad'):
            for image in sorted((source / category / 'test' / label).glob('*')):
                if image.is_file():
                    link(image, destination / category / 'test' / label / image.name)
        actual = sorted(p.name for p in (destination / category / 'train/good').iterdir())
        if actual != refs:
            raise ValueError(f'Unexpected reference files in official view: {category}')
        for shot, seeds in selection['references'].items():
            for seed, names in seeds.items():
                assert actual[int(seed)*int(shot):(int(seed)+1)*int(shot)] == names
    return destination


class FeatureArchive:
    """Cache reference rotations; archive query features but freshly extract for timing."""
    def __init__(self, model):
        self.prepare = model.prepare_image
        self.extract = model.extract_features
        self.reference = False
        self.folder = ROOT / 'data/features/dinov2_vits14_448_fp32'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.stats = {'reference_cache_hits': 0, 'fresh_extractions': 0}
        model.prepare_image = self.prepare_image
        model.extract_features = self.extract_features

    def prepare_image(self, image):
        self.reference = isinstance(image, np.ndarray)
        return self.prepare(image)

    def extract_features(self, tensor):
        array = tensor.numpy()
        identity = hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()
        path = self.folder / (identity + '.npy')
        if self.reference and path.exists():
            self.stats['reference_cache_hits'] += 1
            return np.load(path).copy()
        features = self.extract(tensor)
        self.stats['fresh_extractions'] += 1
        if not path.exists():
            np.save(path, features)
        return features


def dense_map(path, image_shape):
    return utils.dists2map(np.load(path), image_shape)
