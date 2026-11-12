"""Adapter around pinned official AnomalyCLIP code and MVTec-trained prompts."""
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
from torchvision import transforms

from .common import ROOT, MANIFEST, finish_timer, restore_map, sha256, start_timer

VENDOR = ROOT / 'third_party/AnomalyCLIP'
BACKBONE = ROOT / 'models/clip/ViT-L-14-336px.verified.pt'
PROMPT = VENDOR / 'checkpoints/9_12_4_multiscale_visa/epoch_15.pth'
EXPECTED_BACKBONE = '3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02'
PINNED_COMMIT = '3911738c0867544f545a076ad78f3f11d9ecbfdf'
PARAMETERS = {'Prompt_length': 12, 'learnabel_text_embedding_depth': 9, 'learnabel_text_embedding_length': 4}
SIZE = 448


def clip_settings():
    if not BACKBONE.is_file() or sha256(BACKBONE) != EXPECTED_BACKBONE:
        raise RuntimeError('Run tools/download_clip_weights.py to obtain verified official CLIP weights.')
    commit = subprocess.check_output(['git', '-C', str(VENDOR), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != PINNED_COMMIT:
        raise ValueError('AnomalyCLIP source commit changed')
    return {
        'method_id': 'official_anomalyclip', 'protocol': 'zero target reference images; external MVTec-supervised prompts; supplementary comparison',
        'source_repo': 'https://github.com/zqhang/AnomalyCLIP', 'source_commit': commit,
        'adapter_sha256': sha256(__file__), 'backbone': 'OpenAI CLIP ViT-L/14@336px',
        'backbone_sha256': EXPECTED_BACKBONE, 'prompt_checkpoint': PROMPT.relative_to(ROOT).as_posix(),
        'prompt_sha256': sha256(PROMPT), 'prompt_source_training_data': 'MVTec AD, including labeled normal and anomalous source images; no VisA fitting',
        'checkpoint_provenance': 'official train.sh second branch trains dataset=mvtec into _visa directory; test.sh second branch evaluates dataset=visa',
        'prompt_parameters': PARAMETERS, 'features_list': [6, 12, 18, 24], 'DPAM_layer': 20,
        'image_size': SIZE, 'official_default_image_size': 518,
        'geometry': 'official square bicubic resize and center crop, set to 448; differs from aspect-preserving template/AnomalyDINO geometry',
        'comparison_note': 'Optional zero-shot extension, not a matched few-shot reproduction or a Part 3 improvement; geometry and external supervision differ.',
        'image_score': 'official global image-text anomaly softmax probability, temperature .07; not anomaly-map maximum',
        'pixel_score': 'official four layer similarity maps summed, Gaussian sigma=4 at 448, bilinear restore to original image',
        'map_storage': 'float32 448-square NPZ before original-resolution restore', 'map_range': [0, 4],
        'precision': 'FP32', 'calibration': 'same 20 normal training images; score 95th percentile; no prompt selection',
        'manifest_sha256': sha256(MANIFEST), 'target_fit_images': 0,
        'target_normal_references': 0, 'inference_seed': 111, 'test_label_tuning': False,
    }


class AnomalyCLIPPredictor:
    def __init__(self):
        sys.path.insert(0, str(VENDOR))
        import AnomalyCLIP_lib
        from prompt_ensemble import AnomalyCLIP_PromptLearner
        torch.manual_seed(111)
        np.random.seed(111)
        self.library = AnomalyCLIP_lib
        # Local verified path avoids upstream's hardcoded author-specific download directory.
        model, _ = AnomalyCLIP_lib.load(str(BACKBONE), device='cpu', design_details=PARAMETERS)
        model.eval()
        learner = AnomalyCLIP_PromptLearner(model, PARAMETERS)
        checkpoint = torch.load(PROMPT, map_location='cpu', weights_only=True)
        learner.load_state_dict(checkpoint['prompt_learner'], strict=True)
        learner.eval().requires_grad_(False).cuda()
        model.eval().requires_grad_(False).cuda()
        model.visual.DAPM_replace(DPAM_layer=20)
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            prompts, tokenized, deep = learner(cls_id=None)
            text = model.encode_text_learn(prompts, tokenized, deep).float()
            text = torch.stack(torch.chunk(text, dim=0, chunks=2), dim=1)
            self.text = text / text.norm(dim=-1, keepdim=True)
        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize((SIZE, SIZE), interpolation=transforms.InterpolationMode.BICUBIC, antialias=None),
            transforms.CenterCrop((SIZE, SIZE)), transforms.ToTensor(),
            transforms.Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711)),
        ])

    @torch.inference_mode()
    def predict(self, path):
        started = start_timer()
        with Image.open(path) as source:
            source = source.convert('RGB')
            original_size = source.size
            image = self.transform(source).unsqueeze(0).cuda()
        global_feature, patch_features = self.model.encode_image(image, [6, 12, 18, 24], DPAM_layer=20)
        global_feature = global_feature / global_feature.norm(dim=-1, keepdim=True)
        probabilities = ((global_feature @ self.text.permute(0, 2, 1)) / .07).softmax(-1)
        score = float(probabilities[:, 0, 1].cpu()[0])
        maps = []
        for feature in patch_features:
            feature = feature / feature.norm(dim=-1, keepdim=True)
            similarity, _ = self.library.compute_similarity(feature, self.text[0])
            similarity_map = self.library.get_similarity_map(similarity[:, 1:, :], SIZE)
            maps.append((similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2)
        working = torch.stack(maps).sum(dim=0)[0].float().cpu().numpy()
        working = gaussian_filter(working, sigma=4).astype(np.float32)
        dense = restore_map(working, original_size)
        elapsed = finish_timer(started)
        return score, working, dense, elapsed, None
