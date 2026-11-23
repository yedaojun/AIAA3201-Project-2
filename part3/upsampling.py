"""Apply pretrained AnyUp attention to existing patch anomaly scores."""

import subprocess

import numpy as np
import torch

from experiment_utils.common import ROOT, sha256

REPOSITORY = ROOT / 'third_party/anyup'
REVISION = '351807a9c4287368732cc247f26c7c81c9139af4'
CHECKPOINT = ROOT / 'models/anyup/anyup_paper.pth'
CHECKPOINT_SHA256 = '9d035c0f27114a6f32bdd3d8ed93b6cd39dad4b8f8bf94e69fddbfbf022901b2'
CHECKPOINT_URL = 'https://github.com/wimmerth/anyup/releases/download/checkpoint/anyup_paper.pth'
SCALE = 2


def load_upsampler():
    revision = subprocess.check_output(['git', '-C', str(REPOSITORY), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError(f'Use the documented AnyUp revision {REVISION}')
    if not CHECKPOINT.is_file() or sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError(f'Expected verified AnyUp weights at {CHECKPOINT}')
    model = torch.hub.load(str(REPOSITORY), 'anyup', source='local', pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu', weights_only=True))
    return model.eval().requires_grad_(False).cuda()


@torch.inference_mode()
def restore_scores(upsampler, image, features, maps):
    """Use RGB and DINO features to compute weights, then interpolate the maps.

    The pinned AnyUp decoder computes attention from Q/K independently of V.
    Substituting scores only at the decoder input preserves the pretrained
    guidance. Channels contain one or more scored reference configurations.
    """
    channels, height, width = maps.shape
    if features.shape[0] != height * width:
        raise ValueError('Patch maps and DINO features must share the same grid')
    guidance = torch.from_numpy(features.T.copy()).reshape(1, -1, height, width).cuda()
    values = torch.from_numpy(maps.astype(np.float32)).unsqueeze(0).cuda()
    target = (height * SCALE, width * SCALE)

    def replace_values(module, inputs):
        query, key, _ = inputs
        return query, key, values

    with upsampler.cross_decode.register_forward_pre_hook(replace_values):
        result = upsampler(image.unsqueeze(0).cuda(), guidance, output_size=target, q_chunk_size=512)
    output = result.squeeze(0).cpu().numpy()
    if output.shape != (channels, *target) or not np.isfinite(output).all():
        raise ValueError('Invalid learned score map')
    # Attention is a weighted average, so it should not create out-of-range scores.
    if np.any(output.min((1, 2)) < maps.min((1, 2)) - 1e-6) or np.any(output.max((1, 2)) > maps.max((1, 2)) + 1e-6):
        raise ValueError('Learned scores exceed the range of their patch values')
    return output


def intermediate_bilinear(maps):
    return np.stack([
        torch.nn.functional.interpolate(torch.from_numpy(m).reshape(1, 1, *m.shape),
                                       scale_factor=SCALE, mode='bilinear', align_corners=False)[0, 0].numpy()
        for m in maps
    ])
