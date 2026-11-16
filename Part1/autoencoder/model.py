"""Normal-only compact convolutional autoencoder, separate from the few-shot protocol."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
from time import perf_counter

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from anomaly_baseline.core import load_image
from anomaly_baseline.run import save_csv
from .common import (DATA, ROOT, finish_timer, image_tensor, load_manifest,
                     restore_map, sha256, signature, start_timer, top_fraction_score, write_json)

TRAIN_CONFIG = {
    'method': 'Compact convolutional autoencoder', 'training_seed': 111,
    'short_edge': 448, 'patch_multiple': 14,
    'geometry': 'same PIL bicubic short-edge resize and bottom/right crop as template baseline',
    'channels': [12, 24, 48, 64, 96], 'bottleneck_stride': 32,
    'decoder': 'transposed convolutions; no skip connections; sigmoid RGB output',
    'epochs': 20, 'batch_size': 8, 'learning_rate': .001,
    'optimizer': 'Adam', 'loss': 'mean squared RGB reconstruction error',
    'normal_validation_fraction': .1, 'checkpoint_selection': 'lowest normal validation MSE',
    'calibration': 'first 20 official sorted normal training files, excluded from fit and validation',
    'augmentation': 'none', 'mixed_precision_training': 'CUDA float16 GradScaler',
    'inference_precision': 'FP32', 'test_label_tuning': False,
}


class CompactAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        encoder, decoder = [], []
        widths = [3] + TRAIN_CONFIG['channels']
        for a, b in zip(widths[:-1], widths[1:]):
            encoder.extend([nn.Conv2d(a, b, 4, 2, 1), nn.ReLU(inplace=True)])
        for i, (a, b) in enumerate(zip(widths[:0:-1], widths[-2::-1])):
            decoder.append(nn.ConvTranspose2d(a, b, 4, 2, 1))
            decoder.append(nn.Sigmoid() if i == len(widths) - 2 else nn.ReLU(inplace=True))
        self.encoder, self.decoder = nn.Sequential(*encoder), nn.Sequential(*decoder)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode='replicate')
        return self.decoder(self.encoder(x))[..., :h, :w]


def training_split(category):
    calibration = load_manifest()['categories'][category]['calibration']
    pool = sorted(p.name for p in (DATA / category / 'train/good').iterdir() if p.is_file()
                  and p.name not in calibration)
    # Hold the normal fit/validation split fixed when comparing training seeds.
    permutation = np.random.default_rng(TRAIN_CONFIG.get('split_seed', 111)).permutation(len(pool))
    n_val = max(1, int(round(len(pool) * TRAIN_CONFIG['normal_validation_fraction'])))
    validation = sorted(pool[i] for i in permutation[:n_val])
    fitting = sorted(pool[i] for i in permutation[n_val:])
    assert not set(fitting) & set(validation) and not (set(fitting) | set(validation)) & set(calibration)
    return fitting, validation, calibration


def read_training_image(path):
    return load_image(path)[0]


def batch_tensor(arrays, indices):
    # Batch only identical image dimensions; avoids padded pixels changing the objective.
    return torch.from_numpy(np.stack([arrays[i] for i in indices])).permute(0, 3, 1, 2).to(
        'cuda', dtype=torch.float32).div_(255)


def batches_by_shape(arrays, generator, shuffle):
    buckets = {}
    for i, image in enumerate(arrays):
        buckets.setdefault(image.shape, []).append(i)
    batches = []
    for indices in buckets.values():
        if shuffle:
            generator.shuffle(indices)
        batches.extend(indices[i:i + TRAIN_CONFIG['batch_size']] for i in range(0, len(indices), TRAIN_CONFIG['batch_size']))
    if shuffle:
        generator.shuffle(batches)
    return batches


def train_category(category, output):
    output = Path(output) / category
    output.mkdir(parents=True, exist_ok=True)
    fitting, validation, calibration = training_split(category)
    paths = [DATA / category / 'train/good' / n for n in fitting + validation + calibration]
    split_rows = [{'category': category, 'role': role, 'file': p.relative_to(ROOT).as_posix(),
                   'sha256': sha256(p)} for role, names in [('fit', fitting), ('validation', validation), ('calibration', calibration)]
                  for n in names for p in [DATA / category / 'train/good' / n]]
    settings = {'config': TRAIN_CONFIG, 'category': category, 'input_files': split_rows,
                'source_sha256': sha256(__file__), 'torch': torch.__version__,
                'gpu': torch.cuda.get_device_name(0)}
    run_signature = signature(settings)
    config_path = output / 'training_config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8'))['signature'] != run_signature:
        raise ValueError('Autoencoder training configuration changed; use a new directory.')
    write_json(config_path, {**settings, 'signature': run_signature})
    save_csv(output / 'training_images.csv', split_rows)
    if (output / 'training_complete.json').exists():
        print('Reuse completed autoencoder:', category, flush=True)
        return output / 'best.pt'
    seed = TRAIN_CONFIG['training_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = CompactAutoencoder().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'])
    scaler = torch.amp.GradScaler('cuda')
    first_epoch, best, history = 1, float('inf'), []
    if (output / 'last.pt').exists():
        state = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
        if state['signature'] != run_signature:
            raise ValueError('Checkpoint signature mismatch')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['rng_cpu'])
        torch.cuda.set_rng_state_all(state['rng_cuda'])
        first_epoch, best, history = state['epoch'] + 1, state['best'], state['history']
    print(f'Loading {category}: fit={len(fitting)}, validation={len(validation)}, calibration={len(calibration)}', flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        arrays = list(executor.map(read_training_image, paths[:len(fitting) + len(validation)]))
    train_arrays, val_arrays = arrays[:len(fitting)], arrays[len(fitting):]
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(first_epoch, TRAIN_CONFIG['epochs'] + 1):
        started = perf_counter()
        rng = np.random.default_rng(seed + epoch)
        model.train()
        loss_sum, observations = 0., 0
        for indices in batches_by_shape(train_arrays, rng, True):
            x = batch_tensor(train_arrays, indices)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                reconstruction = model(x)
                loss = F.mse_loss(reconstruction, x)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(indices)
            observations += len(indices)
        model.eval()
        val_sum = 0.
        with torch.inference_mode():
            for indices in batches_by_shape(val_arrays, rng, False):
                x = batch_tensor(val_arrays, indices)
                val_sum += float(F.mse_loss(model(x), x)) * len(indices)
        val_mse = val_sum / len(validation)
        record = {'epoch': epoch, 'train_mse': loss_sum / observations,
                  'validation_mse': val_mse, 'elapsed_seconds': perf_counter() - started,
                  'fit_image_count': len(fitting), 'validation_image_count': len(validation)}
        history.append(record)
        if val_mse < best:
            best = val_mse
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'validation_mse': best,
                        'signature': run_signature, 'config': TRAIN_CONFIG}, output / 'best.pt')
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scaler': scaler.state_dict(), 'epoch': epoch, 'best': best, 'history': history,
                 'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(), 'signature': run_signature}
        torch.save(state, output / 'last.tmp.pt')
        (output / 'last.tmp.pt').replace(output / 'last.pt')
        save_csv(output / 'training_history.csv', history)
        print(f'AE {category} epoch {epoch:02d}/{TRAIN_CONFIG["epochs"]}: train={record["train_mse"]:.6f}, val={val_mse:.6f}, {record["elapsed_seconds"]:.1f}s', flush=True)
    write_json(output / 'training_complete.json', {'signature': run_signature,
               'fit_images': len(fitting), 'validation_images': len(validation), 'calibration_images': len(calibration),
               'epochs': TRAIN_CONFIG['epochs'], 'best_validation_mse': best,
               'training_gpu_peak_memory_mb': torch.cuda.max_memory_allocated() / 1024**2,
               'parameter_count': sum(p.numel() for p in model.parameters())})
    return output / 'best.pt'


class AutoencoderPredictor:
    def __init__(self, checkpoint):
        self.checkpoint = Path(checkpoint)
        self.model = CompactAutoencoder().cuda().eval()
        self.model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'])
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def predict(self, path):
        started = start_timer()
        image, original_size = load_image(path)
        x = image_tensor(image).unsqueeze(0).cuda()
        reconstruction = self.model(x).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        working = np.abs(image.astype(np.float32) / 255 - reconstruction).mean(axis=2)
        working = cv2.GaussianBlur(working, (0, 0), sigmaX=2, sigmaY=2)
        score = top_fraction_score(working)
        dense = restore_map(working, original_size)
        elapsed = finish_timer(started)
        return score, working, dense, elapsed, reconstruction
