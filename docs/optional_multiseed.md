# Optional extensions: seeds 0, 1, 2

Both optional methods completed three actual runs on the same six VisA categories and all 960 official test images. The original seed111 experiments remain separate and do not enter the new mean or standard deviation.

| Method | Seed | Image AUROC (%) | Pixel AUROC (%) |
| --- | --- | ---: | ---: |
| Autoencoder | 0 | 84.60 | 64.50 |
| Autoencoder | 1 | 80.93 | 71.32 |
| Autoencoder | 2 | 82.69 | 72.96 |
| **Autoencoder mean ± sample SD** | **0, 1, 2** | **82.74 ± 1.84** | **69.59 ± 4.49** |
| AnomalyCLIP | 0 | 89.54 | 95.81 |
| AnomalyCLIP | 1 | 89.54 | 95.81 |
| AnomalyCLIP | 2 | 89.54 | 95.81 |
| **AnomalyCLIP mean ± sample SD** | **0, 1, 2** | **89.54 ± 0.00** | **95.81 ± 0.00** |

Each seed first receives an equal six-category macro average. The mean and sample standard deviation then use those three macro values (`ddof=1`). Standard deviations in this table are percentage points, not confidence intervals.

The autoencoder uses the same 2,813 normal fitting images, 312 normal validation images and 120 calibration images across all seeds. Each seed trains six separate category models for 20 epochs, so the new study trained 18 models. Initialization and training order change. The fit/validation split stays fixed at split_seed111. Checkpoint selection uses normal validation MSE only.

AnomalyCLIP repeats inference using the same verified CLIP backbone and external MVTec-supervised prompt checkpoint. It does not train prompts or a backbone on VisA. It uses no VisA reference bank, but its decision thresholds still use the same 20 normal calibration images per category. It uses square448 geometry, whereas the required methods preserve aspect ratio. These are separately reported supplementary conditions, not matched few-shot trials.

The cross-seed audit found a maximum AnomalyCLIP image-score difference of 0. Identical saved anomaly-map values across the three seeds: **True**. These are deterministic inference repeats, not a few-shot stability test. Zero observed variation does not establish robustness to new images or other datasets.

Each optional study contains 18 category-level runs and 2,880 test predictions. Combined with the 144 required-method runs, the presentation evidence contains 180 category-level runs and 28,800 test predictions on 960 unique test images.

## Reproduce

```powershell
.\.venv-part2\Scripts\python.exe tools\run_optional_multiseed.py
.\.venv-part2\Scripts\python.exe tools\audit_optional_multiseed.py
```

Use a new output directory if method settings or source fingerprints differ. The run entry points reject changed settings before overwriting saved experiment metadata. Individual commands also accept `--seed 0`, `--seed 1` or `--seed 2` under `Part1/run_autoencoder.py` and `Part2/run_anomalyclip.py`.

## Saved records

Each parent study directory has `summary_by_seed.csv`, `summary_macro.csv`, `runs.csv`, `input_images.csv`, `predictions_all.csv` and `errors_only.csv`. Each `seed_N/category/` keeps its own metrics, calibration scores, per-image predictions and maps. The autoencoder also keeps checkpoints, the 20-row training history and input-image roles with hashes.

## Public evidence

Exact configuration, per-category metrics, training-history and fixed-split copies are in [published_results/optional_multiseed](published_results/optional_multiseed/README.md), with SHA-256 provenance. The old seed111 summaries remain historical records.
