# Part 1 and Part 2 optional extensions

These experiments supplement the mandatory template and AnomalyDINO results. They do
not modify those results and do not substitute for Part 3. The course specification,
page 3, identifies convolutional autoencoding and AnomalyCLIP as the two optional methods.

## Environment and commands

Use the existing project environment; do not replace CUDA torch with the upstream
repository's historical torch requirement.

```powershell
.\.venv-part2\Scripts\python.exe -m pip install -r requirements-extensions.txt
git clone https://github.com/zqhang/AnomalyCLIP.git third_party/AnomalyCLIP
git -C third_party/AnomalyCLIP checkout 3911738c0867544f545a076ad78f3f11d9ecbfdf
.\.venv-part2\Scripts\python.exe tools/download_clip_weights.py
.\.venv-part2\Scripts\python.exe Part1/run_autoencoder.py
.\.venv-part2\Scripts\python.exe Part2/run_anomalyclip.py
.\.venv-part2\Scripts\python.exe tools/benchmark_extensions.py
.\.venv-part2\Scripts\python.exe tools/audit_extensions.py
.\.venv-part2\Scripts\python.exe tools/build_extensions_report.py
.\.venv-part2\Scripts\python.exe Part1/demo_autoencoder.py
.\.venv-part2\Scripts\python.exe Part2/demo_anomalyclip.py
```

The runner supports `--categories candle` for a single category and `--train-only`
for autoencoder training without test evaluation. Completed, signature-matched
training and evaluations resume; changed configurations require a fresh output directory.
The production inference classes are shared by batch evaluation and the live demo.
Example command-line prediction:

```powershell
.\.venv-part2\Scripts\python.exe -m Part1.autoencoder.predict --image data/VisA_pytorch/1cls/candle/test/bad/000.JPG
.\.venv-part2\Scripts\python.exe -m Part2.anomalyclip.predict --image data/VisA_pytorch/1cls/candle/test/bad/000.JPG
```

## Part 1: compact convolutional autoencoder

One independent category-specific model with 342,539 parameters and no skip connections.
Five stride-2 encoder convolutions use channels 12,24,48,64,96; a mirrored transposed
convolution decoder reconstructs RGB with a sigmoid output. Replicate padding to a
multiple of 32 is removed from the output, so the input geometry remains the same
aspect-preserving short-edge-448/bottom-right-crop-to-14 geometry as the template method.

All hyperparameters are fixed before accessing test results: 20 epochs, Adam at
0.001, batch size 8, seed 111, RGB MSE objective, no augmentation, AMP FP16 training
with GradScaler. Validation and inference use FP32. No anomaly labels enter training.
Each category excludes the 20 calibration images, then a fixed NumPy seed-111
permutation reserves round(10%) of the remaining normal images for validation.
The checkpoint with the lowest **normal validation MSE** is selected. This validation
criterion measures reconstruction, not anomaly performance; a lower reconstruction
loss does not guarantee higher test AUROC.

| Category | Fitting | Normal validation | Calibration | Official normal training total |
|---|---:|---:|---:|---:|
| candle | 792 | 88 | 20 | 900 |
| capsules | 470 | 52 | 20 | 542 |
| cashew | 387 | 43 | 20 | 450 |
| chewinggum | 390 | 43 | 20 | 453 |
| fryum | 387 | 43 | 20 | 450 |
| pipe_fryum | 387 | 43 | 20 | 450 |
| Total | 2813 | 312 | 120 | 3245 |

This is a normal-only **many-image trained reconstruction baseline**, not a 1/2/4/8-shot
method. Its training sample counts cannot be hidden in a nominal 4-shot comparison.
There is one prespecified training seed, not three runs or a claimed stability estimate.
No best-test seed, epoch or category is selected.

The map is the mean absolute RGB reconstruction residual in [0,1], Gaussian smoothed
with sigma 2 at working resolution. The image score is the mean of the highest
ceil(1% of working pixels) residuals. The 20 excluded normal calibration scores set
the 95th-percentile threshold. Models, best and last checkpoints, optimizer/scaler/RNG
state, per-epoch CSV losses and hashed input lists are stored by category in
`results/part1_autoencoder/`. Reconstruction PNGs are for visualization; the saved
float32 residual maps are used for evaluation, before any display clipping.

## Part 2: official AnomalyCLIP zero-shot extension

Source: https://github.com/zqhang/AnomalyCLIP at
3911738c0867544f545a076ad78f3f11d9ecbfdf. Python and shell source files are unmodified.
The adapter loads local weights, selects our six categories and official test split,
and evaluates original-resolution masks. Windows checkout has an upstream
case-colliding tracked `__pycache__` pair; these Python cache files are not source
and are not used as a model implementation.

The prompt file is **checkpoints/9_12_4_multiscale_visa/epoch_15.pth** (SHA256
415c5dcb52668b8c33fb9c1a351c686d632b919df5b384d63fa9ce7a2338ced4).
The `_visa` suffix indicates its evaluation destination: upstream `train.sh`
trains this branch on **MVTec AD**; `test.sh` evaluates it on VisA. The unsuffixed
directory is the reverse direction and must not be used for our VisA evaluation.
Upstream `Dataset` defaults to `mode='test'`, and `train.py` does not override that:
the released training recipe uses the labeled **MVTec AD test split** as its auxiliary
source training set. This does not use our VisA test split. These prompts were trained
with external source-domain normal and anomalous labels;
they are not a normal-only VisA-fitted model. They are not selected by our test scores.

The frozen OpenAI CLIP ViT-L/14@336px backbone SHA256 is
3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02.
The original Azure CDN hostname no longer resolved during setup; the download uses
the official `openaipublic.blob.core.windows.net` storage object with the same path
and verifies the exact upstream SHA256. Downloads resume by independent byte ranges.

Inference uses upstream prompt settings (12 context tokens, depth 9, deep text
context length 4), DPAM layer 20, feature layers [6,12,18,24], temperature 0.07,
and Gaussian sigma 4. No prompts are trained, edited or selected on VisA. Image
scores are the official **global image-text anomaly softmax probability**, not the
heatmap maximum. Localization sums the four official layer maps, so raw maps range
from 0 to 4. The image and localization outputs therefore need not agree on every case.

The upstream test.py defaults to square 518. Our supplementary run sets this size
to **448 x 448** while retaining the official square resize/normalization and
FP32 inference. The nominal size is 448 like the mandatory methods, but its square
geometry differs from their aspect-preserving inputs. This is explicitly an optional
zero-shot comparison, **not an exact published-518 reproduction or a strictly
matched few-shot comparison**. Inference does not use the target-domain reference
bank: VisA fitting count = 0, VisA reference count = 0; the additional 20 normal
calibration images per category determine only the demo threshold.

## Evaluation and saved results

Each optional method predicts the same complete official test split: 360 normal and
600 anomalous images, 960 unique inputs, all six required categories. No images are
added to the formal test split from calibration. Pixel AUROC includes all normal
pixels with all-zero masks. Working float32 maps are saved losslessly as NPZ;
bilinear original-size restoration precedes exact floating-point AUROC with tie
handling. No per-image min-max normalization or display quantization enters metrics.
Image AUROC is the main metric; image AP and thresholded TP/TN/FP/FN are also recorded.

Each result folder has per-category predictions/calibration/metrics, unified
`predictions_all.csv`, `errors_only.csv`, `input_images.csv`, `summary_by_category.csv`,
`summary_macro.csv`, `metrics_all.json` and `run_config.json`. The combined page,
paired same-image error transitions and fixed example comparisons are in
`results/extensions/`. Its baseline/AnomalyDINO per-image comparisons use the
prespecified 4-shot/seed-0 configuration for illustration, not a best-performing seed.
The context table also shows the mandatory three-reference-group means and std.

Full evaluation inference time includes file decode, preprocessing, representation,
scoring and original-size map restoration, synchronized at CUDA boundaries, excluding
model load, training, calibration, output I/O and metric sorting. Additional fresh
benchmarks run all four methods serially on the same fixed 120 images (10 good and
10 bad per category). Peak inference memory includes the loaded model and its bank;
the optional benchmark unloads other models before measuring each method.

AE heatmaps use a fixed display range 0..0.20 and CLIP heatmaps 0..4 across all images.
Display clipping does not change any numerical score or metric. Ground-truth-guided
zoom boxes in the report are visualization only and never re-enter inference.

For the progress recording, explain the template failure mechanism, show frozen
feature retrieval's improvement, and use these two extensions to illustrate different
normality models and supervision assumptions. Their measured results and failures
matter more than claiming that every added method beats AnomalyDINO.
