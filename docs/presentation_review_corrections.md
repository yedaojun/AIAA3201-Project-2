# Presentation evidence corrections (2026-11-16)

## Autoencoder: old public code versus new local experiments

The previous public commit was `3b6b2a2`; it only supported the historical training seed111. The local seed0/1/2 training and evaluation records existed but had not been published. The updated entry point accepts `--seed`, fixes `split_seed=111` and changes training initialization/order. All 18 category-model configs and 20-epoch histories are now in `published_results/optional_multiseed`, with their original-file SHA-256 mapping. Original seed111 results are preserved and explicitly excluded from the three-seed mean. Mean image/pixel AUROC: 82.74% / 69.59%; sample SD: 1.84 / 4.49 percentage points.

## AnomalyCLIP: deterministic repeatability, not few-shot stability

Three actual inference runs were saved, but model, external MVTec-trained prompts and target inputs stayed fixed. There is no target reference sampling. Identical results are therefore presented as repeated deterministic inference. The video summary displays one fixed-model result (89.54% image, 95.81% pixel) and no zero-SD stability claim. Detailed repeats remain available in the report.

## Input-specific patch geometry

For candle bad/000.JPG, the saved model input is **H×W = 448×490** and its saved raw patch map is **32×35**. Patch spacing14 is measured in this resized input, not in the original1284×1168 image. A short edge448 alone does not fix the long-side patch count for arbitrary images. Dilution by coarse patches remains a possible explanation of this false negative, not a demonstrated cause.

## Efficiency

The measurement pages include the separate serial 120-input, same-hardware latency check: RGB template 28.2ms/image (CPU); DINO 211.8ms/image, peak allocated GPU120.5MiB. GPU is RTX4060 Laptop. These are 4-shot/group0 timings, separate from the three-group AUROC means. The full table also reports AE and CLIP; measured latency excludes setup/training and result-file/metric I/O, and includes decode, preprocessing, scoring and map restoration.

The final voice edit still awaits four replacement recordings. The pre-existing voice MP4 is preserved; corrected scene sources and offline HTML do not by themselves constitute the final re-export.
