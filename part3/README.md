# Part 3 — Controlled improvements and additional directions

The complete A image- and pixel-level sweep is recorded in [PILOT_RESULTS.md](PILOT_RESULTS.md). The completed D and F comparisons, their limitations, risk–coverage analysis and paired runtime measurement are in [DF_RESULTS.md](DF_RESULTS.md). The six-category, four-shot/group-0 screening results for B, C and E are in [BCE_PILOT.md](BCE_PILOT.md). The [follow-up experiment search](EXPERIMENT_SEARCH_2026-11-22.md) evaluates defect-size strata, calibrated 448+672 fusion across 1/2/4/8 shots and three groups, a conditional second pass, and three-group pixel AP for AnyUp and A+AnyUp. The PDF requires one justified improvement; these branches are separate experiments with different selection and evaluation scopes.

The experiment tests whether an image-guided pretrained upsampler can improve localization of small defects on VisA. The backbone stays frozen. The official test split, 1/2/4/8 normal reference images, three reference groups, Part 2 memory banks, raw patch anomaly scores, and image-level calibration are held fixed. No test masks or labels enter map generation or parameter selection.

| Method | First resizing stage | Final restoration |
| --- | --- | --- |
| Official AnomalyDINO | None | Bilinear resize to the original image size, then Gaussian smoothing with σ = 4 |
| Two-step bilinear control | 2× intermediate grid by bilinear interpolation | Same final restoration |
| AnyUp score upsampling | 2× intermediate grid using pretrained AnyUp attention | Same final restoration |

The AnyUp checkpoint was trained for feature upsampling with DINOv2 ViT-S features. Here the pretrained model computes attention weights from the aligned RGB input and DINOv2 features. Its attention decoder aggregates existing anomaly scores as values. This project adaptation keeps the retrieval scores fixed and tests the spatial redistribution of those scores. The factor of two was fixed before evaluating test masks or labels. Changes in localization do not change the baseline image AUROC or the calibrated OK/NG decisions.

Evaluation computes exact pixel AUROC at original image size, including all valid background and all normal images. The report saves paired differences for the same category, shot count and reference group. A secondary diagnostic records whether the strongest heatmap response hits the ground-truth mask. The exploratory small-defect subset covers masks occupying at most 0.1% of an image. These diagnostics complement AUROC and do not replace it. The five displayed cases include the candle case shown in the presentation and four previously recorded baseline errors. Heatmaps in each row share a raw score scale. Zooms are used only for display.

## Run locally

With the existing Part 2 CUDA environment, dataset and results present, run from the project directory:

```powershell
git clone https://github.com/wimmerth/anyup.git third_party/anyup
git -C third_party/anyup checkout 351807a9c4287368732cc247f26c7c81c9139af4
New-Item -ItemType Directory -Force models/anyup | Out-Null
curl.exe -fL -o models/anyup/anyup_paper.pth https://github.com/wimmerth/anyup/releases/download/checkpoint/anyup_paper.pth
.\.venv-part2\Scripts\python.exe -m part3.run generate
.\.venv-part2\Scripts\python.exe -m part3.run evaluate
.\.venv-part2\Scripts\python.exe -m part3.benchmark
.\.venv-part2\Scripts\python.exe -m part3.report --figures
```

Skip cloning and downloading if the pinned source and weights are already present. Generation checks the checkpoint SHA-256 and reference manifest. Map generation and evaluation resume from completed output files. For a shorter evaluation pass, use `evaluate --shots 4 --seeds 0`. The benchmark compares the methods on the same 120 fixed images, including fresh feature extraction and map restoration. The saved timing notes specify the device and timing boundary.

Results go to `results/part3/anyup_score_maps/`. `protocol.json` records the source revision and SHA-256 hashes. `maps/` stores both experimental grids, `paired/` stores exact per-configuration metrics and localization diagnostics, and the report produces `summary.json`, CSV tables, and `difficult_cases.png`.

Model and checkpoint source: [AnyUp official repository](https://github.com/wimmerth/anyup).

## Direction A pilot: two frozen DINOv2 layers

`multilayer.py` concatenates unit-length patch features from zero-based transformer blocks 8 and 11 before the same nearest-normal-neighbor search. It uses the original final-layer PCA foreground mask, the same resolution and preprocessing, the same rotated reference images, and the same 20 normal calibration images. The script checks that its final-layer vectors and mask match the Part 2 extractor before running. No VisA weights are trained, and the layer pair was fixed before inspecting test results.

```powershell
.\.venv-part2\Scripts\python.exe -m part3.multilayer --shots 4 --seeds 0
.\.venv-part2\Scripts\python.exe -m part3.multilayer_report --pixels
```

The default command is a six-category 4-shot/group-0 pilot. Other shot counts and groups can be selected with `--shots` and `--seeds`. Each run saves calibrated scores, decisions, original-resolution-reconstructible patch maps, timing, peak GPU allocation, and exact reference filenames under `results/part3/multilayer/`.

## Direction D and F pilots: calibrated fusion and review

`fusion_selective.py` joins the 960 same-image saved DINO and AnomalyCLIP predictions for each of the 12 reference settings. Each score is centered by the median and scaled by the interquartile range of its own 20 normal calibration scores, then transformed with `arcsinh`. The fixed pilot fusion uses equal weights; its 95th-percentile decision threshold comes from the 20 paired normal calibration examples. The `REVIEW` output is triggered when the fused score lies within half a calibration interquartile range of that threshold. Risk–coverage also ranks samples by that margin or model disagreement. Test labels are read only to calculate outcomes and metrics. A retrospective test-set 10% FPR operating point is **only** an evaluation metric and is never used as a deployed threshold.

An additional exploratory gate assigns CLIP weight `0.5 × exp(-|z_DINO - t_DINO|)`, where `t_DINO` is the DINO normal-calibration 95th percentile after the same normalization. CLIP contributes most when DINO is close to its own decision boundary. The gate has its own normal-only calibrated 95th-percentile threshold and optional margin-based `REVIEW` output. It was added after inspecting the first equal-fusion pilot, so comparisons between the two candidate fusion rules are exploratory and not an untouched model-selection test.

The margin is a review heuristic, not a calibrated probability that a prediction is correct. Review reduces the number of automatically decided images and depends on a human checking those cases. Retrospective FNR-at-FPR curves are reported for each coverage level so that this trade-off is visible.

```powershell
.\.venv-part2\Scripts\python.exe -m part3.fusion_selective
.\.venv-part2\Scripts\python.exe -m part3.fusion_report
.\.venv-part2\Scripts\python.exe -m part3.compare_d_f
.\.venv-part2\Scripts\python.exe -m part3.fusion_benchmark
```

The image-level results and per-image decisions are in `results/part3/fusion_selective/`; the paired A/D/F audit and runtime records are in `results/part3/df_comparison/`. The figures cover all four shot counts. This is an exploratory image-score fusion and review policy; it does not claim pixel-level fusion. AnomalyCLIP uses MVTec AD anomaly supervision and square 448 input, while the DINO method uses VisA normal references and aspect-preserving 448 input. The saved three CLIP entries are deterministic repeats of the same model, so CLIP variability is not inferred from DINO reference-group changes.

Once a two-layer configuration has finished, the same fixed fusion and review rule can be applied to its saved scores with `python -m part3.multilayer_fusion`. The command writes paired A versus A+D/F results for every completed setting.

After the full A sweep and a refreshed A+D/F fusion, `python -m part3.transitions` writes a per-image CSV with the unchanged Part 2 reference filenames and each branch's score, threshold and TP/TN/FP/FN transition. It also writes counts of corrections and regressions for each complete shot/group setting.

For a single-image CLI demo with an input, image score, OK/NG/REVIEW decision, and DINO heatmap:

```powershell
.\.venv-part2\Scripts\python.exe -m part3.demo --image data/VisA_pytorch/1cls/candle/test/bad/000.JPG --category candle --shots 4 --seed 0 --mode fusion_review --map-restoration anyup
```

The fused score controls the image decision. The displayed heatmap comes from the two-layer DINO branch, optionally restored by AnyUp. Its fixed display range of 0 to 0.4 changes visualization only, never AUROC or thresholding.

## Exploratory composition of A and learned map restoration

`composed.py` takes the completed two-layer patch anomaly scores, keeps their image scores and thresholds, and applies the same fixed 2× AnyUp map restoration used in the earlier independent map experiment. Its exact original-resolution pixel AP/AUROC now covers all three four-shot groups and the complete normal/anomalous test split. The combination was proposed after reviewing the individual branch pilots, so it is exploratory rather than an untouched validation result.

```powershell
.\.venv-part2\Scripts\python.exe -m part3.composed generate --shot 4 --seed 0
.\.venv-part2\Scripts\python.exe -m part3.composed evaluate --shot 4 --seed 0
.\.venv-part2\Scripts\python.exe -m part3.a_anyup_ap_seeds
```

## Post-hoc image-scale and reliability follow-up

The frozen 672-pixel short-edge branch keeps the original AnomalyDINO feature extractor and uses the fixed reference manifest. `high_resolution_sweep.py` caches unchanged query features but performs an exact normal-bank retrieval for every shot/group; six saved four-shot scores per category verify the cache path. `multiscale_fusion.py` combines 448 and 672 scores with fixed equal weights after normal-threshold scaling. Both branches retain the same 20 disjoint normal calibration files. A complete sweep has 72 category/shot/group settings; only the 4-shot paired timing comparison is an end-to-end 120-query benchmark.

```powershell
.\.venv-part2\Scripts\python.exe -m part3.high_resolution_sweep
.\.venv-part2\Scripts\python.exe -m part3.multiscale_fusion --shots 1 2 4 8 --seeds 0 1 2
.\.venv-part2\Scripts\python.exe -m part3.scale_sweep_summary
.\.venv-part2\Scripts\python.exe -m part3.scale_transitions_all_shots
.\.venv-part2\Scripts\python.exe -m part3.multiscale_full_pixel
.\.venv-part2\Scripts\python.exe -m part3.defect_size_pixel_ap
.\.venv-part2\Scripts\python.exe -m part3.calibration_method_audit
.\.venv-part2\Scripts\python.exe -m part3.pure_dino_reliability
```

The scripts save per-image scores, maps, decisions, reference filenames, group-level summaries, area-stratified pixel AP, and normal-only threshold diagnostics under `results/part3/`. The new scale and review ideas were developed after inspecting VisA test outcomes; their matched three-group replication is not an independent holdout. `REVIEW` means human work remains, not that a miss has been automatically corrected.
