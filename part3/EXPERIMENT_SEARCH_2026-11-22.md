# Part 3: follow-up experiment search (22 November 2026)

## Question and protocol

The review correctly questions whether concatenating DINOv2 blocks 8 and 11 is the best main improvement. We investigated **spatial resolution and image-score aggregation** while keeping the official Part 2 reproduction intact. The image-scale experiment uses the same frozen `dinov2_vits14` weights, official agnostic preprocessing, reference filenames, 1/2/4/8-shot reference groups 0/1/2, 20 disjoint normal calibration images, and complete six-category VisA test split. The 672 branch changes only the resized short edge from 448 to 672. The fixed multi-scale method averages `score_448 / threshold_448` and `score_672 / threshold_672` with weights 0.5/0.5. Its final threshold is the 95th percentile of the same 20 paired normal calibration images. Its pixel map is the same equal average of the original-size, threshold-normalized maps. No anomaly labels or test images are used for its model, reference bank, weights, or threshold.

**Updated 23 November 2026:** The 672 branch and fixed fusion now cover **all 1/2/4/8 shots, groups 0/1/2, and six classes** (72 paired settings). This remains a post-hoc exploratory candidate developed after inspecting the same VisA test split. Reference-group replication is not an independent holdout, so it should not replace the pre-fixed A experiment as an unbiased method-selection result.

| References | Official image AUROC | A image AUROC | 672 image AUROC | Fixed fusion image AUROC | Official errors | Fusion errors |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 94.61 ± 1.03% | 94.74 ± 0.99% | 94.97 ± 0.38% | **95.49 ± 0.54%** | 442 | 404 |
| 2 | 95.33 ± 0.57% | 95.47 ± 0.54% | 95.76 ± 0.38% | **96.12 ± 0.39%** | 384 | 368 |
| 4 | 96.15 ± 0.49% | 96.45 ± 0.38% | 96.29 ± 0.24% | **96.74 ± 0.27%** | 365 | 362 |
| 8 | 96.62 ± 0.70% | 96.85 ± 0.74% | 96.82 ± 0.56% | **97.22 ± 0.60%** | 340 | 321 |

Errors are FP+FN over 2,880 category/group test decisions per reference count. `±` is the sample SD of the three group-level six-category macro scores. Each method uses the same reference filenames and disjoint normal calibration images. Full per-setting CSV and references are in `../results/part3/multiscale_fusion/all_shots_category_seed.csv` and `all_shots_macro_by_seed.csv`.

The exact original-resolution fixed-fusion pixel AUROC across the same 1/2/4/8-shot, three-group settings is **98.19 ± 0.08% / 98.32 ± 0.08% / 98.46 ± 0.05% / 98.56 ± 0.03%**, compared with official **97.76% / 97.92% / 98.09% / 98.21%**. Fixed-fusion pixel AP is **48.92 ± 0.56% / 50.26 ± 1.13% / 51.89 ± 0.65% / 52.59 ± 0.87%**. Official pixel AP was computed at four shots only, so the other AP entries are not described as paired AP gains. All normal test images contribute zero masks. The per-category/group values are in `../results/part3/multiscale_fusion/pixel_metrics_all_shots_3seeds.json`.

## What the diagnostic checks found

- Of 600 distinct anomalous test images, **439** have a ground-truth mask smaller than 0.5% of the image, **114** have 0.5–2%, and **47** have more than 2%. Original DINO's three-group, six-category macro image AUROC and normal-calibrated recall in the tiny group are **95.18% / 85.66%**. The medium group is **99.87% / 99.25%**; large-defect results are 100% on the four categories that contain such examples. This is an association with defect size, not proof that patch resolution causes every miss.
- The exact original-size pixel-AP audit now covers all three four-shot reference groups. Within tiny defects, macro category/group AP is **30.08%** for official DINO, **31.21%** for A, and **35.43%** for fixed 448+672 fusion. Within medium defects it is 58.92% / 59.04% / 62.68%; within large defects (four categories only) it is 75.68% / 78.09% / 78.92%. Every bucket includes the category's normal test images with zero masks. Different bucket prevalences make the absolute AP values unsuitable for cross-bucket ranking; compare methods within a bucket.
- Retrospectively rescoring the saved four-shot maps with a top-0.5% image score instead of official top-1% changes original DINO image AUROC from **96.15% to 96.46%** across the three groups; the two-layer method changes from **96.45% to 96.65%**. Subtracting each image's foreground median before peak aggregation *reduces* AUROC to about **95.86–95.90%**. These alternatives were screened using test labels and have not been independently validated or given new normal-calibrated thresholds. The suggested median subtraction is therefore not a supported improvement here.
- The 20-normal-image threshold is sensitive. A leave-one-calibration-image-out diagnostic changes original DINO's candle false positives from 16 to a range of 16–25; the fixed multi-scale candle result varies from 19 to 19–50. These are sensitivity ranges, not confidence intervals or selected thresholds. The calibrated deployment decision remains a harder problem than improving AUROC.
- In a separate 200-resample normal-only bootstrap over all 18 four-shot category/group settings, the empirical 95th percentile, `mean + 1.645 × sample SD`, and fitted Gumbel 95th-percentile rules produced different FP/FN totals: official DINO **154/211**, **152/194**, **142/190**; fixed 448+672 **188/174**, **186/178**, **183/180**. These are retrospective test outcomes under candidate estimators, **not** a test-selected deployment rule. Gaussian and Gumbel assumptions are unverified with 20 normals, and threshold-SD values on the two score scales cannot be directly compared.
- A pure-DINO selective audit (no AnomalyCLIP) ranks four-shot decisions by normal-calibrated threshold margin. At **90% automatic coverage**, accepted-image error is **10.39%** for original 448 and **9.25%** for fixed 448+672; the latter queues 53 of its 174 false negatives for human review across repeated groups. Queued misses are not true positives. Test-derived recall at at most 1%/5% FPR is **70.22%/83.56%** for official and **71.22%/84.67%** for fusion; these are evaluation points only, not deployed thresholds.

## Fixed image-scale comparison

All numbers are six-category macro averages, four normal references, three reference groups. `±` is the sample standard deviation of the three group-level macro scores, **not** a statistical confidence interval.

| Method | Image AUROC | Pixel AUROC | Pixel AP | FP / FN over 2,880 paired test decisions |
| --- | ---: | ---: | ---: | ---: |
| Official 448 Part 2 | 96.15 ± 0.49% | 98.09 ± 0.03% | 47.93 ± 0.48% | 154 / 211 |
| A: DINO blocks 8 + 11 | 96.45 ± 0.38% | 98.12% (all 3 groups) | 48.59 ± 0.30% | 182 / 199 |
| 672 short edge alone | 96.29 ± 0.24% | 98.30% (group 0 only) | 51.85% (group 0 only) | 198 / 191 |
| Fixed 448 + 672 fusion | **96.74 ± 0.27%** | **98.46 ± 0.05%** | **51.89 ± 0.65%** | 188 / 174 |
| Original 448 + pretrained AnyUp score-map restoration | 96.15% (unchanged) | 98.64% (all 3 groups) | **57.25 ± 0.79%** | 154 / 211 (unchanged) |

The A pixel AP now covers all three groups. The 672-alone pixel metrics remain group 0 only. The AnyUp pixel AUROC is from its complete three-group, four-shot result. Pixel AP is computed over exact original-resolution pixels, including all normal images with zero masks.

Per-category image AUROC, averaged over the three fixed reference groups:

| Category | Official 448 | Fixed 448 + 672 | Change |
| --- | ---: | ---: | ---: |
| candle | 92.80% | 94.00% | +1.20 pp |
| capsules | 98.06% | 99.06% | +1.00 pp |
| cashew | 93.31% | 94.96% | +1.65 pp |
| chewinggum | 98.95% | 98.92% | −0.03 pp |
| fryum | 97.41% | 97.79% | +0.38 pp |
| pipe_fryum | 96.38% | 95.71% | −0.67 pp |

For tiny defects, the fixed fusion raises macro image AUROC from **95.18% to 95.96%** and calibrated recall from **85.66% to 88.17%**. The medium and large groups were already close to saturation. On the full test split, the fusion changes **49 FN→TP** and **19 FP→TN**, but also **12 TP→FN** and **53 TN→FP**. Total errors fall only from **365 to 362**. Its gain is clearer in ranking than in the present fixed-threshold OK/NG decision. At retrospective test-set FPR at most 5%/10%, macro recall is **84.67%/91.61%**, versus **83.56%/89.89%** for official DINO. These test-derived operating points are evaluation metrics, not deployment thresholds.

The high-resolution branch by itself is inconsistent: chewinggum falls in all three groups, and pipe_fryum falls in each group. The fixed fusion reduces those regressions but does not remove them. The familiar `candle bad/000.JPG` becomes a TP at 672 alone in group 0, yet remains an FN after the calibrated fusion; `cashew bad/063.JPG` remains an FN. Neither case can be advertised as solved by the proposed final fusion.

On a separate **paired, warmed 120-query** timing pass with both frozen DINO models resident, mean query time is **235 ms/image** for 448, **599 ms/image** for 672, and **833 ms/image** for the full two-scale path. These times include image decode, fresh feature extraction, CPU FAISS retrieval, scoring, and map restoration, and exclude model/reference setup and file output. Thus the 0.59-pp image-AUROC gain costs about **3.55×** the 448 query time in this implementation. This makes the full fusion a scientific improvement but a weak practical trade-off unless accuracy is the primary objective.

## Exploratory conditional second pass

A fixed two-stage rule was also checked: always score at 448, and call 672 only when `abs(score_448 / normal_threshold_448 − 1) <= 0.25`. Triggered scores use the same 50/50 threshold-normalized fusion; other scores remain the normalized 448 score. The final decision threshold is again calibrated on the same 20 normal images. The band was fixed before evaluating this candidate, but the *candidate itself* arose after reviewing the full-scale test results and is not independently validated.

At four shots across the three groups, its image AUROC is **96.68% ± 0.34%**, versus **96.74% ± 0.27%** for always running both scales and **96.15% ± 0.49%** for 448. It triggers the second pass for **44.75%** of all 2,880 test decisions and has **186 FP / 175 FN** (361 errors), compared with full fusion's **188 FP / 174 FN** (362 errors). On the separate 120-query timing list, its trigger rate is **57.5%**, and summing the paired measured 448 time plus measured 672 time only for triggered images yields an **estimated 600 ms/image**, versus 833 ms for full fusion. This is a component-time estimate, not a direct conditional-path benchmark.

The conditional heatmap retains the 448 map when the second pass is skipped. Its three-group pixel AP is only **48.39% ± 0.58%**, barely above official 448's **47.93% ± 0.48%** and well below full fusion's **51.89% ± 0.65%**. Thus it is an image-decision/efficiency ablation, not a strong localization method.

## Decision from the search

1. **Promising but expensive detection candidate:** fixed 448+672 multi-scale fusion. It improves image AUROC at all four reference counts, by **0.88/0.79/0.59/0.60 pp** over official DINO for 1/2/4/8 shots. At four shots it improves pixel AP by **3.95 pp**, but reduces thresholded errors by only three and costs 3.55× query time. The full sweep strengthens its replication across reference groups; it does not remove same-test-set selection bias.
2. **Stronger localization branch:** pretrained AnyUp applied to the original 448 score map raises three-group pixel AP by **9.32 pp**, with unchanged image scores. A+AnyUp now has its own complete three-group four-shot AP of **58.69 ± 0.38%**, versus A's **48.59 ± 0.30%** (+10.10 pp). Both use an extra pretrained model and should be reported separately. Feature-level AnyUp *before* nearest-neighbor matching remains untested.
3. **Keep the conditional second pass as an efficiency ablation.** It preserves most of full fusion's image AUROC with an estimated lower cost, but loses almost all of the full fusion's pixel-AP gain. Do not describe it as a solved small-defect localization method.
4. **Do not pursue the proposed median subtraction as the main method.** Its observed image ranking is worse. PCA compression of the 768-dimensional A descriptor is an efficiency idea, not yet an accuracy improvement. The proposed leave-one-reference-image normal self-distance is undefined at one shot without a different construction and has not been tested.

## Reproducibility files

- Frozen reference choices: `../configs/reference_manifest.json`.
- Size audit: `../results/part3/defect_size_audit/summary.csv` and `category_seed.csv`.
- Aggregation retrospective scan: `../results/part3/aggregation_screen/macro_mean_sd.csv`.
- Three-group AnyUp AP: `../results/part3/anyup_pixel_ap_4shot_3seeds.json`.
- Three-group A+AnyUp AP/AUROC: `../results/part3/a_anyup_pixel_ap_4shot_3seeds.json`.
- All-shot 448/672 comparison and seed dispersion: `../results/part3/multiscale_fusion/all_shots_summary.json` and `all_shots_macro_by_seed.csv`.
- All-shot fixed-fusion original-resolution pixel metrics: `../results/part3/multiscale_fusion/pixel_metrics_all_shots_3seeds.json`.
- Size-bucket pixel AP (all three groups): `../results/part3/defect_size_audit/pixel_ap_by_size_4shot.json`.
- Three-estimator normal-only calibration audit: `../results/part3/calibration_methods_4shot_3seeds/summary.csv`.
- Pure-DINO selective-review audit: `../results/part3/pure_dino_reliability_4shot_3seeds/summary.json`.
- 672 per-image scores, maps, calibration files and protocol: `../results/part3/high_resolution_pilot/`.
- Fixed fusion per-image scores and maps reconstructed from both branches: `../results/part3/multiscale_fusion/`, including `three_seed_summary.json`, `paired_settings_4shot_3seeds.csv`, `per_image_transitions_4shot_3seeds.csv`, and `pixel_metrics_4shot_3seeds.json`.
- Paired runtime boundary: `../results/part3/multiscale_fusion/benchmark_4shot_seed0.json`.
- Conditional second pass and paired-time estimate: `../results/part3/selective_multiscale/`.
- Summary comparison figure: `../docs/figures/part3_experiment_search.png`.
- Leave-one-calibration-image-out audit: `../results/part3/calibration_sensitivity_4shot_seed0.json`.

## Related primary sources

- [AnyUp: Universal Feature Upsampling](https://arxiv.org/abs/2510.12764) motivates learned feature/score-map restoration. Our tested branch restores anomaly *scores* after retrieval, so it cannot change image-level scores.
- [RadioCore: Few-Shot Industrial Anomaly Segmentation with Multi-Scale Radio ViT Features](https://openaccess.thecvf.com/content/CVPR2026W/VISION26/html/Ali_RadioCore_Few-Shot_Industrial_Anomaly_Segmentation_with_Multi-Scale_Radio_ViT_Features_CVPRW_2026_paper.html) motivates a multi-scale feature-bank direction; its method and numbers are not claimed as ours.
