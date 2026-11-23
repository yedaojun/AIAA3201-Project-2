# Part 3 directions D and F: completed comparison

The Project 2 brief asks for **one justified Part 3 improvement**. A–F are alternatives, not six mandatory tasks. We nevertheless evaluated A, D and F to decide which result is strongest and most defensible. This file records the complete D/F image-level sweep; [PILOT_RESULTS.md](PILOT_RESULTS.md) also records the independent A and AnyUp localization experiments.

## Protocol and scope

- Six VisA categories; the same 1/2/4/8 normal reference filenames and reference groups 0/1/2 as Part 2. Each method is assessed on the same 960 distinct test images per shot/group. There are 72 paired category/shot/group evaluations and 11,520 prediction rows across the repeated settings.
- **D equal fusion:** combine normalized Part 2 AnomalyDINO image scores and fixed official AnomalyCLIP image scores with equal weights. Each score is normalized with its own 20 normal calibration images; the combined OK/NG threshold is the 95th percentile of their paired, normal-only fused scores. No VisA test label selects a weight or threshold.
- **D exploratory gate:** CLIP receives more weight when the DINO score approaches its normal-calibrated threshold. This gate was added after inspecting the equal-fusion pilot and is consequently an exploratory ablation, not a winner chosen on an independent holdout.
- **F selective prediction:** a normally calibrated band around the fused threshold outputs `REVIEW`; no automatic OK/NG decision is claimed for those images. Risk–coverage curves also rank ambiguous images retrospectively at fixed 70/80/90/100% automatic coverage. Those fixed-coverage points estimate workload trade-offs and are not deployable thresholds.
- The CLIP model was pretrained with labeled **MVTec AD anomalies** and takes square 448 inputs; DINO uses VisA normal references and aspect-preserving short-edge 448. Thus D is an optional, externally supervised comparison. Its three rows per shot reflect **DINO reference groups**, not three independently trained CLIP models. D/F only changes the image-level score and decision; the displayed pixel heatmap remains DINO's. No pixel-level fusion or localization gain is claimed.

## Direction D: image-level ranking

Six-category macro image AUROC, averaged within each reference group and then over three groups:

| Normal references | Part 2 DINO | A: two DINO layers | D: equal fusion | D: exploratory gate |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 94.61% | 94.74% | **95.53%** | 95.52% |
| 2 | 95.33% | 95.47% | 95.94% | **96.08%** |
| 4 | 96.15% | 96.45% | **96.57%** | 96.53% |
| 8 | 96.62% | 96.85% | 96.86% | **97.03%** |

Equal fusion exceeds A by 0.79, 0.47, 0.11 and 0.01 percentage points at 1/2/4/8 shots. The four-shot advantage is small: equal fusion beats A in only **9 of 18** category/group comparisons, and eight-shot in **8 of 18**. The gate has the highest eight-shot macro AUROC but lacks an untouched validation set. These findings do not justify calling D a fair few-shot replacement for A, because CLIP contributes external anomaly supervision and different image geometry. A is still the controlled Part 2 improvement; D is a potentially useful additional system.

Four-shot image AUROC by product category, averaged across the three DINO reference groups:

| Category | Part 2 | A | D equal | D gate |
| --- | ---: | ---: | ---: | ---: |
| candle | 92.80% | 92.90% | **94.77%** | 94.12% |
| capsules | 98.06% | **98.38%** | 97.57% | 98.23% |
| cashew | 93.31% | **93.61%** | 92.17% | 92.47% |
| chewinggum | 98.95% | **99.05%** | 98.40% | 98.68% |
| fryum | 97.41% | 97.77% | **99.02%** | 98.67% |
| pipe_fryum | 96.38% | 97.01% | **97.46%** | 97.03% |

D's four-shot macro gain comes mainly from candle and fryum; it is worse than A on capsules, cashew and chewinggum. This is why a single macro score should not stand in for the per-category failure analysis.

## Direction F: selective decisions

At **90% automatic coverage**, mean error rates among automatically decided images are:

| References | Part 2 DINO margin | D equal fusion margin | D fusion + disagreement | D exploratory gate margin |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12.13% | **9.31%** | 10.37% | 9.36% |
| 2 | 9.75% | 8.62% | 9.17% | **8.57%** |
| 4 | 10.39% | 8.84% | 8.80% | **8.77%** |
| 8 | 8.62% | 7.86% | 7.90% | **6.88%** |

These are *retained-set* error rates. They are lower partly because about 10% of images receive no automatic decision and require manual review. F does not itself raise image AUROC. At four shots, the retrospective false-negative rate at a test-set FPR no higher than 10% is 9.21% for the original DINO margin ranking, 8.45% for equal fusion, and 7.59% for the exploratory gated policy. At eight shots, equal fusion's corresponding rate is **7.87%**, worse than DINO's **6.36%**, even though its image AUROC is slightly higher. AUROC and a specific operating point need not move together.

The **deployable normal-calibrated review band** has variable coverage. At four shots it marks **426 of 2,880** repeated-setting test predictions for review (14.79%). Among the 2,454 automatically decided predictions, 182 are wrong (7.42%). The review queue contains 189 of the fusion method's 371 errors, including 80 of its 180 false negatives. Those 80 cases are *reviewed false negatives*, not automatically corrected true positives. The 2,880 records represent 960 images assessed in three DINO reference groups.

## Runtime and memory

A warmed, paired four-shot/group-0 measurement on the same 120 VisA queries and an RTX 4060 Laptop GPU measured 228 ms/query for the DINO pass and 223 ms/query for the CLIP pass. The complete D/F query path took **455 ms/query** and reached **1,963 MiB peak allocated GPU tensor memory** with both frozen models resident. This includes decode, both feature passes, retrieval, map restoration, fusion and the review decision; model loading, reference-bank setup, calibration, result writing and metric evaluation are excluded. The previous A benchmark measured 234 ms/query on the same image list in a separate run, so the exact difference between A and D/F should not be read as a controlled head-to-head timing measurement. The direct conclusion is that D/F needs a second large model and substantially more inference work.

## Reproduce and inspect

```powershell
.\.venv-part2\Scripts\python.exe -m part3.fusion_selective
.\.venv-part2\Scripts\python.exe -m part3.multilayer_fusion
.\.venv-part2\Scripts\python.exe -m part3.compare_d_f
.\.venv-part2\Scripts\python.exe -m part3.fusion_report
.\.venv-part2\Scripts\python.exe -m part3.fusion_benchmark
```

The tables and plots are in `results/part3/df_comparison/` and `results/part3/fusion_selective/`: paired category runs, mean/sample-SD AUROC by shot, all-shot risk–coverage, fixed-band workload, per-image outcomes and the timing records. The CSVs retain the negative cases rather than selecting only favorable examples. The saved `protocol.json` and `audit.json` describe provenance and the limitations above.

Representative figures for the public repository: [A versus D image AUROC](../docs/figures/part3_a_d_image_auroc.png) and [F risk–coverage for all shot counts](../docs/figures/part3_d_f_risk_coverage.png).
