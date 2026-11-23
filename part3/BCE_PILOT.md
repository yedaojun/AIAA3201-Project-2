# Part 3 B, C and E: six-category screening experiments

The PDF requires one justified Part 3 improvement; directions A–F are alternatives. We screened the three remaining directions B, C and E using a **prespecified 4-shot, reference-group-0 pilot** on the complete official test split of the six required VisA categories (960 distinct test images). These are exploratory comparisons, **not** full 1/2/4/8-shot, three-group evaluations. No test image or defect mask was used to choose a parameter, train a head or calibrate a threshold.

## Fixed methods

- **B, background suppression:** preserve the Part 2 DINO model, query foreground mask, image resolution and nearest-neighbor score. Apply the existing per-image DINO PCA foreground mask to each rotated *normal reference* before placing patches in the memory bank. No segmentation labels are used. Across categories this keeps 28–55% of the original reference patches.
- **C, synthetic defects:** take the same four normal reference images, make eight CutPaste variants per reference in the already resized 448-short-edge DINO tensor, and label patches overlapping the pasted region. Freeze DINOv2. Train only a class-balanced linear logistic patch head on original normal patches versus synthetic defect patches. There are **4 real normal training images and 32 synthetic training images per category**. Image score is the mean of the top 1% patch probabilities; the original PCA query mask remains. This is a distinct supervised-on-synthetic experiment, not another nearest-neighbor bank.
- **E, bank compression:** keep Part 2 features and image scoring, but select a fixed random quarter of the patches **within each rotated normal reference**. The random seed and 25% budget were fixed before test evaluation. This is a simple reference-stratified coreset control, not an optimized learned clustering method.

All three methods use the same 20 disjoint normal calibration images and the same 95th-percentile threshold rule. The pilot recomputes the original Part 2 score from the identical query features and full bank; the maximum absolute difference from the saved Part 2 scores is **zero** for every category.

## Full-test image-level pilot

Image AUROC, 4-shot, reference group 0:

| VisA category | Part 2 unchanged | B: foreground bank | C: CutPaste head | E: 25% bank |
| --- | ---: | ---: | ---: | ---: |
| candle | 91.64% | 91.33% | 81.27% | 90.42% |
| capsules | 97.82% | 97.15% | 91.60% | **98.30%** |
| cashew | 94.86% | 94.76% | 87.70% | 94.82% |
| chewinggum | 98.84% | 98.64% | 84.66% | 98.64% |
| fryum | 97.00% | 96.28% | 81.78% | 96.88% |
| pipe_fryum | 94.86% | 94.82% | 75.16% | 94.78% |
| **Six-category macro** | **95.84%** | 95.50% | 83.69% | 95.64% |

B falls slightly in all six categories. This is consistent with its relatively aggressive removal of reference features: patches that the PCA mask calls background can still be useful normal neighbors, especially near an object's edge. That interpretation is a hypothesis from the observed scores and bank sizes; no pixel-level causal analysis has been completed.

C is worse in all six categories. CutPaste changes local texture and creates seams, while real VisA anomalies also include product-specific deformations, missing material and contamination. The result suggests a domain gap between this one synthetic generator and real defects. It does not prove that every synthetic-defect method would fail; a richer generator or additional source data would be a different experiment.

E improves capsules by 0.48 percentage points, but the six-category mean falls by 0.20 points. This makes it an efficiency trade-off candidate, not an accuracy improvement. A better spatial or feature-diverse coreset could behave differently and would need its own controlled evaluation.

## Bank and retrieval efficiency

A paired measurement on the same 120 test queries (20 per category, three timing repeats) measured **CPU FAISS nearest-neighbor search only**, with identical fresh DINO query features and four CPU search threads:

| Bank | Mean stored patch-feature memory | Mean retrieval time |
| --- | ---: | ---: |
| Part 2 full bank | 59.75 MiB | 103.54 ms |
| B foreground bank | 22.91 MiB | 43.42 ms |
| E 25% bank | 14.94 MiB | 26.55 ms |

The E bank saves 75% of patch-feature storage and its measured retrieval stage is about four times faster. These figures **exclude** image decoding, DINO extraction, PCA query masking, map restoration and reference-bank construction, so they must not be described as a fourfold end-to-end speedup. The C head has no nearest-neighbor bank and is not part of this retrieval comparison.

## Decision for the final report

The B/C/E pilots do not beat the unchanged Part 2 baseline on the prespecified six-category main metric. We will keep their negative results and explanations as ablations. **A remains the controlled main Part 3 improvement**, with D/F as separately qualified exploratory directions. Promoting any B/C/E candidate would require completing all shot counts, three reference groups, pixel AUROC, representative failures and a matched end-to-end efficiency benchmark.

The full pilot records are in `results/part3/background_memory_pilot/`, `results/part3/synthetic_cutpaste_pilot/` and `results/part3/bce_comparison/`. The [comparison figure](../docs/figures/part3_bce_pilot.png) is a small public-repository artifact.

```powershell
.\.venv-part2\Scripts\python.exe -m part3.background_memory_pilot
.\.venv-part2\Scripts\python.exe -m part3.synthetic_cutpaste_pilot
.\.venv-part2\Scripts\python.exe -m part3.background_memory_benchmark
.\.venv-part2\Scripts\python.exe -m part3.bce_report
```
