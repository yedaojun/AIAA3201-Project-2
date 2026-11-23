# Part 3 experiment results and remaining pilots

All values below are six-category macro averages on the same official VisA test images. Unless a row says otherwise, the shot count is four and the fixed reference group is seed 0. The full 1/2/4/8-shot, three-group Direction A image- and pixel-level evaluations are complete. The A+AnyUp composition remains a four-shot/group-0 exploratory pilot.

| Experiment | Image AUROC | Pixel AUROC | What changed |
| --- | ---: | ---: | --- |
| Official Part 2 DINO | 95.84% | 98.12% | Frozen final-layer patch features |
| A: two DINO layers | 96.29% | 98.10% | Equal unit-length features from blocks 8 and 11 before nearest-neighbor scoring |
| D: official DINO plus CLIP | 96.50% | — | Fixed equal-weight, normal-calibrated image-score fusion |
| D: uncertainty-gated DINO plus CLIP | 96.44% | — | Exploratory normal-calibrated gate around the DINO decision boundary |
| A plus D | 96.41% | — | Same fixed fusion applied to A image scores |
| Original DINO plus AnyUp | 95.84% | 98.68% | Pretrained 2× score-map restoration only |
| A plus AnyUp | 96.29% | 98.63% | A image scores and A patch maps restored by AnyUp |
| A plus D plus AnyUp | 96.41% | 98.63% | Fusion controls image decision; A+AnyUp supplies heatmap |

The image and pixel metrics come from different output branches where indicated. The last row is a runnable combined demo, not evidence that all three changes help equally. In this pilot, adding A to D reduces image AUROC from 96.50% to 96.41%, and adding A to AnyUp reduces pixel AUROC from 98.68% to 98.63%. We retain these ablations and the negative cases rather than hiding them.

The completed 72-setting image-level experiment gives the following six-category macro means across three reference groups:

| Shots | Official DINO | A: two layers | D: equal CLIP fusion | A+D: two layers and CLIP |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 94.61% | 94.74% | 95.53% | 95.39% |
| 2 | 95.33% | 95.47% | 95.94% | 95.96% |
| 4 | 96.15% | 96.45% | 96.57% | 96.52% |
| 8 | 96.62% | 96.85% | 96.86% | 96.98% |

A gives a modest mean image-AUROC gain at every shot count, with individual category/group regressions. Its paired six-category image-AUROC changes for seeds 0/1/2 are −0.03/+0.27/+0.17 percentage points at 1-shot, +0.16/+0.07/+0.19 at 2-shot, +0.46/+0.28/+0.17 at 4-shot, and +0.48/+0.23/−0.02 at 8-shot. D and A+D are additional externally supervised comparisons, with the CLIP caveats below.

The complete original-resolution pixel comparison for A is:

| Shots | Official DINO pixel AUROC | A: two layers pixel AUROC | A change |
| ---: | ---: | ---: | ---: |
| 1 | 97.76% | 97.71% | −0.05 pp |
| 2 | 97.92% | 97.92% | approximately 0 pp |
| 4 | 98.09% | 98.12% | +0.02 pp |
| 8 | 98.21% | 98.25% | +0.04 pp |

Thus A's more consistent gain is image ranking. It does not convincingly solve fine localization on its own. AnyUp is the stronger localization branch, at additional inference time.

On the same warmed 120-image, 4-shot/group-0 benchmark with fresh feature extraction and full heatmap restoration, official DINO averages 178 ms/image, A averages 234 ms/image, and A+AnyUp averages 382 ms/image. Peak allocated GPU tensor memory is 124/126/990 MiB, respectively, with both frozen models resident during all three timings. Reference-bank/model setup, result-file I/O and metric calculation are excluded. These are local RTX 4060 laptop measurements; they are not the original Part 2 timing boundary.

The independent AnyUp score-map experiment has now finished all 72 paired settings. Across three reference groups, the six-category mean pixel AUROC at 1/2/4/8 shots goes from 97.76/97.93/98.09/98.21% for official restoration to 98.41/98.52/98.64/98.73% for AnyUp. The matched two-step bilinear control reaches 97.99/98.13/98.29/98.39%. The image scores and OK/NG decisions are exactly unchanged in this branch. At four shots and group 0, the map's strongest response falls in the annotated defect for 53.5% of 600 anomalous images with official restoration, 60.8% with two-step bilinear, and 72.3% with AnyUp. On 195 images whose masks occupy at most 0.1% of the image, those rates are 28.7/37.9/48.2%. The strongest-response diagnostic is secondary to exact pixel AUROC and does not prove that every small defect becomes detectable.

Across all 1/2/4/8-shot settings and all three DINO reference groups, D alone raises the macro image AUROC from 94.61/95.32/96.15/96.62% to 95.53/95.94/96.57/96.86%. This is a different supervision setting: AnomalyCLIP's prompts were trained on labeled MVTec AD anomalies and its square 448 geometry differs from DINO's aspect-preserving 448 geometry. This result should be presented as an optional exploratory fusion, not as a like-for-like few-shot comparison.

An additional uncertainty gate lets CLIP contribute mainly when DINO is close to its normal-calibrated threshold. The gate's three-group 1/2/4/8-shot macro image AUROC values are 95.52/96.08/96.53/97.03%. It helps more at two and eight shots than the equal blend, but not at one and four. This rule was added after the equal-blend pilot was inspected, so its comparison is exploratory and does not establish a chosen winner on an independent holdout.

At four shots, D's lower calibrated false-negative count comes with more false positives. Across the three reference groups, it changes 211 false negatives and 154 false positives for DINO into 180 false negatives and 191 false positives for fusion. Its mean retrospective false-negative rate at a test-set false-positive rate no higher than 10% is also slightly worse, 11.83% versus 10.94%. The test-set operating point is a reporting metric only, not a deployed threshold.

For the F pilot, a fixed band around the normally calibrated fusion threshold marks ambiguous images for human review. At 90% coverage, the retained-image error rate is 8.84% for fused scores versus 10.39% for original DINO with the same type of margin selection. The retained-image false-negative rate at a retrospective test-set false-positive rate of at most 10% is 8.45% versus 9.21%. The workload cost is that about 10% of images receive no automatic OK/NG decision. These are test-set outcomes, not calibrated guarantees. The separate fixed deployment band reviews 14.79% of images over three four-shot groups.

The exploratory gate's margin policy gives 8.77% retained-image error and 7.59% retrospective false-negative rate at the same 90% coverage. Because this candidate was introduced after initial test inspection, it should not be claimed as independently validated.

In the combined A+D pilot at four shots, seed 0, the fixed band reviews 173 of 960 images and captures 22 of its 52 false negatives for manual review. The candle `bad/000.JPG` example remains an automatic-score false negative but is marked `REVIEW`; it has not become a true positive.

The five fixed failure examples and ground-truth-centered display crops are in `results/part3/multilayer/difficult_cases_4shot_seed0.png` and `difficult_cases_zoom_composed.png`. Ground-truth masks influence crops for presentation only. They never enter feature extraction, score calculation, thresholds, fusion weights or the review rule.

Raw paired records are in `results/part3/multilayer/`, `results/part3/fusion_selective/`, `results/part3/multilayer_fusion/`, and `results/part3/two_layer_anyup/`. The source/reference settings and external-model provenance are documented in the corresponding `protocol.json` files and the Part 3 README.
