# Response to AIAA3201Project2ReportReview.pdf

**Later update, 23 November 2026:** The newer Part 3 review is addressed in [REVIEW_COMPLETION_2026-11-23.md](REVIEW_COMPLETION_2026-11-23.md). In particular, the report now contains three-group A+AnyUp pixel AP, a complete 1/2/4/8-shot 448+672 image-scale sweep, defect-size diagnostics, and a rebuilt PDF. The single-group scope statements below describe the earlier revision, not the current report.

This log records what changed in the report and which suggestions require information or experiment results before they can be claimed. The review's estimates and suggested explanations were treated as hypotheses, not copied as measured findings.

| Review item | Response |
| --- | --- |
| Official CVPR format | `main.tex` uses the official `cvpr.sty` author-kit file in camera-ready mode; checked the rendered two-column PDF. |
| Author identities | Pending the group's real names, student IDs, and affiliation. The PDF still has a group placeholder. |
| 0.02/0.03 and 0.11/0.12 arithmetic | Corrected differences to match the displayed two-decimal means. |
| AnyUp scope | Added the measured A+AnyUp four-shot/seed-0 pixel AUROC beside its timing; retained the distinct official+AnyUp all-setting result. |
| Secondary directions | Added a structured table with supervision, experiment scope, main result, and cost; shortened duplicate prose. |
| Failure cases | Added a seventh case (candle `good/0333.JPG`, TN→FP), explicit diagnostic groups, and method-specific image score/threshold labels. |
| Part 1 visual and timing | Added capsules `good/016.JPG` original/template/difference/DINO comparison. Added RGB time as a separately measured, explicitly non-paired row in Table 4. |
| False-alarm distribution | Counted TN→FP and TP→FN by all six categories from saved four-shot predictions across groups 0/1/2. |
| Latency decomposition | Added a new paired, stage-synchronized 120-image diagnostic and reported its feature-extraction and CPU FAISS components. Its absolute total is kept separate from the original timing protocol. |
| Shot curve | Added a DINO/A image-AUROC zoom inset. |
| Tables 1 and 3 | Added A-minus-DINO columns and highlighted the better paired score. |
| Figure 1 | Added descriptor dimensions, the PCA query mask, and a dashed normal-calibration arrow. |
| Related work and wording | Connected PatchCore's multi-level CNN features to the ViT-layer hypothesis; clarified that groups 0/1/2 are the reference seeds; removed several course-audit phrases. |
| Layer and weight ablation | Completed post-hoc four-shot/seed-0 diagnostics for single blocks 8/11, pairs 5+11/8+11/10+11, and intermediate-layer weights 0.25/0.50/0.75. Added a separate table. The primary A setting remains pre-fixed; nearly tied pairs are not presented as proof of an optimal layer. |
| Localization-sensitive metric | Computed exact original-resolution four-shot/seed-0 pixel AP for Official DINO, A, and A+AnyUp, including all normal zero masks. Added a separate table so its single-group scope cannot be confused with Table 3's three-group averages. The AP calculation was checked against scikit-learn on a tie-containing example. |
| Demo ZIP and GitHub sync | Submission packaging is outside the report text and has not been performed in this revision. The public GitHub URL in the abstract was verified; ensure the latest report and code are pushed before submission. |

The bibliography retains the checked 2026 MVTec AD 2 and AnyUp publication details. The report PDF has been rebuilt and visually reviewed after layout changes.
