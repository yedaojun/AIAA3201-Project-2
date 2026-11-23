# Compact evidence for the Part 3 follow-up

These files are exported by `python -m report.export_review_evidence` from the local experiment records. They include the frozen reference manifest, all 72 matched image-level settings, three-group original-resolution pixel metrics, defect-area strata, normal-only threshold diagnostics, the unified 120-query benchmark, recalibrated top-k score diagnostics, and pure-DINO selective-review curves. Raw VisA images, pretrained model weights, and full-resolution anomaly maps are not included.

The 448+672 candidate and selective-review analysis were developed after examining the same VisA test split. Their reference-group repeats measure sensitivity to normal-bank choice; they are not an independent validation set. `REVIEW` means a human still has to decide. The externally supervised AnomalyCLIP and optional autoencoder results use different training conditions and are described separately in the report.
