# Sources, pretrained models and attribution

This repository contains our course implementation and adapters. The third-party
repositories, datasets and pretrained checkpoints are downloaded separately and
are not redistributed in this Git repository. Their respective upstream terms apply.

| Component | Source | Version used |
|---|---|---|
| VisA / SPot-the-Difference data and official 1cls split | https://github.com/amazon-science/spot-diff | Public VisA archive; six course categories |
| AnomalyDINO | https://github.com/dammsi/AnomalyDINO | b9d1c2648e3a5247437d4d953d907a8f3d994457 |
| DINOv2 and ViT-S/14 weights | https://github.com/facebookresearch/dinov2 | 7764ea0f912e53c92e82eb78a2a1631e92725fc8 |
| AnomalyCLIP | https://github.com/zqhang/AnomalyCLIP | 3911738c0867544f545a076ad78f3f11d9ecbfdf |
| CLIP ViT-L/14@336px | https://github.com/openai/CLIP | Official upstream checkpoint, SHA256 below |

The DINOv2 checkpoint SHA256 is
`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
The CLIP backbone checkpoint SHA256 is
`3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`.
The AnomalyCLIP prompt checkpoint is
`checkpoints/9_12_4_multiscale_visa/epoch_15.pth`, SHA256
`415c5dcb52668b8c33fb9c1a351c686d632b919df5b384d63fa9ce7a2338ced4`.
Its published recipe trains prompts on MVTec AD normal/anomalous source data for
evaluation on VisA. It is not a VisA normal-only fitted model.

Representative visualization panels in `docs/figures/` are derived from VisA test
images and masks using the methods in this repository. They are included as
experiment illustrations, with VisA attribution; the raw dataset is not bundled.

## Papers

- Zou et al., *SPot-the-Difference Self-Supervised Pre-Training for Anomaly Detection and Segmentation*, ECCV 2022.
- Damm et al., *AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2*, WACV 2025.
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024.
- Zhou et al., *AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection*, ICLR 2024.
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021.
