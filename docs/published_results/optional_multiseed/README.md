# Optional experiment provenance

These files copy the actual local seed 0/1/2 records. `source_checksums.json` maps each copy to its original result path and SHA-256. No experiment is rerun or numerically replaced by this export.

Autoencoder: 18 separately trained category models (six categories × three training seeds), 20 epochs each. Training seed changes initialization and order. `split_seed=111` fixes fit/validation membership. Seed-0 training-image manifests apply unchanged to all three seeds; the audit verifies equality. Full per-category training configs and loss histories are included for every seed.

AnomalyCLIP: fixed pretrained model and prompts, with no VisA reference sampling. Seed 0/1/2 are repeated deterministic inference, **not a few-shot reference stability study**. Equal predictions are not evidence of robustness. Each seed has six saved category metrics. The older seed111 summaries elsewhere in `published_results/` are historical runs, excluded from the autoencoder three-seed mean.

Image data, floating maps, checkpoints and personal voice recordings are excluded. Reproduction entry points and commands are in `docs/optional_multiseed.md`.
