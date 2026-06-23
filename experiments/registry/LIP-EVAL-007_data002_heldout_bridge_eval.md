# LIP-EVAL-007 Held-Out LIP-DATA-002 Bridge Evaluation

## Summary

This entry records the first held-out latent-space bridge evaluation. The bridge
was trained with `LIP-TRAIN-004-data002-train16-steps64-lambda025` on the
LIP-DATA-002 train bundle and evaluated on the separate LIP-DATA-002 held-out
eval bundle.

## Recorded Evidence

Training run:

- Experiment: `LIP-TRAIN-004-data002-train16-steps64-lambda025`
- Run ID: `28015568225`
- Artifact ID: `7816492548`
- Artifact digest: `sha256:2463c3e7b30176fb1870af8973da53833c7e4b85cd39b83b97224b36781a5c93`
- Train bundle SHA256: `670c826ca7f83b8785ae157bc9b56232f0ed444ee8181526f11aa5b6c0dde4e6`

Held-out eval bundle:

- Bundle: `LIP-DATA-002`
- Samples: `8`
- Eval bundle SHA256: `964acecf6dd4e801261f2133215334d65a103400a78c2077d503482e2ff011f2`
- Eval shard SHA256: `8b18416d37312e9fbfde53b6b0404c3e69836b3d15fea7ec4807b565661e9fbd`
- Eval zip SHA256: `a79f11ff2d7a526b580face75f7f7959aba6983e058a5fcdac9f9acf18c56850`

## Metrics

- Sample count: `8`
- Latent MSE mean: `1.6045767068862915`
- Latent RMSE mean: `1.254853367805481`
- Cosine diagonal mean: `0.858284592628479`
- Cosine diagonal std: `0.03810800611972809`
- Prediction norm mean: `141.57965087890625`
- Target norm mean: `157.0806427001953`
- Norm ratio mean: `0.9013491868972778`
- Energy drift mean: `15.501007080078125`
- Retrieval top-1: `0.125`
- Retrieval top-k: `{"1": 0.125, "5": 0.875}`
- Off-diagonal cosine mean: `0.8459532856941223`
- Diagonal margin mean: `-0.014098398387432098`

## Interpretation

This is the first held-out latent-space evaluation. The bridge preserves
target-space geometry and norm calibration better than early baselines, but
pair-level generalization is not established. `retrieval_top1` is at chance,
`diagonal_margin_mean` is negative, and most predictions are attracted to
target `2`.

The result supports continuing held-out evaluation before making any semantic
or generalization claim.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Held-out latent-space evaluation only. No semantic transfer, text-level
fidelity, or production readiness claim.
