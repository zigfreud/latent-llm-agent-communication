# LIP-EVAL-012 MBPP16 Held-Out Bridge Evaluation

## Summary

This entry records the held-out latent-space bridge evaluation for
`LIP-TRAIN-006-mbpp32-batch8-steps64-lambda025`. The checkpoint was trained on
the LIP-DATA-003 MBPP32 train bundle and evaluated on the separate MBPP16
validation bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-006-mbpp32-batch8-steps64-lambda025`
- Run ID: `28023800921`
- Artifact ID: `7819833905`
- Artifact digest: `sha256:3333875c0e54a21a63fc805413269c663c3a8d4c60938b3293f3eed1232f585c`

Held-out eval bundle:

- Bundle trace: `LIP-DATA-003-MBPP-EVAL`
- Dataset: `google-research-datasets/mbpp`
- Split: `validation`
- Samples: `16`
- Eval shard SHA256: `0cac44ff2645160b9799ec56e8834e91a020db1b21f295dcf0e1d2ebabab6d12`

## Metrics

- Sample count: `16`
- Latent MSE mean: `2.3398611545562744`
- Latent RMSE mean: `1.5068036317825317`
- Cosine diagonal mean: `0.7847334146499634`
- Cosine diagonal std: `0.08326608687639236`
- Prediction norm mean: `134.0918731689453`
- Target norm mean: `158.0408477783203`
- Norm ratio mean: `0.8483860492706299`
- Energy drift mean: `23.948955535888672`
- Retrieval top-1: `0.3125`
- Retrieval top-k: `{"1": 0.3125, "5": 0.6875}`
- Off-diagonal cosine mean: `0.7324801087379456`
- Diagonal margin mean: `-0.035292334854602814`

## Interpretation

Held-out retrieval_top1 is above chance but margin is negative, indicating
partial latent generalization but weak pair-level separation. Several held-out
errors are attracted to target `8`.

The next diagnostic step is to rerun the same MBPP32 setup with 128 steps to
distinguish undertraining from generalization limits.

This remains latent-space diagnostic evaluation only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, or production readiness claim.
