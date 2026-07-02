# LIP-EVAL-030 LIP-LOSS-001 Reverse NCE 0.0 MBPP32 Held-Out Bridge Evaluation

## Summary

This entry records the held-out latent-space bridge evaluation for
`LIP-TRAIN-015-mbpp64-batch16-steps256-l035-rnce0-margin005-norm01`. The
checkpoint was trained on the LIP-DATA-004 MBPP64 train bundle with
`lambda_reverse_nce=0.0`, `lambda_margin=0.05`, and `lambda_norm=0.10`, then
evaluated on the separate MBPP32 validation bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-015-mbpp64-batch16-steps256-l035-rnce0-margin005-norm01`
- Run ID: `28625926129`
- Artifact ID: `8053318553`
- Artifact digest: `sha256:aae5be93368bfeb1cedcb5027778454db477e71b03a2b834e86858b704f80489`

Held-out eval bundle:

- Bundle trace: `LIP-DATA-004-MBPP-EVAL`
- Dataset: `google-research-datasets/mbpp`
- Split: `validation`
- Samples: `32`
- Eval shard SHA256: `7e58a9afcb2733ee3a7110eff757b3ffd47ae0391c526639f13f30714d60ce8e`
- Eval zip SHA256: `bed8560889ab30dc47de3cb71dc25bad72af5e199683f8816abc37d7fb77e504`

## Metrics

- Sample count: `32`
- Latent MSE mean: `2.0624284744262695`
- Latent RMSE mean: `1.4181245565414429`
- Cosine diagonal mean: `0.8215618133544922`
- Cosine diagonal std: `0.05397816002368927`
- Prediction norm mean: `147.583740234375`
- Target norm mean: `158.02142333984375`
- Norm ratio mean: `0.9339640736579895`
- Energy drift mean: `10.43768310546875`
- Retrieval top-1: `0.375`
- Retrieval top-k: `{"1": 0.375, "5": 0.71875}`
- Off-diagonal cosine mean: `0.731459379196167`
- Diagonal margin mean: `-0.009301887825131416`

## Comparison

Compared with LIP-EVAL-027/028, turning `lambda_reverse_nce` from `0.5` to
`0.0` substantially improved held-out MSE, cosine diagonal, norm ratio, and
energy drift.

- Held-out top1 remained `0.375`.
- Held-out top5 remained `0.71875`.
- Held-out diagonal margin worsened slightly from `-0.007421704009175301` to `-0.009301887825131416`.

Compared with the lambda_mse `0.35` baseline LIP-EVAL-021/022, this run has
slightly better held-out MSE, norm ratio, and energy drift, equal top1, but
worse top5 and diagonal margin.

## Interpretation

Disabling reverse NCE recovered positive-target geometry while preserving
held-out top1. The result suggests reverse NCE was the main source of the
geometry degradation in earlier LIP-LOSS-001 runs.

However, the margin term still did not improve held-out diagonal margin, so
margin calibration remains unresolved. This remains latent-space diagnostic
evaluation only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, generalization, or production readiness
claim.
