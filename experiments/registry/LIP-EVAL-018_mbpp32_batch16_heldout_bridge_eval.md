# LIP-EVAL-018 MBPP32 Batch16 Held-Out Bridge Evaluation

## Summary

This entry records the held-out latent-space bridge evaluation for
`LIP-TRAIN-009-mbpp64-batch16-steps128-lambda025`. The checkpoint was trained on
the LIP-DATA-004 MBPP64 train bundle with batch size 16 and evaluated on the
separate MBPP32 validation bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-009-mbpp64-batch16-steps128-lambda025`
- Run ID: `28597364631`
- Artifact ID: `8041794109`
- Artifact digest: `sha256:6cc7fa9e93324a4fba24d8815dd2cc6477688f1a76631e75962b66a044bea6be`

Held-out eval bundle:

- Bundle trace: `LIP-DATA-004-MBPP-EVAL`
- Dataset: `google-research-datasets/mbpp`
- Split: `validation`
- Samples: `32`
- Eval shard SHA256: `7e58a9afcb2733ee3a7110eff757b3ffd47ae0391c526639f13f30714d60ce8e`
- Eval zip SHA256: `7898d9e4e088b0be2f79b84a74be0f9a1e3edfd5ed7a496c651ce9a6db573c7f`

## Metrics

- Sample count: `32`
- Latent MSE mean: `2.290590763092041`
- Latent RMSE mean: `1.499502420425415`
- Cosine diagonal mean: `0.7984306812286377`
- Cosine diagonal std: `0.05384915694594383`
- Prediction norm mean: `142.42474365234375`
- Target norm mean: `158.02142333984375`
- Norm ratio mean: `0.901314914226532`
- Energy drift mean: `15.596685409545898`
- Retrieval top-1: `0.375`
- Retrieval top-k: `{"1": 0.375, "5": 0.6875}`
- Off-diagonal cosine mean: `0.7059991359710693`
- Diagonal margin mean: `-0.01821288838982582`

## Interpretation

Batch size 16 improved held-out retrieval compared with batch size 8 at matched
total sample exposure, suggesting that additional contrastive negatives help
pair-level ranking. However, reconstruction and norm/energy calibration
worsened, likely because the batch16 run used fewer optimizer updates.

Held-out diagonal margin remains negative, so pair-level generalization is
improved but not robust. This remains latent-space diagnostic evaluation only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, or production readiness claim.
