# LIP-EVAL-024 LIP-LOSS-001 MBPP32 Held-Out Bridge Evaluation

## Summary

This entry records the held-out latent-space bridge evaluation for
`LIP-TRAIN-012-mbpp64-batch16-steps256-l035-rnce1-margin01-norm005`. The
checkpoint was trained on the LIP-DATA-004 MBPP64 train bundle with the
LIP-LOSS-001 margin-aware symmetric objective and evaluated on the separate
MBPP32 validation bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-012-mbpp64-batch16-steps256-l035-rnce1-margin01-norm005`
- Run ID: `28614359665`
- Artifact ID: `8048809744`
- Artifact digest: `sha256:9d5675e392bc7ce23726ffa33ee4fe48aa891de3a0886af1b403b34a5bf723f1`

Held-out eval bundle:

- Bundle trace: `LIP-DATA-004-MBPP-EVAL`
- Dataset: `google-research-datasets/mbpp`
- Split: `validation`
- Samples: `32`
- Eval shard SHA256: `7e58a9afcb2733ee3a7110eff757b3ffd47ae0391c526639f13f30714d60ce8e`
- Eval zip SHA256: `f06e780508db510841ea888ff9191503a8b432f663def44883f43c4aa5a35959`

## Metrics

- Sample count: `32`
- Latent MSE mean: `2.776270627975464`
- Latent RMSE mean: `1.656923532485962`
- Cosine diagonal mean: `0.7438757419586182`
- Cosine diagonal std: `0.05518180876970291`
- Prediction norm mean: `131.98057556152344`
- Target norm mean: `158.02142333984375`
- Norm ratio mean: `0.8351871371269226`
- Energy drift mean: `26.04084014892578`
- Retrieval top-1: `0.34375`
- Retrieval top-k: `{"1": 0.34375, "5": 0.78125}`
- Off-diagonal cosine mean: `0.6383078098297119`
- Diagonal margin mean: `-0.0078982412815094`

## Comparison Against LIP-EVAL-022

Compared with the lambda_mse `0.35` MBPP64 batch16 steps256 held-out baseline:

- Off-diagonal cosine mean decreased from `0.728355884552002` to `0.6383078098297119`.
- Held-out top-5 improved from `0.75` to `0.78125`.
- Held-out top-1 decreased from `0.375` to `0.34375`.
- Latent MSE worsened from `2.0937340259552` to `2.776270627975464`.
- Cosine diagonal mean decreased from `0.8178914785385132` to `0.7438757419586182`.
- Norm ratio moved farther from 1.0, from `0.9280939102172852` to `0.8351871371269226`.
- Energy drift worsened from `11.363945007324219` to `26.04084014892578`.
- Diagonal margin mean moved from `-0.006803490221500397` to `-0.0078982412815094`.

## Interpretation

The LIP-LOSS-001 configuration reduced off-diagonal similarity and improved
held-out top5, suggesting stronger contrastive separation pressure. However, it
worsened MSE, cosine diagonal, norm ratio, energy drift, and held-out top1
compared with the previous lambda_mse `0.35` baseline.

This indicates the new loss is directionally useful but its first weighting
over-penalizes ranking/margin relative to reconstruction and norm calibration.
Held-out diagonal margin remains negative, so pair-level held-out ranking is not
yet robust. This remains latent-space diagnostic evaluation only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, generalization, or production readiness
claim.
