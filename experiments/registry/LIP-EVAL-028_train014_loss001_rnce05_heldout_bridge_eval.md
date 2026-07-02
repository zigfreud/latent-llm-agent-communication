# LIP-EVAL-028 LIP-LOSS-001 Reverse NCE 0.5 MBPP32 Held-Out Bridge Evaluation

## Summary

This entry records the held-out latent-space bridge evaluation for
`LIP-TRAIN-014-mbpp64-batch16-steps256-l035-rnce05-margin005-norm01`. The
checkpoint was trained on the LIP-DATA-004 MBPP64 train bundle with
`lambda_reverse_nce=0.5`, `lambda_margin=0.05`, and `lambda_norm=0.10`, then
evaluated on the separate MBPP32 validation bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-014-mbpp64-batch16-steps256-l035-rnce05-margin005-norm01`
- Run ID: `28617086460`
- Artifact ID: `8049893862`
- Artifact digest: `sha256:987dde166ce79ad3e340951cd0eb83ccbbe8fc6fa79f38eba740e6c2a2501dab`

Held-out eval bundle:

- Bundle trace: `LIP-DATA-004-MBPP-EVAL`
- Dataset: `google-research-datasets/mbpp`
- Split: `validation`
- Samples: `32`
- Eval shard SHA256: `7e58a9afcb2733ee3a7110eff757b3ffd47ae0391c526639f13f30714d60ce8e`
- Eval zip SHA256: `8916c572ea5f5777108f219a1243a0e13ded219d41e96053a48fb71ee2f2294a`

## Metrics

- Sample count: `32`
- Latent MSE mean: `2.4779837131500244`
- Latent RMSE mean: `1.5625848770141602`
- Cosine diagonal mean: `0.7777100205421448`
- Cosine diagonal std: `0.05358319729566574`
- Prediction norm mean: `138.9424591064453`
- Target norm mean: `158.02142333984375`
- Norm ratio mean: `0.8792356252670288`
- Energy drift mean: `19.07895851135254`
- Retrieval top-1: `0.375`
- Retrieval top-k: `{"1": 0.375, "5": 0.71875}`
- Off-diagonal cosine mean: `0.6773086786270142`
- Diagonal margin mean: `-0.007421704009175301`

## Comparison

Compared with LIP-EVAL-025/026, reducing `lambda_reverse_nce` from `1.0` to
`0.5` improved held-out MSE, cosine diagonal, norm ratio, and energy drift.

- Held-out retrieval top-1 remained `0.375`.
- Held-out top-5 decreased from `0.78125` to `0.71875`.
- Held-out diagonal margin mean improved only slightly from `-0.007482936605811119` to `-0.007421704009175301`.

Compared with the previous lambda_mse `0.35` baseline LIP-EVAL-021/022, this
run still has worse held-out MSE, norm ratio, energy drift, top5, and diagonal
margin.

## Interpretation

Reducing reverse NCE from `1.0` to `0.5` recovered part of the reconstruction
and norm/energy degradation seen in earlier LIP-LOSS-001 runs while preserving
held-out top1. However, it still does not outperform the lambda_mse `0.35`
baseline.

The result suggests reverse NCE was too strong at `1.0`, but the current
margin/norm/reverse combination still weakens positive-target geometry. This
remains latent-space diagnostic evaluation only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, generalization, or production readiness
claim.
