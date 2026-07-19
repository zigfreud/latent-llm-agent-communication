# LIP-EVAL-026 LIP-LOSS-001 Adjusted Margin/Norm MBPP32 Held-Out Bridge Evaluation

## Summary

This entry records the held-out latent-space bridge evaluation for
`LIP-TRAIN-013-mbpp64-batch16-steps256-l035-rnce1-margin005-norm01`. The
checkpoint was trained on the LIP-DATA-004 MBPP64 train bundle with an adjusted
LIP-LOSS-001 weighting and evaluated on the separate MBPP32 validation bundle.

The adjusted weighting reduced `lambda_margin` from `0.10` to `0.05` and
increased `lambda_norm` from `0.05` to `0.10` relative to the first LIP-LOSS-001
run.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-013-mbpp64-batch16-steps256-l035-rnce1-margin005-norm01`
- Run ID: `28616101597`
- Artifact ID: `8049502502`
- Artifact digest: `sha256:136313d110ab4103e89f731ad2c5636a5ae70bf6bf237a7895aeb1dd7cfbcff1`

Held-out eval bundle:

- Bundle trace: `LIP-DATA-004-MBPP-EVAL`
- Dataset: `google-research-datasets/mbpp`
- Split: `validation`
- Samples: `32`
- Eval shard SHA256: `7e58a9afcb2733ee3a7110eff757b3ffd47ae0391c526639f13f30714d60ce8e`
- Eval zip SHA256: `5ac0bcc00c0d7034bb2e67fd76a9560c27651020144355b695d3439bcfc1f448`

## Metrics

- Sample count: `32`
- Latent MSE mean: `2.795985221862793`
- Latent RMSE mean: `1.6627753973007202`
- Cosine diagonal mean: `0.7421365976333618`
- Cosine diagonal std: `0.05557094141840935`
- Prediction norm mean: `132.30313110351562`
- Target norm mean: `158.02142333984375`
- Norm ratio mean: `0.8372302055358887`
- Energy drift mean: `25.71828842163086`
- Retrieval top-1: `0.375`
- Retrieval top-k: `{"1": 0.375, "5": 0.78125}`
- Off-diagonal cosine mean: `0.6368019580841064`
- Diagonal margin mean: `-0.007482936605811119`

## Comparison Against LIP-EVAL-024

Compared with the first LIP-LOSS-001 held-out result:

- Held-out retrieval top-1 improved from `0.34375` to `0.375`.
- Held-out top-5 remained `0.78125`.
- Held-out diagonal margin mean improved slightly from `-0.0078982412815094` to `-0.007482936605811119`.
- Held-out norm ratio mean improved slightly from `0.8351871371269226` to `0.8372302055358887`.
- Held-out energy drift mean improved slightly from `26.04084014892578` to `25.71828842163086`.
- Held-out MSE worsened slightly from `2.776270627975464` to `2.795985221862793`.
- Cosine diagonal mean decreased slightly from `0.7438757419586182` to `0.7421365976333618`.

## Interpretation

The adjusted LIP-LOSS-001 weighting slightly improves held-out top1, margin,
norm ratio, and energy drift relative to the first LIP-LOSS-001 run, while
preserving top5. However, reconstruction and cosine diagonal remain worse than
the lambda_mse `0.35` baseline, and held-out diagonal margin remains negative.

This suggests reverse NCE / margin pressure may still be too strong relative to
reconstruction and norm calibration. This remains latent-space diagnostic
evaluation only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, generalization, or production readiness
claim.
