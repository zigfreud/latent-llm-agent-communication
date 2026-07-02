# LIP-EVAL-021 MBPP64 Batch16 256-Step Lambda 0.35 In-Sample Bridge Evaluation

## Summary

This entry records the train-set in-sample latent-space bridge evaluation for
`LIP-TRAIN-011-mbpp64-batch16-steps256-lambda035`. The checkpoint was trained on
the LIP-DATA-004 MBPP64 train bundle with batch size 16 for 256 steps and
evaluated on the same train bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-011-mbpp64-batch16-steps256-lambda035`
- Run ID: `28603839197`
- Artifact ID: `8044498853`
- Artifact digest: `sha256:a64b273dbce6ed364e920a976c32336120e84d59c047456597443f1f61557ba0`

Train eval bundle:

- Bundle trace: `LIP-DATA-004-MBPP-TRAIN`
- Dataset: `google-research-datasets/mbpp`
- Split: `train`
- Samples: `64`
- Eval shard SHA256: `f1dd527799ad2dd0e99edc253ac71213c9db4db36d9cf6b702e3a6ffc126e811`
- Eval zip SHA256: `9f9b91a56bbc48afb25f2c0d8504f31f82480c9285fb7415bd8cecbbb4dfd92f`

## Metrics

- Sample count: `64`
- Latent MSE mean: `0.6976713538169861`
- Latent RMSE mean: `0.8309321403503418`
- Cosine diagonal mean: `0.9411893486976624`
- Cosine diagonal std: `0.010752093978226185`
- Prediction norm mean: `151.78021240234375`
- Target norm mean: `157.84063720703125`
- Norm ratio mean: `0.9616261720657349`
- Energy drift mean: `6.247280597686768`
- Retrieval top-1: `0.96875`
- Retrieval top-k: `{"1": 0.96875, "5": 1.0}`
- Off-diagonal cosine mean: `0.6405885219573975`
- Diagonal margin mean: `0.15557986497879028`

## Comparison Against LIP-EVAL-019

Compared with the lambda_mse `0.25` MBPP64 batch16 steps256 in-sample result:

- In-sample MSE improved.
- Cosine diagonal mean improved.
- Norm ratio improved.
- Energy drift improved.
- Retrieval top-1 remained `0.96875`.

## Interpretation

Increasing lambda_mse from `0.25` to `0.35` in the MBPP64 batch16 steps256
regime improved train-set reconstruction, cosine alignment, norm ratio, and
energy calibration. It preserved the strong in-sample retrieval result.

This remains an in-sample latent-space diagnostic and does not establish
held-out generalization or semantic transfer.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, or production readiness claim.
