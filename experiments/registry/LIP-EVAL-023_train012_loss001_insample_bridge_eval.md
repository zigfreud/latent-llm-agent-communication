# LIP-EVAL-023 LIP-LOSS-001 MBPP64 In-Sample Bridge Evaluation

## Summary

This entry records the train-set in-sample latent-space bridge evaluation for
`LIP-TRAIN-012-mbpp64-batch16-steps256-l035-rnce1-margin01-norm005`. The
checkpoint was trained on the LIP-DATA-004 MBPP64 train bundle with the
LIP-LOSS-001 margin-aware symmetric objective and evaluated on the same train
bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-012-mbpp64-batch16-steps256-l035-rnce1-margin01-norm005`
- Run ID: `28614359665`
- Artifact ID: `8048809744`
- Artifact digest: `sha256:9d5675e392bc7ce23726ffa33ee4fe48aa891de3a0886af1b403b34a5bf723f1`

Train eval bundle:

- Bundle trace: `LIP-DATA-004-MBPP-TRAIN`
- Dataset: `google-research-datasets/mbpp`
- Split: `train`
- Samples: `64`
- Eval shard SHA256: `f1dd527799ad2dd0e99edc253ac71213c9db4db36d9cf6b702e3a6ffc126e811`
- Eval zip SHA256: `626d19d4372689fbd2372e6021600beaa83c3c3115a402e2bde5fa94ce289c2f`

## Metrics

- Sample count: `64`
- Latent MSE mean: `1.3141343593597412`
- Latent RMSE mean: `1.1395597457885742`
- Cosine diagonal mean: `0.8857991099357605`
- Cosine diagonal std: `0.023624025285243988`
- Prediction norm mean: `144.467529296875`
- Target norm mean: `157.84063720703125`
- Norm ratio mean: `0.9153322577476501`
- Energy drift mean: `13.37309741973877`
- Retrieval top-1: `0.96875`
- Retrieval top-k: `{"1": 0.96875, "5": 1.0}`
- Off-diagonal cosine mean: `0.5465688705444336`
- Diagonal margin mean: `0.18054300546646118`

## Comparison Against LIP-EVAL-021

Compared with the lambda_mse `0.35` MBPP64 batch16 steps256 in-sample baseline:

- Off-diagonal cosine mean decreased from `0.6405885219573975` to `0.5465688705444336`.
- Diagonal margin mean improved from `0.15557986497879028` to `0.18054300546646118`.
- Retrieval top-1 remained `0.96875`.
- Latent MSE worsened from `0.6976713538169861` to `1.3141343593597412`.
- Norm ratio moved farther from 1.0, from `0.9616261720657349` to `0.9153322577476501`.

## Interpretation

The LIP-LOSS-001 configuration reduced off-diagonal similarity and improved
in-sample diagonal margin, suggesting stronger contrastive separation pressure.
However, it worsened reconstruction and norm calibration compared with the
previous lambda_mse `0.35` baseline.

This indicates the new loss is directionally useful, but its first weighting
over-penalizes ranking/margin relative to reconstruction and norm calibration.
This remains latent-space diagnostic evaluation only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, generalization, or production readiness
claim.
