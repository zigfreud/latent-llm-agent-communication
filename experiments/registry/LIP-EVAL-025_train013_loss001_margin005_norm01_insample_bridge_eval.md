# LIP-EVAL-025 LIP-LOSS-001 Adjusted Margin/Norm MBPP64 In-Sample Bridge Evaluation

## Summary

This entry records the train-set in-sample latent-space bridge evaluation for
`LIP-TRAIN-013-mbpp64-batch16-steps256-l035-rnce1-margin005-norm01`. The
checkpoint was trained on the LIP-DATA-004 MBPP64 train bundle with an adjusted
LIP-LOSS-001 weighting and evaluated on the same train bundle.

The adjusted weighting reduced `lambda_margin` from `0.10` to `0.05` and
increased `lambda_norm` from `0.05` to `0.10` relative to the first LIP-LOSS-001
run.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-013-mbpp64-batch16-steps256-l035-rnce1-margin005-norm01`
- Run ID: `28616101597`
- Artifact ID: `8049502502`
- Artifact digest: `sha256:136313d110ab4103e89f731ad2c5636a5ae70bf6bf237a7895aeb1dd7cfbcff1`

Train eval bundle:

- Bundle trace: `LIP-DATA-004-MBPP-TRAIN`
- Dataset: `google-research-datasets/mbpp`
- Split: `train`
- Samples: `64`
- Eval shard SHA256: `f1dd527799ad2dd0e99edc253ac71213c9db4db36d9cf6b702e3a6ffc126e811`
- Eval zip SHA256: `a96616738c0304adfa4d9dc1ff91682704a6e833c5d85322aebedf352023ceaa`

## Metrics

- Sample count: `64`
- Latent MSE mean: `1.312156319618225`
- Latent RMSE mean: `1.138636827468872`
- Cosine diagonal mean: `0.8860499262809753`
- Cosine diagonal std: `0.0237271748483181`
- Prediction norm mean: `144.80661010742188`
- Target norm mean: `157.84063720703125`
- Norm ratio mean: `0.9174784421920776`
- Energy drift mean: `13.034022331237793`
- Retrieval top-1: `0.96875`
- Retrieval top-k: `{"1": 0.96875, "5": 1.0}`
- Off-diagonal cosine mean: `0.5471564531326294`
- Diagonal margin mean: `0.18019123375415802`

## Comparison Against LIP-EVAL-023

Compared with the first LIP-LOSS-001 in-sample result:

- Latent MSE improved slightly from `1.3141343593597412` to `1.312156319618225`.
- Cosine diagonal mean improved slightly from `0.8857991099357605` to `0.8860499262809753`.
- Norm ratio mean improved slightly from `0.9153322577476501` to `0.9174784421920776`.
- Energy drift improved slightly from `13.37309741973877` to `13.034022331237793`.
- Retrieval top-1 remained `0.96875`.
- Diagonal margin mean moved from `0.18054300546646118` to `0.18019123375415802`.

## Interpretation

The adjusted LIP-LOSS-001 weighting slightly improves in-sample reconstruction,
cosine diagonal, norm ratio, and energy drift relative to the first LIP-LOSS-001
run. In-sample retrieval remains strong and diagonal margin remains positive.

This is a marginal in-sample improvement, not a semantic-fidelity claim.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, generalization, or production readiness
claim.
