# LIP-EVAL-029 LIP-LOSS-001 Reverse NCE 0.0 MBPP64 In-Sample Bridge Evaluation

## Summary

This entry records the train-set in-sample latent-space bridge evaluation for
`LIP-TRAIN-015-mbpp64-batch16-steps256-l035-rnce0-margin005-norm01`. The
checkpoint was trained on the LIP-DATA-004 MBPP64 train bundle with
`lambda_reverse_nce=0.0`, `lambda_margin=0.05`, and `lambda_norm=0.10`, then
evaluated on the same train bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-015-mbpp64-batch16-steps256-l035-rnce0-margin005-norm01`
- Run ID: `28625926129`
- Artifact ID: `8053318553`
- Artifact digest: `sha256:aae5be93368bfeb1cedcb5027778454db477e71b03a2b834e86858b704f80489`

Train eval bundle:

- Bundle trace: `LIP-DATA-004-MBPP-TRAIN`
- Dataset: `google-research-datasets/mbpp`
- Split: `train`
- Samples: `64`
- Eval shard SHA256: `f1dd527799ad2dd0e99edc253ac71213c9db4db36d9cf6b702e3a6ffc126e811`
- Eval zip SHA256: `178ca37d48ba9da37146be9934dba4e83c9da8c1b523bb2bdae9fb3ec120613c`

## Metrics

- Sample count: `64`
- Latent MSE mean: `0.6863064765930176`
- Latent RMSE mean: `0.8237009048461914`
- Cosine diagonal mean: `0.9422373175621033`
- Cosine diagonal std: `0.011055461131036282`
- Prediction norm mean: `152.45980834960938`
- Target norm mean: `157.84063720703125`
- Norm ratio mean: `0.965922474861145`
- Energy drift mean: `5.600724220275879`
- Retrieval top-1: `0.96875`
- Retrieval top-k: `{"1": 0.96875, "5": 1.0}`
- Off-diagonal cosine mean: `0.6418929100036621`
- Diagonal margin mean: `0.15451475977897644`

## Comparison Against LIP-EVAL-027

Compared with the reverse NCE `0.5` in-sample result:

- Latent MSE improved from `1.0395598411560059` to `0.6863064765930176`.
- Cosine diagonal mean improved from `0.9110040068626404` to `0.9422373175621033`.
- Norm ratio mean improved from `0.9402974247932434` to `0.965922474861145`.
- Energy drift improved from `9.502910614013672` to `5.600724220275879`.
- Retrieval top-1 remained `0.96875`.
- Diagonal margin mean decreased from `0.17066794633865356` to `0.15451475977897644`.

## Interpretation

Disabling reverse NCE recovered in-sample positive-target geometry and
calibration while preserving retrieval. In-sample diagonal margin decreased,
which is consistent with less pair-separation pressure.

The result suggests reverse NCE was the main source of geometry degradation in
earlier LIP-LOSS-001 runs. This remains latent-space diagnostic evaluation only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, generalization, or production readiness
claim.
