# LIP-EVAL-027 LIP-LOSS-001 Reverse NCE 0.5 MBPP64 In-Sample Bridge Evaluation

## Summary

This entry records the train-set in-sample latent-space bridge evaluation for
`LIP-TRAIN-014-mbpp64-batch16-steps256-l035-rnce05-margin005-norm01`. The
checkpoint was trained on the LIP-DATA-004 MBPP64 train bundle with
`lambda_reverse_nce=0.5`, `lambda_margin=0.05`, and `lambda_norm=0.10`, then
evaluated on the same train bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-014-mbpp64-batch16-steps256-l035-rnce05-margin005-norm01`
- Run ID: `28617086460`
- Artifact ID: `8049893862`
- Artifact digest: `sha256:987dde166ce79ad3e340951cd0eb83ccbbe8fc6fa79f38eba740e6c2a2501dab`

Train eval bundle:

- Bundle trace: `LIP-DATA-004-MBPP-TRAIN`
- Dataset: `google-research-datasets/mbpp`
- Split: `train`
- Samples: `64`
- Eval shard SHA256: `f1dd527799ad2dd0e99edc253ac71213c9db4db36d9cf6b702e3a6ffc126e811`
- Eval zip SHA256: `1145d0b0882118f9643bd017678dd3fc94212241e5386630b1468e4a1843dc14`

## Metrics

- Sample count: `64`
- Latent MSE mean: `1.0395598411560059`
- Latent RMSE mean: `1.0125335454940796`
- Cosine diagonal mean: `0.9110040068626404`
- Cosine diagonal std: `0.019608071073889732`
- Prediction norm mean: `148.40916442871094`
- Target norm mean: `157.84063720703125`
- Norm ratio mean: `0.9402974247932434`
- Energy drift mean: `9.502910614013672`
- Retrieval top-1: `0.96875`
- Retrieval top-k: `{"1": 0.96875, "5": 1.0}`
- Off-diagonal cosine mean: `0.5869627594947815`
- Diagonal margin mean: `0.17066794633865356`

## Comparison Against LIP-EVAL-025

Compared with the adjusted LIP-LOSS-001 in-sample result using
`lambda_reverse_nce=1.0`:

- Latent MSE improved from `1.312156319618225` to `1.0395598411560059`.
- Cosine diagonal mean improved from `0.8860499262809753` to `0.9110040068626404`.
- Norm ratio mean improved from `0.9174784421920776` to `0.9402974247932434`.
- Energy drift improved from `13.034022331237793` to `9.502910614013672`.
- Retrieval top-1 remained `0.96875`.
- Diagonal margin mean decreased from `0.18019123375415802` to `0.17066794633865356`.

## Interpretation

Reducing reverse NCE from `1.0` to `0.5` recovered part of the reconstruction
and norm/energy degradation seen in earlier LIP-LOSS-001 runs while preserving
in-sample retrieval. In-sample margin decreased but remained positive.

This suggests reverse NCE was too strong at `1.0`, but this remains an
in-sample latent-space diagnostic, not a semantic-fidelity claim.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, generalization, or production readiness
claim.
