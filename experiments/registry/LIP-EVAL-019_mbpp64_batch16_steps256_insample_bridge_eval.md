# LIP-EVAL-019 MBPP64 Batch16 256-Step In-Sample Bridge Evaluation

## Summary

This entry records the train-set in-sample latent-space bridge evaluation for
`LIP-TRAIN-010-mbpp64-batch16-steps256-lambda025`. The checkpoint was trained on
the LIP-DATA-004 MBPP64 train bundle with batch size 16 for 256 steps and
evaluated on the same train bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-010-mbpp64-batch16-steps256-lambda025`
- Run ID: `28601028796`
- Artifact ID: `8043328665`
- Artifact digest: `sha256:50822147964e3527fbf2abe56b3c6b9811239dcd9bc5a58ee5dc02c423f13d3a`

In-sample eval bundle:

- Bundle trace: `LIP-DATA-004-MBPP-TRAIN`
- Dataset: `google-research-datasets/mbpp`
- Split: `train`
- Samples: `64`
- Train shard SHA256: `f1dd527799ad2dd0e99edc253ac71213c9db4db36d9cf6b702e3a6ffc126e811`
- Eval zip SHA256: `a5e664f7d7c2050d8bb6bcf40263cb1337edb7b3dac304780850c516ada2a762`

## Metrics

- Sample count: `64`
- Latent MSE mean: `0.957990825176239`
- Latent RMSE mean: `0.9728941917419434`
- Cosine diagonal mean: `0.9182648062705994`
- Cosine diagonal std: `0.016101522371172905`
- Prediction norm mean: `149.13897705078125`
- Target norm mean: `157.84063720703125`
- Norm ratio mean: `0.9449038505554199`
- Energy drift mean: `8.823211669921875`
- Retrieval top-1: `0.96875`
- Retrieval top-k: `{"1": 0.96875, "5": 1.0}`
- Off-diagonal cosine mean: `0.5994787216186523`
- Diagonal margin mean: `0.167476087808609`

## Interpretation

Batch16 steps256 improved in-sample reconstruction, margin, norm ratio, and
energy calibration relative to batch16 steps128. In-sample retrieval_top1
remained `0.96875`, while additional optimizer updates improved train fit.

This remains latent-space diagnostic evaluation only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, or production readiness claim.
