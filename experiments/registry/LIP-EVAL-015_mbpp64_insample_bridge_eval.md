# LIP-EVAL-015 MBPP64 In-Sample Bridge Evaluation

## Summary

This entry records the train-set in-sample latent-space bridge evaluation for
`LIP-TRAIN-008-mbpp64-batch8-steps256-lambda025`. The checkpoint was trained on
the LIP-DATA-004 MBPP64 train bundle and evaluated on the same train bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-008-mbpp64-batch8-steps256-lambda025`
- Run ID: `28028527051`
- Artifact ID: `7821777879`
- Artifact digest: `sha256:ad4624dd539c84930ef938cb8cf56f3af3bc5435833eb71694c505fbfe328ba4`

In-sample eval bundle:

- Bundle trace: `LIP-DATA-004-MBPP-TRAIN`
- Dataset: `google-research-datasets/mbpp`
- Split: `train`
- Samples: `64`
- Train shard SHA256: `f1dd527799ad2dd0e99edc253ac71213c9db4db36d9cf6b702e3a6ffc126e811`
- Eval zip SHA256: `0aa62311a3baf357fcf57b78c9ad09566683ae059fff6a346bb59bc71d14b110`

## Metrics

- Sample count: `64`
- Latent MSE mean: `0.7554762363433838`
- Latent RMSE mean: `0.8656440377235413`
- Cosine diagonal mean: `0.9363442659378052`
- Cosine diagonal std: `0.010999363847076893`
- Prediction norm mean: `151.9931182861328`
- Target norm mean: `157.84063720703125`
- Norm ratio mean: `0.9629573822021484`
- Energy drift mean: `5.992755889892578`
- Retrieval top-1: `0.96875`
- Retrieval top-k: `{"1": 0.96875, "5": 1.0}`
- Off-diagonal cosine mean: `0.6739262938499451`
- Diagonal margin mean: `0.1016358882188797`

## Interpretation

Scaling from MBPP32/16 to MBPP64/32 improved reconstruction, norm ratio, and
energy calibration. In-sample retrieval remains strong at `0.96875`, indicating
that the bridge learns useful latent geometry on the larger train bundle.

This remains latent-space diagnostic evaluation only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, or production readiness claim.
