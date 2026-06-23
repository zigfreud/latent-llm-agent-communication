# LIP-EVAL-013 MBPP32 128-Step In-Sample Bridge Evaluation

## Summary

This entry records the train-set in-sample latent-space bridge evaluation for
`LIP-TRAIN-007-mbpp32-batch8-steps128-lambda025`. The checkpoint was trained on
the LIP-DATA-003 MBPP32 train bundle for 128 steps and evaluated on the same
train bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-007-mbpp32-batch8-steps128-lambda025`
- Run ID: `28024709060`
- Artifact ID: `7820197833`
- Artifact digest: `sha256:708953416f86020f287df6c1d8166f40cd5cee9cc87091e95abd2b474eef6d52`

In-sample eval bundle:

- Bundle trace: `LIP-DATA-003-MBPP-TRAIN`
- Dataset: `google-research-datasets/mbpp`
- Split: `train`
- Samples: `32`
- Train shard SHA256: `c0e3eb2c8832be89f61e8033407853e48d402f761d3918b75c3a0291b02373a5`
- Eval zip SHA256: `d69035ae7d67dd5978749be50777b6b7ba09c3a27aa366991155418a78e70113`

## Metrics

- Sample count: `32`
- Latent MSE mean: `0.8109225630760193`
- Latent RMSE mean: `0.8954616189002991`
- Cosine diagonal mean: `0.9310228824615479`
- Cosine diagonal std: `0.0140335438773036`
- Prediction norm mean: `146.5041961669922`
- Target norm mean: `157.84262084960938`
- Norm ratio mean: `0.9281337261199951`
- Energy drift mean: `11.338427543640137`
- Retrieval top-1: `1.0`
- Retrieval top-k: `{"1": 1.0, "5": 1.0}`
- Off-diagonal cosine mean: `0.6541506052017212`
- Diagonal margin mean: `0.1489599645137787`

## Interpretation

Doubling steps from 64 to 128 improved in-sample reconstruction, margin, and
calibration. In-sample retrieval_top1 remained perfect while diagonal margin
increased substantially relative to LIP-EVAL-011.

This suggests training time helps the MBPP32 train-set latent fit. This remains
latent-space diagnostic evaluation only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, or production readiness claim.
