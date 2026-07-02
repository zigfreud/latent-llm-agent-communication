# LIP-EVAL-020 MBPP32 Batch16 256-Step Held-Out Bridge Evaluation

## Summary

This entry records the held-out latent-space bridge evaluation for
`LIP-TRAIN-010-mbpp64-batch16-steps256-lambda025`. The checkpoint was trained on
the LIP-DATA-004 MBPP64 train bundle with batch size 16 for 256 steps and
evaluated on the separate MBPP32 validation bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-010-mbpp64-batch16-steps256-lambda025`
- Run ID: `28601028796`
- Artifact ID: `8043328665`
- Artifact digest: `sha256:50822147964e3527fbf2abe56b3c6b9811239dcd9bc5a58ee5dc02c423f13d3a`

Held-out eval bundle:

- Bundle trace: `LIP-DATA-004-MBPP-EVAL`
- Dataset: `google-research-datasets/mbpp`
- Split: `validation`
- Samples: `32`
- Eval shard SHA256: `7e58a9afcb2733ee3a7110eff757b3ffd47ae0391c526639f13f30714d60ce8e`
- Eval zip SHA256: `f7ae83ef1e812ae8a591da5f2e4c74abc7cbbda9387cc97a1521db5fa8aca328`

## Metrics

- Sample count: `32`
- Latent MSE mean: `2.3901524543762207`
- Latent RMSE mean: `1.5336017608642578`
- Cosine diagonal mean: `0.7872247695922852`
- Cosine diagonal std: `0.05290355160832405`
- Prediction norm mean: `141.07305908203125`
- Target norm mean: `158.02142333984375`
- Norm ratio mean: `0.8927412629127502`
- Energy drift mean: `16.948368072509766`
- Retrieval top-1: `0.34375`
- Retrieval top-k: `{"1": 0.34375, "5": 0.71875}`
- Off-diagonal cosine mean: `0.6915251016616821`
- Diagonal margin mean: `-0.007311269640922546`

## Interpretation

On held-out evaluation, top5 and diagonal margin improved relative to batch16
steps128, but top1 decreased slightly from `0.375` to `0.34375` and
reconstruction/norm calibration worsened.

The result suggests additional optimizer updates improve train fit and move
correct held-out targets closer to the top, but do not yet produce robust top1
held-out ranking. This remains latent-space diagnostic evaluation only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, or production readiness claim.
