# LIP-EVAL-014 MBPP16 128-Step Held-Out Bridge Evaluation

## Summary

This entry records the held-out latent-space bridge evaluation for
`LIP-TRAIN-007-mbpp32-batch8-steps128-lambda025`. The checkpoint was trained on
the LIP-DATA-003 MBPP32 train bundle for 128 steps and evaluated on the separate
MBPP16 validation bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-007-mbpp32-batch8-steps128-lambda025`
- Run ID: `28024709060`
- Artifact ID: `7820197833`
- Artifact digest: `sha256:708953416f86020f287df6c1d8166f40cd5cee9cc87091e95abd2b474eef6d52`

Held-out eval bundle:

- Bundle trace: `LIP-DATA-003-MBPP-EVAL`
- Dataset: `google-research-datasets/mbpp`
- Split: `validation`
- Samples: `16`
- Eval shard SHA256: `0cac44ff2645160b9799ec56e8834e91a020db1b21f295dcf0e1d2ebabab6d12`
- Eval zip SHA256: `13b4a1d352cf6a303ca3509a4ae25e09003fe8b6a8ef20b34c90bbda551e017b`

## Metrics

- Sample count: `16`
- Latent MSE mean: `2.2415237426757812`
- Latent RMSE mean: `1.4806479215621948`
- Cosine diagonal mean: `0.8029100894927979`
- Cosine diagonal std: `0.06228332221508026`
- Prediction norm mean: `145.42359924316406`
- Target norm mean: `158.0408477783203`
- Norm ratio mean: `0.920122504234314`
- Energy drift mean: `12.617236137390137`
- Retrieval top-1: `0.375`
- Retrieval top-k: `{"1": 0.375, "5": 0.6875}`
- Off-diagonal cosine mean: `0.740384042263031`
- Diagonal margin mean: `-0.017510171979665756`

## Interpretation

Held-out retrieval improved from `0.3125` to `0.375`, and held-out energy/norm
calibration improved substantially relative to LIP-EVAL-012. However, held-out
diagonal margin remains negative, so pair-level generalization is still not
robust.

The result suggests training time helps, but data scale or loss/architecture
changes are likely needed next.

This remains latent-space diagnostic evaluation only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, or production readiness claim.
