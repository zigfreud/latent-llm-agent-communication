# LIP-EVAL-006 Lambda MSE 0.25 Bridge Evaluation

## Summary

This entry records the latent-space bridge evaluation for
`LIP-TRAIN-003-data001-steps32-lambda025`, a configurable 32-step remote
training run using `lambda_mse=0.25` over the LIP-DATA-001 8-sample real latent
bundle.

## Recorded Evidence

Input checkpoint:

- Source: LIP-TRAIN-003 configurable remote training run
- Experiment ID: `LIP-TRAIN-003-data001-steps32-lambda025`
- Run ID: `28010180058`
- Artifact ID: `7814359573`
- Artifact digest: `sha256:2be644900eea715c9baa39682f916df5bb316f04fe01685178b50bdbfc4a664e`
- Checkpoint used: `best_model.pth`

Evaluation artifact:

- Eval zip SHA256: `b86fdbbb00a2d3063f87483ace9c2328f1d355026498e1b09fa78f6fb242f9b8`

Input bundle:

- Trace ID: `LIP-DATA-001`
- Source model: `deepseek-ai/deepseek-coder-1.3b-base`
- Target model: `NousResearch/Meta-Llama-3-8B-Instruct`
- Dataset origin: `LIP-DATA-001 curated 8-prompt code task set`
- Dimensions: `2048 -> 4096`
- Samples: `8`
- Source layer: `-1`
- Target layer: `-1`
- Shard path: `shards/shard_0.pt`
- Shard SHA256: `015d0bec1889f3aeb2b92356def0c5a9abaca83fbee9f8f2a6081619bd1ea0d6`
- Validation status: `passed`

## Metric Values

- Sample count: `8`
- Latent MSE mean: `0.9348428249359131`
- Latent RMSE mean: `0.9603360295295715`
- Cosine diagonal mean: `0.9219813942909241`
- Cosine diagonal std: `0.01689634472131729`
- Prediction norm mean: `138.89547729492188`
- Target norm mean: `157.0719757080078`
- Norm ratio mean: `0.8842504620552063`
- Energy drift mean: `18.17650032043457`
- Retrieval top-1: `0.625`
- Retrieval top-k: `{"1": 0.625, "5": 1.0}`
- Off-diagonal cosine mean: `0.8649646639823914`
- Diagonal margin mean: `0.019980110228061676`

## Metric Comparison

Compared with `lambda_mse=0.1` from LIP-EVAL-004:

- MSE/RMSE improved.
- Cosine diagonal mean improved.
- Norm ratio improved from `0.7536500096321106` to `0.8842504620552063`.
- Energy drift improved from `38.70719909667969` to `18.17650032043457`.
- Retrieval top-1 was preserved at `0.625`.
- Retrieval top-5 was preserved at `1.0`.
- Diagonal margin improved from `0.013561271131038666` to `0.019980110228061676`.

Compared with `lambda_mse=0.5` from LIP-EVAL-005:

- MSE/RMSE and energy calibration were not as strong as `lambda_mse=0.5`.
- Retrieval top-1 improved from `0.5` to `0.625`.
- Retrieval top-5 improved from `0.875` to `1.0`.
- Diagonal margin improved from `0.001561969518661499` to `0.019980110228061676`.
- The diagonal/offdiag cosine gap improved from approximately `0.03242` to approximately `0.05702`.

## Interpretation

`lambda_mse=0.25` is the best current compromise between reconstruction/energy
calibration and pair-level discrimination. It preserves `retrieval_top1` from
`lambda_mse=0.1`, improves energy/norm substantially, and avoids the
discrimination drop observed at `lambda_mse=0.5`.

This remains in-sample, tiny-scale, latent-only evaluation.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space, in-sample, tiny-scale evaluation only. No semantic transfer or
generalization claim.
