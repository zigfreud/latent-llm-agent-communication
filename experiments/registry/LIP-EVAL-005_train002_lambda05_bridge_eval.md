# LIP-EVAL-005 Lambda MSE 0.5 Bridge Evaluation

## Summary

This entry records the latent-space bridge evaluation for the LIP-TRAIN-002
configurable 32-step remote training run using `lambda_mse=0.5` over the
LIP-DATA-001 8-sample real latent bundle. It compares the result with
LIP-EVAL-004, which used the same bundle and training length with
`lambda_mse=0.1`.

## Recorded Evidence

Input checkpoint:

- Source: LIP-TRAIN-002 configurable remote training run
- Experiment ID: `LIP-TRAIN-002-data001-steps32-lambda05`
- Checkpoint used: `best_model.pth`
- Device: `cpu`
- Samples: `8`
- Batch size: `2`
- Epochs requested: `8`
- Max steps: `32`
- Steps completed: `32`
- Learning rate: `0.0001`
- Lambda MSE: `0.5`
- Best loss: `1.088736355304718`
- Final loss: `1.088736355304718`
- Final accuracy: `0.5`

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
- Extraction notes: Real hidden-state extraction; raw prompts and model text
  outputs are not stored in shards.
- Validation status: `passed`

Evaluation command:

```bash
python -m src.scripts.evaluate_bridge \
  --config config/LIP-EVAL-001_bridge_eval.yaml \
  --checkpoint /content/drive/MyDrive/lip-artifacts/LIP-TRAIN-002-data001-steps32-lambda05/best_model.pth \
  --bundle-dir datasets/LIP-H0-003/latent_bundle \
  --output-dir runs/LIP-EVAL-005
```

## What Was Evaluated

- The LIP-TRAIN-002 adapter checkpoint trained for 32 CPU steps with
  `lambda_mse=0.5`.
- The paired source and target latent vectors from the real LIP-DATA-001 bundle.
- CPU evaluation with `input_dim=2048`, `hidden_dim=1024`, and `output_dim=4096`.
- The latent bundle validation status was `passed`.

Metric values:

- Sample count: `8`
- Latent MSE mean: `0.7958155870437622`
- Latent RMSE mean: `0.887549877166748`
- Cosine diagonal mean: `0.9325576424598694`
- Cosine diagonal std: `0.01429613959044218`
- Prediction norm mean: `145.36990356445312`
- Target norm mean: `157.0719757080078`
- Norm ratio mean: `0.9254684448242188`
- Energy drift mean: `11.702067375183105`
- Retrieval top-1: `0.5`
- Retrieval top-k: `{"1": 0.5, "5": 0.875}`
- Off-diagonal cosine mean: `0.9001398682594299`
- Diagonal margin mean: `0.001561969518661499`

## Metric Comparison

Compared with LIP-EVAL-004 (`lambda_mse=0.1`):

- `latent_mse_mean` improved from `1.9026975631713867` to `0.7958155870437622`.
- `latent_rmse_mean` improved from `1.3739054203033447` to `0.887549877166748`.
- `cosine_diag_mean` improved from `0.8337277770042419` to `0.9325576424598694`.
- `norm_ratio_mean` improved from `0.7536500096321106` to `0.9254684448242188`.
- `energy_drift_mean` improved from `38.70719909667969` to `11.702067375183105`.
- `retrieval_top1` decreased from `0.625` to `0.5`.
- `retrieval_top5` decreased from `1.0` to `0.875`.
- `diagonal_margin_mean` decreased from `0.013561271131038666` to `0.001561969518661499`.
- `cosine_diag_minus_offdiag` decreased from approximately `0.08955` to approximately `0.03242`.

## Interpretation

- `lambda_mse=0.5` substantially improved reconstruction, norm ratio, and
  energy calibration compared with `lambda_mse=0.1`.
- Pair-level discrimination weakened: `retrieval_top1` and `retrieval_top5`
  decreased, and `diagonal_margin_mean` became barely positive.
- `cosine_diag_mean` and `offdiag_cosine_mean` are both high, indicating strong
  global target-space alignment but weak separation between correct and
  incorrect targets.
- This suggests a tradeoff between MSE-driven reconstruction/energy calibration
  and contrastive pair discrimination.
- The result motivates an intermediate `lambda_mse` test, likely
  `lambda_mse=0.25`, before changing dataset size or architecture.
- This remains in-sample, tiny-scale, latent-only evaluation.

## What The Metrics Do Not Prove

- They do not prove semantic transfer.
- They do not prove text-level fidelity.
- They do not prove generalization beyond this tiny in-sample bundle.
- They do not prove model-to-model alignment.
- They do not prove production readiness.

## Next Recommended Experiment

Run the same configurable training and evaluation path with an intermediate
`lambda_mse`, likely `0.25`, to test whether it preserves the improved
norm/energy calibration while recovering stronger pair-level retrieval and
diagonal margin.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

This PR registers a latent-space bridge evaluation result for a configurable
32-step remote training run with `lambda_mse=0.5`. It does not claim semantic
transfer, text-level fidelity, model-to-model alignment, generalization, or
production readiness.
