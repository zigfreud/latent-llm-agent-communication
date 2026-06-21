# LIP-EVAL-004 Configurable 32-Step Bridge Evaluation

## Summary

This entry records the latent-space bridge evaluation over the
LIP-TRAIN-001 configurable 32-step remote training run on the LIP-DATA-001
8-sample real latent bundle. It compares that result with the LIP-EVAL-003
two-step H0-003 smoke-training baseline over the same bundle.

## Recorded Evidence

Input checkpoint:

- Source: LIP-TRAIN-001 configurable remote training artifact
- Run ID: `27904764823`
- Job ID: `82571317401`
- Artifact ID: `7775415799`
- Artifact name: `LIP-TRAIN-001-27904764823-LIP-TRAIN-001-data001-steps32-lambda01`
- Artifact digest: `sha256:9874bf1c34206d36abf0deee1c7ab961ee7976088bf47c528551df219ae7fc69`
- Checkpoint used: `best_model.pth`
- Head branch: `main`
- Head SHA: `5ad298de516e29ec8c0973d2630cd5a5b9abd9d8`

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
- Bundle zip SHA256: `c733242c173ce23837d540f29b1769e668e000009c0eee80e8d173076d44b9e6`
- Extraction notes: Real hidden-state extraction; raw prompts and model text
  outputs are not stored in shards.

Training metadata:

- Experiment ID: `LIP-TRAIN-001-data001-steps32-lambda01`
- Samples: `8`
- Batch size: `2`
- Epochs requested: `8`
- Max steps: `32`
- Steps completed: `32`
- Learning rate: `0.0001`
- Lambda MSE: `0.1`
- Best loss: `0.7296363562345505`
- Final loss: `0.7296363562345505`
- Final accuracy: `0.75`

Evaluation command:

```bash
python -m src.scripts.evaluate_bridge \
  --config config/LIP-EVAL-001_bridge_eval.yaml \
  --checkpoint runs/LIP-TRAIN-001-data001-steps32-lambda01/best_model.pth \
  --bundle-dir datasets/LIP-H0-003/latent_bundle \
  --output-dir runs/LIP-EVAL-004
```

## What Was Evaluated

- The LIP-TRAIN-001 adapter checkpoint trained for 32 CPU steps.
- The paired source and target latent vectors from the real LIP-DATA-001 bundle.
- CPU evaluation with `input_dim=2048`, `hidden_dim=1024`, and `output_dim=4096`.
- The latent bundle validation status was `passed`.

Metric values:

- Sample count: `8`
- Latent MSE mean: `1.9026975631713867`
- Latent RMSE mean: `1.3739054203033447`
- Cosine diagonal mean: `0.8337277770042419`
- Cosine diagonal std: `0.034103695303201675`
- Prediction norm mean: `118.3647689819336`
- Target norm mean: `157.0719757080078`
- Norm ratio mean: `0.7536500096321106`
- Energy drift mean: `38.70719909667969`
- Retrieval top-1: `0.625`
- Retrieval top-k: `{"1": 0.625, "5": 1.0}`
- Off-diagonal cosine mean: `0.7441755533218384`
- Diagonal margin mean: `0.013561271131038666`

Per-sample retrieval:

- Correct top-1 samples: `0`, `1`, `4`, `6`, `7`
- Incorrect top-1 samples: `2`, `3`, `5`
- Incorrect samples `2`, `3`, and `5` were nearest to target `7`.

## Metric Comparison

Compared with LIP-EVAL-003:

- `latent_mse_mean` improved from `4.753047943115234` to `1.9026975631713867`.
- `latent_rmse_mean` improved from `2.177975654602051` to `1.3739054203033447`.
- `cosine_diag_mean` improved from `0.45896223187446594` to `0.8337277770042419`.
- `norm_ratio_mean` improved from `0.46201658248901367` to `0.7536500096321106`.
- `energy_drift_mean` improved from `84.504638671875` to `38.70719909667969`.
- `retrieval_top1` improved from `0.125` to `0.625`.
- `retrieval_top5` improved from `0.625` to `1.0`.
- `diagonal_margin_mean` improved from `-0.11237937957048416` to `+0.013561271131038666`.
- `cosine_diag_minus_offdiag` improved from approximately `0.00148` to approximately `0.08955`.

## Interpretation

- The 32-step configurable training run substantially improved latent-space
  reconstruction and alignment relative to the 2-step H0-003 run.
- `retrieval_top1=0.625` over 8 candidates is above random chance of `0.125`,
  indicating emerging pair-level discrimination.
- `diagonal_margin_mean` is now positive, but still small, indicating weak
  separation between correct and strongest incorrect targets.
- `cosine_diag_mean` and `offdiag_cosine_mean` are both high, suggesting
  predictions occupy a shared target-space region; however, the diagonal/offdiag
  gap is now meaningfully larger than in LIP-EVAL-003.
- `norm_ratio` and `energy_drift` improved substantially, suggesting better
  target-space energy calibration, but `norm_ratio` remains below `1.0`.
- Samples `2`, `3`, and `5` were misretrieved toward target `7`, suggesting
  possible local attraction/collapse around one target or insufficient
  pair-level separation.
- This is not yet semantic validation because evaluation remains in-sample,
  tiny-scale, and latent-only.

## What The Metrics Do Not Prove

- They do not prove semantic transfer.
- They do not prove text-level fidelity.
- They do not prove generalization beyond this tiny in-sample bundle.
- They do not prove model-to-model alignment.
- They do not prove production readiness.

## Next Recommended Experiment

Evaluate a held-out latent bundle after configurable training and track whether
retrieval top-1, diagonal margin, and the diagonal/offdiag cosine gap remain
above chance outside the training bundle. Also test whether additional
norm/energy calibration can bring `norm_ratio` closer to `1.0`.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

This PR registers a latent-space bridge evaluation result for a configurable
32-step remote training run. It does not claim semantic transfer, text-level
fidelity, model-to-model alignment, generalization, or production readiness.
