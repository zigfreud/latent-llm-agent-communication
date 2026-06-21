# LIP-EVAL-003 8-Sample Real Bridge Evaluation Baseline

## Summary

This entry records the latent-space bridge evaluation over the LIP-DATA-001
8-sample real latent bundle and the H0-003 remote CPU checkpoint produced from
that bundle. It extends the 2-sample LIP-EVAL-002 baseline with a larger real
bundle while preserving the same evaluation boundary: latent-space metrics
only.

## Recorded Evidence

Input checkpoint:

- Source: H0-003 remote run artifact for LIP-DATA-001
- Run ID: `27903029567`
- Artifact ID: `7774881054`
- Artifact name: `LIP-H0-003-27903029567-remote-cpu-from-latent-bundle`
- Artifact digest: `sha256:125c803450f6cbd5288bf90ef6fca63faa254336ab18d3d70ffe4aee9f68263a`
- Checkpoint used: `best_model.pth`
- Head branch: `main`
- Head SHA: `a95427fe300b696c16016237658388238d30e576`

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

Remote training summary:

- Experiment ID: `LIP-H0-003`
- Samples: `8`
- Batch size: `2`
- Max steps: `2`
- Steps completed: `2`
- Best loss: `1.2974371314048767`
- Final loss: `1.2974371314048767`
- Final accuracy: `0.5`
- Step 1: `loss=1.305346`, `nce_loss=0.680488`, `mse_loss=6.248584`, `accuracy=0.5`
- Step 2: `loss=1.289528`, `nce_loss=0.700090`, `mse_loss=5.894382`, `accuracy=0.5`

Evaluation command:

```bash
python -m src.scripts.evaluate_bridge \
  --config config/LIP-EVAL-001_bridge_eval.yaml \
  --checkpoint runs/LIP-H0-003-data-001-remote/best_model.pth \
  --bundle-dir datasets/LIP-H0-003/latent_bundle \
  --output-dir runs/LIP-EVAL-003
```

## What Was Evaluated

- The H0-003 adapter checkpoint trained from the LIP-DATA-001 8-sample bundle.
- The paired source and target latent vectors from the real LIP-DATA-001 bundle.
- CPU evaluation with `input_dim=2048`, `hidden_dim=1024`, and `output_dim=4096`.
- The latent bundle validation status was `passed`.

Metric values:

- Sample count: `8`
- Latent MSE mean: `4.753047943115234`
- Latent RMSE mean: `2.177975654602051`
- Cosine diagonal mean: `0.45896223187446594`
- Cosine diagonal std: `0.07422509789466858`
- Prediction norm mean: `72.56733703613281`
- Target norm mean: `157.0719757080078`
- Norm ratio mean: `0.46201658248901367`
- Energy drift mean: `84.504638671875`
- Retrieval top-1: `0.125`
- Retrieval top-k: `{"1": 0.125, "5": 0.625}`
- Off-diagonal cosine mean: `0.4574820101261139`
- Diagonal margin mean: `-0.11237937957048416`

## Metric Comparison

Compared with LIP-EVAL-002:

- `latent_mse_mean` improved from `6.012866973876953` to `4.753047943115234`.
- `latent_rmse_mean` improved from `2.452113151550293` to `2.177975654602051`.
- `cosine_diag_mean` improved from `0.11214403808116913` to `0.45896223187446594`.
- `norm_ratio_mean` improved from `0.22647474706172943` to `0.46201658248901367`.
- `energy_drift_mean` improved from `121.36544799804688` to `84.504638671875`.
- `retrieval_top1` dropped from `1.0` at `n=2` to `0.125` at `n=8`, which equals random chance for 8 candidates.
- `diagonal_margin_mean` moved from `+0.04042129963636398` to `-0.11237937957048416`.

## Interpretation

- The 8-sample run shows stronger global latent-space alignment than the
  2-sample baseline: lower MSE/RMSE, higher matching cosine, better norm ratio,
  and lower energy drift.
- Retrieval metrics do not yet show pair-level discrimination.
  `retrieval_top1=0.125` equals chance for 8 candidates; `retrieval_top5=0.625`
  equals chance for top-5 among 8 candidates.
- `cosine_diag_mean` and `offdiag_cosine_mean` are nearly identical, indicating
  that predictions are entering a shared target-space region but not separating
  the correct target from incorrect targets.
- `diagonal_margin_mean` is negative, indicating that the correct target is
  often not more similar than the strongest incorrect target.
- The likely cause is undertraining: the H0-003 workflow used `max_steps=2`, so
  the model saw only two mini-batches.
- The next technical step should be a configurable remote training workflow
  with larger `max_steps`/epochs and possibly stronger norm/MSE calibration.

## What The Metrics Do Not Prove

- They do not prove semantic transfer.
- They do not prove text-level fidelity.
- They do not prove generalization beyond this tiny bundle.
- They do not prove model-to-model alignment.
- They do not prove production readiness.

## Next Recommended Experiment

Run a configurable remote H0-003 training workflow over LIP-DATA-001 or a larger
held-out bundle with more steps and epochs. Track whether additional training
improves diagonal margin and retrieval above chance while preserving the
observed improvements in MSE, cosine, norm ratio, and energy drift.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

This PR registers a latent-space bridge evaluation baseline for an 8-sample
real bundle. It does not claim semantic transfer, text-level fidelity,
model-to-model alignment, generalization, or production readiness.
