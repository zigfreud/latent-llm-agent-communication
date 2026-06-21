# LIP-EVAL-002 First Real Bridge Evaluation Baseline

## Summary

This entry records the first latent-space bridge evaluation over the first real
remote H0-003 checkpoint and the real H0-005 latent bundle. It builds on the
LIP-EXP-001 remote training record and uses the deterministic bridge evaluator
added in LIP-EVAL-001.

## Recorded Evidence

Input checkpoint:

- Source: H0-003 real remote run artifact
- Run ID: `27833132397`
- Job ID: `82374579828`
- Artifact ID: `7752805337`
- Artifact name: `LIP-H0-003-27833132397-remote-cpu-from-latent-bundle`
- Artifact digest: `sha256:18dcdb307a7e992d113e0b0f3f89fafc1dbd680749ffb4ee64b4a1bae7d8d579`
- Checkpoint used: `best_model.pth`

Input bundle:

- Trace ID: `LIP-H0-005`
- Source model: `deepseek-ai/deepseek-coder-1.3b-base`
- Target model: `NousResearch/Meta-Llama-3-8B-Instruct`
- Dataset origin: `LIP-H0-005 curated tiny prompt set`
- Dimensions: `2048 -> 4096`
- Samples: `2`
- Source layer: `-1`
- Target layer: `-1`
- Shard path: `shards/shard_0.pt`
- Shard SHA256: `f40cd861383dfd2dadfd58f0b9fe727a03df1b976e5f26d63891a28b3496dfc3`
- Extraction notes: Real hidden-state extraction; raw prompts and model text
  outputs are not stored in shards.

Evaluation command:

```bash
python -m src.scripts.evaluate_bridge \
  --config config/LIP-EVAL-001_bridge_eval.yaml \
  --checkpoint runs/LIP-H0-003-real-remote/best_model.pth \
  --bundle-dir datasets/LIP-H0-003/latent_bundle \
  --output-dir runs/LIP-EVAL-002
```

## What Was Evaluated

- The trained H0-003 adapter checkpoint was evaluated against the paired
  source and target vectors from the H0-005 real tiny latent bundle.
- The evaluator ran on CPU with `input_dim=2048`, `hidden_dim=1024`, and
  `output_dim=4096`.
- The latent bundle validation status was `passed`.

Metric values:

- Sample count: `2`
- Latent MSE mean: `6.012866973876953`
- Latent RMSE mean: `2.452113151550293`
- Cosine diagonal mean: `0.11214403808116913`
- Cosine diagonal std: `0.013238072395324707`
- Prediction norm mean: `35.53398132324219`
- Target norm mean: `156.8994140625`
- Norm ratio mean: `0.22647474706172943`
- Energy drift mean: `121.36544799804688`
- Retrieval top-1: `1.0`
- Retrieval top-k: `{"1": 1.0}`
- Off-diagonal cosine mean: `0.07172273844480515`
- Diagonal margin mean: `0.04042129963636398`

## What The Metrics Suggest

- `retrieval_top1=1.0` over two samples is a positive operational signal, but
  it is statistically weak because `sample_count=2`.
- `cosine_diag_mean=0.1121` indicates weak absolute directional alignment.
- `diagonal_margin_mean=0.0404` indicates the correct target was only slightly
  closer than the best incorrect target.
- `norm_ratio_mean=0.2265` and `energy_drift_mean=121.3654` indicate substantial
  target-space energy/norm mismatch.
- This baseline suggests the bridge has early discriminative signal, but needs
  larger bundles, more training, and energy/norm calibration.

## What The Metrics Do Not Prove

- They do not prove semantic transfer.
- They do not prove text-level fidelity.
- They do not prove generalization beyond the two evaluated samples.
- They do not prove model-to-model alignment.
- They do not prove production readiness.

## Next Recommended Experiment

Run the same evaluator on a larger held-out latent bundle after longer bridge
training. The next run should separate training and evaluation prompts, preserve
the same manifest and checksum discipline, and track both retrieval margins and
norm/energy calibration.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

This PR registers the first real latent-space bridge evaluation baseline. It
does not claim semantic transfer, text-level fidelity, model-to-model
alignment, or production readiness.
