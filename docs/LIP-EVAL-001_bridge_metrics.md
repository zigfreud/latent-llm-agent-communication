# LIP-EVAL-001 Bridge Metrics

## Purpose

LIP-EVAL-001 adds a deterministic bridge-level evaluator for trained
`LIPAdapter` checkpoints and validated latent bundles. The evaluator measures
latent-space behavior before scaling training or making any semantic claims.

## Inputs

The evaluator takes:

- a YAML config, such as `config/LIP-EVAL-001_bridge_eval.yaml`;
- a trained adapter checkpoint, either a raw `state_dict` or a trainer checkpoint
  containing `model_state`;
- a latent bundle directory that passes `src.scripts.validate_latent_bundle`.

Example:

```bash
python -m src.scripts.evaluate_bridge \
  --config config/LIP-EVAL-001_bridge_eval.yaml \
  --checkpoint runs/LIP-H0-003/best_model.pth \
  --bundle-dir datasets/LIP-H0-003/latent_bundle \
  --output-dir runs/LIP-EVAL-001
```

The evaluator does not run base LLMs, generate text, or download models.

## Outputs

The evaluator writes:

- `eval_metrics.json`: aggregate metrics and bundle validation metadata;
- `eval_pairs.csv`: per-sample metrics;
- `eval_summary.md`: short human-readable summary.

## Metrics

- `sample_count`: number of evaluated source/target vector pairs.
- `input_dim`: source vector dimension.
- `output_dim`: target vector dimension.
- `latent_mse_mean`: mean per-sample squared error between adapter prediction
  and target vector.
- `latent_rmse_mean`: mean per-sample square root of latent MSE.
- `cosine_diag_mean`: mean cosine similarity for matching prediction/target
  pairs.
- `cosine_diag_std`: population standard deviation of matching-pair cosine
  similarities.
- `prediction_norm_mean`: mean L2 norm of predicted target-space vectors.
- `target_norm_mean`: mean L2 norm of target vectors.
- `norm_ratio_mean`: mean `||prediction|| / ||target||`.
- `energy_drift_mean`: mean absolute difference between prediction and target
  vector norms.
- `retrieval_top1`: fraction where the matching target vector is the nearest
  target by cosine similarity.
- `retrieval_topk`: configured top-k retrieval fractions when `k <= sample_count`.
- `offdiag_cosine_mean`: mean non-matching prediction/target cosine similarity
  when `sample_count > 1`.
- `diagonal_margin_mean`: mean difference between matching cosine and best
  non-matching cosine when `sample_count > 1`.

## Interpretation Boundary

These are latent-space bridge metrics. They can identify whether an adapter
checkpoint numerically maps source vectors toward paired target vectors under a
given bundle. They do not establish semantic transfer, text-level fidelity,
model-to-model alignment, or production readiness.

## Data Policy

Evaluation outputs belong under ignored run directories such as
`runs/LIP-EVAL-001`. Do not commit latent bundles, `.pt` shards, checkpoints,
model weights, caches, generated run outputs, or downloaded artifacts.
