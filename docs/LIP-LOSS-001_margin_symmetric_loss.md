# LIP-LOSS-001 Margin-Aware Symmetric Contrastive Loss

## Purpose

LIP-LOSS-001 adds optional loss terms for the bridge training objective. Recent
MBPP latent-space evaluations showed that the bridge can learn train-set
geometry and achieve held-out retrieval above chance, while held-out diagonal
margin remains negative or close to zero. That means the correct target is often
near the top-k set but does not reliably beat the hardest incorrect target.

This loss extension keeps the existing forward contrastive plus MSE behavior by
default, and enables controlled experiments with symmetric ranking,
hard-negative margin pressure, and norm calibration.

## InfoNCE Is Not Cross-Attention

The existing contrastive objective compares a batch of predicted target vectors
against the true target vectors with a cosine-similarity matrix. This is an
InfoNCE-style ranking loss over vector pairs.

This PR does not add cross-attention, hidden-state mixing, a memory bank, or a
new bridge architecture. It only changes optional scalar loss terms used during
adapter training.

## Forward InfoNCE

Given predicted target vectors `pred` and true target vectors `target`, both
with shape `[B, D]`, the loss normalizes both sides and computes:

```text
cosine = normalize(pred) @ normalize(target).T
logits = cosine / temperature
labels = arange(B)
loss_nce_forward = CrossEntropyLoss(logits, labels)
```

This is the existing ranking direction: each predicted vector should retrieve
its paired target vector from the batch.

## Reverse InfoNCE

The optional reverse term applies the same cross-entropy loss to the transposed
logit matrix:

```text
loss_nce_reverse = CrossEntropyLoss(logits.T, labels)
```

It is controlled by `lambda_reverse_nce`, defaulting to `0.0`. When enabled, the
batch objective also pressures each target vector to retrieve its paired
prediction.

## Hard-Negative Margin

The margin term directly targets the observed failure mode where the correct
target does not beat the strongest incorrect target:

```text
positive_i = cosine[i, i]
hardest_negative_i = max cosine[i, j] for j != i
margin_i = positive_i - hardest_negative_i
loss_margin = mean(relu(margin_target - margin_i))
```

The implementation also computes the column-side margin and averages the row
and column margin losses. If `B < 2`, no hard negative exists, so the margin
loss returns zero without crashing.

The term is controlled by `lambda_margin`, defaulting to `0.0`. The default
`margin_target` is `0.05`.

## Norm Calibration

The norm calibration term measures whether predictions have similar target-space
energy to their paired target vectors:

```text
norm_ratio_i = ||pred_i|| / (||target_i|| + eps)
loss_norm = mean((norm_ratio_i - 1)^2)
```

It is controlled by `lambda_norm`, defaulting to `0.0`.

## Total Loss

When optional terms are enabled, the total objective is:

```text
loss_total =
    loss_nce_forward
  + lambda_reverse_nce * loss_nce_reverse
  + lambda_mse * loss_mse
  + lambda_margin * loss_margin
  + lambda_norm * loss_norm
```

The trainer logs the existing loss fields and the additional components when
available:

- `reverse_nce_loss`
- `margin_loss`
- `norm_loss`
- `cosine_diag_mean`
- `offdiag_cosine_mean`
- `diagonal_margin_mean`
- `hard_negative_cosine_mean`
- `norm_ratio_mean`

## Backward Compatibility

The new coefficients default to zero:

```yaml
lambda_reverse_nce: 0.0
lambda_margin: 0.0
lambda_norm: 0.0
```

With those defaults, `HybridContrastiveLoss` preserves the existing return shape
and behavior: `(total_loss, loss_nce, loss_mse, accuracy)`. Existing configs do
not need to change and old experiment definitions are not modified.

## Suggested First Real Run

Trigger `LIP-LOSS-001 Configurable Remote Bundle Training` with the LIP-DATA-004
MBPP64 train bundle:

```text
experiment_id: LIP-TRAIN-012-mbpp64-batch16-steps256-l035-rnce1-margin01-norm005
google_drive_file_id: <LIP-DATA-004 train bundle file id>
latent_bundle_sha256: <LIP-DATA-004 train bundle sha256>
max_steps: 256
epochs: 64
batch_size: 16
learning_rate: 0.0001
lambda_mse: 0.35
lambda_reverse_nce: 1.0
lambda_margin: 0.1
margin_target: 0.05
lambda_norm: 0.05
device: cpu
```

The workflow verifies the external bundle SHA256, safely unpacks it, validates
the manifest and shards, writes `effective_config.yaml`, trains the adapter, and
uploads the run artifact.

## Evaluation

Evaluate the resulting checkpoint with `evaluate_bridge.py` on both:

- LIP-DATA-004 MBPP64 train bundle for in-sample diagnostics.
- LIP-DATA-004 MBPP32 eval bundle for held-out latent-space diagnostics.

Use the same evaluation config pattern as earlier LIP-EVAL registry entries and
record the bundle trace IDs, shard digests, run artifact digest, and metric
values in a registry PR.

## Data Policy

Do not commit generated configs, datasets, bundles, shards, zips, checkpoints,
model weights, caches, or run artifacts. External bundles must be trusted and
content-addressed with SHA256 before workflow execution.

## Scientific Claim Boundary

This PR adds a margin-aware symmetric contrastive training objective. It does
not claim semantic transfer, text-level fidelity, model-to-model alignment,
generalization, or production readiness.
