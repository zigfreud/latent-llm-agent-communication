# LIP-EVAL-032 — receiver-unrolled gradient geometry

Status: executed; the predeclared scale-limited route was selected.

## Why this is an evaluation

H0-013 supplied a tenfold induced-trajectory margin coefficient. It improved
the paired development gate, but it did not reduce the batch-level core hinge
loss and it did not pass the core Holm test. Another training intervention is
therefore underidentified. The immediate question is diagnostic: what happens
to the core-margin gradient after it passes through receiver evolution, and how
does it interact with the other objectives?

This is not PROTO-015 and it is not a confirmation test. It is a
development-only evaluation of already frozen H0-011 and H0-013 checkpoints.

## Frozen inputs

- Checkpoints: H0-011 seed 4007 and H0-013 seed 4007.
- Receiver: the frozen PROTO-014 target revision, evolved through blocks 0–7.
- Data: train split only; confirmation and development-gate examples are
  prohibited.
- Batch partition: all 256 train tasks, sorted by task id and divided into 16
  deterministic disjoint batches of 16.
- Packet regions: joint, core, and name under the existing masks.
- Precision and accelerator: the same L4/AMP receiver path, with diagnostic
  gradient accumulations evaluated in float32.

## Measurements

For each checkpoint and batch, compute gradients with respect to the predicted
layer-0 entry condition after unrolling through receiver blocks 0–7:

1. core hard-negative margin;
2. symmetric NCE;
3. reconstruction (`huber + cosine + norm`);
4. the non-margin objective; and
5. the configured total objective.

Record loss values, gradient norms, pairwise cosine similarities, active-hinge
fractions, and the effective margin/non-margin norm ratio after applying the
checkpoint's configured coefficient.

Because the configured trajectory margin is the mean of joint, core, and name,
the diagnostic reports the actual core contribution to the configured total as
`lambda_margin / 3`. The unscaled core-gradient norm and the aggregate
configured-total gradient are retained separately, so this accounting choice
is auditable rather than implicit.

Separately, construct the full 256-task train candidate bank from each frozen
checkpoint. For every anchor, compare its all-train hardest negative with the
hardest negative available inside its assigned batch. Report global-hardest
coverage, local/global margin difference, and hard-negative identity agreement
between H0-011 and H0-013.

## Predeclared routing

- **Scale-limited:** median effective core-margin/non-margin gradient ratio is
  below 0.10 and median alignment with the non-margin objective is at least
  -0.10. Route H0-014 to an explicit core-only or adaptive gradient weight.
- **Conflict-limited:** core-margin versus non-margin gradient cosine is below
  -0.10 in at least 12 of 16 batches. Route H0-014 to staged optimization or a
  conflict-aware update, not a larger scalar.
- **Coverage-limited:** neither condition above holds and fewer than 25% of
  anchors see their all-train hardest negative inside the assigned batch.
  Route H0-014 to cross-batch memory or explicit hard-negative mining.
- **Mixed or unresolved:** more than one condition holds without a dominant
  pattern, or bootstrap intervals cross the routing thresholds. Do not open a
  training experiment; refine the diagnostic.

All summaries must include per-batch rows, bootstrap intervals over batches,
checkpoint hashes, bundle hash, receiver revision, code commit, peak VRAM, wall
time, and visible Colab compute units.

Bootstrap intervals are 95% percentile intervals with 10,000 deterministic
resamples and seed 4501. Any interval crossing a routing threshold yields the
predeclared mixed/unresolved outcome.

## Decision boundary

LIP-EVAL-032 may authorize one H0-014 development intervention. It cannot
authorize replication, functional confirmation, or PROTO-015. Those remain
blocked until a learned bridge passes the existing joint/core/name family on a
replicated development result.

## Executed result

The H0-013 median effective core/non-margin gradient ratio was `0.0380`
(`95% bootstrap [0.0328, 0.0411]`), below the `0.10` threshold, while the
median gradient cosine was positive at `+0.4093`
(`95% bootstrap [+0.3743, +0.4305]`). Zero of 16 batches met the conflict
criterion, and no bootstrap interval crossed a routing threshold.

The selected route is therefore **scale-limited**, authorizing one H0-014
development intervention using an explicit core-only objective or adaptive
core-gradient weighting. Candidate-bank coverage was only `4.30%` in H0-013,
but remains a secondary deficit under the frozen routing precedence.

Full provenance and artifact hashes are recorded in
`experiments/registry/LIP-EVAL-032_gradient_geometry.json`.
