# LIP-H0-015 — frozen global-hard-negative batches

Status: design frozen, not yet executed.

## Why this experiment

H0-014 established that increasing the core-margin contribution can reduce the
optimized local hinge without improving held identity geometry. The remaining
candidate is not another scalar but which negatives are present when that
hinge is evaluated.

LIP-EVAL-032 identified the global hardest H0-013 train negative for every one
of the 256 anchors. Its reported 4.30% coverage describes the static partition
used by that evaluation, not cumulative coverage across the randomly shuffled
H0-013 epochs. H0-015 therefore does not claim that H0-013 historically saw
exactly 4.30% of its hard negatives.

Instead, H0-015 creates a direct causal contrast: replace random membership
with a frozen balanced partition whose within-batch global-hardest coverage is
known before training.

## Frozen intervention

The deterministic partition search uses the frozen H0-013 candidate bank from
LIP-EVAL-032. With batch size 16, seed 4007, eight restarts, and at most 128
improving swaps per restart, it colocates the global hardest negative for
224/256 anchors: 87.50% coverage. A random balanced partition has expected
coverage 15/255 = 5.88%.

Every task appears exactly once per epoch. The 16 batch memberships stay
frozen, while batch order and row order are shuffled deterministically each
epoch. No task is duplicated or upweighted.

Everything else returns to and remains frozen at H0-013: equal regional margin
weights, `lambda_margin=1.0`, reconstruction, NCE, entry regularization,
receiver evolution, architecture, optimizer, seed, batch size, 128 updates,
2,048 examples seen, checkpoint selection, data splits, and gate.

## Decision boundary

Directional success requires core margin above H0-013 while core retrieval and
mean retrieval do not decrease. Strong success additionally requires the full
joint/core/name Holm family.

Even strong success authorizes only exact replication on the two existing
development seeds. Confirmation and PROTO-015 remain prohibited. A negative
result rejects this static frozen-coverage mechanism; it does not by itself
reject dynamic hard-negative mining or a cross-batch memory bank.
