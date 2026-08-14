# LIP-H0-015 — frozen global-hard-negative batches

Status: complete; strong identity-geometry gate passed on the development seed.

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

## Result

The pilot passed and the complete L4 screen selected step 120. Against H0-013:

| Metric | H0-013 | H0-015 | Delta |
|---|---:|---:|---:|
| Core margin | 0.010905 | 0.017437 | +0.006532 |
| Core retrieval | 71.88% | 75.00% | +3.12 pp |
| Mean retrieval | 81.25% | 84.38% | +3.12 pp |
| Normalized RMSE | 1.418561 | 1.455715 | +0.037154 |

All three identity regions passed Holm correction: joint
`p_Holm=0.000030`, core `p_Holm=0.005530`, and name
`p_Holm=0.000040`. The directional and strong gates therefore passed.

Hard batching increased median train core hinge from `0.03781` to `0.05212`,
as expected when harder negatives are actually presented. By the final update,
the hinge had fallen to `0.01639`, below H0-013's `0.02473`. This change, unlike
core-only scaling, generalized to the held identity gate.

RMSE worsened, so the result identifies an identity-fidelity tradeoff rather
than uniform packet improvement. The next authorized step is an exact H0-016
replication on seeds 4001 and 4003 using the same frozen partition. Functional
confirmation and PROTO-015 remain premature.
