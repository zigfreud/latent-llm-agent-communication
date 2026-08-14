# LIP-H0-014 — core-only trajectory margin

Status: complete; directional gate failed.

## Why this experiment

LIP-EVAL-032 found that the receiver-unrolled core-margin gradient is active
and positively aligned with the non-margin objective, but its actual configured
contribution is too small. H0-013 applies `lambda_margin = 1.0` to the mean of
joint, core, and name margins, so only one third of that coefficient belongs to
the persistently failing core region.

The H0-013 median raw core/non-margin gradient-norm ratio was `0.1143`; after
equal regional averaging its effective ratio was only `0.0380`. H0-014 removes
that dilution without increasing the global scalar.

## Frozen intervention

Only the induced-trajectory margin-region weights change:

- H0-013: `joint = 1/3`, `core = 1/3`, `name = 1/3`;
- H0-014: `joint = 0`, `core = 1`, `name = 0`.

`lambda_margin` remains `1.0`. Reconstruction, cosine, symmetric NCE, norm,
entry regularization, receiver evolution, architecture, optimizer, batch size,
examples seen, seed, checkpoint selection, splits, and development gate remain
frozen to H0-013.

Joint and name remain constrained by their reconstruction and NCE terms and by
the complete joint/core/name checkpoint and gate family. Core-only therefore
does not remove them from training or evaluation; it removes them only from the
margin contribution whose scale was diagnosed as limiting.

## Why not 85/10/5 or 90/5/5 now

A soft blend is a plausible fallback if core-only produces an observed joint or
name regression. It is not the cleanest first intervention:

- 85% core projects an effective ratio near `0.097`, still below the frozen
  `0.10` threshold;
- 90% core projects about `0.103`, barely across the threshold;
- core-only projects about `0.114`, a modest rather than explosive update.

Using core-only first maximizes identifiability. A later blend is authorized
only by an observed regional tradeoff, not by a precautionary assumption.

## Execution and decision boundary

Run a four-update L4 pilot. If finite gradients, AMP, and VRAM gates pass, run
the single frozen seed-4007 screen for 128 updates at batch size 16.

Directional success requires core margin above H0-013 while core retrieval and
mean retrieval do not decrease. Strong success additionally requires the
complete joint/core/name Holm family. Even strong success authorizes exact
replication only; it does not directly authorize confirmation or PROTO-015.

## Result

The pilot passed and the full L4 screen completed 128 updates. Core-only
weighting modestly reduced the training core hinge (median `0.03781 ->
0.03592`; final `0.02473 -> 0.02158`) without numeric instability beyond one
initial AMP scale adjustment.

The held development geometry moved in the wrong direction:

| Metric | H0-013 | H0-014 | Delta |
|---|---:|---:|---:|
| Core margin | 0.010905 | 0.004402 | -0.006503 |
| Core retrieval | 71.88% | 65.62% | -6.25 pp |
| Mean retrieval | 81.25% | 76.04% | -5.21 pp |
| Normalized RMSE | 1.418561 | 1.405544 | -0.013017 |

Core remained non-significant after Holm correction (`p_Holm=0.2863`), so
both the directional and strong gates failed. Confirmation data were not used.

## Decision update

The result does not support a `90/5/5` fallback: the stronger core-only form
already reduced the local hinge but failed to improve held identity geometry,
and joint/name did not show a selective collapse that would justify restoring
small regional margins.

The next identified intervention is coverage. LIP-EVAL-032 observed that only
4.30% of H0-013 anchors included their global hardest negative in the assigned
training batch. A development-only H0-015 may therefore return to the H0-013
loss and isolate explicit hard-negative batch construction. A cross-batch
memory bank is reserved as a more stateful fallback. PROTO-015 remains
premature.
