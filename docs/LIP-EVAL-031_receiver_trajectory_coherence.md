# LIP-EVAL-031 — receiver trajectory coherence

## Status and purpose

This is a post-confirmation exploratory evaluation. It localizes the causal gap
left by LIP-PROTO-014; it is not a new confirmatory protocol and it does not
reuse the exposed confirmation tasks for a fresh efficacy claim.

PROTO-014 established two facts under its frozen contracts: target-oracle task
identity was causally active, while learned source-to-target identity was not.
At the same time, the primary nonlinear bridge and the structured linear
baseline passed their registered packet-geometry gates. EVAL-031 asks whether
that separation is explained by a missing receiver trajectory constraint.

## Mechanistic hypothesis

At token position `p` and receiver layer `l`, the relevant evolution is

`h[l+1,p] = F[l](h[l,<=p])`.

The current query is computed from the current residual and attends to keys and
values built from the available causal context. Prior queries are not a memory
bank. Therefore the diagnostic separates two axes that are easy to conflate:

1. token-position causality inside a layer;
2. residual evolution from one layer to the next.

The current bridge emits a complete stack of residual snapshots. During replay,
each snapshot is imposed independently at a block boundary. If layer `l-1`
produces an incoming state that is far from the scheduled snapshot for layer
`l`, replay performs a large corrective jump. A packet may consequently be
close to teacher snapshots pointwise but fail to describe a trajectory the
receiver would naturally realize.

## Measurements

For every condition and task, the target runs once on the native task prompt and
once on the neutral carrier with packet replay. The capture covers target layers
0–7 and the final 24 prompt positions.

Layer 0 is reported separately as the **carrier-entry jump**. It measures the
distance between the neutral carrier and the first imposed state. Layers 1–7 are
the primary **cross-layer transition jumps**: the distance between the residual
actually produced by the preceding receiver computation and the next scheduled
snapshot.

The evaluation also compares replay with the native teacher execution at:

- residual input and output;
- query, key, and value projections before rotary/cache transformation;
- attention output;
- the next-token distribution at the final prompt position.

Oracle replay is an empirical reference, not a zero-discontinuity assumption.
Only the selected packet positions are replaced; the rest of the context remains
the neutral carrier and can itself induce transition mismatch.

## Conditions and staging

The pilot uses two exposed PROTO-014 confirmation tasks and seed 4001 for the
learned variants. It compares:

- matched target-oracle packets;
- the primary component-contrastive bridge;
- the structured-linear regression bridge;
- the training mean scaffold.

Only after the pilot validates tensor capture, memory use, runtime, and metric
scale may the evaluation expand to all 32 tasks and all three registered seeds.
The L4 is the preferred accelerator. There is no silent downgrade to another
GPU. Colab compute-unit readings are recorded from the UI before and after the
run; code records model/GPU identity, VRAM peaks, wall time, and throughput.

## Decision boundary

Evidence supports a receiver-aware corrector experiment when the learned bridge
has a reproducible coherence deficit relative to oracle and at least one
downstream receiver-state family moves in the same direction. The structured
linear comparison helps distinguish a general replay problem from a
query-conditioned architecture problem.

If oracle and learned packets show comparable discontinuity and comparable
receiver-state alignment, this hypothesis is weakened and the next experiment
must localize a different missing property. No EVAL-031 outcome by itself proves
that discontinuity causes the functional floor.

