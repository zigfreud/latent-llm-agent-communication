# LIP-H0-017 — closed-loop trajectory corrector

## Why this is a development branch

The next step is not PROTO-015 and not yet EVAL-038. The operator itself is
still unvalidated, so H0-017 is a development-only feasibility and causal
comparison on the existing training/development bundle. Confirmation tasks and
fresh holdouts are prohibited.

Four earlier results constrain the design:

- H0-007 rejected adding one static packet-minus-scaffold delta to the live
  receiver at every layer.
- H0-008 rejected a layer-0 anchor followed by the same static delta rule.
- H0-009/H0-010 showed that receiver evolution is useful, but a learned layer-0
  initial condition alone is insufficient.
- EVAL-037 showed that a scalar convex mixture of native and oracle states does
  not restore the receiver-capacity gate.

H0-017 therefore tests a property none of those operators had: the correction
is recomputed from the receiver state that actually exists at that moment.

## Operator

The source packet is encoded once into a task code `c`. Before each frozen
receiver block `l`, at each registered position `p`, the corrector observes the
live residual `u[l,p]` and emits a normalized delta:

```text
delta[l,p] = C(c, normalize(u[l,p]), l, p)
z[l,p] = u[l,p] + site_scale[l,p] * delta[l,p]
u[l+1] = ReceiverBlock[l](z[l])
```

The source encoder is initialized from the H0-015 seed-4007 checkpoint and
frozen. The receiver is also frozen. Only `C` is trained. Its output head starts
at zero, so the untrained operator is exactly a no-op rather than a random
intervention.

## What makes the loss causal

A loss only on `z[l]` could teach eight independent snapshot regressions. The
primary loss is instead evaluated on `u[l]` for layers 1–7, before the current
layer correction. Error at `u[l]` can only be reduced by corrections applied at
earlier layers and propagated through frozen receiver blocks.

A smaller auxiliary loss keeps each corrected state near the corresponding
teacher state, and a weak normalized-delta energy penalty discourages gratuitous
large updates.

## Causal control

The paired control has the same architecture, source code, parameters, data,
seed, optimizer, and sequential injection schedule, but the live-state input is
replaced with zeros. It can learn source-, layer-, and position-dependent
deltas, but cannot adapt them to the receiver trajectory it actually caused.

The comparison therefore isolates live-state conditioning from merely adding a
trainable multi-layer delta decoder.

## Sequential gate

The first run is a four-update L4 feasibility pilot for the primary operator.
It gates only numeric stability, differentiability, frozen-parameter boundaries,
memory, and completion.

Only a passing pilot authorizes the paired 128-update development screen. The
primary operator must reduce incoming-trajectory RMSE by at least 10% relative
to the state-blind control, without lowering core or mean retrieval, and must
pass the joint/core/name Holm family. A pass would authorize design—not
execution—of EVAL-038 on the already open functional cohort.

H0-017 cannot upgrade EVAL-033 through EVAL-037, spend a fresh holdout, or
authorize PROTO-015.
