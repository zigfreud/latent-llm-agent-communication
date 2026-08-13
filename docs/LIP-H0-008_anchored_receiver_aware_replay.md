# LIP-H0-008 — anchored receiver-aware replay

## Why this is the next test

H0-007 lowered the apparent layer-to-layer intervention while making the
receiver's attention and residual states worse on every task. The same failure
occurred with exact oracle packets. That falsifies the static affine-origin
assumption, not the broader idea that task transmission must respect receiver
dynamics.

Layer 0 has always been a distinct carrier-entry boundary. H0-008 asks whether
the receiver first needs an absolute task-conditioned anchor before relative
updates become meaningful.

## Frozen intervention

The comparator remains absolute replay at all eight target layers. The new
operator is:

```text
layer 0:    h_injected = packet
layers 1–7: h_injected = h_live_incoming + (packet - training_scaffold)
```

Everything else is held fixed: exposed eight-task screen, neutral receiver
carrier, target sites and offsets, bridge checkpoints, three training seeds,
and both bridge variants.

## Two-stage gate

The oracle diagnostic is logically prior to the learned-packet result. Against
the frozen unanchored-oracle means from H0-007, anchoring must:

- not worsen transition jump;
- reduce both attention-output and residual-output NRMSE by at least 25%.

The learned operator is then compared taskwise with absolute replacement. A
replica passes only if all three metrics have a lower mean and improve on at
least 6/8 tasks. At least two of the three component-contrastive replicas must
pass. Both the oracle-origin gate and learned-operator gate are required before
any functional generation.

## Interpretation matrix

- Oracle fails: layer-0 anchoring does not repair the arithmetic operator; move
  to a learned live-state-conditioned corrector.
- Oracle passes, learned fails: the operator is dynamically coherent for exact
  packets, but current bridge packets do not encode usable layerwise deltas.
- Both pass: run the already bounded matched-versus-shuffled identity pilot.
- Functional pilot passes: only then consider a confirmatory PROTO-015.

This remains an exploratory operator screen on exposed tasks. It cannot support
a new confirmatory claim.
