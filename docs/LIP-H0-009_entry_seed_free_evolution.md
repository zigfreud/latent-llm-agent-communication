# LIP-H0-009 — entry seed with free receiver evolution

## Rationale

H0-008 showed that an absolute layer-0 anchor repairs the receiver locally, but
the first packet-minus-scaffold update immediately spends that advantage. This
raises a simpler possibility: once task identity is seeded, the receiver should
perform the timeline evolution itself.

H0-009 therefore replaces only layer 0. At layers 1–7 the hook adds a zero
vector, which is exactly equivalent to leaving the live incoming state
unchanged while preserving identical capture instrumentation.

## Frozen gate

The same exposed eight-task screen, bridge checkpoints, seeds, packet sites,
neutral carrier, and oracle absolute-replay reference are retained.

Transition jump is diagnostic only: it is zero after entry by construction and
cannot validate semantic dynamics. The gate uses attention-output and
residual-output NRMSE.

The oracle free evolution must reduce both errors by at least 25% relative to
the anchored repeated-delta oracle from H0-008. A learned replica passes only
if both errors are below absolute replacement in the mean and on at least 6/8
tasks. At least two primary replicas and the oracle gate must pass before any
functional generation.

## Decision meaning

- Oracle and learned pass: test matched-versus-shuffled functional identity.
- Oracle passes, learned fails: entry seeding is mechanically viable, but the
  learned layer-0 packet is insufficient.
- Oracle fails: a single static seed is insufficient; proceed to a trained
  corrector conditioned on the receiver's live state.

The task set is already exposed. H0-009 is an operator screen, not a new
confirmatory protocol.
