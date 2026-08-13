# LIP-H0-011 — core negative-coverage screen

## Why this is development, not PROTO-015

H0-010 established a replicated advantage for learning through receiver
evolution, but failed its frozen quality gate because core diagonal margins
remained negative. Functional confirmation is therefore still prohibited.

## Isolated intervention

H0-011 changes only the number of in-batch alternatives. The H0-010 reference
uses batch four for 512 updates; H0-011 uses batch sixteen for 128 updates.
Both systems see 2,048 training examples, while the contrastive set grows from
three to fifteen negatives per prediction. Architecture, source/target data,
loss weights, seed 4007, receiver evolution, and checkpoint rule are unchanged.

Because fixed example budget cannot preserve both batch size and optimizer-step
count, this is a practical-intervention screen rather than a pure asymptotic
batch-size estimate. A positive result supports negative coverage as part of
the missing property; a negative result moves the next development test to
explicit core-discriminative weighting.

## Gates

The four-update pilot gates only numeric feasibility and L4 memory. If it
passes, one 128-update development screen is authorized. Directional success
requires core margin to improve over the H0-010 seed-4007 reference without
lower core or mean retrieval. Strong success additionally requires positive
core margin and a passing joint/core/name Holm family.

Even strong success authorizes replication, not functional confirmation.
