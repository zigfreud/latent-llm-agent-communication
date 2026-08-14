# LIP-H0-014 — core-only margin improves the local hinge, not the held geometry

H0-014 changed only the induced-trajectory regional margin weights relative to
H0-013: equal `joint/core/name` weighting became `0/1/0`. The global margin
coefficient remained `1.0`; receiver evolution, all non-margin losses, data,
seed, update count, checkpoint selection, and gate were frozen. Confirmation
data were not used.

The four-update pilot passed, and the L4 screen completed all 128 updates. The
intervention modestly reduced the local batch-level core hinge: its median fell
from `0.03781` to `0.03592` and its final value from `0.02473` to `0.02158`.
This confirms that the extra core weight reached the optimizer and did not
produce numeric instability.

It did not generalize to the held development gate. Against H0-013, core
margin fell from `+0.01090` to `+0.00440`, core retrieval from `71.88%` to
`65.62%`, and mean regional retrieval from `81.25%` to `76.04%`. RMSE improved
from `1.41856` to `1.40554`, but the predeclared directional gate required all
three identity conditions and therefore failed. Core also remained non-robust
under Holm correction (`p_Holm=0.2863`).

This result argues against spending the next run on a `90/5/5` soft blend. A
blend would weaken an intervention whose stronger core-only form already moved
the local hinge without moving the held identity geometry. Joint and name did
not collapse selectively; all three retrieval regions declined.

The scale diagnosis from LIP-EVAL-032 was locally valid but insufficient. Its
secondary result now becomes decisive: only `4.30%` of H0-013 anchors saw their
global hardest train negative in the assigned batch. The next isolated
development experiment should return to the H0-013 objective and change
hard-negative coverage, preferably through explicit hard-negative batch
construction. A cross-batch memory bank remains a fallback because it adds
state and staleness as extra mechanisms.

- Run commit: `566d722a8c11054364403e7429b77a46574fad67`.
- Pilot: 4 updates, numeric gate passed, best step 2, 27.63 seconds.
- Screen: 128 updates, best step 128, 109.15 seconds.
- Peak allocated VRAM: 8.24 GB.
- Screen SHA-256: `08c6a12e5daa6df148568eefa000da502e90411a1f87535c5cc8fe7a4ae715a9`.
- Confirmation data used: no.
- Decision: authorize development-only H0-015 in the hard-negative-coverage
  family; do not authorize replication, functional confirmation, or PROTO-015.
