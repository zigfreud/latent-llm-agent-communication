# LIP-H0-009 — entry seed with free receiver evolution result

## Outcome

H0-009 produced a decisive mechanistic split.

With an exact oracle layer-0 packet, one entry intervention followed by free
receiver evolution passed the frozen oracle gate. Relative to the anchored
repeated-delta oracle from H0-008:

| Metric | Repeated delta | Free evolution | Ratio | Gate |
|---|---:|---:|---:|---:|
| Attention-output NRMSE | 1.258 | 0.397 | 0.315 | pass |
| Residual-output NRMSE | 1.316 | 0.323 | 0.245 | pass |

The same operator failed with every learned layer-0 packet. Attention and
residual alignment were worse than learned absolute replacement on all 8/8
tasks for all six replicas. Therefore no functional generation was run.

| Variant | Seed | Attention absolute → free | Residual absolute → free |
|---|---:|---:|---:|
| Component contrastive | 4001 | 0.781 → 1.287 | 0.800 → 1.046 |
| Component contrastive | 4003 | 0.773 → 1.267 | 0.802 → 1.041 |
| Component contrastive | 4007 | 0.786 → 1.322 | 0.804 → 1.061 |
| Structured linear | 4001 | 0.710 → 1.099 | 0.705 → 0.947 |
| Structured linear | 4003 | 0.716 → 1.124 | 0.713 → 0.958 |
| Structured linear | 4007 | 0.708 → 1.117 | 0.702 → 0.951 |

## Timeline interpretation

The receiver does not need a layerwise oracle trajectory imposed on it to
remain near the oracle-replay regime. From an exact target entry state it can
construct most of that trajectory itself. Oracle attention error grows
gradually across layers 0–7 (`0.000, 0.160, 0.349, 0.441, 0.470, 0.549, 0.615,
0.589`), as does residual error (`0.000, 0.153, 0.271, 0.354, 0.387, 0.433,
0.477, 0.510`). This is dramatically better than injecting a static residual
at every boundary.

The learned bridge fails before that evolution begins. At layer 0, the primary
replicas have mean Q/K/V NRMSE ranges of 0.712–0.768, 0.747–0.798, and
0.916–0.920; layer-0 attention and residual NRMSE are approximately 0.66–0.68
and 0.87–0.88. The structured-linear seed is closer but still well outside the
oracle regime. Its downstream free trajectory is correspondingly wrong.

The missing property is therefore best stated as a **causally sufficient
receiver initial condition**, not merely a stack of target-space snapshots and
not a hand-written per-layer drift rule.

## Consequence for training

The next bridge should emit only the target layer-0 packet and be optimized
through the frozen receiver dynamics. Its loss should be computed on states
induced at later target layers, so gradients reward an entry state for what it
causes the receiver to become rather than only for static similarity to the
teacher snapshot.

This is an initial-value problem:

`source packet → learned target initial condition → frozen target dynamics → induced trajectory`

PROTO-015 remains premature. The next step is a development-only experimental
branch using the existing 256 train, 32 development-selection, and 32
development-gate tasks. The exposed confirmation tasks from H0-007–009 must not
be used for model selection.

## Claim boundary

H0-009 is an exploratory causal operator screen over eight exposed tasks. It
establishes a positive oracle-mechanics result under this replay contract and a
negative result for the existing learned entry packets. It does not establish
functional task identity for layer-0-only replay or learned cross-model
communication.

## Execution record

- Run commit: `c739c0234885945a5c2aefb2af162f7b579ce3f5`.
- Sample: 8 exposed tasks; 112 result rows.
- Accelerator: NVIDIA L4; peak allocated VRAM 6.16 GB.
- Wall time: 85.23 seconds.
- Observed Colab use, including post-run diagnostics: 0.38 compute units.
- Final artifact SHA-256:
  `62381432a277cbb612522730523858f292f4869939cb7ef91843f02417500c78`.

The canonical artifact is
`lip-artifacts/LIP-H0-009/trajectory_gate.json` in Drive. The L4 runtime was
deleted after the artifact and hash were finalized.
