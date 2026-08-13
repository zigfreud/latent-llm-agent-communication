# LIP-H0-008 — anchored receiver-aware replay result

## Outcome

H0-008 is negative. Absolute replacement at layer 0 followed by task-delta
addition at layers 1–7 did not pass either the frozen oracle-origin gate or the
learned-operator gate. No functional generation was run.

Anchoring the entry state produced a real but incomplete oracle improvement:

| Oracle metric | Unanchored add | Anchored hybrid | Ratio | Gate |
|---|---:|---:|---:|---:|
| Transition jump NRMSE | 0.289 | 0.310 | 1.073 | fail |
| Attention-output NRMSE | 1.348 | 1.258 | 0.933 | fail |
| Residual-output NRMSE | 1.713 | 1.316 | 0.768 | fail |

The preregistered thresholds required no worse transition jump and at least a
25% reduction in each downstream error. None passed.

## Learned packets

The learned result repeated the H0-007 dissociation. The hybrid operator
reduced transition jump on 8/8 tasks for every replica, yet improved attention
and residual alignment on 0/8 tasks for every replica.

| Variant | Seed | Jump absolute → anchored | Attention absolute → anchored | Residual absolute → anchored |
|---|---:|---:|---:|---:|
| Component contrastive | 4001 | 0.621 → 0.263 | 0.781 → 1.352 | 0.800 → 1.161 |
| Component contrastive | 4003 | 0.603 → 0.273 | 0.773 → 1.356 | 0.802 → 1.193 |
| Component contrastive | 4007 | 0.637 → 0.259 | 0.786 → 1.363 | 0.804 → 1.136 |
| Structured linear | 4001 | 0.554 → 0.262 | 0.710 → 1.114 | 0.705 → 1.025 |
| Structured linear | 4003 | 0.563 → 0.256 | 0.716 → 1.110 | 0.713 → 0.993 |
| Structured linear | 4007 | 0.550 → 0.263 | 0.708 → 1.108 | 0.702 → 1.012 |

Zero of three primary replicas passed.

## Layerwise localization

The absolute layer-0 anchor does what it should locally: oracle attention and
residual error are zero at layer 0, and residual-output error at layer 1 falls
from 1.598 to 0.629 relative to the unanchored operator. But the first relative
update does not preserve that advantage. Oracle attention error is already
0.866 at layer 1 and spikes to 2.355 at layer 2. The learned primary bridge
shows the same layer-2 spike (2.117 versus 1.256 when unanchored).

This suggests that the receiver may already be evolving the task-bearing state
after the entry anchor, while adding a fresh packet-minus-scaffold residual at
every layer double-counts or miscoordinates that evolution. The next smallest
test is therefore not another arithmetic correction: seed layer 0 once and let
the receiver evolve freely through layers 1–7.

## Claim boundary

This is an exploratory paired screen over eight already exposed PROTO-014
confirmation tasks. It falsifies the anchored static-delta operator under this
contract. It does not falsify receiver dynamics, learned cross-model task
identity in general, or a trained live-state-conditioned corrector.

## Execution record

- Run commit: `3915a18e8bc3f8a05f06a835adbe340c05fc85f9`.
- Sample: 8 exposed tasks; 112 result rows.
- Accelerator: NVIDIA L4; peak allocated VRAM 6.16 GB.
- Wall time: 84.81 seconds.
- Observed Colab use: 0.13 compute units.
- Final artifact SHA-256:
  `4b18469a8e1eced1588312744c54ddcb95dc9124dc5e249fdf30e8f83340e493`.

The canonical artifact is
`lip-artifacts/LIP-H0-008/trajectory_gate.json` in Drive.
