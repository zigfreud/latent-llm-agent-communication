# LIP-EVAL-031 — receiver trajectory coherence result

## Outcome

EVAL-031 supports the receiver-trajectory hypothesis strongly enough to justify
a small causal operator experiment. It does not yet justify PROTO-015.

The learned packets were not merely farther from native teacher states. At every
replayed layer after layer 0, they required a much larger correction between the
state actually produced by the preceding receiver block and the next snapshot
scheduled by the bridge.

| Condition | Seed | Mean cross-layer jump NRMSE | Attention-output NRMSE | Residual-output NRMSE |
|---|---:|---:|---:|---:|
| Oracle matched | — | 0.170 | 0.367 | 0.168 |
| Component contrastive | 4001 | 0.617 | 0.818 | 0.797 |
| Component contrastive | 4003 | 0.601 | 0.808 | 0.797 |
| Component contrastive | 4007 | 0.636 | 0.827 | 0.803 |
| Structured linear | 4001 | 0.557 | 0.764 | 0.711 |
| Structured linear | 4003 | 0.567 | 0.770 | 0.720 |
| Structured linear | 4007 | 0.555 | 0.762 | 0.710 |
| Mean scaffold | — | 0.678 | 0.871 | 0.845 |

The paired result is unusually clean for an exploratory localization: for each
of the three seeds, learned cross-layer jump, attention-output error, and
residual-output error exceeded oracle on all 32/32 tasks. The median learned to
oracle jump ratio ranged from 3.70 to 3.91 for the primary bridge and from 3.48
to 3.55 for the structured-linear bridge.

## Timeline shape

Layer 0 remains a separate carrier-entry problem. Its jump was large even for
oracle (1.20), because replay enters from a neutral text carrier. The useful
separation appears at layers 1–7:

- oracle: 0.158, 0.213, 0.193, 0.141, 0.157, 0.196, 0.131;
- component contrastive, pooled across seeds: 0.746, 0.582, 0.585, 0.555,
  0.605, 0.656, 0.596;
- structured linear, pooled across seeds: 0.613, 0.532, 0.548, 0.493, 0.554,
  0.617, 0.559.

This is not a single bad boundary. The coherence deficit persists across the
whole replayed depth.

## What it means

The result fits the proposed mechanism: a bridge can learn a geometrically
plausible stack of target snapshots without learning a trajectory that the
receiver dynamics can realize. The structured-linear bridge is somewhat closer
to oracle than the nonlinear primary bridge, but remains far outside the oracle
regime and was functionally inert in PROTO-014. That points to a missing
property shared across the existing contracts, not merely a bad nonlinear
architecture.

Oracle Q/K/V alignment to native execution was nearly exact (NRMSE 0.004,
0.004, and 0.008), while its attention-output error remained 0.367 because the
non-packet context was still the neutral carrier. Perfect reproduction of the
native text-prompt trajectory is therefore unnecessary. The useful target is
the **oracle replay regime**, not zero error.

Absolute first-token comparison with the native text prompt is not useful here.
Total variation was approximately 0.937 in every condition, and KL was not
ordered by functional usefulness. This endpoint should be demoted; causal task
identity and receiver-trajectory measures are the relevant endpoints.

## Claim boundary

The 32 tasks are the already exposed PROTO-014 confirmation cohort. EVAL-031 is
descriptive and post-confirmation. It establishes a robust association between
learned replay, discontinuous layer-to-layer correction, and downstream
receiver-state divergence. It does not prove that lowering discontinuity will
restore task identity. That requires an intervention.

## Recommended next experiment

The next step should be `exp/LIP-H0-007-receiver-aware-corrector`, beginning
with a no-training causal operator test:

`h_injected[l] = h_live_incoming[l] + (packet[l] - training_scaffold[l])`

Instead of replacing the live receiver state with an absolute snapshot, this
operator preserves the receiver's realized carrier evolution and adds only the
task-conditioned residual predicted by the bridge. The existing hook already
supports additive injection, so the test is small and directly targets the
property localized here.

The operator should be compared with absolute replacement for oracle matched /
shuffled and learned matched / shuffled conditions. Primary endpoints are the
matched-minus-shuffled functional identity effect and the change in cross-layer
coherence. Only a positive causal result should graduate into a trainable
receiver-aware corrector and, later, PROTO-015.

## Execution record

- Run commit: `e6f21d2f24c67d7ec0014ce4b70609884bad7600`.
- Full sample: 32 tasks, 3 seeds, 256 condition rows.
- Accelerator: NVIDIA L4; peak allocated VRAM 6.16 GB.
- Full cached run: 161.85 seconds.
- Observed Colab use for setup, pilot, and full run: 0.40 compute units.
- Pilot artifact SHA-256: `a3acc8fcd6304954690f218b76915a2ddf0d20ace98369fdb4f5013cd68119fc`.
- Full artifact SHA-256: `71c76b9e73e300f3781666dc11e1fa5869651608f981fc110439c2a312f2de1d`.

The canonical raw artifacts are under `lip-artifacts/LIP-EVAL-031` in Drive.
The L4 runtime was deleted after both finalized artifacts and hashes were
verified; only reproducible scratch state and the downloaded model cache were
removed.
