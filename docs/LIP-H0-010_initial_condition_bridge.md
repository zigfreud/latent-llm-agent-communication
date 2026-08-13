# LIP-H0-010 — unrolled initial-condition bridge

## The shift in formulation

H0-009 turns the cross-model transport problem into an initial-value problem.
An exact layer-0 target packet is enough for the frozen receiver to construct a
near-oracle trajectory through layers 1–7. Existing bridges fail because their
layer-0 output is not a causally sufficient initial condition.

H0-010 trains the bridge for what its entry state **causes**, not only for its
static distance to a teacher tensor:

```text
source trajectory
    ↓ bridge
target layer-0 initial condition
    ↓ frozen target blocks 0..7
induced target trajectory
    ↓ loss against teacher trajectory
```

The receiver weights remain frozen. Gradients pass through the first eight
receiver blocks into the bridge. Execution stops before block 8, avoiding the
irrelevant final 24 blocks during training.

## Causal comparison

Two systems share the exact architecture, data, seeds, and optimization
settings:

- `static_entry_snapshot`: fit only the normalized target layer-0 packet;
- `unrolled_initial_condition`: fit induced states at layers 1–7, with a 0.25
  auxiliary penalty on the layer-0 teacher snapshot.

Both are selected and evaluated on the trajectory they induce under free
receiver evolution. This isolates the value of the unrolled objective from the
value of merely reducing the decoder output from eight layers to one.

## Split discipline

The real PROTO-014 training bundle contains 256 train, 32
development-selection, and 32 development-gate tasks and no confirmation
records. H0-010 uses only these splits. The eight H0-007–009 tasks and the full
PROTO-014 confirmation cohort are exposed and prohibited for model selection.

The first run is a 16-update, one-seed feasibility pilot on L4. It gates only
memory, differentiability, numeric stability, and completion. It is not a model
quality decision. It uses the same task batch of four as the full matrix so the
contrastive objective retains the original PROTO-014 negative-set size. If
feasible, the frozen matrix is two objectives by three seeds, 512 updates each.

The first two numeric pilots completed all 16 updates but failed their AMP
gates. The v2 split loss localized the failure precisely: the single-layer
entry auxiliary had relative norm loss `8.31e18`, while the seven-layer induced
trajectory stayed near `1.01`. The original PROTO-014 norm term aggregated over
eight layers; a layer-0-only teacher boundary can instead have near-zero norm.
Protocol v3 therefore disables the relative-norm term only for the layer-0
snapshot and retains it for the induced trajectory. Huber, cosine, symmetric
NCE, and margin losses remain unchanged. Disabled norm terms are not evaluated,
preventing `0 * inf` from becoming NaN. Prior artifacts remain preserved.

## Development decision

Checkpoint selection uses joint/core/name retrieval and margins on the induced
development-selection trajectory, then RMSE. The untouched development gate
applies the existing Holm-corrected margin family.

The primary system must pass in at least two of three replicas and improve
induced-trajectory RMSE over the paired static control in at least two seeds
without lowering mean retrieval. Only then do we design a new untouched
functional confirmation protocol. That future protocol may become PROTO-015;
H0-010 itself cannot.
