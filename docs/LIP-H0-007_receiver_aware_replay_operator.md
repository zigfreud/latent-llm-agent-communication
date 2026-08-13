# LIP-H0-007 — receiver-aware replay operator

## Question

EVAL-031 showed that the PROTO-014 bridges predict individually plausible
target snapshots but require large corrective jumps after every receiver block.
H0-007 asks the causal next question: does preserving the receiver's live
trajectory while adding only the bridge's task delta repair that regime?

The tested operator is:

`h_injected[l] = h_live_incoming[l] + (packet[l] - training_scaffold[l])`

The comparator remains the PROTO-014 absolute replacement:

`h_injected[l] = packet[l]`

No corrector is trained in this stage. That keeps the intervention focused on
the representation property identified by EVAL-031 rather than introducing a
new architecture and objective at the same time.

## Frozen trajectory gate

The gate uses eight already exposed PROTO-014 confirmation tasks, balanced
across the two tokenizer strata, and all three registered training seeds. The
component-contrastive bridge is the decision variant; structured linear is a
diagnostic architecture comparator.

For each task and replica, additive replay is paired with absolute replacement.
It must improve all three quantities:

1. mean intervention-jump NRMSE across layers 1–7;
2. attention-output NRMSE relative to oracle absolute replay;
3. residual-output NRMSE relative to oracle absolute replay.

A replica passes only when the additive operator has a lower pooled mean and a
lower value on at least 6/8 paired tasks for every quantity. The trajectory gate
passes when at least 2/3 primary replicas pass. Layer 0 remains a separately
reported carrier-entry diagnostic.

This is an exploratory decision rule, not an inferential confirmation test. The
tasks were exposed in PROTO-014 and EVAL-031.

## Conditional functional test

If the trajectory gate passes, the same frozen sample advances to a
matched-versus-shuffled task-identity pilot. Donors are first derived by the
full 32-task PROTO-014 stratified derangement and only then subset, so the donor
contract cannot change in response to the gate result. All three training seeds
and three generation seeds are retained.

If the gate fails, generation is skipped. A lower-norm intervention alone is
not evidence of transmitted task identity, and a positive trajectory gate does
not itself justify PROTO-015. Only a positive matched-minus-shuffled functional
effect can motivate training a receiver-aware corrector.

## Claim boundary

H0-007 can show that changing the replay operator causally changes receiver
trajectory coherence and, conditionally, task identity. It cannot establish
generalization to unseen tasks, models, layer mappings, or packet widths. Any
positive result remains a branch experiment whose property must later be
confirmed under a new protocol.
