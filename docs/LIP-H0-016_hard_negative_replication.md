# LIP-H0-016 — cross-seed replication of frozen hard-negative batches

Status: design frozen, not yet executed.

H0-016 repeats the exact H0-015 system on bridge/dropout seeds 4001 and 4003.
The candidate bank, 224/256 balanced partition, H0-013 loss, receiver,
architecture, batch size, update count, example exposure, checkpoint rule,
splits, and held identity gate remain unchanged. Only the training seed and the
deterministic batch/row order induced by that seed vary.

The frozen seed-4007 H0-015 cell already passed the complete joint/core/name
Holm family. System replication requires at least one of the two new cells to
pass the same complete family, yielding at least two strong seeds among the
three total. Passing both new cells is reported separately as 3/3 robustness.

This is a replication of system robustness, not a new paired causal estimate.
There are no seed-matched H0-013 random-batch controls with
`lambda_margin=1.0` for seeds 4001 and 4003. The causal batch-membership
contrast remains directly identified on seed 4007 only.

If the aggregate strong gate passes, H0-016 authorizes design—not execution—of
a bounded functional matched-versus-shuffled confirmation evaluation.
Confirmation data remain prohibited during H0-016, and PROTO-015 remains
premature until functional causal evidence exists.
