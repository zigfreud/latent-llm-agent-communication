# LIP-H0-016 — cross-seed replication of frozen hard-negative batches

Status: executed; aggregate strong gate passed at the minimum preregistered
threshold (2/3 total strong seeds, 1/2 new strong seeds).

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

## Result

Seed 4003 passed the complete joint/core/name Holm family. Seed 4001 retained
positive margins in all three regions but failed the joint and core decisions
at `p_Holm=0.06102`. Combined with the frozen strong seed 4007 from H0-015,
the aggregate gate passed 2/3 at its exact threshold; the stricter 3/3 outcome
did not occur.

The replicated property is therefore real but heterogeneous. Name separation
was strong in every seed and core retrieval was 75% in every seed. Joint
retrieval and joint/core statistical strength remained initialization-sensitive.

Only seed 4007 retains a paired causal comparison against the H0-013 random
batch policy. The two new cells test system robustness, not cross-seed causal
replication. The result authorizes preregistration of LIP-EVAL-033 as a bounded
functional learned-matched versus learned-shuffled test across all three fixed
bridge seeds. It does not authorize execution inside H0-016.
