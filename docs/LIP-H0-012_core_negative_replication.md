# LIP-H0-012 — negative-coverage replication extension

H0-012 freezes the H0-011 intervention and runs only the missing seeds 4001
and 4003. Their results are combined with the already frozen H0-011 seed 4007;
that successful screen is not rerun or selected again.

Each new cell uses batch 16 for 128 updates, preserving 2,048 examples and 15
in-batch negatives per prediction. Architecture, objective, data, receiver,
checkpoint selection, and statistical family remain unchanged.

Directional replication requires improvement in core margin without loss of
core or mean retrieval relative to the paired H0-010 result in at least two of
the three seeds. Confirmation design requires the stronger condition: at least
two seeds must pass the complete joint/core/name Holm family. A merely
directional replication keeps negative coverage and motivates an explicit
core-margin objective in the next development experiment.

Confirmation data remain prohibited throughout H0-012.
