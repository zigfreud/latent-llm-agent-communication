# LIP-H0-010 matrix v3 — dynamic learning signal, development gate failed

The unrolled initial-condition objective beat static layer-0 regression in all
three paired seeds. On the untouched development gate, induced-trajectory RMSE
fell from 2.376 to 1.269, 2.319 to 1.274, and 2.225 to 1.288. Mean retrieval
also rose in every pair, so the frozen paired comparison passed 3/3.

The overall gate nevertheless failed because none of the three unrolled
replicas passed the full joint/core/name Holm family. The failure is sharply
localized to the core region: its mean diagonal margins were -0.0543, -0.0308,
and -0.0042. Joint and name margins became positive, and the final seed passed
Holm for both of those regions.

Core top-1 retrieval improved substantially over the static control, reaching
46.9%, 53.1%, and 56.3% versus a 3.125% chance rate, but the remaining hard
negative errors were large enough to keep the mean margin negative. This is a
positive mechanistic result for the timeline hypothesis and a negative quality
result for the present contract: receiver evolution is worth retaining, while
the learned entry state still lacks robust separation of causal core identity.

A likely objective mismatch is that training sees only three in-batch negatives
at batch size four, whereas the gate compares each task against 31 alternatives.
The margin term is also weighted only 0.1. The next experiment should remain in
development and test broader negative coverage plus explicitly stronger core
margin pressure. PROTO-015 and functional confirmation remain premature.

- Run commit: `050848f28ec8c94c8ac0d637a51d6ea8a65a6cdb`.
- Six cells: two variants x three seeds x 512 accepted updates.
- Wall time summed across cells: 1,181.67 seconds on NVIDIA L4.
- Peak allocated VRAM: 6.98 GB.
- Colab usage: 0.64 compute units.
- Artifact: `lip-artifacts/LIP-H0-010/matrix-v3/matrix_summary.json`.
- SHA-256: `74cf5a3d3e2dc9173103d199d0d5cf393696133c3fd2c0f88c0837a3b9f7f474`.
- Confirmation data used: no.

All six cells had completed before aggregation encountered a one-literal Python
`false`/`False` error. Commit `08f25160` fixes it. The same literal-only patch
was applied to the Colab working tree without changing the frozen run commit;
the runner then verified and skipped all complete cells before producing the
matrix summary. The error had no effect on training or evaluation outputs.
