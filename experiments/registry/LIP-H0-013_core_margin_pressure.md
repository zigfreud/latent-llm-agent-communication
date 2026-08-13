# LIP-H0-013 — stronger margin pressure improves core directionally, not robustly

H0-013 changed one scalar relative to the frozen H0-011 seed-4007 screen:
induced-trajectory `lambda_margin` increased from 0.10 to 1.00. Receiver
evolution, negative coverage, architecture, entry loss, every other trajectory
term, data, seed, update count, example exposure, checkpoint rule, and gate
remained fixed. Confirmation data were not used.

The intervention passed its directional gate. Core margin rose from +0.00335
to +0.01090, core retrieval from 68.75% to 71.88%, and mean retrieval from
73.96% to 81.25%. RMSE also improved slightly from 1.42517 to 1.41856. Joint
and name passed their Holm-adjusted tests.

Core remained below the frozen strong threshold (`p_Holm=0.0813`), so the
complete family failed. The contract therefore prohibits exact multi-seed
replication, functional confirmation, and PROTO-015.

The training history supplies a more important diagnostic. Despite the
tenfold coefficient, median and final batch-level core hinge losses did not
fall relative to H0-011 (median 0.03830 to 0.03781; final 0.02298 to 0.02473).
The selected solution moved in the desired direction on the development gate,
but the optimized violation itself was not resolved. Another blind coefficient
increase would not identify the missing property.

The next step should be an evaluation rather than another training branch:
measure gradient norms and cosine alignment among core margin, NCE, and
reconstruction through the frozen receiver trajectory on the H0-011 and
H0-013 checkpoints. That separates three live explanations: insufficient
gradient scale, destructive objective conflict, or mismatch between in-batch
and gate hard negatives.

- Run commit: `32e97fcc57d973d3f2a08fad3fc5fc767130cec4`.
- Pilot: 4 updates, numeric gate passed, best step 2, 27.16 seconds.
- Screen: 128 updates, best step 120, 107.75 seconds.
- Peak allocated VRAM: 8.24 GB.
- H0-013 interval: 0.13 visible Colab compute units.
- Pilot SHA-256: `90852aeb08edac3c091b1f2f1b29d851c26b71940a8688499281abc0f16c60bb`.
- Screen SHA-256: `80792ddaa130ea9d9de9d2baddbe433cafb7b497bc09ec1949841f932d49d0b6`.
- Confirmation data used: no.
