# LIP-H0-012 — negative coverage replicates directionally, not strongly

H0-012 froze the H0-011 intervention, ran only the missing seeds 4001 and
4003, and combined them with the already frozen seed 4007. Each seed therefore
used batch 16, 128 updates, 2,048 examples, and fifteen in-batch alternatives.
No confirmation task was opened.

The directional result replicated in all three seeds. Relative to the paired
batch-4 H0-010 cells, every core margin improved and changed from negative to
positive. Core retrieval was also non-lower in all three. The final core
margins were +0.00741, +0.00758, and +0.00335 for seeds 4001, 4003, and 4007.

The strong result did not replicate: the Holm-adjusted core test failed in all
three seeds (`p_Holm=0.123`, `0.173`, and `0.373`), although joint and name
passed in every cell. Thus broader negative coverage is a reproducible
component of identity learning, but it is not sufficient to make the core a
robust carrier.

This sharpens the mechanistic account. Training through frozen receiver
evolution solves much of the trajectory problem; broader negative coverage
corrects the sign of core discrimination; the remaining deficit is the
magnitude and reliability of core-specific separation. The next authorized
step is one development-only screen that retains coverage and adds an explicit
core-margin pressure. Functional confirmation and PROTO-015 remain premature.

- Run commit: `0330044a8aba72dc684c29a3a8c8fce3322bbf64`.
- Seed 4001: RMSE 1.39216, core retrieval 71.88%, best step 128.
- Seed 4003: RMSE 1.39866, core retrieval 62.50%, best step 112.
- Frozen seed 4007: RMSE 1.42517, core retrieval 68.75%, best step 120.
- New-cell wall time: 217.26 seconds; peak allocated VRAM: 8.24 GB.
- H0-012 interval: 0.51 visible Colab compute units.
- Seed 4001 SHA-256: `ea85d0677ea67e8348b578b90189f5e06ca3a08bab3ef980cc36ea6e87b4ec83`.
- Seed 4003 SHA-256: `732fec87a73f7705bde2275145ba3d6cb0e12ba0bdb38833877633e44fe80b9b`.
- Confirmation data used: no.
