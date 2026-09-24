# H0-017 joint objective ablation v1

Frozen before execution, 2026-09-24. Exploratory development diagnostic; does not change the historical H0-017 gate.

## Question

With the same full packet and frozen models, does removing standalone core/name identity objectives improve joint task retrieval under closed-loop online backpropagation?

Core and name are evaluated as a unit. The existing joint metric includes boundary positions; this definition remains fixed in both arms. Joint retrieval measures packet discrimination, not demonstrated functional communication.

## Paired design

- Both arms: `closed_loop_live`, seed 4007, fresh AdamW, shared initial corrector, same frozen source encoder, receiver, candidate bank, batches, 256 training tasks and 32 development-selection tasks.
- `regional`: historical symmetric NCE averaged over joint/core/name, and historical weighted regional margin.
- `joint_only`: symmetric NCE and margin use joint only, with their original overall coefficients (no division by three).
- Incoming-trajectory and corrected-state losses both receive this change. Reconstruction, per-site cosine, norm regularization and correction energy retain all previous coefficients and component weights. This isolates identity supervision; it does not remove reconstruction of packet regions.
- Both arms run 512 updates, evaluation every eight updates, endpoint and selected snapshots at 128/256/512. No resume from old weights.
- Both select checkpoints using the same lexicographic key: joint top-1, joint mean diagonal margin, negative normalized residual RMSE, earlier step. Standalone core/name scores have no role in selection or acceptance.

## Readout and bounds

Primary comparison is joint retrieval at the fixed 512-update endpoint, with paired task wins/losses and joint margin changes. Report RMSE and the 128/256/512 trajectory, plus best-through-budget as secondary descriptive results. The common selection set is reused development data: no confirmatory p-value or generalization claim, no winner declared from a cherry-picked checkpoint. One seed and 32 tasks limit precision.

The historical regional arm is rerun in the same current environment to avoid confounding objective changes with environment changes. Historical selection can be rescored descriptively, but historical and new runs must not be silently mixed. Do not test other methods or change data, learning rate, loss coefficients, duration or joint definition within this comparison.

No development-gate dataset or confirmation dataset is constructed. No functional generation, new data cohort, RL or response-supervised loss is authorized by this policy. Save the frozen policy, code commit, environment, shared initialization, batch plans, histories, summaries, endpoint/selected weights and checksums in persistent storage.
