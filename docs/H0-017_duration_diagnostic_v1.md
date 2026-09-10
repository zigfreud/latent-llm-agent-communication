# H0-017 duration diagnostic v1

Exploratory duration comparison specified before execution on 2026-09-10. Separate from the frozen H0-017 paired screen; cannot authorize functional evaluation.

Run control then treatment from the same seeded corrector, each for 512 successful updates. Preserve data, optimizer, constant learning rate, architecture, loss, dropout and batch partition/order. Validate every eight updates on development_selection only. Never construct or evaluate gate or confirmation datasets.

Save endpoint weights at 128, 256 and 512 and the selected best weights within each prefix using the original selection key. Primary comparison: equal-step endpoint relative RMSE advantage and mean/core retrieval differences; 512 is primary, 256 intermediate. Report all budgets without selecting a winner retrospectively. Secondary: selected checkpoint within each prefix, separately reported.

Audit the first 128 updates against the archived corrected run, including environment and numeric differences. Across the new pair verify initial tensor hashes, batch plans and precision events. Repeated checkpoints are not independent seeds. No new training tasks are introduced.

Exploratory support requires greater relative RMSE advantage at 512 than 128, treatment mean/core retrieval no lower than control at 512, and no decline in treatment mean/core from 128. Report magnitudes; directional conditions establish neither robustness nor functional transport. Stop at 512 without extensions after results. Stop earlier only for integrity, numeric or frozen-parameter failure and preserve partial outputs.

Use --duration-policy config/H0-017_duration_diagnostic_v1.json. Frozen base configuration remains unchanged. Summary stage is duration_diagnostic_cell, gate metrics are null, and the effective duration policy has its own provenance hash alongside the base hash. No resume is implemented: endpoint weights omit optimizer/RNG/scaler state and cannot represent an equivalent continuation. Interrupted runs require a recorded fresh namespace and restart from initialization.

Validation: policy drift and gate access rejected; snapshots preserve current weights, optimizer and RNG; existing trajectory/aggregation tests pass.
