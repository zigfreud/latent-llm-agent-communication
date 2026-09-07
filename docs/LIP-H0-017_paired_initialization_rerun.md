# H0-017: initialization repair and paired rerun

Authorized on 2026-09-07 after the original 128-update screen exposed two implementation defects: seeding occurred after corrector construction, and the producer omitted the selected variant from resolved_stage. Original outputs are preserved separately; their observed effects do not certify the causal effect of live-state conditioning.

This repair seeds the bridge before construction, loads identical corrector tensors from one shared initialization checkpoint in both arms, and records tensor and file SHA-256 values. The aggregator rejects missing or differing initialization evidence. The producer records the selected variant directly, so no metadata adapter is needed.

Run namespace: `screen-v2-paired-initialization`. Order: preflight shared initialization verification, archive initialization, control, archive control, treatment, archive treatment, aggregate and archive complete results.

Keep the original experiment and aggregation YAML files byte-identical: seed 4007, 128 successful updates per arm, batch size 16, source checkpoint, receiver, data, objective, checkpoint-selection policy and all scientific thresholds unchanged. The previous pilot remains evidence of numeric feasibility; this rerun adds explicit initialization checks before training.

This is a corrected development comparison on already observed data, not independent confirmation. Do not tune the configuration using its gate result. The frozen result routes remain design-only EVAL-038 on pass, no progression on fail, and no confirmation use.

Regression tests exercise the real bridge builder after unrelated random-number consumption, exact shared weights across both variants, rejection of a different seed or corrupted initialization, and rejection of missing or mismatched initialization evidence during aggregation.
