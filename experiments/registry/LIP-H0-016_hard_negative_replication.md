# LIP-H0-016 — the frozen system replicates at the minimum cross-seed gate

H0-016 repeated the exact H0-015 system at bridge/dropout seeds 4001 and
4003. Both cells used the same 224/256 frozen hard-negative partition, loss,
receiver evolution, architecture, batch size, 128-update budget, checkpoint
rule, and untouched development gate. Confirmation data were not used.

The preregistered aggregate gate passed, but at its minimum threshold. Seed
4003 passed the complete joint/core/name Holm family; seed 4001 did not. With
the frozen strong H0-015 seed 4007, the total is therefore 2/3 strong seeds and
1/2 new strong seeds. The separate all-new-seeds criterion failed.

| Seed | Joint retrieval / margin / Holm p | Core retrieval / margin / Holm p | Name retrieval / margin / Holm p | Complete family |
| --- | --- | --- | --- | --- |
| 4001 | 65.62% / +0.01711 / 0.06102 | 75.00% / +0.01052 / 0.06102 | 87.50% / +0.07045 / 0.000030 | no |
| 4003 | 81.25% / +0.02394 / 0.000100 | 75.00% / +0.00952 / 0.03883 | 78.12% / +0.07072 / 0.000030 | yes |
| 4007 (frozen H0-015) | 87.50% / +0.03344 / 0.000030 | 75.00% / +0.01744 / 0.00553 | 90.62% / +0.07285 / 0.000040 | yes |

The useful signal is now sharper. Name identity is stable and strongly
separated in all three cells, while core retrieval is exactly 75% in every
cell. The remaining sensitivity is chiefly in joint retrieval and the
inferential strength of joint/core margins. Seed 4001 is a preregistered
failure even though all its mean margins are positive: joint and core both
landed just outside the corrected threshold at `p_Holm=0.06102`.

A post-hoc diagnostic found slightly more late core hinge pressure in seed
4001 than in 4003 or 4007. The difference is small and was not part of the
gate, so it is a clue about optimization variance, not an explanation.

This result establishes threshold-level cross-seed robustness of the frozen
system. It does not replicate the causal batch-membership contrast across
seeds: only seed 4007 has a matched H0-013 random-batch control. Nor does
identity geometry establish functional task transmission.

H0-016 authorizes preregistration of a bounded LIP-EVAL-033 design, not its
execution. The existing P014 functional cohort has already been opened, so a
new evaluation on it must remain development-only rather than masquerade as
independent confirmation. The recommended primary endpoint is learned-matched
minus learned-shuffled functional performance, task-clustered across all three
fixed bridge seeds. A positive result may authorize design of a fresh-cohort
PROTO-015; it cannot itself supply that confirmation. Dynamic mining and
PROTO-015 remain premature.

- Run commit: `7ec2ed17a3b45a30c8d84b8a28fcafe971c5326c`.
- Seed 4001: best step 128, 117.13 seconds, strong gate failed.
- Seed 4003: best step 120, 118.76 seconds, strong gate passed.
- Peak allocated VRAM: 8.24 GB per cell on NVIDIA L4.
- Aggregate SHA-256: `141083dab2d22156b64059c19fbf4be62a2a692e2529ab32b236c65404df25bd`.
- Frozen batch-plan SHA-256: `2057fbc1f146058cbf4da4687d1eae7e158d3c48afa788cdad02827172963783`.
