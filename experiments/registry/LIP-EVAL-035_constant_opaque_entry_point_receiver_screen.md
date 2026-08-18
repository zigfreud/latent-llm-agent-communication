# LIP-EVAL-035 — constant opaque entry-point receiver screen

## Registered result

LIP-EVAL-035 completed on 2026-08-18 over the frozen 864-cell development
grid. All 32 tasks, three bridge seeds, five conditions, generation metadata,
and source hashes validated. Functional scoring completed inside the hardened
Linux namespace.

The prespecified diagnostic route is:

`constant_carrier_oracle_capacity_failure`

This route takes precedence over the learned contrasts below. The result is
development-only, `claim_eligible=false`, and cannot upgrade EVAL-033 or
EVAL-034.

## Condition results

| Condition | Rows | Exact passes | Alias passes | Core passes | Binding gaps |
|---|---:|---:|---:|---:|---:|
| canonical no packet | 96 | 1 | 1 | 1 | 0 |
| learned matched | 288 | 18 | 21 | 21 | 3 |
| learned shuffled | 288 | 0 | 1 | 1 | 1 |
| oracle teacher matched | 96 | 6 | 65 | 68 | 62 |
| oracle teacher shuffled | 96 | 0 | 0 | 0 | 0 |

The no-packet and shuffled-oracle specificity controls passed. The matched
oracle reached `68/96 = 70.83%` core recovery, below the frozen 75% capacity
gate by four passing rows. Therefore the protocol does not permit the learned
contrast to select a positive mechanism route.

## Learned contrasts

The exact-binding endpoint nevertheless contains a prespecified matched signal:

- 18 learned-matched passes versus 0 learned-shuffled passes;
- task-clustered mean difference `0.0625`;
- task-bootstrap 95% interval `[0.0138889, 0.125]`;
- one-sided exact sign-flip `p=0.03125`;
- five nonzero task clusters;
- all three bridge seeds positive (`0.0520833`, `0.0520833`, `0.0833333`).

The core-recovery endpoint is also positive:

- 21 learned-matched passes versus 1 learned-shuffled pass;
- task-clustered mean difference `0.0694444`;
- task-bootstrap 95% interval `[0.0208333, 0.1284722]`;
- one-sided exact sign-flip `p=0.015625`;
- six nonzero task clusters;
- all three bridge seeds positive (`0.0833333`, `0.0520833`, `0.0729167`).

These are real secondary observations under the frozen grid, not authorization
to relabel the registered result. The oracle capacity gate failed first.

## The useful property

The constant prompt alone declared `f_0` in all 96 no-packet rows. Under the
matched oracle intervention, only 11 of 96 rows declared `f_0`, while 68 rows
contained functionally correct task content after conservative alias exposure.
Sixty-two oracle rows therefore had a binding gap.

The `f_0` prompt tokens were outside the overwritten 24-token suffix, so this
is not literal token replacement. It is downstream representational drift:
the injected task packet changes the receiver trajectory strongly enough to
displace the supplied symbolic readout. The constant name successfully
separates the channels at input, but it does not keep them independent through
generation.

At the same time, learned matched exceeded learned shuffled in exact and core
functionality. That is the closest result so far to a causal, executable bridge
signal, but it remains sparse and lives under a failed receiver-capacity gate.

## Concentration

Only tasks 46, 49, 62, 152, 372, and 418 contribute a learned core difference.
Five of them contribute an exact difference. Task 152 alone contributes six
exact matched passes and the only shuffled core pass. The result does not show
broad transport across the 32-task cohort.

Matched-oracle core recovery was complete in 20 tasks, partial in five, and
zero in seven. Exact oracle binding was complete in only two tasks. This
supports a receiver-carrier redesign before a fresh trajectory-training spend.

## Decision

Do not start PROTO-015 and do not spend a holdout from this result. The next
step should be a small development mechanism branch that makes symbolic
binding invariant to the injected semantic trajectory, while preserving the
same matched, shuffled, no-packet, and oracle controls.

The most direct candidate is a constrained receiver carrier that fixes only
the lexical prefix `def f_0` outside the latent channel and leaves the argument
list and body to the receiver. A deterministic alias-only rescore remains a
diagnostic baseline, not the bridge itself. The dynamic trajectory branch
remains motivated for coverage after receiver capacity is restored.

No next experiment is authorized by this registry entry; its contract should
be reviewed first.

## Provenance

- run commit: `6811f6c7ec98b5306e599c645b8ebe10bc2d3b32`;
- config SHA-256: `b7f676123af59d28c1870a7e9103f8e6151a4e454d98a8275afdc80db9fb5a8f`;
- design SHA-256: `0cf81f599ca473e9626d623cdef1aed8f6d4eb02a6bbae6fb8cc3ade0b9e14e8`;
- generations SHA-256: `a7cc355c0eff3fb36a729cf5f2882c2b7621ef087267965a2687c3fcbd61ee07`;
- generation metadata SHA-256: `2578a6b61aea70d8adc2ad2c56235b61ab0c3d08904d0ec0dff919e13c5124dd`;
- functional summary SHA-256: `9442e05b2fff7b423d7e7744412e41d5de8bb2d2af0d6c49051c891e3502ac51`;
- scored rows SHA-256: `cad2be57fe77e64909b46bb31c9ffa4089a6126b36339d4707e4ccad5389181e`;
- sandbox report SHA-256: `c481c955fefd72b9d53edea38b78c3bea7395df30f13d1615e21e35322782989`.
