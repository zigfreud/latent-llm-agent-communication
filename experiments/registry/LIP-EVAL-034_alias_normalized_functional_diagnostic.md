# LIP-EVAL-034 — alias-normalized functional diagnostic

## Registered result

LIP-EVAL-034 completed on 2026-08-14 over the exact 576 frozen EVAL-033
outputs. The hardened namespace and source hashes validated. No generation,
GPU compute, fresh cohort, or holdout was consumed.

The diagnostic route is:

`matched_specific_alias_recovery_candidate`

This is a positive post-hoc diagnostic, not a positive confirmatory result.
`claim_eligible=false`, `can_upgrade_EVAL_033=false`, and the EVAL-033 endpoint
remains negative.

## Result

| Condition | Rows | Alias eligible | Functional passes | All-row rate | Eligible-only rate |
|---|---:|---:|---:|---:|---:|
| learned matched | 288 | 208 | 29 | 10.07% | 13.94% |
| learned shuffled | 288 | 206 | 0 | 0.00% | 0.00% |

Across the 32 task clusters, the matched-minus-shuffled mean was
`0.1006944`, with exploratory task-bootstrap 95% interval
`[0.0347222, 0.1875]`. Eight tasks had a nonzero difference. The one-sided
exact sign-flip value was `0.00390625`; it is descriptive because the alias
hypothesis was selected after inspecting EVAL-033.

All three bridge seeds were positive:

- seed 4001: 12 matched passes, difference `0.125`;
- seed 4003: 5 matched passes, difference `0.0520833`;
- seed 4007: 12 matched passes, difference `0.125`.

The three generation seeds contributed 10, 9, and 10 passes. This rules out a
single bridge seed or generation seed as the sole source of recovery within
the frozen matrix.

## What the alias changed

Of 576 outputs, 414 were syntactically valid programs with exactly one
top-level function and received one appended name binding. Another 40 had
multiple top-level functions and were conservatively ineligible; 122 were
syntax-invalid. There were no body, argument, control-flow, or test rewrites.

The recovered generated names are semantically related to their expected
entry points, for example:

- `sort_list` or `sort_lists` for `merge_sort`;
- `count_digits` for `count_Digit`;
- `remove_characters` for `remove_Char`;
- `find_max_sum_of_subarray` for `Find_Max`.

This signature is consistent with executable task content reaching the target
without reliable binding to the evaluator's exact symbol.

## Boundary

Recovery remains sparse: 8 of 32 tasks. Task 152 contributed all nine of its
cells and task 62 contributed six, together accounting for 15 of 29 passes.
Therefore the result does not establish broad functional transport or show
that name binding is the only remaining bottleneck. It establishes a narrower
property on the reused open cohort: some task-specific executable computation
is present in matched outputs and absent in all shuffled outputs once the
symbolic alias mismatch is normalized.

## Decision

The immediate next design should not be PROTO-015. The lowest-cost causal
discriminator is a development screen with one constant opaque entry point,
shared across every task and stated to the receiver. This removes variable
symbol transmission without leaking semantic task names.

Provisional name:

`LIP-EVAL-035_constant_opaque_entry_point_receiver_screen`

Its central contrast should remain learned matched versus the frozen learned
shuffled donor control on the open P014 cohort. A no-packet constant-name
control should measure generic prior, while an oracle-matched constant-name
anchor should verify receiver capacity. EVAL-035 design is authorized;
execution, fresh holdout spend, and PROTO-015 remain unauthorized until its
contract is reviewed.

The dynamic/closed-loop trajectory bridge remains a valid architectural branch
for increasing coverage beyond the eight recovered tasks. EVAL-034 changes
the order of operations: isolate binding first, then decide whether the next
training branch should combine a semantic trajectory channel with an explicit
symbol/readout channel.

## Provenance

- run commit: `fe8208aa97cdd813e1312156aad35153164cb223`;
- config SHA-256: `63dbb739986cd2f7a5e52e68b3cb2fb10534a50fed9b4055f8ae3051b724e190`;
- design SHA-256: `94d3dfd0fdf9d9cfda304129ae57418c0e725c8194772c4ba6333617c0028e9c`;
- summary SHA-256: `6ebb679928e387d2adc43e099aa8b341fec039b4dd1df4c6193bafdab652dd30`;
- scored rows SHA-256: `bf4bb4cdbba308b766e4e00bf9fc45074aacdae7aefc58159a20098c50b0b866`;
- sandbox report SHA-256: `c481c955fefd72b9d53edea38b78c3bea7395df30f13d1615e21e35322782989`.
