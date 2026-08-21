# LIP-EVAL-037 — oracle native-to-packet blend screen

## Registered result

LIP-EVAL-037 completed its prespecified 192-row screen on 2026-08-21 and
stopped at the frozen gate. Generation completed on an NVIDIA L4, all 192 rows
were present, and hardened scoring completed inside the Linux security
namespace.

The registered route is:

`oracle_blend_screen_no_candidate`

No interior alpha reached the required `24/32 = 75%` matched-oracle core
recovery. The confirmation phase was therefore not authorized and its 128 rows
were not generated. The result remains development-only and
`claim_eligible=false`.

## Screen results

| Alpha | Matched exact/alias/core | Shuffled exact/alias/core | Prefix | Eligible |
|---:|---:|---:|---:|:---:|
| 0.25 | 22/32 | 0/32 | 64/64 | no |
| 0.50 | 23/32 | 0/32 | 64/64 | no |
| 0.75 | 22/32 | 0/32 | 64/64 | no |

There were no binding gaps. The evaluator classified 191 rows as already
exact and one row as syntactically invalid. The prefix and shuffled-specificity
gates passed for every alpha; only matched capacity failed.

## Endpoint comparison

The cryptographically reused EVAL-036 endpoints for generation seed 4127 were:

- `alpha=0`: no-packet core `0/32`;
- `alpha=1`: matched core `23/32`, shuffled core `0/32`.

The full screen curve was therefore:

`0, 22, 23, 22, 23` matched core passes at
`alpha = 0, 0.25, 0.50, 0.75, 1`.

The best interior blend, `alpha=0.50`, tied full replacement and remained one
pass below the frozen gate. Retaining a scalar fraction of the receiver-native
residual did not expand oracle capacity at the frozen packet sites.

## Causal interpretation

The negative result is narrower than “the timeline hypothesis failed.” The
screen rejected one static operator:

`h_intervened = (1 - alpha) * h_native + alpha * h_oracle_packet`

That operator uses the live native state only through a fixed scalar mixture.
It does not learn a state-dependent transition conditioned jointly on the live
receiver residual, layer, position, and source signal. The result therefore
weakens the hypothesis that full replacement fails merely because it erases too
much native state, while leaving the stronger closed-loop trajectory hypothesis
untested.

The flat, task-specific plateau is consistent with a packet that already
contains usable content for a stable subset of tasks, but whose missing cases
are not repaired by changing only signal magnitude. The next discriminator
should separate two remaining possibilities:

1. the intervention is applied at the wrong layers or positional sites;
2. the receiver needs a state-dependent update law rather than an absolute or
   convexly mixed state.

This is an interpretation and next hypothesis, not a demonstrated mechanism.

## Decision

Do not run the locked confirmation, a learned bridge, PROTO-015, or a fresh
holdout. The next candidate is a newly frozen, development-only mechanism test
on the same open cohort that discriminates packet-site mismatch from missing
closed-loop trajectory evolution. This registry entry does not itself authorize
that experiment.

## Provenance

- evaluation commit: `76b81522edb53ce6d5659f5e1b74cf1cd65692eb`;
- initial one-row probe commit: `5817d17a028f2c2efe04fb2248b076df7d010b95`;
- config SHA-256: `b48f0238dcff434afe1349fed178fccffffb8362c36df0a6ee0272c9c9de610a`;
- design SHA-256: `0848ec3c6743a06b80c0405c7eef834a53a94aff83b678dc78197749bbe56a53`;
- generations SHA-256: `fb5fc6d0187b44c82948907a733f5b9a34b50a50a012b6598174234531d7675b`;
- generation metadata SHA-256: `082ef590349caa77d5d9a90918f545a05ce6f68ff6abe5c57f73af2877389748`;
- screen summary SHA-256: `a253c8cb8224377b360142dd8517435dc8c029d1578d992192f928a7710e42aa`;
- scored rows SHA-256: `457f0d627929dc0b636cf22e642495df3fffb8665f45229a5b257e292c145d27`;
- sandbox report SHA-256: `c481c955fefd72b9d53edea38b78c3bea7395df30f13d1615e21e35322782989`.

The initial probe produced one generation row on `5817d17a`; the remaining
191 rows and hardened evaluation ran on `76b81522`. The intervening commit
changed only partial-probe evaluator comparison layout and its regression test;
the generation path was unchanged.
