# LIP-EVAL-036 — constrained-prefix receiver screen

## Registered result

LIP-EVAL-036 stopped after its prespecified 288-row control phase on
2026-08-18. The forced `def f_0` prefix realized in all rows, all source and
generation hashes validated, and scoring completed inside the hardened Linux
namespace.

The registered route is:

`constrained_prefix_oracle_capacity_failure`

Matched-oracle core recovery reached `69/96 = 71.875%`, three rows below the
frozen 75% capacity gate. The learned phase was therefore not authorized and
was not executed. The result remains development-only and
`claim_eligible=false`.

## Control results

| Condition | Rows | Exact | Alias | Core | Binding gaps | Prefix realized |
|---|---:|---:|---:|---:|---:|---:|
| canonical no packet | 96 | 1 | 1 | 1 | 0 | 96 |
| oracle teacher matched | 96 | 69 | 69 | 69 | 0 | 96 |
| oracle teacher shuffled | 96 | 0 | 0 | 0 | 0 | 96 |

The prefix-realization, shuffled-oracle specificity, and no-packet specificity
gates passed. Only the matched-oracle capacity gate failed.

## What changed relative to EVAL-035

The constrained prefix solved the symbolic-binding problem. On the same open
cohort, matched-oracle exact recovery rose from 6/96 in EVAL-035 to 69/96,
while binding gaps fell from 62 to zero. Exact, alias, and core recovery are
now identical.

But core recovery rose only from 68/96 to 69/96. The intervention already
contained executable task content in most EVAL-035 oracle rows; forcing the
name made that content directly callable, but did not materially expand which
tasks the packet could support. This separates two properties:

- lexical readout/binding is now functional;
- receiver capacity under full packet replacement is still insufficient.

This is progress even though the frozen gate is negative: the failure can no
longer be attributed mainly to the output name.

## Coverage signature

Matched-oracle recovery was complete in 21 tasks, absent in seven, and partial
in four. The non-full tasks were 49, 98, 174, 208, 258, 362, 392, 427, 450,
453, and 476. This task-structured concentration argues against treating the
three-row shortfall as ordinary sampling noise.

The shuffled oracle remained 0/96 and no-packet remained 1/96, so the recovered
behavior is highly task-specific. That observation does not override the
capacity gate and does not authorize a learned-bridge claim.

## Decision

Do not append the 576 learned rows, start PROTO-015, or spend a fresh holdout.
The next development candidate is an oracle-only convex blend between the
receiver-native residual and the oracle packet. The intended question is
whether full replacement destroys native receiver state needed by the
remaining tasks:

`h_intervened = (1 - alpha) * h_native + alpha * h_packet`

The blend contract must be frozen before execution. It should reuse this open
cohort, the constrained prefix, matched/shuffled oracle controls, and the
existing `alpha=1` and no-packet endpoints where cryptographic reuse is valid.
No next experiment is authorized by this registry entry alone.

## Provenance

- run commit: `c9fd47dfb838d8e4ef7a07d26714a33cd9697de9`;
- config SHA-256: `f417da2327e432b54a4beee78cc1ebf6a07d13602f0bf22baa0cd8548591ee54`;
- design SHA-256: `9f7b4d19f1d5a101f42619cafb16ba0e14345f413b2954b1280411c69f862ec2`;
- generations SHA-256: `bfef48894fd8f007abd3defdf546eaec10985e175871c7eeae477e2095416871`;
- generation metadata SHA-256: `bbc54e193938ee1733866d6451cefe9f8003de1e8d6c65a93cec0621fe9fc5b6`;
- control summary SHA-256: `336dbd99b588551ee56917216625dc5c0813e27c63e5692abb479da93cb5b600`;
- scored rows SHA-256: `86e4cc47fa84451d2793b4ccf442e4f3fffe274ee5b94754fd0375c48f4b94b1`;
- sandbox report SHA-256: `c481c955fefd72b9d53edea38b78c3bea7395df30f13d1615e21e35322782989`.
