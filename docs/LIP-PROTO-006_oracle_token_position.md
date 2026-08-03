# LIP-PROTO-006 target-oracle token-position recovery

## Why this protocol exists

`LIP-PROTO-004` established a real capacity curve under the exact
single-layer suffix interface: recovery rose from approximately zero at
`K=1` to 27.54% at `K=8`, about 60% at `K=16`, and a plateau near 69% at
`K=32`. `LIP-PROTO-005`, however, found zero functional successes for every
non-text condition at the frozen `K=8`; all such generations missed the
required function entry point.

Those results are compatible if the aggregate teacher-forced metric is mostly
recovering *late* continuation tokens. At position `t`, teacher forcing gives
the model every correct reference token before `t`. The experiment may
therefore become easy only after text has already supplied the syntax, function
name, and part of the algorithm. Free generation receives none of that help.

This protocol locates the recovered information along the continuation. It is
a mechanistic diagnosis, not a new bridge-training experiment.

## Frozen experimental factors

- Same target model revision, prompt protocol, 16 held-out tasks, and frozen
  8/8 selection-confirmation split as `LIP-PROTO-004`.
- Same deterministic 64-token target-model references.
- Same selected layer `-16` and exact block-output replacement.
- Same length-controlled, attention-visible neutral carrier.
- Same packet grid `K = [1, 2, 4, 8, 16, 32, 48]`.
- Same exact self-replacement check.
- No source model, learned bridge, gain, loss, textual side channel, or
  functional execution.

The unused tasks 16:32 of the 32-task held-out bundle remain untouched. They
are reserved for a later functional confirmation if this protocol identifies
an early-token-capable packet.

## Per-token measurement

Let `y_1, ..., y_T` be the deterministic reference continuation. For condition
`c` in `{task, neutral, injected}`, the teacher-forced token loss at relative
continuation position `t` is

```text
ell_c(t) = -log p_c(y_t | prompt_c, y_<t).
```

The causal alignment is important: logits at absolute position
`prompt_length - 1 + t` predict reference token `y_t`. The raw artifact stores
every `ell_c(t)` and every top-1 correctness indicator, so later paper figures
do not depend on recomputing model forwards.

For a token set `W`, such as the first eight continuation tokens, pool all
eligible token observations inside a split and calculate

```text
L_c(W) = mean_{task i, token t in W_i} ell_{c,i}(t).
```

The task-prompt predictive advantage is

```text
A(W) = L_neutral(W) - L_task(W).
```

The fraction recovered by a packet is

```text
R_K(W) = [L_neutral(W) - L_injected,K(W)] / A(W).
```

Interpretation:

- `R = 0`: the latent packet is no better than the neutral prompt.
- `R = 1`: it recovers the full task-text advantage for that token region.
- `R < 0`: injection is harmful relative to neutral.
- `R > 1`: injection outperforms task text on the fixed reference; possible,
  but not automatically evidence of better free generation.

The estimator is a ratio of pooled NLL differences rather than a mean of
unstable single-token per-task ratios. A window is informative only when its
pooled task advantage is at least `0.05` NLL and has the pre-registered minimum
task support.

## Registered views and gate

The summary reports cumulative prefixes of 1, 4, and 8 tokens; the continuation
after token 8; the full sequence; and every individual relative token position.
Reference lengths may vary, so each position records its own task support.

On the selection split, choose the smallest `K` whose pooled recovery over the
first eight tokens is at least 10%. Test only that frozen choice against the
same 10% threshold on the disjoint confirmation split. A preflight is never
claim-eligible.

The first-eight-token window is the gate because it normally covers the opening
syntax and required Python function name. The first-token curve remains a
stricter diagnostic, while the full-sequence curve is retained as an exact
conceptual bridge to `LIP-PROTO-004`.

## Execution

```bash
python -m src.scripts.run_oracle_packet_audit \
  --config config/LIP-PROTO-006_oracle_token_position.yaml \
  --preflight

python -m src.scripts.run_oracle_packet_audit \
  --config config/LIP-PROTO-006_oracle_token_position.yaml
```

## Decision rule

- **Confirmed early-prefix crossing:** the suffix interface contains usable
  bootstrap information. Run a functional target-oracle confirmation at the
  selected `K` on the unused held-out tasks 16:32.
- **No early-prefix crossing, despite high late/full recovery:** the previous
  capacity result was dependent on teacher-forced textual state. Reject this
  suffix interface for autonomous control and move to a KV-cache or learned
  soft-prefix carrier.
- **Early recovery only at `K=32` or `K=48`:** test that packet functionally
  before changing interface; `K=8` was then a premature functional choice.

None of these outcomes decides whether latent agent communication is possible
in general. The protocol discriminates between a capacity failure and a
bootstrap/interface failure under one precisely defined intervention.

## Frozen full-run result

The 16-task full run selected `K=8` on the first eight tasks and confirmed the
first-eight-token recovery gate on the disjoint final eight tasks. Exact
self-replacement had maximum absolute NLL delta `0.0` over all seven packet
sizes. The preflight was not used for selection or inference.

| K | First 8 selection | First 8 confirmation | After token 8 selection | After token 8 confirmation | Full selection | Full confirmation |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.26% | 0.46% | -0.09% | 0.65% | 0.18% | 0.51% |
| 2 | 0.34% | 0.89% | -0.49% | 0.39% | 0.13% | 0.75% |
| 4 | 7.31% | 7.71% | 9.56% | 3.29% | 7.87% | 6.48% |
| 8 | 37.08% | 31.17% | 10.63% | 5.13% | 30.50% | 23.92% |
| 16 | 68.84% | 71.26% | 13.37% | 8.80% | 55.04% | 53.86% |
| 32 | 76.73% | 77.20% | 26.17% | 27.36% | 64.15% | 63.32% |
| 48 | 77.10% | 76.14% | 25.83% | 27.34% | 64.34% | 62.54% |

The result rejects the narrow hypothesis that `LIP-PROTO-004` recovery was
created mainly by the late teacher-forced continuation. Recovery is stronger
inside the first eight tokens than after token eight for every `K >= 8`, and
the capacity curve again saturates near `K=32`.

The first continuation token is normally a Markdown code-fence token and has
no informative task-text advantage on the selection split. Token positions
1--3 therefore cannot serve as a stable semantic bootstrap statistic. Position
4 is usually the first task-specific function-name token: its pooled task-text
advantage is 13.54/13.48 NLL on the selection/confirmation splits. Recovery at
that exact position rises from 49%/25% for `K=8` to 86%/89% for `K=16` and
90%/95% for `K=32`.

This reconciles the NLL and functional results more precisely. `K=8` carries
substantial information about the required entry point, but partial likelihood
recovery need not change the free-generation argmax; `LIP-PROTO-005` still
observed zero correct non-text entry points at that capacity. The sharp rise at
`K=16` and `K=32` motivates a functional packet-size escalation rather than an
immediate switch to a different carrier.

Generate the registered vector figure directly from the archived summary:

```bash
python -m src.scripts.plot_oracle_token_position \
  --summary runs/LIP-PROTO-006/oracle-token-position/summary.json
```

Immutable artifacts are stored at
`lip-artifacts/LIP-H0-005/LIP-PROTO-006/v1-token-position-full` on Drive:

- `summary.json`: `90206a52c0c607da05df311fc7c03da2b30852044aaf3b4c6377e4b05a29e0a8`
- `oracle_packet_records.jsonl`: `35cb58a4eea89484b17807ead3109a065420ac01729c55a0ad6401212d110fbb`
- `references.jsonl`: `4b490be0ebfc547d551a7596577bb760f580c9c7ec36426ec925e7a624d68d66`
- `resolved_config.yaml`: `fa01c47707b1d4ccccfc8ce5af130eefc13b3a55f416ddc95a372549b7d74c15`

The next protocol must keep layer `-16` and the suffix-replacement carrier,
reserve the unused held-out tasks 16:32, and compare functional matched and
task-mismatched packets at `K = [8, 16, 32]`. `K=8` is the replication anchor;
`K=16` and `K=32` test whether the early-token likelihood curve crosses a
functional decision boundary.
