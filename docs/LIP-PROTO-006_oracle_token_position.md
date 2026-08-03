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
