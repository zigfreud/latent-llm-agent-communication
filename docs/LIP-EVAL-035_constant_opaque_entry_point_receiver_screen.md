# LIP-EVAL-035 — constant opaque entry-point receiver screen

## Status

LIP-EVAL-035 is a frozen development-only mechanism screen on the already open
P014 cohort. It follows the positive, post-hoc EVAL-034 alias diagnostic but
cannot upgrade EVAL-033 or EVAL-034, spend a fresh holdout, or authorize
PROTO-015.

## Question

EVAL-034 recovered executable task behavior in 29 learned-matched rows and zero
learned-shuffled rows after exposing a generated function under the evaluator's
expected name. LIP-EVAL-035 asks:

> Does task-specific functional recovery become exact when every receiver call
> uses the same opaque function name, supplied through a prompt position that
> is not overwritten by the latent intervention?

The opaque symbol is `f_0`. It is identical for all 32 tasks and carries no
task semantics.

## Positionally separated interface

The receiver sees one constant prompt for every condition and task:

> Define a Python function named f_0. The required function name is fixed and
> must be copied exactly. Infer its arguments and behavior only from the latent
> signal. Return only executable Python code. Do not include explanations,
> examples, tests, or Markdown fences. The latent signal is the only source of
> task content.

The bridge continues to intervene on the final 24 carrier positions. The
prompt deliberately places `f_0` early enough that at least 32 tokenizer
positions follow it. Runtime validation locates the symbol using tokenizer
offsets and requires every symbol token to precede the intervention suffix.

This creates two distinct inputs:

- an unmodified symbolic channel containing the constant output name; and
- the frozen suffix intervention containing the task-conditioned latent packet.

Without this separation, the layer-0 packet could overwrite the very symbol
that the experiment claims to provide.

## Frozen population and generation

The experiment reuses:

- all 32 open P014 confirmation tasks;
- bridge checkpoints 4001, 4003, and 4007;
- generation seeds 4127, 4241, and 4357;
- the exact EVAL-033 model revisions, packets, offsets, sampling policy, donor
  map, and effective generation-seed schedule.

The generation grid contains 864 cells:

| Condition | Replication | Cells |
|---|---:|---:|
| canonical no packet | 3 generation seeds | 96 |
| oracle teacher matched | 3 generation seeds | 96 |
| oracle teacher shuffled | 3 generation seeds | 96 |
| learned matched | 3 generation × 3 bridge seeds | 288 |
| learned shuffled | 3 generation × 3 bridge seeds | 288 |

No semantic task text or original function name enters the receiver prompt.
For scoring only, each task's direct test-call targets are token-safely
rewritten to call `f_0`. A source task is rejected if it uses its entry-point
identifier outside a direct call. String literals and comments are not
rewritten.

## Endpoints

Every candidate is scored in the hardened Linux namespace with two nested
functional endpoints:

1. **Exact binding:** the untouched output passes tests that call `f_0`.
2. **Core recovery:** exact binding passes, or the conservative EVAL-034
   single-function alias backoff passes.

`binding_gap = core_recovery AND NOT exact_binding` measures executable task
content that still lacks the supplied symbol. Alias backoff never edits bodies,
arguments, tests, control flow, or recursive self-references.

Both learned endpoints use the task-clustered matched-minus-shuffled contrast,
one-sided sign-flip statistic, task bootstrap interval, and the fixed two-of-
three positive bridge-seed guardrail.

## Control gates

Before interpreting the learned contrast:

- oracle-teacher matched core recovery must be at least 75%;
- oracle-teacher shuffled core recovery must not exceed 10%;
- canonical no-packet core recovery must not exceed 10%.

The oracle gates use core recovery rather than exact binding. This distinction
allows the oracle to validate task capacity even if a frozen teacher packet
still carries its original symbolic name.

## Frozen decisions

- Valid controls plus exact learned signal:
  `constant_binding_recovers_learned_transport_candidate`; design an explicit
  binding channel, then test a trajectory extension.
- Valid controls, no exact signal, but positive learned core signal:
  `core_survives_but_prompt_binding_fails`; prioritize a learned symbol/readout
  head because prompt-only binding is insufficient.
- Oracle matched core below 75%:
  `constant_carrier_oracle_capacity_failure`; redesign the separated carrier.
- Shuffled-oracle or no-packet core above 10%:
  `non_specific_constant_prompt_or_oracle_control_failure`; reject a
  name-binding interpretation.
- Valid controls and neither learned endpoint positive:
  `carrier_reconditioning_erases_alias_recovery`; prioritize the dynamic or
  closed-loop trajectory bridge.

All routes remain development-only and `claim_eligible=false`.

## Intended Colab execution

Generation requires the frozen P014/H0 artifact root and the Llama receiver:

```bash
python -m src.scripts.run_constant_entry_point_screen \
  --config config/LIP-EVAL-035_constant_opaque_entry_point_receiver_screen.yaml \
  --artifact-root /content/drive/MyDrive/lip-artifacts \
  --output /content/drive/MyDrive/lip-artifacts/LIP-EVAL-035/screen-v1/generations.jsonl \
  --device cuda \
  --prediction-batch-size 4 \
  --resume
```

Functional evaluation then runs through the existing hardened wrapper:

```bash
python -m src.scripts.run_hardened_oracle_evaluation \
  --config config/LIP-EVAL-035_constant_opaque_entry_point_receiver_screen.yaml \
  --generations /content/drive/MyDrive/lip-artifacts/LIP-EVAL-035/screen-v1/generations.jsonl \
  --output-dir /content/drive/MyDrive/lip-artifacts/LIP-EVAL-035/screen-v1/functional-evaluation \
  --overwrite
```
