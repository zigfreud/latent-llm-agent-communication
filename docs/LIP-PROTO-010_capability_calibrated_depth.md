# LIP-PROTO-010 capability-calibrated layer-depth replication

## Purpose

`LIP-PROTO-009` produced a sharp task-identity pattern but did not cross its
registered confirmatory threshold. On 16 fresh tasks, every task-shuffled
condition failed and every matched replay condition declared the required
entry point. Functional successes were nevertheless concentrated in four
tasks, almost exactly the same tasks that the target model solved from text.
The first fixed-sequence hypothesis therefore stopped at `p=0.0625`; the
descriptive 24-layer contrast had five positive task clusters and `p=0.03125`
but was correctly left untested.

This result mixes two mechanisms:

1. **receiver competence:** whether the target model can solve a task at all;
2. **channel fidelity:** whether matched latent replay supplies the identity of
   that task rather than merely perturbing generation.

`LIP-PROTO-010` separates them prospectively. A text-only calibration phase
selects a deterministic registry of tasks on which the receiver demonstrated
some functional capacity. Confirmation then tests matched versus equal-capacity
task-shuffled latent replay with new seeds and no adaptive task choice.

This protocol does not estimate performance over arbitrary MBPP tasks. Its
target population is narrower: tasks selected by the frozen capability rule
for this exact target model, revision, prompt protocol, and sampling regime.
That restriction is a feature of the mechanistic question and must remain
explicit in every paper claim.

## Frozen candidate registry

Materialize 192 MBPP `test` candidates using sampling seed `1010`. The registry
excludes the complete task manifests from both `LIP-PROTO-008` and
`LIP-PROTO-009`; exclusion paths, hashes, the union-of-IDs hash, and disjointness
are recorded in the candidate manifest. The target remains pinned to revision
`53346005fb0ef11d3b6a83b12c895cca40156b6c`.

The pool size is fixed rather than adaptively expanded. If it does not yield 32
eligible tasks, the protocol stops and records an insufficient-calibration
result. It must not sample additional candidates after observing the shortfall.

## Capability screening and deterministic selection

Every candidate receives the ordinary task text under two screening seeds,
`[17, 29]`, with the same generation settings as confirmation. Let

`Y(i, text, s) = 1`

when candidate `i` passes its functional tests under screening seed `s`, and
zero otherwise. Candidate `i` is eligible exactly when

`max(Y(i, text, 17), Y(i, text, 29)) = 1`.

The confirmation registry is the **first 32 eligible tasks in candidate-manifest
order**. The ordering was fixed by the materialization seed before any model
output existed. No rank by pass count, code quality, task topic, prompt length,
or apparent latent suitability is permitted. The selector accepts only a
complete functional evaluation from the validated Linux namespace sandbox and
binds the resulting registry to hashes of the candidate manifest, generations,
metadata, scored outputs, and screening summary.

Screening seeds are not inferential replicates and never appear in confirmation.
Screening itself is always `claim_eligible=false`.

## Confirmation design

The selected 32 tasks receive the unchanged `K=32` cumulative replay ladder:

| Scope | Replayed block inputs | Depth |
|---|---:|---:|
| `early_quarter_input` | `0:8` | 8 |
| `early_half_input` | `0:16` | 16 |
| `early_three_quarters_input` | `0:24` | 24 |
| `all_layer_input` | `0:32` | 32 |

Each matched scope has a Sattolo-deranged task-shuffled control with identical
tensor count, shape, hook boundary, norm scale, carrier, and generation seed.
Neutral and task-text controls are retained. Confirmation seeds are
`[401, 509, 631]`, producing `32 × 10 × 3 = 960` records.

The candidate tasks have been observed under text screening, but no latent
condition, shuffled control, neutral control, or confirmation seed is observed
before selection. Consequently, selection and confirmation randomness are
disjoint, while the estimand remains explicitly conditional on demonstrated
text capability.

## Pre-confirmation operational amendment A1

The screening completed on 2026-08-04 with 81 eligible candidates (70 passed
both screening seeds and 11 passed one). The deterministic eligible prefix
therefore selected 32 tasks without expanding or reordering the candidate pool.

The first confirmation invocation stopped before writing any confirmation
record. The formatted neutral prompt contained 51 tokens, while selected tasks
`257` and `263` each contained 50. A masked length-matched carrier can add
invisible left padding, but cannot remove a visible neutral token without
truncating the prompt. This was an input-compatibility failure, not an observed
confirmation outcome.

Amendment `LIP-PROTO-010-A1` replaces only the confirmation neutral user text
with `Use the latent signal.`. Under the pinned tokenizer this formats to 39
tokens; the selected task range is 50--78 tokens, leaving every carrier valid
and preserving the last 32 visible positions used by `K=32`. Task selection,
conditions, layer scopes, shuffled permutation, seeds, generation settings,
inference, and output paths are unchanged. Screening and selection retain the
original frozen configuration and hashes. Confirmation uses the separately
versioned
`config/LIP-PROTO-010_capability_calibrated_depth_confirmation.yaml`.

This amendment was fixed after inspecting only prompt lengths and the thrown
compatibility error. No matched, shuffled, neutral, text, or confirmation-seed
generation existed when it was recorded. The runner now checks the entire
registry for carrier compatibility before any expensive state capture.

## Primary estimand and inference

For selected task `i`, depth `d`, and confirmation seed `s`, let `Y(i,d,s)` be
the binary functional-pass outcome. The task-level matched-minus-shuffled
difference is

`D(i,d) = mean_s Y(i,d,s) - mean_s Y(i,shuffle(d),s)`.

The reported effect is the average of `D(i,d)` over the 32 selected tasks.
Seeds are averaged *within* task; they reduce noisy generation variance but do
not turn 32 independent tasks into 96 independent observations.

The prospective primary family is tested one-sided at `alpha=0.05` in the fixed
sequence:

1. first 24 blocks versus its 24-block shuffled control;
2. first 16 blocks versus its 16-block shuffled control;
3. first 8 blocks versus its 8-block shuffled control.

Testing stops after the first non-rejection. Each test uses a task-clustered
sign-flip randomization distribution. Zero task differences are omitted from
sign enumeration because multiplying zero by either sign produces the same
statistic. Up to 20 nonzero task differences are enumerated exactly; larger
sets use the registered Monte Carlo approximation. Bootstrap intervals resample
tasks, never individual generations.

The 24-layer hypothesis is prospective for `LIP-PROTO-010`, although motivated
by the descriptive `LIP-PROTO-009` pattern. Its evidence comes entirely from a
new exclusion-bound candidate registry and new confirmation seeds.

All-layer replay remains a descriptive positive anchor and is not a member of
the primary family. The earlier 32-first sequence answered replication before
localization and masked the planned 24-layer test when replication narrowly
missed alpha. The new family directly answers the registered question: whether
task-specific capacity survives when refresh is removed from the final decoder
quarter. The protocol will not claim non-inferiority or no performance loss,
because no scientifically justified non-inferiority margin has been specified.

## Claim gate and registered interpretations

A task-specific early-prefix result requires:

- nonzero confirmation performance from task text;
- matched 24-layer replay above neutral and its task-shuffled control;
- rejection of the first registered 24-layer hypothesis.

If that gate passes, later fixed-sequence results may support 16 or 8 layers.
The smallest rejected tested scope is reported as `smallest_confirmed_scope`;
failure at a shorter depth is absence of support, not proof that the longer
depth is mathematically minimal.

Registered interpretations:

- **24 fails:** no confirmatory evidence that the final quarter can be removed
  for the capability-calibrated population;
- **24 passes, 16 fails:** task-specific state survives without refresh in the
  final quarter, but sufficiency of the first half is not established;
- **16 passes, 8 fails:** the first half is supported, while the first quarter
  is not established;
- **8 passes:** all tested early-prefix depths are supported, motivating a
  token-position reduction before learned-bridge training;
- **matched and shuffled rise together:** replay changes generation without
  transmitting target-task identity;
- **confirmation text is zero:** calibration did not reproduce under new seeds,
  invalidating the channel interpretation;
- **fewer than 32 eligible candidates:** stop without adaptive pool expansion.

## Execution

Materialize the exclusion-bound candidates:

```bash
python -m src.scripts.materialize_oracle_tasks \
  --config config/LIP-PROTO-010_mbpp_test_sampling.yaml
```

Generate and score the text-only screen:

```bash
python -m src.scripts.run_oracle_capability_screen \
  --config config/LIP-PROTO-010_capability_calibrated_depth.yaml

python -m src.scripts.run_hardened_oracle_evaluation \
  --config config/LIP-PROTO-010_capability_calibrated_depth.yaml \
  --generations runs/LIP-PROTO-010/screening/generations.jsonl \
  --output-dir runs/LIP-PROTO-010/screening/functional-evaluation \
  --overwrite
```

Apply the deterministic selection rule:

```bash
python -m src.scripts.select_oracle_capability_tasks \
  --config config/LIP-PROTO-010_capability_calibrated_depth.yaml
```

Only after selection succeeds, run confirmation and hardened scoring:

```bash
python -m src.scripts.run_oracle_memory_functional \
  --config config/LIP-PROTO-010_capability_calibrated_depth_confirmation.yaml

python -m src.scripts.run_hardened_oracle_evaluation \
  --config config/LIP-PROTO-010_capability_calibrated_depth_confirmation.yaml \
  --generations runs/LIP-PROTO-010/generations.jsonl \
  --output-dir runs/LIP-PROTO-010/functional-evaluation \
  --overwrite
```

Render the paper-facing depth curve and state maps:

```bash
python -m src.scripts.plot_oracle_layer_depth \
  --summary runs/LIP-PROTO-010/functional-evaluation/summary.json \
  --output-stem runs/LIP-PROTO-010/LIP-PROTO-010_functional_layer_depth

python -m src.scripts.plot_oracle_state_diagnostics \
  --diagnostics runs/LIP-PROTO-010/state-diagnostics.json \
  --output-stem runs/LIP-PROTO-010/LIP-PROTO-010_state_diagnostics
```

## Result

Pending preregistered execution. This section must be appended without changing
the frozen design above.
