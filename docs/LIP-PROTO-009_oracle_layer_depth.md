# LIP-PROTO-009 oracle layer-depth localization

## Purpose

`LIP-PROTO-008` established a narrow oracle feasibility result: exact target
prompt state controlled functional generation when replayed at the input of all
32 decoder blocks, while a single block-output packet and replay over only the
late 16 block inputs failed. The next question is whether early task state can
become self-sustaining after a finite number of blocks, or whether it must be
refreshed through the entire decoder.

This protocol changes one axis: **the length of a contiguous replay prefix over
decoder depth**. Token packet size remains `K=32`; hook boundary, target model,
carrier, task prompt protocol, sampling settings, and generation budget remain
fixed. No source model or learned bridge is present.

## Frozen depth ladder

The target has 32 decoder blocks, numbered `0:31`. Every matched scope replays
the exact final 32 prompt states from the same task before a prefix of blocks.

| Scope | Replayed blocks | Layers | Scalars relative to all-layer |
|---|---:|---:|---:|
| `early_quarter_input` | `0:8` | 8 | 25% |
| `early_half_input` | `0:16` | 16 | 50% |
| `early_three_quarters_input` | `0:24` | 24 | 75% |
| `all_layer_input` | `0:32` | 32 | 100% |

Each scope has a Sattolo-deranged task-shuffled control with the same boundary,
layers, tensor shapes, norm scale, and generation seed. Neutral and task-text
controls are unchanged, giving ten conditions and `16 × 10 × 3 = 480` records
in the claim-oriented run.

The ladder is intentionally cumulative. If replay succeeds through block 23
but not through block 15, the latent task state needs correction beyond the
first half but no longer needs direct intervention in the final quarter. If
only all-layer replay succeeds, the present interface behaves as persistent
layer-wise memory rather than as a state that can be injected once and carried
forward autonomously.

This is still an oracle localization experiment. Reducing layers reduces
payload linearly, but even eight layers at `K=32` are not a useful compression
claim.

## Fresh tasks and provenance

The registry contains 18 new public MBPP `test` tasks sampled with seed `809`.
The sampling configuration binds the complete `LIP-PROTO-008` task manifest as
an exclusion set, and the new manifest records both manifest hashes and a hash
of the excluded task IDs. This makes disjointness machine-verifiable rather
than dependent on a lucky change of sampling seed.

- tasks `0:2`: sacrificial preflight, never claim-eligible;
- tasks `2:18`: untouched 16-task confirmation set;
- generation seeds: `[101, 202, 303]`.

The target revision remains pinned to
`53346005fb0ef11d3b6a83b12c895cca40156b6c`.

## Primary inference

The primary family contains four directional, task-paired hypotheses comparing
matched replay with its equal-capacity task-shuffled control. They are tested in
this fixed sequence:

1. all 32 layers;
2. first 24 layers;
3. first 16 layers;
4. first 8 layers.

Testing stops after the first non-rejection. Each contrast averages the three
generation seeds within task and applies a one-sided sign-flip randomization
test at `alpha=0.05`. Fixed-sequence gatekeeping controls family-wise error at
the same alpha under the registered order without applying a post-hoc search
over successful depths.

The full table also reports two-sided task-bootstrap intervals and Holm-adjusted
secondary contrasts against neutral, adjacent depths, and task text. Those
secondary results describe effect size and shape; they do not replace the
primary decision rule.

A depth is supported only when its matched mean exceeds both neutral and its
shuffled control and its primary fixed-sequence hypothesis is rejected. The
run supports the mechanistic conclusion only if task text is nonzero and the
all-layer result from `LIP-PROTO-008` replicates first.

## Registered outcomes

- **32 fails:** the earlier all-layer effect does not replicate on new tasks;
  stop interface reduction and audit task/model sensitivity.
- **32 passes, 24 fails:** memory must be refreshed into the final decoder
  quarter under the present carrier.
- **24 passes, 16 fails:** the final quarter is dispensable, but correction
  beyond the first half is necessary.
- **16 passes, 8 fails:** early layers are privileged, and the first half is
  sufficient.
- **8 passes:** a compact early-depth interface exists; the next registered
  ablation should reduce token positions before learning a bridge.
- **Matched and shuffled rise together:** layer replay perturbs generation but
  does not transmit task identity.
- **Text is zero:** the task sample is uninformative, so no interface conclusion
  is claim-eligible.

## Execution

Materialize the exclusion-bound registry:

```bash
python -m src.scripts.materialize_oracle_tasks \
  --config config/LIP-PROTO-009_mbpp_test_sampling.yaml
```

Run the sacrificial preflight, inspect raw outputs, and score them:

```bash
python -m src.scripts.run_oracle_memory_functional \
  --config config/LIP-PROTO-009_oracle_layer_depth.yaml \
  --preflight

python -m src.scripts.evaluate_oracle_packet_semantics \
  --config config/LIP-PROTO-009_oracle_layer_depth.yaml \
  --generations runs/LIP-PROTO-009/preflight/generations.jsonl \
  --output-dir runs/LIP-PROTO-009/preflight/evaluation \
  --allow-incomplete --overwrite

python -m src.scripts.run_hardened_oracle_evaluation \
  --config config/LIP-PROTO-009_oracle_layer_depth.yaml \
  --generations runs/LIP-PROTO-009/preflight/generations.jsonl \
  --output-dir runs/LIP-PROTO-009/preflight/functional-evaluation \
  --allow-incomplete --overwrite
```

The two-task preflight authorizes the confirmation only when text and matched
all-layer replay show nonzero raw capacity, shuffled all-layer replay does not
match that capacity, all replay self-checks remain within `1e-4`, and the output
grid and provenance pass inspection. It is not expected to pass the inferential
fixed sequence.

The executed preflight exposed a high-false-stop failure mode in that
execution-only rule before any confirmation task was inspected. The original
artifact and rule are preserved; the registered replacement authorization is
documented in [LIP-PROTO-009 amendment 1](LIP-PROTO-009_amendment_1.md). The
confirmation design and scientific claim gate remain unchanged.

After authorization, execute the unchanged confirmation:

```bash
python -m src.scripts.run_oracle_memory_functional \
  --config config/LIP-PROTO-009_oracle_layer_depth.yaml

python -m src.scripts.evaluate_oracle_packet_semantics \
  --config config/LIP-PROTO-009_oracle_layer_depth.yaml \
  --generations runs/LIP-PROTO-009/generations.jsonl \
  --output-dir runs/LIP-PROTO-009/evaluation \
  --overwrite

python -m src.scripts.run_hardened_oracle_evaluation \
  --config config/LIP-PROTO-009_oracle_layer_depth.yaml \
  --generations runs/LIP-PROTO-009/generations.jsonl \
  --output-dir runs/LIP-PROTO-009/functional-evaluation \
  --overwrite

python -m src.scripts.plot_oracle_layer_depth \
  --summary runs/LIP-PROTO-009/functional-evaluation/summary.json \
  --output-stem runs/LIP-PROTO-009/LIP-PROTO-009_functional_layer_depth
```

## Result

The original design above was frozen before task materialization or generation.
The post-registration execution completed all `480/480` confirmation records
over 16 untouched tasks and three seeds. The hardened namespace evaluator
validated the complete grid, sandbox, model revision, task-manifest binding,
and design fingerprint, so the run is claim-eligible. Every replay self-check
had maximum absolute logit delta `0.0`.

The preregistered confirmatory gate did **not** pass. All-layer replay was the
first fixed-sequence hypothesis and produced a positive task-clustered
difference of `0.229167` over its shuffled control, with bootstrap interval
`[0.0625, 0.4375]`. Its one-sided exact sign-flip test gave `p=0.0625`, so it
was not rejected at `alpha=0.05`; testing therefore stopped before the 24-,
16-, and 8-layer hypotheses. `semantic_transport_supported` is consequently
`false`, `supported_scopes` is empty, and no minimum sufficient replay depth is
declared.

| Condition | Functional passes | Task-clustered mean | 95% task-bootstrap CI | Entry point declared |
|---|---:|---:|---:|---:|
| Neutral carrier | 0/48 | 0.00% | [0.00%, 0.00%] | 0/48 |
| Matched early quarter, blocks `0:8` | 9/48 | 18.75% | [4.17%, 37.50%] | 48/48 |
| Shuffled early quarter | 0/48 | 0.00% | [0.00%, 0.00%] | 0/48 |
| Matched early half, blocks `0:16` | 10/48 | 20.83% | [4.17%, 41.67%] | 48/48 |
| Shuffled early half | 0/48 | 0.00% | [0.00%, 0.00%] | 0/48 |
| Matched early three quarters, blocks `0:24` | 12/48 | 25.00% | [6.25%, 45.83%] | 48/48 |
| Shuffled early three quarters | 0/48 | 0.00% | [0.00%, 0.00%] | 0/48 |
| Matched all-layer input, blocks `0:32` | 11/48 | 22.92% | [6.25%, 43.75%] | 48/48 |
| Shuffled all-layer input | 0/48 | 0.00% | [0.00%, 0.00%] | 0/48 |
| Task text | 10/48 | 20.83% | [4.17%, 39.58%] | 48/48 |

The causal pattern is nevertheless sharp and must be reported separately from
the negative confirmatory decision. All four matched depths exceeded both
neutral and their equal-capacity task-shuffled controls; all 192 shuffled
depth observations failed, while every matched and text observation declared
the correct entry point. Text, 8, 16, and 32 layers succeeded on the same four
tasks (`30`, `38`, `252`, and `458`). The 24-layer scope added one isolated
success on task `469` and had a descriptive fixed-sequence result of
`difference=0.25`, interval `[0.0625, 0.458333]`, and `p=0.03125`, but it was
correctly marked `tested=false` after the first hypothesis failed. The 16- and
8-layer descriptive tests each gave `p=0.0625`.

This is absence of sufficient confirmatory evidence under the registered
sequence, not evidence that early-prefix replay has zero capacity. The exact
test is discrete because the unit is task rather than generation: four task
clusters with consistently nonnegative paired differences yield
`2^-4 = 0.0625`, whereas the fifth positive cluster at 24 layers yields
`2^-5 = 0.03125`. Replicate seeds improve within-task rate estimation but do
not manufacture additional independent task clusters. The non-monotonic point
estimate at 24 layers may be sampling variation or interference from continued
late-layer refresh; this run does not distinguish those explanations.

The diagnostic maps reproduce the task-structured latent geometry seen in
`LIP-PROTO-008`. Averaged over layer and suffix position, task-centered energy
was `0.5793` in values before cache construction, `0.4763` in residual inputs,
and `0.1694` in keys before RoPE. Corresponding angular separation was
`0.6132`, `0.5048`, and `0.1795`; normalized task effective rank was `0.6257`,
`0.6448`, and `0.5412`. Value signal peaked at suffix positions `-22`, `-21`,
`-20`, `-18`, and `-19`; residual signal peaked at `-8`, `-19`, `-21`, `-31`,
and `-32`. These maps describe where task variation exists; the zero shuffled
controls provide the causal task-identity evidence.

The next registered experiment should not simply rerun the same 16-task test
or reorder hypotheses after seeing this result. It should increase the number
of independently informative task clusters and explicitly separate receiver
competence from channel fidelity, while retaining an untouched confirmation
split. The observed 24-layer peak is a hypothesis for that replication, not a
licensed positive claim from this run.

The canonical artifact is stored at `lip-artifacts/LIP-PROTO-009` on Drive.
It includes the exclusion-bound task registry, immutable preflight and
pre-confirmation manifest, all full outputs, hardened sandbox reports, vector
and raster figures, configs, amendment, source/authorization/confirmation/
analysis commit records, and a verified final manifest.

- `SHA256SUMS`: `68428c63356e345cf39999de2773ef0d4abb1e92e14047e0e29944c3c8e0033b`
- `runs/generations.jsonl`: `1523a1622ace78a37c7d45e5730dbb8d2d613bee694d15ba98bbe54dcadb355f`
- `runs/functional-evaluation/summary.json`: `ae72f235a1675436504377ea0c033496c197f98675e50d6f5cf983a0c478aeff`
- `runs/state-diagnostics.json`: `d80702dd23411b57fe4ce2b03f5c5a68a5687eb445db7872a919a7780d9716e1`
- `runs/LIP-PROTO-009_functional_layer_depth.svg`: `25f7e92a4510c8d7c3c316ee0bdcb78b5a607d7f606315633746b20196c72e680`
- `runs/LIP-PROTO-009_state_diagnostics.svg`: `0f71aded97b1f3faf66792b837c3bbe271a0c06b70078c0a06929436017ad5c9`
