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

Pending. This document is frozen before task materialization or generation.
