# LIP-PROTO-012 leave-one-octet-out latent packet

## Purpose

`LIP-PROTO-011` independently replicated task-specific functional transport
with a 32-position native latent packet replayed through only the first eight
decoder blocks. On a latent-unseen 32-task holdout, task text passed 91.67%, the
matched full packet passed 85.42%, and its equal-capacity task-shuffled control
passed 0%. The preregistered diagnostic and peak-window `K=8` packets both
passed 0%; a terminal `K=8` suffix produced only a non-confirmatory 10.42%
exploratory signal.

That result rejects the particular claim that the eight positions with greatest
task-centered energy are sufficient. It does not identify why the full packet
works. At least three explanations remain:

1. a packet-size threshold lies somewhere between 8 and 32 positions;
2. particular regions of the prompt suffix supply necessary information;
3. useful information is synergistic, so positions that are weak alone become
   useful only in combination.

`LIP-PROTO-012` moves from sparse selection to complementary deletion. It
partitions the 32-position packet into four contiguous octets and removes one
octet at a time. Every deletion leaves 24 positions. This tests whether each
octet is dispensable in the context of the other three while holding packet
size, replay depth, boundary, carrier, and generation protocol fixed.

## Final latent-unseen capability holdout

The immutable `LIP-PROTO-010` text screen produced 81 eligible tasks in
candidate-manifest order. The first 32 eligible tasks received latent
conditions in `LIP-PROTO-010`; eligible ranks 33--64 received their first latent
conditions in `LIP-PROTO-011`. This protocol uses every remaining eligible
task: zero-based ranks `[64:81]`, equivalently human ranks 65--81.

The 17 selected tasks must equal that exact suffix of the frozen eligible
registry. Selection may not expand the candidate pool, reorder by text pass
count, inspect latent suitability, or omit a remaining eligible task. The
selector verifies:

- the hash-bound `LIP-PROTO-010` candidate registry and hardened text screen;
- the first 32-task calibration selection;
- the hash-bound `LIP-PROTO-011` selected registry and selection report;
- `LIP-PROTO-011` claim eligibility and full-`K=32` replication;
- disjointness of the new 17 tasks from all 64 earlier latent tasks.

The calibration artifact has `SHA256SUMS` hash
`80ea9320defa471a69b891affd8b58f761daffef94d64b95a5abec2419a4016c`.
The `LIP-PROTO-011` artifact has `SHA256SUMS` hash
`5ee64230177e7fe8f09d3c643d9fb7ba5a1f5d7399965f4af85862079e6aabe9`.

The estimand remains conditional on this receiver having demonstrated some
text capability under the frozen screen. Seventeen task clusters are fewer than
the predecessor confirmations, but they exhaust rather than sample the
remaining eligible registry.

## Frozen intervention

Capture covers the last 32 prompt positions at every decoder block so aggregate
state diagnostics remain comparable. Replay remains frozen at the input
boundary of decoder blocks `0:8`, represented by layer indices
`[-32, ..., -25]`. This is the earliest quarter confirmed in
`LIP-PROTO-010` and replicated in `LIP-PROTO-011`.

The packet partition is:

| Octet | Prompt-suffix offsets |
|---|---|
| 1 | `-32 ... -25` |
| 2 | `-24 ... -17` |
| 3 | `-16 ... -9` |
| 4 | `-8 ... -1` |

The five matched packet patterns are:

| Pattern | Deleted offsets | Replayed vectors |
|---|---|---:|
| `full_k32` | none | 32 |
| `drop_octet_1_k24` | `-32 ... -25` | 24 |
| `drop_octet_2_k24` | `-24 ... -17` | 24 |
| `drop_octet_3_k24` | `-16 ... -9` | 24 |
| `drop_octet_4_k24` | `-8 ... -1` | 24 |

Unselected carrier positions retain their native neutral states. They are not
zeroed and are not replaced by a constant. Each pattern has a Sattolo-deranged
task-shuffled control with the same offsets, tensor count, layer count, target
carrier, and generation seed. Neutral and task-text conditions remain in the
design.

Confirmation seeds `[1103, 1217, 1301]` are disjoint from screening seeds
`[17, 29]`, `LIP-PROTO-010` seeds `[401, 509, 631]`, and `LIP-PROTO-011` seeds
`[743, 887, 991]`. The complete design is:

```text
17 tasks x 12 conditions x 3 seeds = 612 records.
```

## Estimand

For task `i`, packet pattern `p`, and generation seed `s`, let
`Y(i,p,s)` equal one when the generated program passes its functional tests.
For every pattern define the task-level matched-minus-shuffled difference:

```text
D(i,p) = mean_s Y(i,matched(p),s) - mean_s Y(i,shuffled(p),s).
```

The estimated effect is the arithmetic mean of `D(i,p)` over the 17 tasks.
Generation seeds are repeated observations within a task, not independent
tasks. Confidence intervals therefore resample task clusters, and sign-flip
randomization changes the signs of task-level differences.

The task-shuffled contrast is the causal identity test. If matched and shuffled
packets behave alike, the injection may perturb decoding but does not show that
the target task's information was transmitted. If matched replay beats
task-shuffled replay while all physical properties remain fixed, task identity
is the differing factor.

## Gatekept primary family

The primary inference is one-sided at familywise `alpha=0.05`.

First test the replication anchor:

```text
full_k32 matched > full_k32 task-shuffled.
```

If and only if that anchor rejects, open the four equal-status deletion
hypotheses:

```text
drop_octet_j_k24 matched > drop_octet_j_k24 task-shuffled,
for j in {1, 2, 3, 4}.
```

The four raw one-sided sign-flip values receive Holm step-down correction.
There is no arbitrary fixed order among octets. Holm sorts their `p` values and
compares the smallest against `0.05/4`, the next against `0.05/3`, and so on,
while reporting adjusted values in the original octet order. This controls the
probability of at least one false rejection across the deletion family.

The 17-task design permits exact enumeration whenever at most 17 task
differences are nonzero. For example, if all 17 task signs are positive, the
smallest one-sided value is `1 / 2^17 = 0.0000076294`, which remains below the
first Holm threshold.

The broader descriptive table also includes matched-versus-neutral,
full-versus-each-deletion, and task-text-versus-neutral contrasts. Its
two-sided values receive Holm correction across the configured descriptive
family. They do not replace the registered primary tests.

## Claim gate and interpretations

The `K=24` block-deletion semantic gate requires:

- nonzero task-text performance under the new confirmation seeds;
- confirmed full-`K=32` matched-over-shuffled replication;
- at least one Holm-confirmed `K=24` matched-over-shuffled deletion pattern.

Registered interpretations are:

- **full `K=32` fails:** the known channel did not replicate on the final
  holdout; make no claim about block deletion;
- **full passes and no deletion passes:** no tested `K=24` complement is
  confirmed sufficient; the result remains compatible with a capacity
  threshold above 24 or global positional synergy;
- **some deletions pass:** at least one 24-vector packet transmits task identity,
  and deletion outcomes depend on which octet is removed;
- **all deletions pass:** no single contiguous octet in this partition is
  necessary for bounded functional transport; the packet contains substantial
  positional redundancy at `K=24`.

A confirmed deletion establishes that its removed octet is dispensable in the
context of the other 24 positions. A non-confirmed deletion does **not** by
itself prove that the missing octet is necessary: absence of evidence may also
come from sampling noise or reduced power. Direct claims of necessity require
positive evidence from a contrast designed for necessity, not merely failure
to reject sufficiency.

Passing any deletion would establish a 24-vector native oracle carrier in this
bounded system. It would not establish a learned sender-to-receiver bridge,
non-inferiority to text, a universal 24-vector minimum, arbitrary text
replacement, or interoperability between different model families.

## Visual outputs

The paper-facing block-deletion figure contains:

1. matched and task-shuffled functional rates with task-bootstrap intervals;
2. the task-text interval and neutral baseline;
3. the anchor raw `p` value and Holm-adjusted deletion values;
4. a 5-by-32 mask showing exactly which prompt positions were replayed.

The existing aggregate state-diagnostic figure is rendered separately for
residual inputs, pre-RoPE keys, and pre-cache values. These geometric maps are
descriptive. Functional matched-versus-shuffled behavior remains the causal
communication evidence.

## Execution

Restore the hash-bound `LIP-PROTO-010` calibration files and
`LIP-PROTO-011` predecessor files at the paths frozen in the config, then
materialize the final holdout:

```bash
python -m src.scripts.select_oracle_block_deletion_tasks \
  --config config/LIP-PROTO-012_block_deletion.yaml
```

Run generation and hardened scoring:

```bash
python -m src.scripts.run_oracle_memory_functional \
  --config config/LIP-PROTO-012_block_deletion.yaml

python -m src.scripts.run_hardened_oracle_evaluation \
  --config config/LIP-PROTO-012_block_deletion.yaml \
  --generations runs/LIP-PROTO-012/generations.jsonl \
  --output-dir runs/LIP-PROTO-012/functional-evaluation \
  --overwrite
```

Render the functional deletion figure and aggregate state maps:

```bash
python -m src.scripts.plot_oracle_block_deletion \
  --summary runs/LIP-PROTO-012/functional-evaluation/summary.json \
  --output-stem runs/LIP-PROTO-012/LIP-PROTO-012_block_deletion

python -m src.scripts.plot_oracle_state_diagnostics \
  --diagnostics runs/LIP-PROTO-012/state-diagnostics.json \
  --output-stem runs/LIP-PROTO-012/LIP-PROTO-012_state_diagnostics
```

## Result

Pending preregistered execution. This section must be appended without changing
the frozen design above.
