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

Execution completed on 2026-08-04. The selector restored and hash-verified all
nine frozen inputs from the canonical `LIP-PROTO-010` and `LIP-PROTO-011`
artifacts. It reproduced the 81-task eligible registry, proved that the first
64 eligible tasks were exactly the two earlier latent cohorts, and selected
every remaining eligible task at ranks `[64:81]`. The resulting 17-task
registry was disjoint from both predecessors. Generation produced all 612
registered records. Hardened scoring reported no missing cells, a validated
Linux namespace sandbox, `claim_eligible=true`, and
`semantic_transport_supported=true`.

Task-clustered functional results were:

| Condition | Passes | Rate | 95% task-bootstrap interval |
|---|---:|---:|---:|
| Neutral carrier | 0/51 | 0.00% | [0.00%, 0.00%] |
| Task text | 46/51 | 90.20% | [82.35%, 96.08%] |
| Matched full `K=32` | 42/51 | 82.35% | [66.67%, 96.08%] |
| Shuffled full `K=32` | 0/51 | 0.00% | [0.00%, 0.00%] |
| Matched drop octet 1, keep `-24 ... -1` | 43/51 | 84.31% | [66.67%, 100.00%] |
| Shuffled drop octet 1 | 0/51 | 0.00% | [0.00%, 0.00%] |
| Matched drop octet 2 | 29/51 | 56.86% | [35.29%, 76.47%] |
| Shuffled drop octet 2 | 0/51 | 0.00% | [0.00%, 0.00%] |
| Matched drop octet 3 | 17/51 | 33.33% | [13.73%, 54.90%] |
| Shuffled drop octet 3 | 0/51 | 0.00% | [0.00%, 0.00%] |
| Matched drop terminal octet 4, keep `-32 ... -9` | 1/51 | 1.96% | [0.00%, 5.88%] |
| Shuffled drop octet 4 | 0/51 | 0.00% | [0.00%, 0.00%] |

The preregistered replication anchor rejected. Full-`K=32`
matched-minus-shuffled replay had mean task-level difference `0.823529`, 16
nonzero task clusters, interval `[0.666667, 0.960784]`, and exact one-sided
`p=0.0000152588`. Matched replay passed 42 programs while the equal-capacity
task-shuffled control passed none. The known early-quarter channel therefore
replicated on the final latent-unseen holdout and opened the four-member
deletion family.

Three deletion hypotheses survived Holm correction:

| Deleted octet | Mean matched-minus-shuffled difference | Nonzero tasks | Raw exact `p` | Holm `p` | Confirmed dispensable in context |
|---|---:|---:|---:|---:|---:|
| 1: `-32 ... -25` | 0.843137 | 15 | 0.0000305176 | 0.0001220703 | yes |
| 2: `-24 ... -17` | 0.568627 | 12 | 0.0002441406 | 0.0007324219 | yes |
| 3: `-16 ... -9` | 0.333333 | 7 | 0.0078125 | 0.015625 | yes |
| 4: `-8 ... -1` | 0.019608 | 1 | 0.5 | 0.5 | no |

Thus a 24-vector packet is prospectively confirmed sufficient for bounded
task-specific transport, and octets 1, 2, and 3 are each dispensable when the
other 24 positions remain. The deletion outcomes are not exchangeable in
practice: observed performance declines monotonically as the deleted block
moves toward the prompt boundary, from 84.31% after deleting the earliest
octet to 1.96% after deleting the terminal octet. Removing octet 1 happened to
score slightly above full `K=32`, but the design did not register a superiority
or non-inferiority claim for that contrast and the difference must not be
interpreted as beneficial compression.

Deleting octet 4 was not confirmed sufficient. Its near-zero observed rate is
strong descriptive evidence that the tested complement `-32 ... -9` lacks the
functional capacity retained by the other complements, but failure to reject
does not by itself prove that every vector in octet 4 is necessary. A necessity
claim requires a positive test that distinguishes the two task-dependent tail
positions from the task-invariant generation boundary and resolves
within-octet interactions.

The positional gradient was stable across generation seeds. Full-`K=32` rates
were 76.47%, 94.12%, and 76.47% for seeds 1103, 1217, and 1301. Drop-octet-1
rates were 88.24%, 82.35%, and 82.35%; drop-octet-2 rates were 52.94%, 70.59%,
and 47.06%; drop-octet-3 rates were 29.41%, 41.18%, and 29.41%; and
drop-octet-4 rates were 0.00%, 5.88%, and 0.00%. Task text scored 88.24%,
94.12%, and 88.24%. Neutral and every shuffled condition remained 0.00% under
every seed.

### Exploratory mechanism analysis

The aggregate state maps do not rank causal necessity. Within the eight replay
layers, residual-input task-signal fractions averaged `0.6500`, `0.7053`,
`0.6012`, and only `0.2006` across octets 1 through 4. Corresponding angular
separations were `0.6738`, `0.7404`, `0.6296`, and `0.2125`. Mean residual
norms were nevertheless similar (`2.0999`, `2.2628`, `2.1527`, and `2.0718`).
The lowest-energy, least-separated octet was therefore the one whose deletion
most severely reduced function. This directly reinforces the `LIP-PROTO-011`
result: local geometric salience is not a causal sufficiency or necessity
score.

A post-result tokenizer audit explains why the terminal octet deserves finer
resolution. Offsets `-8` and `-7` contained task-dependent required-function
name tokens, with 16 distinct token IDs at each position across the 17 tasks.
Offsets `-6 ... -1` were identical across every task: the closing backtick and
period, end-of-turn marker, assistant-header start, `assistant` role token,
header end, and final newline. Their hidden states are still contextualized by
the complete task even though their visible token IDs are constant. Octet 4
therefore straddles the final task-specific identifiers and a fixed generation
boundary that may act as a contextual integration site. This audit is
exploratory and was not part of the primary family.

The supported claim remains bounded. For this model revision, chat template,
capability-calibrated final holdout, native oracle source, and replay through
only the first eight decoder blocks, at least three distinct 24-position
packets transmitted task identity above equal-capacity shuffled controls. The
smallest confirmed packet size is 24 among the packet sizes tested here; it is
not a mathematical minimum. The result does not establish non-inferiority to
text, arbitrary task coverage, a learned sender-to-receiver bridge, cross-model
interoperability, or that the terminal octet is uniformly necessary. The next
prospective position experiment should decompose offsets `-8 ... -1`, while
separating the two function-name positions from the six fixed chat-boundary
positions on a newly frozen task population.

The canonical artifact is stored at `lip-artifacts/LIP-PROTO-012` on Drive. It
contains 17 payload files plus `SHA256SUMS`: the frozen config and source
commit, selected task registry and selection report, generations and metadata,
hardened scores and summary, aggregate state diagnostics, and raster/vector/PDF
figures. All 17 manifest entries were verified after final rendering, and the
folder was independently visible through the authenticated Drive API.

- `SHA256SUMS`: `507b8a99255227999edd55475b25257332bd0798bd726a9d9d73e0b5bd2b8773`
- `runs/generations.jsonl`: `0705a539ccb0cbede1312fadc8502c11c2558d6ef2af9d655f6c898f88d70f01`
- `runs/functional-evaluation/summary.json`: `294abb90ee428fb6add650ce489cd910a611d93ff5147cca5a6babb817263533`
- `runs/state-diagnostics.json`: `c109d9ef63d2e45ef04f1ef7bef4eafd598af8492317c71cc85d1e4d2a13760c`
- `runs/LIP-PROTO-012_block_deletion.svg`: `9d9b85803f73bad0029a31377c73c46d9d9abc2d2ccdffbd3ba72b60ba04cbcc`
- `runs/LIP-PROTO-012_state_diagnostics.svg`: `43d0270640297d1745099bcc0ab8e0e52d3ac4eeefc4c2940901aa0aaf026db7`
