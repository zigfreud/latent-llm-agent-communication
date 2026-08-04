# LIP-PROTO-011 position-sparse latent packet

## Purpose

`LIP-PROTO-010` supplied claim-eligible causal evidence that native latent
states can replace task text for functional program synthesis under a bounded
oracle design. On capability-calibrated tasks, task text passed 93.75%, a
matched `K=32` latent packet replayed only through the first 8 of 32 decoder
blocks passed 88.54%, and the equal-capacity task-shuffled control passed 0%.
Changing only task identity therefore destroyed function, while preserving
packet shape, scalar count, boundary, layers, carrier, and generation seed.

That result establishes functional latent substitution in the tested system;
it does not establish arbitrary text replacement, a learned sender-to-receiver
bridge, non-inferiority to text, or compression to eight vectors. Replay depth
and packet length remained separate: `LIP-PROTO-010` reduced receiving decoder
blocks to 8 but kept all 32 prompt-suffix vectors.

`LIP-PROTO-011` tests whether task-specific function survives with only eight
of those 32 vectors and whether *which* prompt positions are replayed matters.
Depth remains frozen at the confirmed first eight decoder blocks. The new axis
is packet position.

## Sealed calibration reuse and latent-unseen holdout

No text screen is rerun. The experiment reuses the immutable
`LIP-PROTO-010` candidate registry and hardened text-only screen from the
canonical Drive artifact whose `SHA256SUMS` hash is
`80ea9320defa471a69b891affd8b58f761daffef94d64b95a5abec2419a4016c`.
Every reused file has an individually frozen hash in the configuration.

The predecessor screen yielded 81 eligible tasks in candidate-manifest order.
The first 32 were used in `LIP-PROTO-010`. This protocol selects zero-based
eligible ranks `[32:64]`, equivalently human ranks 33--64. These 32 tasks were
observed under text screening but have never received matched, shuffled,
neutral, or other latent conditions. The selector must prove that the new
registry equals the frozen slice and is disjoint from the predecessor's first
32 tasks. It may not reorder by text pass count, prompt length, topic,
diagnostic score, or anticipated latent suitability.

The estimand remains conditional: performance among tasks for which this exact
receiver showed some text capability under the frozen `LIP-PROTO-010` screen.
It is not an estimate over arbitrary MBPP tasks.

## Prospectively selected prompt positions

`LIP-PROTO-010` stored aggregate-only state diagnostics for 32 suffix positions
and 32 decoder layers. For the residual input states that are causally replayed,
compute task-centered energy at each position and take the arithmetic mean
across layers. Rank positions by descending mean, breaking exact ties by
ascending packet offset.

The eight highest positions are frozen as:

```text
[-32, -30, -23, -22, -21, -20, -19, -18]
```

The contiguous eight-position window with the greatest mean energy is frozen
as:

```text
[-23, -22, -21, -20, -19, -18, -17, -16]
```

The source diagnostic hash is
`23e1b1052f1865d8572a84e9d9d9df553a15efb38b575a1a90adb8e957f2fef2`.
The selector independently recomputes both choices and stops if they differ.
These positions were selected after observing `LIP-PROTO-010`, but before any
latent output on the new 32-task holdout. They are therefore hypotheses derived
from one task sample and prospectively tested on another.

## Frozen intervention

All latent conditions replay residual states at the input boundary of decoder
blocks `0:8`, represented by layer indices `[-32, ..., -25]`. Capture spans the
last 32 positions at every decoder layer so the aggregate diagnostic grid
remains comparable, but generation injects only the rows named by each
condition. Unselected carrier positions retain their native neutral states;
they are not zeroed or silently replaced.

| Pattern | Replayed offsets | Vectors |
|---|---|---:|
| `full_k32` | `-32 ... -1` | 32 |
| `diagnostic_top_k8` | `-32, -30, -23 ... -18` | 8 |
| `peak_window_k8` | `-23 ... -16` | 8 |
| `suffix_k8` | `-8 ... -1` | 8 |

`full_k32` is a fresh-task replication anchor. `diagnostic_top_k8` asks whether
the most task-variable residual positions support sparse transport.
`peak_window_k8` tests a contiguous implementation-friendly window near the
observed peak. `suffix_k8` retains the historical last-eight interface and
tests whether a naive suffix loses information that a diagnostic selection
preserves.

Every pattern has a Sattolo-deranged task-shuffled control using the same
offsets, scalar count, layer count, target carrier, and generation seed.
Neutral and task-text conditions are retained. Confirmation seeds
`[743, 887, 991]` are disjoint from both the screening seeds `[17, 29]` and the
`LIP-PROTO-010` confirmation seeds `[401, 509, 631]`. The complete design is
`32 tasks x 10 conditions x 3 seeds = 960` records.

## Estimand and clustered inference

For task `i`, pattern `p`, and generation seed `s`, let `Y(i,p,s)` be one when
the generated program passes its functional tests. Define the per-task paired
difference

```text
D(i,p) = mean_s Y(i,matched(p),s) - mean_s Y(i,shuffled(p),s).
```

The reported effect averages `D(i,p)` over tasks. Seeds reduce generation
noise within a task; they do not turn 32 independent task clusters into 96
independent samples. Confidence intervals therefore bootstrap tasks, and
sign-flip randomization changes the signs of task-level differences.

The one-sided primary family uses fixed-sequence gatekeeping at `alpha=0.05`:

1. `full_k32` versus its shuffled control;
2. `diagnostic_top_k8` versus its shuffled control;
3. `peak_window_k8` versus its shuffled control;
4. `suffix_k8` versus its shuffled control.

Testing stops after the first non-rejection. The order first verifies that the
known 32-vector interface transfers to the new holdout, then tests the
highest-signal sparse packet, then progressively more constrained positional
rules. Up to 20 nonzero task differences are enumerated exactly; larger sets
use the registered 100,000-sample Monte Carlo sign-flip approximation.

The broader comparison table also includes direct diagnostic-versus-suffix,
peak-window-versus-suffix, diagnostic-versus-peak, and full-versus-diagnostic
contrasts. Its two-sided sign-flip values receive Holm correction. Those
contrasts test performance differences between packet layouts; the primary
matched-versus-shuffled family tests task-identity transport.

No non-inferiority claim is registered. A sparse packet may carry causal task
identity while still performing worse than text or the full packet.

## Claim gate and interpretations

The position-sparse semantic gate requires:

- nonzero task-text performance under the new confirmation seeds;
- confirmed `full_k32` matched-over-shuffled replication;
- confirmed `diagnostic_top_k8` matched-over-shuffled transport.

Registered interpretations:

- **full `K=32` fails:** the predecessor effect did not replicate on the new
  capability-calibrated holdout; make no sparse-packet claim;
- **full passes, diagnostic `K=8` fails:** eight vectors selected by the prior
  geometric diagnostic are not confirmed sufficient;
- **diagnostic `K=8` passes:** task-specific functional information can be
  conveyed by eight prospectively selected latent vectors at the confirmed
  early-quarter depth;
- **peak window also passes:** a contiguous eight-vector packet around the
  diagnostic peak is supported;
- **suffix also passes:** positional selection may be unnecessary at `K=8`
  under multi-layer early-quarter replay;
- **diagnostic/peak pass while suffix fails:** useful capacity is position
  dependent, and the historical terminal suffix discards causal task signal;
- **matched and shuffled rise together:** sparse injection perturbs generation
  without transmitting the target task's identity.

Passing `K=8` would prove an eight-vector oracle carrier in this bounded
system, not a learned protocol. Each vector still has the model hidden width,
and fixed position metadata is part of the interface.

## Execution

Restore the hash-bound `LIP-PROTO-010` source files from its canonical artifact,
then materialize the registered latent-unseen holdout:

```bash
python -m src.scripts.select_oracle_position_tasks \
  --config config/LIP-PROTO-011_position_sparse_packet.yaml
```

Run confirmation and hardened scoring:

```bash
python -m src.scripts.run_oracle_memory_functional \
  --config config/LIP-PROTO-011_position_sparse_packet.yaml

python -m src.scripts.run_hardened_oracle_evaluation \
  --config config/LIP-PROTO-011_position_sparse_packet.yaml \
  --generations runs/LIP-PROTO-011/generations.jsonl \
  --output-dir runs/LIP-PROTO-011/functional-evaluation \
  --overwrite
```

Render the joint functional/position figure and the aggregate state maps:

```bash
python -m src.scripts.plot_oracle_position_packet \
  --summary runs/LIP-PROTO-011/functional-evaluation/summary.json \
  --output-stem runs/LIP-PROTO-011/LIP-PROTO-011_position_sparse_packet

python -m src.scripts.plot_oracle_state_diagnostics \
  --diagnostics runs/LIP-PROTO-011/state-diagnostics.json \
  --output-stem runs/LIP-PROTO-011/LIP-PROTO-011_state_diagnostics
```

## Result

Pending preregistered execution. This section must be appended without changing
the frozen design above.
