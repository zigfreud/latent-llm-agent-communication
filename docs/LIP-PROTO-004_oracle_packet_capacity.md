# LIP-PROTO-004 target-oracle packet capacity

## Purpose

`LIP-PROTO-002` and `LIP-PROTO-003` showed that one exact target-model block
output at the generation boundary recovers less than one percent of the task
prompt's predictive advantage. This protocol asks the next narrow question:
is that failure caused by the capacity of a one-vector carrier?

The audit replaces a contiguous suffix of exact task-prompt block outputs in a
length-controlled neutral carrier. Packet size is the only experimental axis:
`K = [1, 2, 4, 8, 16, 32, 48]`.

## Frozen choices

- Layer `-16`, selected only on the `LIP-PROTO-003` selection split.
- Same 16 held-out tasks and frozen 8/8 selection-confirmation split.
- Same deterministic, self-generated 64-token references.
- Same exact block-output capture and `replace` intervention.
- Same masked, length-controlled neutral carrier.
- Same recovery statistic and 10% decision threshold.
- No source model, learned bridge, gain, loss, functional execution, or extra
  textual instruction.

`K=1` is an internal replication anchor for the selected-layer result from
`LIP-PROTO-003`. Every packet occupies the suffix ending at the generation
boundary. The maximum packet is constrained to fit entirely within visible
neutral-carrier positions; no injected vector is hidden behind the padding
attention mask.

## Selection and confirmation

On the first eight tasks, select the smallest `K` whose mean recovery is at
least 10%, provided at least two tasks are informative. Test that single frozen
`K` on the disjoint final eight tasks. A preflight is never claim-eligible.

## Execution

```bash
python -m src.scripts.run_oracle_packet_audit --preflight

python -m src.scripts.run_oracle_packet_audit
```

## Decision

- A confirmed crossing identifies a minimum sufficient packet under this
  single-layer suffix interface and unlocks a functional target-oracle test at
  the frozen `K`.
- A monotonic curve without a crossing motivates a larger/full-sequence packet.
- A flat near-zero curve rejects this single-layer suffix interface and moves
  the next protocol to a KV-cache or soft-prefix carrier. It does **not** reject
  latent communication in general.

## Frozen full-run result

The 16-task full run selected `K=8` on the first eight tasks and passed the
pre-registered confirmation gate on the disjoint final eight tasks. All tasks
were informative. Exact self-replacement had maximum absolute NLL delta `0.0`
across all seven packet sizes.

| Packet size | Selection recovery | Confirmation recovery |
|---:|---:|---:|
| 1 | 0.22% | 0.52% |
| 2 | 0.14% | 0.90% |
| 4 | 7.67% | 7.34% |
| 8 | 31.26% | 27.54% |
| 16 | 57.47% | 60.25% |
| 32 | 66.20% | 69.02% |
| 48 | 66.42% | 68.54% |

`K=1` exactly reproduces the `LIP-PROTO-003` layer-`-16` confirmation result,
providing an internal cross-protocol anchor. The curve rises sharply between
`K=4` and `K=16` and then saturates around `K=32`; this supports a distributed
capacity bottleneck under the tested interface. It does not yet establish that
the recovered predictive context is sufficient for correct executable code.

Immutable artifacts are stored at
`lip-artifacts/LIP-H0-005/LIP-PROTO-004/v1-packet-capacity-full` on Drive:

- `summary.json`: `4308c44cca4d7dea78956eaaedd03286d7fa28d46a43c595a716cd22225567c5`
- `oracle_packet_records.jsonl`: `07a4f604d9dfa4de7a77306b43992e3a20f8281af4f61238184fe19eb4a0763a`
- `references.jsonl`: `4b490be0ebfc547d551a7596577bb760f580c9c7ec36426ec925e7a624d68d66`

The next protocol must freeze `K=8` and test functional target-oracle
generation against task-text, neutral, and task-mismatched packet controls.
