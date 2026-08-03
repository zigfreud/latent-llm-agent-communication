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
