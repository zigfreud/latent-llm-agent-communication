# LIP-PROTO-005 functional target-oracle packet test

## Purpose

`LIP-PROTO-004` selected the smallest sufficient exact target-state packet,
`K=8`, using predictive NLL and confirmed 27.54% recovery on a disjoint split.
`LIP-PROTO-005` asks whether that recovered context changes executable task
behavior. It is an oracle interface test, not a learned source-to-target bridge
result.

## Frozen design

Only the final eight `LIP-PROTO-004` confirmation tasks are used. Layer `-16`,
packet size `K=8`, the length-controlled neutral carrier, and exact `replace`
intervention are frozen before generation. Three stochastic generation seeds
reuse the `LIP-PROTO-001` settings (`256` tokens, temperature `0.2`, top-p
`0.95`). Conditions are:

- `neutral_no_lip`: length-controlled neutral carrier without a packet;
- `text_only_no_lip`: original task text, the target-model upper control;
- `oracle_packet_k1`: one-vector replication control;
- `oracle_packet_k8`: selected same-task packet;
- `shuffled_oracle_packet_k8`: a Sattolo task derangement preserving packet
  form and capacity while destroying task identity.

The same effective random seed is used across conditions within each
task/replicate. Generation records include the complete task specification,
immutable target revision, bundle/config/design hashes, packet provenance, and
the exact condition plan. Resume mode validates that every existing record
belongs to the same frozen design.

## Execution

```bash
python -m src.scripts.run_oracle_packet_functional --preflight --overwrite

python -m src.scripts.run_oracle_packet_functional --resume

python -m src.scripts.evaluate_oracle_packet_semantics \
  --generations runs/LIP-PROTO-005/generations.jsonl \
  --overwrite
```

Generation and code execution are deliberately separated. Syntax scoring does
not execute model output. Functional scoring must run without Drive mounts,
credentials, or network access in a disposable sandbox; the legacy
resource-limited subprocess is explicitly not a security boundary.

## Decision

A semantic oracle result requires `oracle_packet_k8` to improve functional pass
rate over neutral, shuffled-`K=8`, and same-task `K=1` controls. Task text must
also demonstrate that the target/model/token budget can solve a nonzero part of
the frozen task set. With eight tasks, uncertainty and raw per-task outcomes
remain primary; a gate pass is evidence for this interface, not a broad claim
that text can be replaced generally.
