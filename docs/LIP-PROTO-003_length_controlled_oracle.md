# LIP-PROTO-003 length-controlled target-oracle transport

## Purpose

`LIP-PROTO-002` found that replacing one generation-boundary state under the
native neutral prompt recovered almost none of the task-prompt predictive
advantage. The native neutral prompt was 3–23 tokens shorter than the task
prompts, leaving absolute position and rotary-position behavior as an explicit
confound.

`LIP-PROTO-003` changes only that factor. It left-pads the tokenized neutral
prompt to the task prompt's exact length using the tokenizer pad token and an
attention mask of zero for every added position. The target therefore receives
the same visible neutral tokens, but the generation boundary and teacher-forced
continuation occupy the same sequence indices as in the task condition.

## Invariants

- Same 16 held-out tasks and 8/8 selection-confirmation split.
- Same deterministic 64-token references.
- Same exact block-output capture and `replace` intervention.
- Same layer grid and thresholds.
- Added carrier tokens are masked and cannot be attended to.
- No source model, bridge, loss, gain, functional execution, or extra text.

The shared runner records both native and carrier token counts, the pad token,
masking policy, configuration hash, and immutable model revision.

## Execution

```bash
python -m src.scripts.run_oracle_transport_audit \
  --config config/LIP-PROTO-003_length_controlled_oracle.yaml \
  --preflight

python -m src.scripts.run_oracle_transport_audit \
  --config config/LIP-PROTO-003_length_controlled_oracle.yaml
```

## Decision

- Positive confirmation recovery would identify position mismatch as a material
  failure mode and justify a position-controlled functional oracle probe.
- Near-zero recovery again would reject the current single-state carrier more
  strongly and move the protocol to a separately versioned multi-vector soft
  prefix or KV packet.
