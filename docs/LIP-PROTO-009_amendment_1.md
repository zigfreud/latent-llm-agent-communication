# LIP-PROTO-009 pre-confirmation amendment 1

## Status and timing

This amendment was registered after the sacrificial two-task preflight and
before any of the 16 confirmation tasks were generated or inspected. The
original preflight artifact remains immutable. This amendment changes only the
operational authorization rule; it does not change the confirmation task set,
conditions, seeds, model, decoding contract, primary hypothesis order,
family-wise alpha, functional endpoint, or claim gate.

## Trigger

The original operational gate required nonzero text-only and matched all-layer
functional capacity on two tasks. Both task-text outputs failed, and every
matched replay reproduced the same task-specific candidate program. The
failures were receiver errors rather than channel errors:

- task `112`: text and matched replay omitted the squared-radius term;
- task `145`: text and matched replay omitted the second function argument;
- every shuffled replay emitted the candidate program belonging to its
  registered source task rather than the target task;
- all four replay self-checks had maximum absolute logit delta `0.0`.

With the `LIP-PROTO-008` text-only functional rate of `0.4375`, two consecutive
text failures have probability `(1 - 0.4375)^2 = 0.31640625`. The original
two-task nonzero rule therefore has an unacceptably high false-stop probability
for an operational check.

## Amended execution-only gate

The confirmation is authorized only if a machine-readable audit establishes
all of the following on the original preflight:

1. the complete 2-task, 10-condition, 1-seed grid and provenance are valid;
2. all four replay self-checks remain within `1e-4`;
3. text declares the target entry point and neutral does not for both tasks;
4. every matched depth emits exactly the extracted text-control program for
   that task and declares its entry point;
5. every shuffled depth names the other registered oracle task, emits exactly
   that source task's text-control program, and does not declare the target
   task's entry point.

This is a channel-identity authorization, not evidence for the scientific
claim. Its output is always marked `claim_eligible: false`. The unchanged full
run must still show nonzero text capacity, replicate all-layer functional
transport, pass the preregistered fixed sequence, and satisfy every original
gate before any mechanistic conclusion is supported.

Run the audit with:

```bash
python -m src.scripts.audit_layer_depth_preflight \
  --config config/LIP-PROTO-009_oracle_layer_depth.yaml \
  --metadata runs/LIP-PROTO-009/preflight/generations.metadata.json \
  --scored-generations \
    runs/LIP-PROTO-009/preflight/functional-evaluation/scored_generations.jsonl \
  --output runs/LIP-PROTO-009/preflight/preflight-authorization.json
```
