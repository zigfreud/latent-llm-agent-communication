# LIP-PROTO-002 target-oracle transport audit

## Question

`LIP-PROTO-002` asks whether a single exact hidden state from the target model
can carry task-conditioned predictive information from a task-bearing prompt
into the neutral prompt used by `LIP-PROTO-001`. It does not load the source
model or a learned adapter. Failure therefore localizes the problem to the
single-state intervention channel rather than the bridge loss.

The audit is teacher-forced. For each held-out task, the target model first
produces one deterministic reference continuation under the complete task
prompt. The same continuation is then scored under:

1. the complete task prompt;
2. the fixed neutral prompt;
3. the neutral prompt after replacing its generation-boundary state with the
   exact task-prompt state from a configured target layer.

This avoids running a large stochastic generation grid merely to discover that
the intervention cannot affect the target distribution.

## Primary metric

Let `NLL_task`, `NLL_neutral`, and `NLL_injected` be the mean token negative
log-likelihoods of the deterministic reference continuation. Define:

```text
task_advantage = NLL_neutral - NLL_task
recovery = (NLL_neutral - NLL_injected) / task_advantage
```

A task is informative only when `task_advantage` meets the configured minimum.
Recovery near zero means injection behaves like the neutral prompt; recovery
near one means it recovers the task-prompt predictive advantage. Negative
recovery means the intervention makes the continuation less likely.

## Frozen first-stage design

The checked-in audit changes one factor only:

- target layers: `[-1, -2, -4, -8, -16, -24, -32]`;
- injection mode: exact-state `replace`;
- token site: the last non-padding chat-generation boundary;
- reference decoding: deterministic greedy decoding;
- tasks: 16 from the already materialized 32-task held-out bundle;
- layer selection: first 8 tasks;
- confirmation: next 8 tasks.

The source bridge, source vectors, adapter loss, gain, random controls, and
functional code execution are intentionally absent. Adding those factors before
the oracle channel works would confound diagnosis and multiply an invalid run.

The audit records a same-prompt self-replacement check for the first task. An
exact state replaced at its original site should leave continuation NLL
unchanged within tolerance. Failure of that check invalidates the hook/layer
alignment before any cross-prompt interpretation.

Oracle states are captured directly from the output of each configured
transformer block. They are not selected from the model-wide
`output_hidden_states` tuple, whose final entry may include a post-block final
normalization and therefore does not represent the `model.layers[-1]` hook
interface.

## Known scope boundary

The task and neutral prompts have different token lengths. The summary records
this position/context confound explicitly. The first-stage question is whether
the current `LIP-PROTO-001` channel works at any layer under its actual neutral
prompt. If it does not, the next protocol must distinguish positional mismatch
from insufficient channel capacity using a length-controlled carrier or a
multi-vector latent packet. This limitation is not silently absorbed into a
layer or gain search.

## Execution

Run the non-claim-eligible two-task plumbing check first:

```bash
python -m src.scripts.run_oracle_transport_audit \
  --config config/LIP-PROTO-002_oracle_transport.yaml \
  --preflight
```

Only after its self-reconstruction checks pass, run the frozen 16-task design:

```bash
python -m src.scripts.run_oracle_transport_audit \
  --config config/LIP-PROTO-002_oracle_transport.yaml
```

The full gate selects the best layer using only the selection split and applies
the configured recovery threshold to the disjoint confirmation split. A
preflight or diagnostic subset is always marked `claim_eligible=false` and
cannot pass the final gate.

## Decision

- If a layer confirms positive recovery, use that frozen layer for a small
  functional oracle generation before returning to the learned bridge.
- If no layer recovers the target-prompt advantage, do not tune the bridge
  loss. Advance to a separately versioned carrier experiment: length-controlled
  state injection, a short soft prefix, or a KV packet.
- Treat failure as a measured capacity/interface limit of the single-state
  protocol, not as evidence that latent agent communication is impossible.
