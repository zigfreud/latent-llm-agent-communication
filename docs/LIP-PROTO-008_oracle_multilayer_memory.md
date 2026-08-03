# LIP-PROTO-008 multi-layer oracle memory replay

## Purpose

`LIP-PROTO-007` showed that increasing a single residual suffix packet from
`K=8` to `K=32` does not produce a callable program, even when the packet is an
exact same-task target-model state. `LIP-PROTO-006` nevertheless found strong
teacher-forced recovery over the first eight continuation tokens. The remaining
mechanistic question is whether the task state must persist across transformer
depth rather than merely occupy more token positions at one layer.

This protocol changes one axis: **where in the decoder the exact target-oracle
state is replayed**. Packet size remains fixed at the previous ceiling `K=32`.
No source model or learned bridge is present.

## Why the hook boundary matters

For prompt position `p` and decoder block `l`, the block-input residual
`h[l,p]` determines that block's cached attention memory approximately as

```text
k[l,p] = W_K[l] norm(h[l,p])
v[l,p] = W_V[l] norm(h[l,p]).
```

The previous block-output replacement at layer `-16` can affect later blocks,
but it cannot retroactively change the prompt keys and values already computed
by blocks `0:-16`. Replaying the exact task-conditioned **block input** before a
set of layers forces each selected layer to rebuild its own prompt `K/V` memory
from task-specific states. The hook fires once during prompt prefill and becomes
a no-op during token-by-token decoding.

This is implemented through block-input hooks rather than version-specific
mutation of Hugging Face cache objects. The causal state written into the cache
is the same, while the intervention remains testable across compatible cache
implementations.

## Frozen scopes and controls

The target has 32 decoder blocks. Each task contributes the exact final 32
prompt states at the registered boundaries.

| Scope | Boundary | Replayed layers | Role |
|---|---|---:|---|
| `single_layer_output` | Output | `[-16]` | Exact `LIP-PROTO-007` anchor |
| `late_half_input` | Input | `[-16, ..., -1]` | Task memory in the final half |
| `all_layer_input` | Input | `[-32, ..., -1]` | Task memory throughout decoder depth |

Every scope has a same-task condition and a Sattolo-deranged equal-capacity
condition. The derangement preserves boundary, layer count, tensor shape, norm
scale, and generation seed while destroying task identity. Neutral and task-text
controls remain unchanged. The complete design has eight conditions.

The scalar payload grows with layer count, so this is not a compression claim.
It is an oracle localization experiment: first establish the minimum causal
state that can control the target, then optimize or learn a smaller carrier.
Before generation, exact same-task replay on one task must reproduce baseline
logits within maximum absolute delta `1e-4` at all three registered scopes.

## Registered state diagnostics

The same capture pass records aggregate-only maps over the final 32 prompt
positions and all 32 decoder layers for three state types:

- `residual_input`: the residual stream entering each decoder block;
- `key_pre_rope`: the output of that block's key projection before RoPE;
- `value_pre_cache`: the value projection immediately before cache shaping.

For every state type × layer × suffix-position cell, the run reports mean L2
norm, task-centered energy fraction, mean cross-task cosine, and entropy
effective rank (raw and normalized by its task-limited maximum). Raw hidden
states are not written to the diagnostic artifact.

These are localization diagnostics, not estimates of channel capacity in bits.
High task-centered energy or rank shows that tasks occupy distinguishable state
directions at a boundary; it does not establish that downstream generation uses
those directions. Only the matched-versus-task-shuffled functional comparison
tests task-specific causal control.

The two-task preflight validates capture and plotting only. Its centered task
matrix has rank at most one, so effective-rank maps become scientifically
interpretable only in the frozen 32-task confirmation run.

## Fresh task registry

The previous 32 validation tasks have all been consumed. This protocol samples
32 new tasks from the disjoint public MBPP `test` split with seed `808`, appends
only the benchmark entry-point name to each natural-language task, and freezes
IDs and prompt hashes in a lightweight task manifest. No source/target latent
bundle is extracted because this is a target-only oracle experiment.

Entry points are inferred from tests without exposing assertions to the model.
If a benchmark callable intentionally shadows a Python builtin (MBPP task 126
uses `sum`), the resolver intersects names called by every test with top-level
functions declared by the reference implementation. Reference code is used
only to resolve that metadata and is never persisted in the task registry or
transmitted to the model. This policy is frozen as
`tests_then_reference_code` in the sampling configuration and manifest.

The target revision is pinned to
`53346005fb0ef11d3b6a83b12c895cca40156b6c`, exactly the revision recorded in
the archived `LIP-PROTO-007` metadata. Model identity is therefore held fixed
across the single-layer and multi-layer comparisons.

- tasks `0:2`: sacrificial preflight, never claim-eligible;
- tasks `2:16`: reserved diagnostics;
- tasks `16:32`: untouched 16-task confirmation set;
- generation seeds: `[101, 202, 303]`.

The claim-oriented run therefore contains `16 × 8 × 3 = 384` generations. A
preflight can never be resumed into the confirmation run because task slices,
run scope, design fingerprint, and expected grids differ.

## Decision rule

For a scope to carry task-specific functional control, its task-clustered pass
mean must be strictly greater than both the neutral carrier and its own
task-shuffled control. The registered gate also requires nonzero task-text
capacity. The summary reports the smallest supported scope in the fixed order
single-layer output, late-half input, all-layer input.

- late-half or all-layer replay succeeds: the missing variable was persistent
  multi-layer memory; subsequent bridge work should target that minimum scope;
- matched and shuffled replay improve together: additional state changes model
  dynamics without transmitting task identity;
- every latent scope fails while text succeeds: even exact prompt-state replay
  is insufficient under this carrier, motivating direct cache-prefix or
  cross-attention interfaces;
- text fails: the sampled task/model budget is not an informative functional
  test, so no interface conclusion is claim-eligible.

Nine paired task-level contrasts retain exact sign-flip tests, task bootstrap
intervals, and Holm correction. The raw gate and adjusted inference are both
reported; neither silently substitutes for the other.

## Execution

Freeze the real task registry:

```bash
python -m src.scripts.materialize_oracle_tasks \
  --config config/LIP-PROTO-008_mbpp_test_sampling.yaml
```

Run the sacrificial generation preflight and syntax inspection:

```bash
python -m src.scripts.run_oracle_memory_functional \
  --config config/LIP-PROTO-008_oracle_multilayer_memory.yaml \
  --preflight

python -m src.scripts.evaluate_oracle_packet_semantics \
  --config config/LIP-PROTO-008_oracle_multilayer_memory.yaml \
  --generations runs/LIP-PROTO-008/preflight/generations.jsonl \
  --output-dir runs/LIP-PROTO-008/preflight/evaluation \
  --allow-incomplete --overwrite
```

Functional scoring must use the versioned hardened namespace evaluator. Run it
only after the generation grid and raw outputs pass inspection:

```bash
python -m src.scripts.run_hardened_oracle_evaluation \
  --config config/LIP-PROTO-008_oracle_multilayer_memory.yaml \
  --generations runs/LIP-PROTO-008/preflight/generations.jsonl \
  --output-dir runs/LIP-PROTO-008/preflight/functional-evaluation \
  --allow-incomplete --overwrite
```

Render the preregistered layer × suffix-position × state-type maps after the
diagnostic JSON has been inspected:

```bash
python -m src.scripts.plot_oracle_state_diagnostics \
  --diagnostics runs/LIP-PROTO-008/preflight/state-diagnostics.json \
  --output-stem runs/LIP-PROTO-008/preflight/LIP-PROTO-008_state_diagnostics
```

The full confirmation command remains intentionally unexecuted until the
preflight verifies capture/replay provenance and at least partial task-text
capacity.

## Result

Pending frozen preflight.
