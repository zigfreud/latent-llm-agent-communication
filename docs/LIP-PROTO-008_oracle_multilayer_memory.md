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
interpretable only in the frozen 16-task confirmation run.

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

After the preflight verifies capture/replay provenance and at least partial
task-text capacity, run the frozen confirmation and claim-oriented scoring:

```bash
python -m src.scripts.run_oracle_memory_functional \
  --config config/LIP-PROTO-008_oracle_multilayer_memory.yaml

python -m src.scripts.evaluate_oracle_packet_semantics \
  --config config/LIP-PROTO-008_oracle_multilayer_memory.yaml \
  --generations runs/LIP-PROTO-008/generations.jsonl \
  --output-dir runs/LIP-PROTO-008/evaluation \
  --overwrite

python -m src.scripts.run_hardened_oracle_evaluation \
  --config config/LIP-PROTO-008_oracle_multilayer_memory.yaml \
  --generations runs/LIP-PROTO-008/generations.jsonl \
  --output-dir runs/LIP-PROTO-008/functional-evaluation \
  --overwrite

python -m src.scripts.plot_oracle_state_diagnostics \
  --diagnostics runs/LIP-PROTO-008/state-diagnostics.json \
  --output-stem runs/LIP-PROTO-008/LIP-PROTO-008_state_diagnostics
```

## Result

The preflight completed all 16 expected records and passed its internal
authorization gate: text, single-layer output replay, and all-layer input
replay each achieved a task-clustered functional mean of `0.5`, while their
neutral or shuffled controls remained at zero. It was not claim-eligible and
was used only to authorize the already frozen confirmation.

The 16-task confirmation completed all `384/384` registered generations. The
hardened namespace evaluator validated the complete grid and marked the run
claim-eligible. The preregistered semantic gate passed, with
`all_layer_input` as the only supported scope.

| Condition | Functional passes | Task-clustered mean | 95% task-bootstrap CI | Entry point declared |
|---|---:|---:|---:|---:|
| Neutral carrier | 0/48 | 0.00% | [0.00%, 0.00%] | 0/48 |
| Matched single-layer output | 0/48 | 0.00% | [0.00%, 0.00%] | 5/48 |
| Shuffled single-layer output | 0/48 | 0.00% | [0.00%, 0.00%] | 0/48 |
| Matched late-half input | 0/48 | 0.00% | [0.00%, 0.00%] | 3/48 |
| Shuffled late-half input | 0/48 | 0.00% | [0.00%, 0.00%] | 0/48 |
| Matched all-layer input | 22/48 | 45.83% | [22.92%, 68.75%] | 48/48 |
| Shuffled all-layer input | 0/48 | 0.00% | [0.00%, 0.00%] | 0/48 |
| Task text | 21/48 | 43.75% | [18.75%, 68.75%] | 48/48 |

Matched all-layer replay exceeded both neutral and its task-shuffled control
by `0.4583`. The paired task-bootstrap intervals were `[0.2083, 0.6875]`
against neutral and `[0.2292, 0.6875]` against shuffled. Both exact two-sided
sign-flip tests gave `p=0.0078125`; Holm adjustment across all nine registered
contrasts gave `p=0.0703125`. The raw preregistered gate therefore passes, but
no all-layer contrast crosses a family-wise `alpha=0.05` threshold after the
conservative multiplicity adjustment. Both facts are part of the result.

The effect is stable at the task level rather than dispersed across lucky
samples. Seven tasks passed all three seeds under both task text and matched
all-layer replay; the same nine tasks failed all text seeds. All-layer replay
added one isolated success for task `409` at seed `202`, producing 22 rather
than 21 total passes. Every all-layer generation declared the required entry
point, whereas every equal-capacity shuffled generation failed to do so.

The 16-task diagnostic maps localize substantial task variation before cache
construction. Averaged over all layers and suffix positions, task-centered
energy was `0.5697` for `value_pre_cache`, `0.4604` for `residual_input`, and
`0.1647` for `key_pre_rope`; normalized task effective rank was respectively
`0.6392`, `0.6553`, and `0.5524`. Value-state signal peaked over suffix
positions `-23`, `-22`, `-17`, and `-16` at `0.755--0.765`, with normalized
effective-rank fractions `0.816--0.877`. These maps are descriptive
localization evidence; the matched-versus-shuffled functional result supplies
the causal evidence.

This confirms the narrow mechanistic hypothesis left open by
`LIP-PROTO-007`: exact target-model prompt state can replace task text for
functional control, but only when replayed persistently at the input of every
decoder layer under this carrier. A one-layer packet or replay limited to the
late half is insufficient. The result establishes feasibility of a latent
control channel in this target-only oracle setting; it does not yet establish
source-to-target bridge learnability, compression, or arbitrary agent
communication. The next bridge should target the all-layer input memory
interface and treat layer/position reduction as a new registered ablation.

The canonical artifact is stored at `lip-artifacts/LIP-PROTO-008` on Drive.
It contains the task registry, preflight and full outputs, hardened sandbox
reports, three vector/raster diagnostic figures, frozen configs, source commit,
and a verified manifest covering every payload without duplicating run data.
Its provenance binds Git commit
`1e3cbc7dcf0ee0ab5694a5a8b382a43ac4a15143`.

- `SHA256SUMS`: `1499340005988082c7a535339e44cde8817e07848c355b5afa03c058d394cf53`
- `runs/generations.jsonl`: `c95bf45befed4608f03d9141c5d444f3e21a35ddc2be14c3c61feeb559856c5b`
- `runs/functional-evaluation/summary.json`: `7ec537a49ca2fcde4450c28b54e03aa2f96d69b93c26a45919a9fe9fad302349`
- `runs/state-diagnostics.json`: `f9c52595f9a40d2ead048989d0acfe2ee204eec7ac534991ee0d086b37635ed0`
- `runs/LIP-PROTO-008_state_diagnostics.svg`: `8a9a6d6783f8608ea7123c89fc7496e0343413d3efe2299b87b1c4335ac80c0d`
