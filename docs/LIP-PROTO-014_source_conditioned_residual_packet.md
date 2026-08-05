# LIP-PROTO-014 source-conditioned residual packet distillation

Status: conditional design draft; implementation and execution are blocked on
the sealed `LIP-PROTO-013` result.

## Research question

`LIP-PROTO-010` established that native target-model prompt states can replace
task text for bounded functional program synthesis. `LIP-PROTO-012` localized
that capacity to a terminally weighted 24-position packet, and
`LIP-PROTO-013` is testing which parts of that packet must preserve target-task
identity when structural capacity is held constant.

`LIP-PROTO-014` asks the first learned cross-model question after that oracle
sequence:

> Can a bridge that sees only source-model latent states predict the
> task-specific residual of a causally validated target-model packet well
> enough for the receiver to solve unseen tasks without receiving their text?

This is not a generic interoperability or text-equivalence claim. It is a
bounded test between one pinned source, one pinned receiver, one prompt
protocol, one packet carrier, and one public coding benchmark.

## Dependency and branch contract

The protocol is drafted before the `LIP-PROTO-013` confirmation result is read.
No `LIP-PROTO-014` checkpoint training, final task materialization, or
claim-eligible generation may begin until the predecessor artifact and source
commit are sealed.

The target packet is selected by this registered decision table:

| `LIP-PROTO-013` outcome | `LIP-PROTO-014` action |
| --- | --- |
| Full `K=32` replication gate fails | Stop. Do not train a bridge against an unreplicated carrier. |
| Full `K=32` passes, terminal `K=24` gate fails | Use the confirmed full `K=32`, first-eight-layer packet. |
| Both gates pass | Use the terminal `K=24`, first-eight-layer packet. |

Component-level `LIP-PROTO-013` results inform interpretation and later
ablations, but they do not remove positions from the primary learned packet.
This prevents a post-result component choice from becoming an unregistered
compression claim.

The development branch is stacked on the frozen `LIP-PROTO-013` implementation
only while that PR is open. Its PR must not be opened until the predecessor is
merged or the stack is otherwise rebased to a clean mainline.

## Why predict residuals instead of raw states

Let the native target packet for task (t), receiver layer \(\ell\), and prompt
position \(p\) be

\[
H_{t,\ell,p} = \mu_{\ell,p} + \Delta_{t,\ell,p}.
\]

The scaffold

\[
\mu_{\ell,p} = \frac{1}{N_{train}}
\sum_{t \in train} H_{t,\ell,p}
\]

contains target-side structure shared across training tasks. The centered
residual

\[
\Delta_{t,\ell,p} = H_{t,\ell,p} - \mu_{\ell,p}
\]

contains the task-dependent displacement that the bridge must predict.

The earlier single-vector bridge was optimized against raw target states. Raw
state similarity can be dominated by a large task-invariant component: a model
may reconstruct \(\mu\) well, obtain a favorable cosine, and still miss the
small direction that identifies the task. `LIP-PROTO-014` turns the oracle
mechanism result into an architectural inductive bias by predicting
\(\Delta\), then reconstructing

\[
\widehat H_{t,\ell,p}
= \mu_{\ell,p} + \widehat\Delta_{t,\ell,p}.
\]

The scaffold is computed from the training split only and is stored once at
the receiver. No evaluation-task target state contributes to it.

## Frozen communication surface

### Source

- Model family: `deepseek-ai/deepseek-coder-1.3b-base`.
- Revision: immutable Hugging Face commit, to be sealed in the config and
  bundle manifest before extraction.
- Prompt: raw MBPP task description plus only the benchmark-required callable
  name, under the existing role-specific prompt contract.
- State: final decoder-layer residual states at the final 32 non-padding source
  positions.
- Shape per task: `[32, 2048]` in the recorded source dtype.
- The source model is frozen and is not present during bridge training after
  packet extraction.

The 32-position source suffix is intentionally not claimed to align token by
token with the receiver suffix. It is a fixed source-side latent message from
which a query-conditioned bridge reads.

### Receiver teacher and carrier

- Model: `NousResearch/Meta-Llama-3-8B-Instruct`.
- Revision: the immutable receiver revision inherited from the sealed oracle
  protocols unless the dependency table stops the experiment.
- Carrier: the same neutral target prompt and exact chat-template rendering as
  the selected oracle predecessor.
- Injection boundary: decoder-block residual inputs.
- Replay depth: the first eight of 32 receiver decoder blocks.
- Positions: `K=24`, offsets `-24 ... -1`, if the terminal gate passes;
  otherwise the confirmed full `K=32`, offsets `-32 ... -1`.
- Target states are captured in a separate extraction pass; source and target
  base models need not be resident simultaneously.

## Bridge architecture

The bridge is a shared query-conditioned packet mapper, not 192 unrelated
linear heads and not one monolithic million-dimensional output layer.

1. Project the source packet from width 2048 to bridge width \(d_b\).
2. Add learned source-position embeddings.
3. Construct one learned query for every receiver `(layer, position)` site by
   combining receiver-layer and relative-position embeddings.
4. Cross-attend the receiver queries to the projected source packet.
5. Decode every query through one shared residual head to width 4096.
6. Add the frozen training-only scaffold \(\mu_{\ell,p}\).

In notation,

\[
Z_t = E_\theta(S_t),
\qquad
z_{t,\ell,p} = \operatorname{CrossAttn}_\theta(q_{\ell,p}, Z_t),
\]

\[
\widehat\Delta_{t,\ell,p} = D_\theta(z_{t,\ell,p}),
\qquad
\widehat H_{t,\ell,p} = \mu_{\ell,p} + \widehat\Delta_{t,\ell,p}.
\]

The initial implementation freezes bridge width, attention-head count,
dropout, optimizer, and step budget before the final holdout is generated.
Any finite hyperparameter comparison is restricted to the development split
and recorded in a selection report. Final task functional output must not be a
selection metric.

### Frozen initial implementation

- source projection: `Linear(2048, 512)` followed by `LayerNorm`;
- bridge width: `512`;
- source-position embeddings: 32 learned embeddings;
- receiver queries: the sum of 8 learned layer embeddings and either 24 or 32
  learned relative-position embeddings, as resolved by the predecessor gate;
- decoder: two pre-norm transformer-decoder blocks;
- attention heads: 8;
- feed-forward width: 2048 with GELU;
- dropout: 0.10 in training and disabled in evaluation;
- shared residual head: `Linear(512, 4096)`;
- optimizer: AdamW, learning rate `1e-4`, weight decay `1e-2`;
- task batch size: 4;
- maximum updates: 2048;
- validation interval: 64 updates;
- gradient clipping: global norm 1.0;
- bridge computation: FP32 parameters with FP16 autocast on T4;
- no gradient is propagated into either base model.

The proposed objective freezes `temperature=0.07`, `margin_target=0.05`,
`lambda_huber=1.0`, `lambda_cosine=0.25`, `lambda_symmetric_nce=1.0`,
`lambda_margin=0.10`, and `lambda_norm=0.05`.

Checkpoint selection is lexicographic on the development split: highest
task-level packet retrieval top-1, then highest paired diagonal margin, then
lowest centered normalized residual RMSE, then earliest update. This rule is
identical across replicas. No manual checkpoint choice is permitted.

## Training data and split topology

All splits are by task. Sites, positions, layers, and augmented views from the
same task may never cross a split boundary.

- Training: a deterministic 256-task sample from the MBPP train split.
- Development: a deterministic 64-task sample from the MBPP validation split.
- Final confirmation: 32 latent-unseen, text-capable tasks from the sealed
  `LIP-PROTO-013` candidate screen.

The confirmation selector takes eligible ranks `[16:32]` separately inside
the two-token and three-token tokenizer strata. It therefore selects 16 tasks
per stratum after the 16+16 predecessor selection. The selector must prove:

- membership in the sealed candidate manifest;
- functional text capability under the sealed screening rule;
- exact rank-slice equality within each stratum;
- disjointness from every `LIP-PROTO-013` confirmation task;
- absence of any earlier learned-bridge or oracle-latent generation for the
  selected task IDs.

The already observed screen counts (35 capable two-token tasks and 32 capable
three-token tasks) make this slice structurally feasible without consulting a
`LIP-PROTO-013` latent outcome.

## Packet bundle contract

The existing single-vector latent bundle format is insufficient. The new
content-addressed bundle records, per task:

- task ID and prompt SHA-256;
- source and target model IDs and immutable revisions;
- role-specific formatted-prompt hashes;
- source token IDs, attention-mask hash, layer, and selected offsets;
- target token IDs, attention-mask hash, selected offsets, replay layers, and
  capture boundary;
- source packet `[32, 2048]`;
- native target teacher packet `[L, K, 4096]`;
- split identity and task-level provenance.

Manifests record tensor dtype, shape, per-shard SHA-256, task IDs, prompt
protocols, dataset revision, source commit, and the sealed predecessor artifact
hash. Validation rejects mock extraction for claim-oriented runs.

## Training objectives

For numerical stability, each receiver site has a training-only scalar RMS

\[
\sigma_{\ell,p}
= \sqrt{\frac{1}{N_{train}d}
\sum_{t,j}\Delta_{t,\ell,p,j}^{2} + \epsilon}.
\]

Residual regression operates on
\(\widetilde\Delta_{t,\ell,p}=\Delta_{t,\ell,p}/\sigma_{\ell,p}\).
The proposed loss is

\[
\mathcal L =
\lambda_H \mathcal L_{Huber}
+ \lambda_C \mathcal L_{cos}
+ \lambda_N \mathcal L_{symNCE}
+ \lambda_M \mathcal L_{margin}
+ \lambda_R \mathcal L_{norm}.
\]

- `Huber` reconstructs the centered, site-normalized residual packet.
- `cos` preserves residual direction at each receiver site.
- `symNCE` makes predicted packets retrieve their paired target residual packet
  in both prediction-to-target and target-to-prediction directions.
- `margin` requires the paired packet to beat the hardest task-negative packet.
- `norm` calibrates residual packet energy without allowing energy alone to
  satisfy the contrastive objective.

Flattened packet similarities are computed only after site normalization.
Training batches are sampled by task; the 8 x K sites do not masquerade as
independent examples.

### Registered objective ablation

The paper-facing ablation trains the same architecture and data with:

1. `raw_state`: the earlier raw-state MSE plus forward InfoNCE principle,
   generalized to the packet;
2. `centered_regression`: training-only scaffold plus centered Huber and cosine;
3. `centered_contrastive`: the proposed full centered symmetric/margin-aware
   objective.

Architecture, parameter count, source packet, target carrier, optimizer budget,
and training seeds remain fixed. The proposed objective is primary; the other
two are registered ablations, not candidates selected after final generation.

The `raw_state` objective uses `temperature=0.10`, forward InfoNCE weight 1.0,
and raw-state MSE weight 0.35, matching the earlier bridge principle. The
`centered_regression` objective uses centered normalized Huber weight 1.0 and
sitewise cosine weight 0.25, with every contrastive, margin, and norm term set
to zero. These constants are not retuned on the final cohort.

## Independent replicas

Every objective is trained with three independent seeds. Checkpoint selection
uses development latent metrics only and applies the same deterministic rule to
all objectives. The final holdout is generated once per selected checkpoint.

Suggested fresh seed registry, to be copied into the executable config after
dependency resolution:

- data selection: `4013`;
- bridge replicas: `[4001, 4003, 4007]`;
- generation: `[4127, 4241, 4357]`;
- statistics/bootstrap: `4481`;
- shuffled-task derangement: `4513`.

No seed overlaps the screening, confirmation, donor, or statistics seeds of
`LIP-PROTO-010` through `LIP-PROTO-013`.

## Functional conditions

The receiver never receives task-specific text except in the explicit text
control.

| Condition | Target-visible task text | Injected packet | Role |
| --- | ---: | --- | --- |
| `neutral_no_lip` | No | None | No-message baseline |
| `text_only_no_lip` | Yes | None | Receiver capability ceiling |
| `oracle_teacher_matched` | No | Native matching teacher | Carrier replication gate |
| `oracle_teacher_shuffled` | No | Native same-stratum donor | Oracle identity control |
| `mean_scaffold` | No | Training-only \(\mu\) | Shared-structure control |
| `learned_matched` | No | \(\mu+\widehat\Delta(S_t)\) | Primary treatment |
| `learned_shuffled` | No | \(\mu+\widehat\Delta(S_{\pi(t)})\) | Learned identity control |
| `random_residual_norm_matched` | No | \(\mu+R_t\) | Residual-energy control |

`learned_*` and random controls are generated for each bridge replica. No-vector
baselines, the oracle controls, and the mean scaffold are generated once per
task/generation seed and are explicitly marked as replica-independent rather
than duplicated as new observations.

The random residual uses an isotropic direction with the primary treatment's
layer-wise Frobenius norm. Shuffled donors are assigned by a same-stratum
Sattolo derangement and retain their natural learned residuals. Both natural
and norm diagnostics are recorded.

The two ablation objectives receive matched and shuffled learned conditions
under the same tasks, seeds, carrier, and functional scorer. They are labeled
secondary and cannot replace the proposed objective in the primary claim.

## Functional scoring and statistical unit

Generated Python is scored only inside the hardened Linux namespace sandbox.
No normal subprocess scorer is claim-eligible.

The task is the inferential unit. Generation seeds and bridge replicas are
averaged within task before uncertainty or hypothesis testing. Bootstrap
intervals resample tasks. Paired one-sided sign-flip tests use the exact task
pairing; zero task differences do not create artificial resolution.

The analysis has an ordered oracle gate:

\[
H_{oracle}: Y_{oracle\ matched} > Y_{oracle\ shuffled}.
\]

If it fails, learned semantic claims are closed. If it passes, one
Holm-adjusted primary family opens:

\[
\begin{aligned}
H_{identity} &: Y_{learned\ matched} > Y_{learned\ shuffled},\\
H_{structure} &: Y_{learned\ matched} > Y_{mean\ scaffold},\\
H_{energy} &: Y_{learned\ matched} > Y_{random\ norm\ matched}.
\end{aligned}
\]

All three primary hypotheses must reject in the predicted direction to set
`learned_cross_model_transport_supported=true`. This makes the claim require
task correspondence, improvement beyond shared target structure, and
improvement beyond residual energy.

One registered secondary contrast compares the proposed objective with the
raw-state objective. It is labeled an objective ablation and is not required
for the transport claim. Per-replica directions, absolute functional rates,
the gap to native oracle replay, and the gap to text are always reported.

The protocol does not claim non-inferiority to text unless a separate future
design registers a non-inferiority margin and adequate power before data.

## Stop rules

Stop without claim-eligible final generation when any of the following occurs:

- predecessor packet gate fails under the dependency table;
- either confirmation tokenizer stratum has fewer than 16 remaining capable
  tasks;
- source or target bundle provenance does not bind to immutable model revisions
  and exact prompt hashes;
- native oracle replay fails on the development preflight;
- the hardened sandbox probe fails;
- fewer than two of three proposed-objective replicas satisfy the frozen
  development checkpoint rule;
- output completeness, task disjointness, or replica/seed cell checks fail.

Development failure is reported as a learnability limit of this bridge and
objective. It is not repaired by inspecting final-task behavior, changing the
packet after training, increasing model width, or trying unregistered seeds.

## Compute policy

- Use a standard T4 unless measured memory requires a documented change.
- Extract source and target bundles in separate model-loading passes.
- Store generated bundles, checkpoints, and run outputs only under ignored
  runtime paths and the canonical Drive artifact.
- Run bridge-only training from cached packets; do not keep either base model
  loaded during training.
- Run a two-task, one-replica, all-condition real-model preflight before the
  full confirmation budget.
- Do not use a premium GPU merely to reduce wall-clock time.

## Artifact contract

The canonical artifact must contain:

- frozen executable config and source commit;
- predecessor artifact hash and dependency decision;
- train/development/final task registries;
- source/target packet bundle manifests and validation reports;
- training-only scaffold and site-scale hashes;
- resolved configs, logs, metrics, and best checkpoints for every replica;
- development checkpoint-selection report;
- complete generations and metadata;
- hardened scored rows, sandbox report, and inferential summary;
- paper-facing functional, geometry, and ablation figures;
- top-level `SHA256SUMS` verified after rendering.

## Claim boundary

A positive result would establish that, in this pinned system, a learned bridge
can transform source-model hidden states into a target-side latent packet that
causally improves unseen functional behavior without exposing task text to the
receiver, beyond task-shuffled, shared-structure, and energy-matched controls.

It would not establish universal latent language, arbitrary agents, arbitrary
tasks, text non-inferiority, security against adversarial messages, or
interoperability across untested models. A negative result would localize the
failure to the tested source representation, query-conditioned mapper,
residual objective, data scale, and carrier; it would not erase the native
oracle channel established by the predecessor sequence.
