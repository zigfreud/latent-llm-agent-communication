# LIP-PROTO-014 source-conditioned residual packet

Status: frozen train-side contract and implementation; real packet extraction
and bridge training have not yet been executed. Confirmation tasks remain
sealed behind the registered development gate.

## Research question

`LIP-PROTO-013` established that task identity in the causally effective
receiver packet is concentrated in its terminal core and function-name
components. The full first-eight-layer carrier remained effective, while the
boundary component did not show an independently detectable positive
contribution.

`LIP-PROTO-014` asks the first learned heterogeneous-model question after that
oracle result:

> Can a bridge that sees only DeepSeek-Coder source residual states reconstruct
> enough of a causally validated Llama receiver packet to transport task
> identity when the receiver sees no task-specific text?

This is a bounded pairwise transport experiment. A positive result supports a
learned semantic bridge for the pinned source, receiver, carrier, prompt
protocols, and MBPP task family. It does not by itself establish a universal
model-independent latent language.

## Why the universal-protocol claim is deferred

The implementation is deliberately modular:

\[
S_A \xrightarrow{E_A} Z_{LIP} \xrightarrow{D_B} \widehat\Delta_B.
\]

Here, `E_A` is a sender encoder, `Z_LIP` is a fixed-shape 32 x 512 code, and
`D_B` is a receiver decoder. This is the architecture needed for future
endpoint composition.

With only one observed pair `A -> B`, however, the internal code is not
identifiable. For any invertible map `Q`, the pair

\[
E'_A = Q E_A, \qquad D'_B = D_B Q^{-1}
\]

produces exactly the same endpoint behavior. Nothing in one pair tells us which
coordinate system is the reusable protocol. A model-agnostic claim therefore
requires at least another independently trained endpoint and an unseen
composition such as `E_A -> D_C` or `E_C -> D_B`.

The 014 can prove that heterogeneous latent translation is learnable. A later
protocol gate must prove that its middle representation is reusable.

## Dependency and evidence boundary

The carrier is frozen from the completed `LIP-PROTO-013` result:

- receiver boundary: transformer block input;
- receiver layers: first eight decoder blocks, indices `[0, ..., 7]`;
- positions: terminal contiguous `K=24` suffix;
- hidden width: 4096;
- neutral carrier: `Use the latent signal.` rendered with the exact target chat
  protocol and left-padded, with padding masked, to the task-prompt length.

The executable config binds the predecessor `SHA256SUMS` digest. Any change to
the predecessor artifact, endpoint revisions, prompt protocols, dataset
revision, packet shapes, or task registry creates a different experiment.

## What the NVIDIA result changes

The design incorporates three lessons from NVIDIA's
[Cross-Model KV Cache Transfer in LLM Families](https://arxiv.org/abs/2608.03893):

1. A structured linear transfer is a required baseline, not an optional weak
   comparator.
2. Source information may be distributed across depth, so the sender captures
   all 24 DeepSeek-Coder block inputs instead of only its final layer.
3. Reconstruction error alone is insufficient. Checkpoint selection and final
   claims are functional and identity-sensitive because where error lands can
   matter more than its aggregate magnitude.

The paper's closed-form per-head ridge map cannot be copied literally. Its
endpoints have matched KV geometry and a very large token-level calibration
set. The 014 has heterogeneous residual widths and 256 task-level training
examples; a fully flattened ridge map would be severely underdetermined. The
registered linear comparator therefore shares a width projection and learns a
structured source-site mixture.

## Frozen communication surfaces

### Sender

- model: `deepseek-ai/deepseek-coder-1.3b-base`;
- immutable revision:
  `e5babb80b8539a4e85dd2418c0ee611522276987`;
- prompt: raw MBPP task with the required function name appended;
- state: residual input to every one of the 24 decoder blocks;
- positions: last 32 active prompt tokens;
- task tensor: `[24, 32, 2048]`, stored as `float16`.

The 24 x 32 sites are a fixed sender observation, not a claim that sender and
receiver tokens or layers are aligned.

### Receiver teacher

- model: `NousResearch/Meta-Llama-3-8B-Instruct`;
- immutable revision:
  `53346005fb0ef11d3b6a83b12c895cca40156b6c`;
- prompt: the same task under `lip-prompt-v1`, target chat template, generation
  marker, and the frozen Python-code system prompt;
- state: residual input to layers `[0, ..., 7]`;
- positions: last 24 active prompt tokens;
- task tensor: `[8, 24, 4096]`, stored as `float16`.

For every task, tokenizer offsets must prove that the required function-name
tokens immediately precede the same six-token chat boundary found in 013, and
that at least one terminal core position remains.

## Residual target

Let the native target teacher packet for task `t`, layer `l`, and position `p`
be `H[t,l,p]`. The receiver stores a task-independent scaffold computed only
from training tasks:

\[
\mu_{l,p} = \frac{1}{N_{train}}
\sum_{t \in train} H_{t,l,p}.
\]

The bridge predicts the task-specific residual

\[
\Delta_{t,l,p} = H_{t,l,p} - \mu_{l,p}.
\]

Each receiver site is normalized by one training-only scalar RMS:

\[
\sigma_{l,p} =
\sqrt{\frac{1}{N_{train}d}
\sum_{t,j}\Delta_{t,l,p,j}^{2} + \epsilon}.
\]

The learned target is
`Delta_tilde[t,l,p] = Delta[t,l,p] / sigma[l,p]`. At inference,

\[
\widehat H_{t,l,p} =
\mu_{l,p} + \sigma_{l,p}\widehat{\widetilde\Delta}_{t,l,p}.
\]

This decomposition matters because raw residual states contain a large shared
receiver scaffold. A predictor can achieve low raw MSE by reproducing that
shared structure while missing the smaller task-specific direction that is
causally useful.

## Query-conditioned bridge

The primary bridge has two explicit modules.

### Sender encoder `E_A`

1. Project each 2048-wide sender state to width 512.
2. Add learned layer and position embeddings.
3. Flatten the 24 x 32 source sites into 768 memory sites.
4. Use 32 learned protocol queries in two pre-normalized cross-attention blocks.
5. Emit a fixed `[32, 512]` LIP code.

### Receiver decoder `D_B`

1. Form 8 x 24 receiver queries from learned layer and position embeddings.
2. Cross-attend them to the LIP code in two pre-normalized blocks.
3. Project each decoded query to width 4096.
4. Emit the normalized receiver residual `[8, 24, 4096]`.

The frozen width is 512, with 8 attention heads, feed-forward width 2048,
dropout 0.10 during training, and dropout disabled during evaluation.

## Structured linear baseline

For target site `s`, source site `m`, and source vector `x[m]`, the baseline is

\[
\widehat y_s = a_s \odot W
\left(\sum_m M_{s,m}x_m\right) + b_s.
\]

`M` is a content-independent site-mixing matrix, `W` is one shared
2048-to-4096 linear projection, and `a_s,b_s` are target-site scale and bias.
The map is affine in the sender packet. It is expressive enough to learn
cross-layer source selection without pretending that 256 tasks identify an
unconstrained flattened map with billions of coefficients.

## Component-aware objective

The 24 receiver positions are partitioned for every task into:

- `core`: task text before the terminal function name;
- `name`: the tokenizer span of the required function name;
- `boundary`: the fixed final six chat-template tokens.

Site losses are averaged inside each component before components are combined
with weights `core=0.45`, `name=0.45`, and `boundary=0.10`. Thus six easy
boundary positions cannot numerically overwhelm two or three causally critical
name positions.

The primary loss is

\[
\mathcal L =
1.00\mathcal L_{Huber}
+0.25\mathcal L_{cos}
+1.00\mathcal L_{symNCE}
+0.10\mathcal L_{margin}
+0.05\mathcal L_{norm}.
\]

Contrastive similarity is calculated separately over the joint packet, core,
and name regions, then averaged equally. For a batch similarity matrix `C`,
the symmetric InfoNCE term is

\[
\mathcal L_{symNCE} = \tfrac12
\left[CE(C/\tau, I) + CE(C^T/\tau, I)\right],
\qquad \tau=0.07.
\]

The hardest-negative margin requires each matched packet to exceed its closest
task-negative by 0.05 in both retrieval directions. Batches are sampled by
task; packet sites never masquerade as independent observations.

## Data topology

All splits are by task.

- `train`: 256 deterministic tasks from MBPP `train`;
- `development_selection`: first 32 selected tasks from MBPP `validation`;
- `development_gate`: next 32 selected tasks from MBPP `validation`;
- `confirmation`: 32 sealed capable MBPP `test` tasks opened only after the
  multi-replica development gate.

Within each source dataset split, tasks are ordered by SHA-256 using the frozen
selection salt. Train/development tasks are materialized before packet
extraction. The training bundle must declare exactly zero confirmation records.

The confirmation selector takes eligible ranks `[16:32]` inside each of the
two-token and three-token terminal-layout strata from the sealed 013 capability
screen. It selects 16 tasks per stratum and proves that ranks `[0:16]` are
exactly the predecessor cohort and that the new tasks overlap neither it nor
any bridge train/development task.

## Content-addressed bundle

Every task record includes:

- task, raw-prompt, role-formatted-prompt, input-ID, and attention-mask hashes;
- the actual source/target input IDs and masks;
- source and target token counts;
- terminal function-name token count;
- source packet `[24,32,2048]`;
- receiver teacher packet `[8,24,4096]`;
- split and task identity.

The manifest binds immutable model and dataset revisions, prompt protocols,
packet contracts, registry digest, config digest, predecessor digest, task
order, shard hashes, dtypes, and split counts. Shards are loaded only with
`torch.load(..., weights_only=True)`. Claim-oriented training rejects dry-run
bundles.

Source and target extraction are sequential. Neither base model is present
during bridge-only training.

## Registered systems and replicas

Three systems are trained with seeds `[4001, 4003, 4007]`:

1. `component_contrastive`: query-conditioned bridge and the full primary loss;
2. `centered_regression`: the same nonlinear bridge with Huber and cosine only;
3. `structured_linear_regression`: the affine structured baseline with the same
   centered Huber and cosine regression objective.

The first comparison isolates the effect of identity-aware loss terms while
holding architecture fixed. The second compares linear and nonlinear capacity
under the same regression target and loss.

Training uses AdamW, learning rate `2e-4`, weight decay `0.01`, batch size 4,
gradient clipping at 1.0, mixed precision on CUDA, 2048 updates, and validation
every 64 updates. Base models remain frozen and unloaded.

## Checkpoint selection and development gate

Checkpoint selection consults only `development_selection`. The lexicographic
key maximizes, in order:

1. weakest retrieval top-1 across joint/core/name;
2. mean retrieval top-1;
3. weakest diagonal margin;
4. mean diagonal margin;
5. negative normalized RMSE;
6. earlier step.

After selection, the chosen checkpoint is evaluated exactly once on
`development_gate`. Task-level matched-minus-hardest-negative margins for
joint, core, and name undergo one-sided sign-flip tests in one Holm family.
One replica passes only if all three adjusted tests reject with positive mean
margins. Confirmation opens only if at least two of the three primary replicas
pass.

No failed gate may be repaired by inspecting confirmation tasks, adding seeds,
changing packet geometry, widening the bridge, or selecting a different
checkpoint rule.

## Functional confirmation conditions

The receiver sees task text only in the explicit text control.

| Condition | Task text at receiver | Packet |
| --- | ---: | --- |
| `neutral_no_lip` | no | none |
| `text_only_no_lip` | yes | none |
| `oracle_teacher_matched` | no | native matching teacher |
| `oracle_teacher_shuffled` | no | same-stratum teacher donor |
| `mean_scaffold` | no | training-only `mu` |
| `learned_matched` | no | `mu + predicted residual` |
| `learned_shuffled` | no | prediction from a same-stratum source donor |
| `random_residual_norm_matched` | no | isotropic residual with matched layer norms |

The task is the inferential unit. Generation seeds and bridge replicas are
averaged within task before hypothesis tests or task-bootstrap intervals.
Generated Python is claim-eligible only when evaluated in the hardened,
network-isolated Linux namespace used by the preceding protocols.

The oracle identity gate is tested first. If it passes, one Holm-adjusted
primary family tests learned matched against learned shuffled, mean scaffold,
and random norm. All three must reject in the predicted direction to support
learned cross-model semantic transport.

## Functional recovery measures

Absolute functional rates remain primary. Two unbounded descriptive ratios
make the remaining gaps interpretable:

\[
R_{identity} =
\frac{P_{learned\ matched}-P_{learned\ shuffled}}
{P_{oracle\ matched}-P_{oracle\ shuffled}},
\]

\[
R_{text} =
\frac{P_{learned\ matched}-P_{neutral}}
{P_{text}-P_{neutral}}.
\]

Ratios are not clipped. A zero denominator is reported as undefined rather
than silently replaced. These quantities describe recovered functional effect;
they do not create a non-inferiority claim.

## Stop rules

Stop before claim-eligible confirmation if any of the following occurs:

- immutable endpoint, prompt, dataset, registry, or predecessor provenance
  fails validation;
- target terminal-layout evidence fails;
- native target self-replay exceeds the frozen logit threshold;
- fewer than two primary replicas pass the development gate;
- the sealed confirmation rank slice is unavailable or overlaps prior tasks;
- generation cells are incomplete or duplicated;
- the hardened functional-evaluation probe fails.

Negative outcomes are results. They localize the limit to the tested sender
representation, bridge family, receiver carrier, data budget, and task family.

## Compute and artifact policy

- Prefer a standard T4.
- Do not use a premium GPU only to reduce wall-clock time.
- Keep packet bundles, checkpoints, and run outputs on the mounted Drive or RAID,
  not on full local disks.
- Run a two-task, one-replica real-model preflight before the full extraction
  and confirmation budgets.
- Preserve configs, registries, manifests, validation reports, target
  statistics, checkpoints, selection histories, generation grids, hardened
  evaluation output, plots, environment evidence, and `SHA256SUMS`.

### Executable preflight sequence

From the repository root, set `LIP_RUNTIME_ROOT` to an ignored Drive/RAID
directory and run:

```bash
python -m src.scripts.materialize_packet_bridge_tasks \
  --config config/LIP-PROTO-014_source_conditioned_residual_packet.yaml

python -m src.scripts.extract_packet_bridge_bundle \
  --config config/LIP-PROTO-014_source_conditioned_residual_packet.yaml \
  --bundle-dir "$LIP_RUNTIME_ROOT/preflight-bundle" \
  --preflight-tasks-per-split 2

python -m src.scripts.run_packet_bridge_matrix \
  --config config/LIP-PROTO-014_source_conditioned_residual_packet.yaml \
  --bundle-dir "$LIP_RUNTIME_ROOT/preflight-bundle" \
  --output-dir "$LIP_RUNTIME_ROOT/preflight-training" \
  --variants component_contrastive \
  --seeds 4001 \
  --max-updates 2 \
  --allow-nonclaim-bundle
```

The preflight bundle is marked `extraction_scope=preflight`; claim-oriented
training rejects it even though its tensors came from the real endpoints. Only
after this path passes should the same extraction command be run without the
preflight limit, followed by the complete registered matrix.

## Claim boundary

A positive 014 permits the statement that, in this bounded environment, a
learned bridge transformed source-model latent states into a receiver-side
latent packet carrying causally useful task identity without transmitting the
task text to the receiver.

It does not yet permit the statement that the 32 x 512 internal code is a
universal language, that arbitrary models can communicate through it, or that
text has been replaced generally. Those require unseen endpoint composition,
broader task families, and an independently registered non-inferiority design.
