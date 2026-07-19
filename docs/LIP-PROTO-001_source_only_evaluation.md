# LIP-PROTO-001 source-only semantic evaluation

## Purpose

This protocol tests a narrower question than latent reconstruction: can a task
that is visible to the source model affect target-model code behavior when the
target never receives that task as text?

`LIP-PROTO-001` is an experiment harness, not a semantic-transfer result. No
paper or README claim should change until the experiment has been run and its
controls support a new claim.

## Protocol invariant

Bundle construction, source extraction, target-control extraction, and
generation use `lip-prompt-v1`. The checked-in configuration uses raw prompts,
`last_non_padding`, source layer `-1`, and target layer `-2`. The target vector
layer is aligned with the output of target transformer layer `-2`, where the
generation hook intervenes. Both bundle extraction and probing use the same
recorded bitsandbytes 4-bit loading policy for source and target models.
Real bundle construction records the resolved immutable Hugging Face commit for
each model. The probe reloads those exact revisions and rejects manifests that
contain only a mutable branch name or no revision.

The materializer also writes the exact held-out task specifications to
`datasets/LIP-PROTO-001/mbpp_validation_tasks.jsonl`. Bundle manifests record
the sampled public task IDs and SHA-256 digests of their prompts. The probe
loads source and oracle vectors from the validated held-out bundle by task ID,
checks those prompt digests, and records both training- and held-out-manifest
digests in every generation row. It therefore does not perform a second,
untracked source-vector extraction during generation.

Changing prompt mode, layers, tokenizer special-token behavior, or token
position requires rebuilding both train and held-out bundles and retraining all
adapters. Earlier checkpoints and bundles are not interchangeable with that new
protocol even when tensor dimensions match.

The primary `source_latent` condition follows this path:

1. The source model receives the task prompt.
2. The adapter maps the recorded source vector to target space.
3. The target receives only the fixed neutral prompt from the config.
4. The mapped vector replaces the neutral prompt's hidden state once at the
   configured target layer and token.

Replacement is used because the adapter is trained to predict the target hidden
state itself, not an additive residual. The protocol does not recalibrate that
vector against input-embedding energy: `vector_scaling=none` and `gain=1.0`
preserve the learned target-space norm. The random control is norm-matched to
the corresponding translated vector.

Each result records `target_prompt_kind`, SHA-256 digests of both target-visible
user text and fully formatted target input, vector provenance, training seed,
and generation seed. The full task
specification is attached only after generation so scoring remains auditable.

## Conditions

| Condition | Target-visible task text | Injected vector | Role |
|---|---:|---|---|
| `neutral_no_lip` | No | None | Neutral baseline |
| `text_only_no_lip` | Yes | None | Oracle textual baseline |
| `source_latent` | No | Matching translated source | Primary treatment |
| `shuffled_source_latent` | No | Another task's translated source | Semantic correspondence control |
| `random_norm_matched` | No | Random vector with matched norm | Energy-only control |
| `oracle_target_latent` | No during generation | Target hidden state extracted from task text | Injection-channel upper-bound diagnostic |

The shuffled mapping is a deterministic derangement, so it has no accidental
fixed points. Sampling is reset to the same task-level seed across conditions,
which provides common random numbers for paired comparisons.

## Rebuild and train independent replicas

Install the portable protocol environment. This requirement set works with
CPU PyTorch in a local/current runtime and with the CUDA-enabled PyTorch already
present in Colab. DirectML remains a Windows-only optional dependency in the
general requirements and is not installed into Linux or Colab environments.

```bash
python -m pip install -r requirements-protocol.txt
```

Only when attempting 4-bit model loading on CPU, install the separate CPU
kernel extra. Do not install this extra in CUDA/Colab runtimes:

```bash
python -m pip install -r requirements-protocol-cpu.txt
```

The intended execution split is:

| Stage | Current CPU runtime | Colab/CUDA |
|---|---:|---:|
| Mock materialization, tests, validation | Yes | Yes |
| Real DeepSeek/Llama bundle extraction | Only if models/hardware are available | Preferred |
| Three-seed adapter training from downloaded shards | Yes | Yes |
| Llama-3 controlled generation | Impractically slow here | Preferred |
| Syntax/statistical scoring | Yes | Yes |

All scripts use `device=auto`; CUDA is selected when available and CPU remains
a supported path for the lightweight stages.

Materialize the MBPP train/eval prompt selections and build new bundles under
the current protocol:

```bash
python -m src.scripts.materialize_mbpp_prompt_configs \
  --config config/LIP-PROTO-001_mbpp_sampling.yaml

python -m src.scripts.build_real_tiny_latent_bundle \
  --config datasets/LIP-PROTO-001/generated_configs/LIP-PROTO-001_train_mbpp_64.yaml \
  --device auto

python -m src.scripts.build_real_tiny_latent_bundle \
  --config datasets/LIP-PROTO-001/generated_configs/LIP-PROTO-001_eval_mbpp_32.yaml \
  --device auto
```

Train three replicas whose only intended difference is the training seed:

```bash
python -m src.scripts.run_multiseed_training \
  --config config/LIP-PROTO-001_multiseed_training.yaml \
  --seeds 41 42 43 \
  --output-root runs/LIP-PROTO-001/training \
  --device auto
```

The source-only config points at the three resulting `best_model.pth` files.
Generated bundles, model weights, and run outputs remain untracked.
The multi-seed runner refuses non-empty seed directories so an earlier replica
cannot be silently resumed or overwritten under the same identity.

Before loading the target model, the probe validates the configured training
and held-out bundles and checks their models, immutable revisions, layers,
token position, prompt protocol, dimensions, sampled task IDs, prompt digests,
and shard hashes. Dry-run/mock bundles are explicitly rejected. It also requires each checkpoint's `metrics.json` and
`resolved_config.yaml`, verifies the training seed and dataset path, and records
checkpoint/manifest SHA-256 digests in run metadata.

## Generate all controls

```bash
python -m src.scripts.run_source_only_probe \
  --config config/LIP-PROTO-001_source_only_eval.yaml
```

The runner refuses to replace an existing JSONL. Use `--resume` after an
interruption; use `--overwrite` only when the existing run is intentionally
being discarded.

If `data.tasks_jsonl` exists, it is used. Otherwise the script deterministically
samples the configured public dataset split with the same count, seed, and
prompt-length filter as the bundle materializer.

The default design has 32 held-out tasks, three independent adapter seeds,
three stochastic generation seeds, and six conditions: 1,728 result records.
No-vector baselines are deterministic for a fixed task/generation seed and do
not depend on an adapter, so their text is generated once and copied across
adapter-seed records with an explicit reuse flag.

## Score and summarize

Syntax-only scoring is safe to run as a normal local analysis:

```bash
python -m src.scripts.evaluate_source_only_semantics \
  --config config/LIP-PROTO-001_source_only_eval.yaml
```

Functional scoring executes generated Python. Its resource-limited subprocess
is not a security sandbox. Run it only in a disposable, network-isolated
environment with no credentials or sensitive mounts:

```bash
python -m src.scripts.evaluate_source_only_semantics \
  --config config/LIP-PROTO-001_source_only_eval.yaml \
  --functional \
  --allow-unsafe-execution
```

The summary first averages generation and training replicas within task, then
bootstraps tasks. Paired source/control differences use shared held-out tasks
and a two-sided task-level sign-flip test; configured comparison p-values also
receive a Holm multiplicity correction. Per-seed consistency must be
inspected; three training seeds are too few to characterize a broad population
of training runs.

## Decision rule

A positive semantic-transport claim is not supported unless all of the
following hold on functional pass rate:

- `source_latent` improves over `neutral_no_lip`;
- `source_latent` improves over `shuffled_source_latent`;
- `source_latent` improves over `random_norm_matched`;
- the direction is consistent across independent adapter seeds;
- paired uncertainty intervals and raw per-task outputs do not indicate that a
  small number of tasks or malformed generations explain the effect;
- the text-only and oracle-target controls confirm that the target and
  injection channel can solve at least part of the selected task set.

Failure of any check is still a useful result. It should be recorded as a limit
of the current adapter/protocol rather than converted into a stronger claim.
