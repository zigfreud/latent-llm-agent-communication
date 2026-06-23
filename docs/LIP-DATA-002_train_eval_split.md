# LIP-DATA-002 Train/Eval Latent Bundle Split

## Purpose

LIP-DATA-002 adds separate train and held-out evaluation latent bundle configs
for the next LIP experiment. Both configs use the existing
`src.scripts.build_real_tiny_latent_bundle` builder and keep generated bundles
outside git.

## Why A Split Is Needed

LIP-DATA-001 enabled the first 8-sample real bundle experiments, but those
evaluations were in-sample and tiny-scale: the same bundle was used for
training and evaluation. LIP-DATA-002 separates a 16-prompt train set from an
8-prompt held-out eval set so the next bridge run can start measuring
out-of-sample latent-space behavior.

## Prompt Policy

The train and eval prompts are small, non-sensitive, self-contained programming
tasks. They avoid private data, credentials, personal information,
project-specific secrets, and generated text capture. The builder does not
write raw prompts into shard records and does not store model completions.

The train and eval prompt sets are intentionally disjoint.

## Dry-Run

Dry-run mode does not download or load Hugging Face models. It writes
deterministic mock vectors with the configured dimensions, packages each bundle,
and validates it.

Train bundle dry-run:

```bash
python -m src.scripts.build_real_tiny_latent_bundle --config config/LIP-DATA-002_train_bundle_16.yaml --dry-run
```

Eval bundle dry-run:

```bash
python -m src.scripts.build_real_tiny_latent_bundle --config config/LIP-DATA-002_eval_bundle_8.yaml --dry-run
```

Validate both dry-run bundles:

```bash
python -m src.scripts.validate_latent_bundle --bundle-dir datasets/LIP-DATA-002/train_bundle_16
python -m src.scripts.validate_latent_bundle --bundle-dir datasets/LIP-DATA-002/eval_bundle_8
```

## Real Extraction In Colab

Run real extraction only in a local or Colab environment where downloading and
loading the configured models is intentional. Do not run real extraction in
GitHub Actions.

```bash
python -m src.scripts.build_real_tiny_latent_bundle --config config/LIP-DATA-002_train_bundle_16.yaml --device cuda --max-samples 16
python -m src.scripts.build_real_tiny_latent_bundle --config config/LIP-DATA-002_eval_bundle_8.yaml --device cuda --max-samples 8
```

The configs use `batch_size: 1`, `sequential_model_loading: true`,
`device_map: "auto"`, and `load_in_4bit: true` to make extraction more practical
on constrained GPU environments.

## Training With LIP-TRAIN-001

Upload the train bundle zip to trusted user-managed storage and trigger
`LIP-TRAIN-001 Configurable Remote Bundle Training` with the train bundle file
ID or URL plus its SHA256 digest.

Suggested first held-out experiment inputs:

```text
experiment_id: LIP-TRAIN-004-data002-train16-steps32-lambda025
google_drive_file_id: <LIP-DATA-002 train bundle file ID>
latent_bundle_sha256: <LIP-DATA-002 train bundle zip sha256>
max_steps: 32
epochs: 8
batch_size: 2
learning_rate: 0.0001
lambda_mse: 0.25
device: cpu
```

## Held-Out Evaluation

After downloading the training artifact, evaluate the checkpoint against the
held-out eval bundle:

```bash
python -m src.scripts.evaluate_bridge \
  --config config/LIP-EVAL-001_bridge_eval.yaml \
  --checkpoint runs/LIP-TRAIN-004-data002-train16-steps32-lambda025/best_model.pth \
  --bundle-dir datasets/LIP-DATA-002/eval_bundle_8 \
  --output-dir runs/LIP-EVAL-DATA002-heldout
```

Use the eval bundle manifest and shard digest in the resulting registry entry
so the held-out evaluation remains traceable.

## Expected Generated Outputs

Train bundle:

```text
datasets/LIP-DATA-002/train_bundle_16/manifest.json
datasets/LIP-DATA-002/train_bundle_16/shards/shard_0.pt
datasets/LIP-DATA-002/LIP-DATA-002_train_latent_bundle_16.zip
```

Eval bundle:

```text
datasets/LIP-DATA-002/eval_bundle_8/manifest.json
datasets/LIP-DATA-002/eval_bundle_8/shards/shard_0.pt
datasets/LIP-DATA-002/LIP-DATA-002_eval_latent_bundle_8.zip
```

## Data Policy

No generated latent bundle, shard, checkpoint, model weight, dataset, cache,
zip, or run artifact should be committed. Generated outputs belong under
ignored paths such as `datasets/` or another local scratch directory.

## Scientific Claim Boundary

This PR only adds train/eval latent bundle configs. It does not claim semantic
transfer, text-level fidelity, model-to-model alignment, generalization, or
production readiness.
