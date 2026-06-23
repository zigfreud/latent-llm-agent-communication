# LIP-DATA-003 MBPP Public Prompt Sampling

## Purpose

LIP-DATA-003 adds a reproducible prompt sampling path for the public MBPP
dataset. It materializes train and held-out eval configs that are compatible
with `src.scripts.build_real_tiny_latent_bundle`, while keeping generated prompt
configs and bundles outside git under ignored `datasets/` paths.

## Why MBPP Is Next

LIP-DATA-002 showed that the bridge can learn a small curated train set while
held-out pair-level generalization remains weak. MBPP is a small public
code-generation dataset with train, validation, and test splits, which makes it
a practical next step for controlled prompt diversity without moving directly
to larger benchmark suites.

Synthetic data remains useful for smoke tests and CI because it is deterministic
and does not require model or dataset downloads. It is not a substitute for
public-dataset prompt coverage. APPS is deferred because it is larger and more
operationally expensive; MBPP is the smaller controlled step.

## Dataset Policy

The sampler uses `google-research-datasets/mbpp` with config `full`. It samples
only the natural-language `text` field from the train and validation splits. It
does not copy MBPP code, tests, completions, or model outputs into the generated
bundle configs.

Generated configs include sampled prompt IDs for traceability. They are written
under `datasets/LIP-DATA-003/generated_configs`, which is ignored by git.

## Materialize Prompt Configs

Install the Hugging Face `datasets` package in the local or Colab environment
where public dataset access is intended, then run:

```bash
python -m src.scripts.materialize_mbpp_prompt_configs --config config/LIP-DATA-003_mbpp_sampling.yaml
```

This writes:

```text
datasets/LIP-DATA-003/generated_configs/LIP-DATA-003_train_mbpp_32.yaml
datasets/LIP-DATA-003/generated_configs/LIP-DATA-003_eval_mbpp_16.yaml
```

For offline repo validation without loading Hugging Face datasets, use mock
prompt rows:

```bash
python -m src.scripts.materialize_mbpp_prompt_configs --config config/LIP-DATA-003_mbpp_sampling.yaml --mock-data
```

## Dry-Run Bundles

Dry-run mode does not download or load DeepSeek, Llama, TinyLlama, or any large
model. It produces deterministic mock vectors and validates bundle structure.

```bash
python -m src.scripts.build_real_tiny_latent_bundle --config datasets/LIP-DATA-003/generated_configs/LIP-DATA-003_train_mbpp_32.yaml --dry-run
python -m src.scripts.build_real_tiny_latent_bundle --config datasets/LIP-DATA-003/generated_configs/LIP-DATA-003_eval_mbpp_16.yaml --dry-run
```

Validate the generated dry-run bundles:

```bash
python -m src.scripts.validate_latent_bundle --bundle-dir datasets/LIP-DATA-003/mbpp_train_bundle_32
python -m src.scripts.validate_latent_bundle --bundle-dir datasets/LIP-DATA-003/mbpp_eval_bundle_16
```

## Real Extraction In Colab

Run real extraction only in Colab or a local environment where model downloads
and access are intentional. Authenticate with Hugging Face if the configured
target model requires access.

```bash
python -m src.scripts.build_real_tiny_latent_bundle --config datasets/LIP-DATA-003/generated_configs/LIP-DATA-003_train_mbpp_32.yaml --device cuda --max-samples 32
python -m src.scripts.build_real_tiny_latent_bundle --config datasets/LIP-DATA-003/generated_configs/LIP-DATA-003_eval_mbpp_16.yaml --device cuda --max-samples 16
```

Expected zip outputs:

```text
datasets/LIP-DATA-003/LIP-DATA-003_mbpp_train_latent_bundle_32.zip
datasets/LIP-DATA-003/LIP-DATA-003_mbpp_eval_latent_bundle_16.zip
```

## Training With LIP-TRAIN-001

Upload the train bundle zip to trusted user-managed storage and trigger
`LIP-TRAIN-001 Configurable Remote Bundle Training` with the zip SHA256 digest.

Suggested first MBPP run:

```text
experiment_id: LIP-TRAIN-006-mbpp32-batch8-steps64-lambda025
google_drive_file_id: <LIP-DATA-003 MBPP train bundle file ID>
latent_bundle_sha256: <LIP-DATA-003 MBPP train bundle zip SHA256>
max_steps: 64
epochs: 16
batch_size: 8
learning_rate: 0.0001
lambda_mse: 0.25
device: cpu
```

## Evaluation

Evaluate the resulting checkpoint against both the MBPP train bundle and the
held-out validation bundle:

```bash
python -m src.scripts.evaluate_bridge \
  --config config/LIP-EVAL-001_bridge_eval.yaml \
  --checkpoint runs/LIP-TRAIN-006-mbpp32-batch8-steps64-lambda025/best_model.pth \
  --bundle-dir datasets/LIP-DATA-003/mbpp_train_bundle_32 \
  --output-dir runs/LIP-EVAL-MBPP32-insample

python -m src.scripts.evaluate_bridge \
  --config config/LIP-EVAL-001_bridge_eval.yaml \
  --checkpoint runs/LIP-TRAIN-006-mbpp32-batch8-steps64-lambda025/best_model.pth \
  --bundle-dir datasets/LIP-DATA-003/mbpp_eval_bundle_16 \
  --output-dir runs/LIP-EVAL-MBPP16-heldout
```

Record bundle manifests, shard digests, training artifact digests, and eval
metrics in later registry PRs.

## Data Policy

Do not commit generated MBPP prompt configs, latent bundles, shards, zip files,
datasets, model caches, checkpoints, or run artifacts. Generated outputs belong
under ignored local paths such as `datasets/` and `runs/`.

## Scientific Claim Boundary

This PR adds a public-dataset prompt sampling path only. It does not claim
semantic transfer, text-level fidelity, model-to-model alignment,
generalization, or production readiness.
