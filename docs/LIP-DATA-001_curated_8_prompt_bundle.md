# LIP-DATA-001 Curated 8-Prompt Bundle Config

## Purpose

LIP-DATA-001 adds a separate configuration for generating the next real latent
bundle with eight curated prompt pairs. It uses the existing H0-005 latent
bundle builder and keeps the generated bundle outside git.

## Why This Exists Separately

The H0-005 config is a tiny smoke-test config with four prompts. The first real
remote run used two samples, which was enough to validate the ingestion and
evaluation path but not enough to scale the next experiment. This config keeps
the eight-prompt input set traceable without changing the H0-005 baseline.

## Prompt Policy

The prompt set contains eight small, non-sensitive, self-contained programming
tasks. The prompts avoid private data, credentials, personal information,
project-specific secrets, and requests for model completions. The builder does
not write raw prompts into shard records and does not store generated text.

## Dry-Run

Dry-run mode does not download or load Hugging Face models. It creates
deterministic mock vectors with the configured dimensions, writes the bundle,
packages it, and validates it.

```bash
python -m src.scripts.build_real_tiny_latent_bundle --config config/LIP-DATA-001_real_bundle_8.yaml --dry-run
```

Expected dry-run outputs:

```text
datasets/LIP-DATA-001/real_bundle_8/manifest.json
datasets/LIP-DATA-001/real_bundle_8/shards/shard_0.pt
datasets/LIP-DATA-001/LIP-DATA-001_real_latent_bundle_8.zip
```

## Real Extraction In Colab

Run real extraction only in a local or Colab environment where downloading and
loading the configured models is intentional. Do not run real extraction in
GitHub Actions.

```bash
python -m src.scripts.build_real_tiny_latent_bundle --config config/LIP-DATA-001_real_bundle_8.yaml --device cuda --max-samples 8
```

The config keeps `batch_size: 1`, `sequential_model_loading: true`,
`device_map: "auto"`, and `load_in_4bit: true` to make extraction more practical
on constrained GPU environments. If extraction runs out of memory, use a
smaller `--max-samples` value for infrastructure testing and keep generated
outputs outside git.

## Validation

Validate the generated bundle directory before using the zip in a remote
workflow:

```bash
python -m src.scripts.validate_latent_bundle --bundle-dir datasets/LIP-DATA-001/real_bundle_8
```

The builder also validates the bundle automatically after packaging and prints
the final zip SHA256 digest.

## H0-003 Workflow Dispatch

Upload the produced zip to trusted user-managed storage, such as Google Drive,
and trigger `LIP-H0-003 Remote CPU From Latent Bundle` with:

- `google_drive_file_id`: Google Drive file ID for
  `LIP-DATA-001_real_latent_bundle_8.zip`.
- `latent_bundle_sha256`: SHA256 digest printed by the builder or recomputed
  locally from the zip.

Do not pass a browser/share URL as `latent_bundle_url`; Google Drive share URLs
can return HTML or confirmation pages. Use the first-class
`google_drive_file_id` input for Drive-hosted bundles.

## Data Policy

No generated latent bundle, shard, checkpoint, model weight, dataset, cache,
zip, or run artifact should be committed. Generated outputs belong under
ignored paths such as `datasets/` or another local scratch directory.

## Scientific Claim Status

This config only prepares a curated prompt set for generating a larger real
latent bundle. It does not claim semantic transfer, text-level fidelity,
model-to-model alignment, or production readiness.
