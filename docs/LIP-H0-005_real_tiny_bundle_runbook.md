# LIP-H0-005 Real Tiny Latent Bundle Runbook

## Purpose

LIP-H0-005 adds a small local builder for producing an H0-003-compatible latent
bundle from a curated tiny prompt set. The builder supports a dry-run mode that
uses deterministic mock tensors and a real mode that extracts hidden-state
vectors from the configured source and target models.

## Scope

This tooling validates that a tiny latent bundle can be produced, packaged, and
validated against the H0-003 ingestion contract. It does not prove semantic
transfer, latent communication, model-to-model alignment, or performance
improvement.

The script does not write raw prompts into shard records, does not write model
text outputs, and does not include model weights in the bundle.

## Dry-Run

Dry-run mode does not download or load Hugging Face models. It creates
deterministic mock tensors with the configured `input_dim` and `output_dim`.

```bash
python -m src.scripts.build_real_tiny_latent_bundle \
  --config config/LIP-H0-005_real_tiny_bundle.yaml \
  --dry-run
```

Expected outputs:

```text
datasets/LIP-H0-005/real_tiny_bundle/manifest.json
datasets/LIP-H0-005/real_tiny_bundle/shards/shard_0.pt
datasets/LIP-H0-005/LIP-H0-005_real_tiny_latent_bundle.zip
```

The command packages the bundle, validates it with
`src.scripts.validate_latent_bundle`, and prints the final zip `sha256`.

## Real Extraction

Run real extraction only in a local or Colab environment where downloading and
loading the configured models is intentional. Do not run real extraction in
GitHub Actions.

Example:

```bash
python -m src.scripts.build_real_tiny_latent_bundle \
  --config config/LIP-H0-005_real_tiny_bundle.yaml \
  --device cuda
```

Use `--device cpu` only if the host has enough memory and runtime budget. The
builder loads the source model, extracts source vectors, frees it, then loads
the target model and extracts target vectors for the same prompts.

Optional overrides:

```bash
python -m src.scripts.build_real_tiny_latent_bundle \
  --config config/LIP-H0-005_real_tiny_bundle.yaml \
  --device cuda \
  --max-samples 2 \
  --output-dir datasets/LIP-H0-005/real_tiny_bundle \
  --output-zip datasets/LIP-H0-005/LIP-H0-005_real_tiny_latent_bundle.zip
```

## Zip Digest

The builder prints the final zip digest:

```text
sha256: <digest>
```

To recompute it:

```bash
python -c "import hashlib, pathlib; p=pathlib.Path('datasets/LIP-H0-005/LIP-H0-005_real_tiny_latent_bundle.zip'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

Use this digest as the `latent_bundle_sha256` input when triggering the H0-003
workflow manually.

## H0-003 Workflow Dispatch

Upload the generated zip to a trusted location and trigger
`LIP-H0-003 Remote CPU From Latent Bundle` with:

- `latent_bundle_url`: URL for the uploaded zip.
- `latent_bundle_sha256`: sha256 digest printed by the builder.

External latent bundles must be trusted and content-addressed with sha256. The
H0-003 workflow rejects external URLs without a digest.

## Data Policy

Do not commit generated bundles, shards, model caches, checkpoints, run
artifacts, model weights, or datasets. Keep generated files under ignored paths
such as `datasets/` or another local scratch directory.
