# LIP-H0-003 Latent Bundle Contract

This document defines the file contract for an external latent bundle accepted by
the LIP-H0-003 CPU smoke workflow.

## Layout

A bundle is either an unpacked directory or a zip archive with these paths at the
archive root:

```text
manifest.json
shards/shard_0.pt
shards/shard_1.pt
...
```

`shards/shard_0.pt` is required. Additional shard files are optional and must be
listed in `manifest.json`. Zip bundles are intentionally narrow: entries outside
`manifest.json` and direct `shards/*.pt` files are rejected by the workflow.

## manifest.json

Required fields:

```json
{
  "bundle_format": "lip_latent_bundle",
  "schema_version": 1,
  "trace_id": "LIP-H0-003",
  "source_model": "synthetic-source",
  "target_model": "synthetic-target",
  "dataset_origin": "synthetic-pr-validation",
  "input_dim": 2048,
  "output_dim": 4096,
  "num_samples": 8,
  "created_at": "2026-06-19T00:00:00Z",
  "license_notes": "Synthetic generated data for CI validation.",
  "shards": [
    {
      "path": "shards/shard_0.pt",
      "records": 8,
      "sha256": "optional"
    }
  ]
}
```

Field rules:

- `bundle_format` must be `lip_latent_bundle`.
- `schema_version` must be `1`.
- `trace_id`, `source_model`, `target_model`, `dataset_origin`, `created_at`,
  and `license_notes` must be non-empty strings.
- `input_dim`, `output_dim`, and `num_samples` must be positive integers.
- `shards` must be a non-empty list of objects.
- `shards/shard_0.pt` must be listed.
- Each shard object must include a relative direct-child `path` matching
  `shards/*.pt` using forward slashes.
- `records` is optional. When any shard provides `records`, every shard must
  provide it, and the sum must equal `num_samples`.
- Loaded shard record counts must equal `num_samples`.
- `sha256` is optional per shard, but when present it must match the shard file
  digest.

Optional top-level fields may include:

- `extraction_commit`
- `extraction_notes`
- `source_layer`
- `target_layer`
- `prompt_policy`

## Shard Format

Each shard file is a `.pt` file loadable with:

```python
torch.load(path, map_location="cpu", weights_only=True)
```

The loaded object must be a non-empty list of dictionaries. Each dictionary must
include:

- `src_vector`: a `torch.Tensor` with shape `(input_dim,)` after `squeeze()`.
- `tgt_vector`: a `torch.Tensor` with shape `(output_dim,)` after `squeeze()`.

For LIP-H0-003, the expected dimensions are `2048` for `src_vector` and `4096`
for `tgt_vector`.

## Data Policy

Real latent shards, datasets, checkpoints, model weights, and generated run
artifacts must not be committed to the repository. Manual external bundles must
be supplied with a `latent_bundle_sha256` workflow input so the downloaded zip is
content-addressed before extraction.
