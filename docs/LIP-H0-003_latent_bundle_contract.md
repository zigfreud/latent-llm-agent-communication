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
listed in `manifest.json`.

## manifest.json

Required fields:

```json
{
  "bundle_format": "lip_latent_bundle",
  "schema_version": 1,
  "input_dim": 2048,
  "output_dim": 4096,
  "shards": [
    {
      "path": "shards/shard_0.pt",
      "records": 8
    }
  ]
}
```

Field rules:

- `bundle_format` must be `lip_latent_bundle`.
- `schema_version` must be `1`.
- `input_dim` and `output_dim` must be positive integers.
- `shards` must be a non-empty list of objects.
- Each shard object must include a relative `path` under `shards/`.
- `records` is optional, but when present it must match the number of examples in
  the shard.
- `sha256` is optional, but when present it must match the shard file digest.

## Shard Format

Each shard file is a `.pt` file loadable with:

```python
torch.load(path, map_location="cpu")
```

The loaded object must be a non-empty list of dictionaries. Each dictionary must
include:

- `src_vector`: a `torch.Tensor` with shape `(input_dim,)` after `squeeze()`.
- `tgt_vector`: a `torch.Tensor` with shape `(output_dim,)` after `squeeze()`.

For LIP-H0-003, the expected dimensions are `2048` for `src_vector` and `4096`
for `tgt_vector`.

## Data Policy

Real latent shards, datasets, checkpoints, model weights, and generated run
artifacts must not be committed to the repository. External bundles should be
provided to manual workflows by URL, preferably with `latent_bundle_sha256`.
