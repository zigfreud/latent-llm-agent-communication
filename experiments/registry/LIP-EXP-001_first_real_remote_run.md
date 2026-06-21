# LIP-EXP-001 First Real Remote Latent-Bundle Run

## Summary

This entry records the first successful remote GitHub Actions run that consumed
a real H0-005 latent bundle and trained the H0-003 bridge in CPU mode.

## Recorded evidence

- Workflow: `LIP-H0-003 Remote CPU From Latent Bundle`
- Run ID: `27833132397`
- Job ID: `82374579828`
- Head branch: `main`
- Head SHA: `f7f5cd5f769217ab70d2faf1e49a72bdf843897d`
- Artifact ID: `7752805337`
- Artifact name: `LIP-H0-003-27833132397-remote-cpu-from-latent-bundle`
- Artifact digest: `sha256:18dcdb307a7e992d113e0b0f3f89fafc1dbd680749ffb4ee64b4a1bae7d8d579`
- Artifact expires at: `2026-06-26T15:00:16Z`
- External artifact backup: Google Drive / user-managed storage

## What Was Validated

- A real H0-005 latent bundle was accepted by the H0-003 remote workflow.
- The bundle manifest described a `2048 -> 4096` latent-vector pair format.
- The run used CPU training for the H0-003 bridge.
- The workflow produced a temporary GitHub Actions artifact.
- The recorded training metrics show one completed CPU training step.

Training summary:

- Experiment ID: `LIP-H0-003`
- Device: `cpu`
- Samples: `2`
- Batch size: `2`
- Max steps: `2`
- Steps completed: `1`
- Best loss: `1.301304578781128`
- Final loss: `1.301304578781128`
- Final accuracy: `0.5`

## What Was Not Validated

- Semantic fidelity was not evaluated.
- Generalization was not evaluated.
- Held-out prompts were not evaluated.
- Model-to-model alignment quality was not established.
- Production readiness was not established.

## Operational Milestone

This is an operational milestone because it confirms that a real latent bundle
generated outside GitHub Actions can be supplied to the remote H0-003 workflow,
validated, trained on CPU, and captured as a temporary artifact without
committing datasets, shards, checkpoints, zips, or run artifacts to the
repository.

## Scientific Claim Status

This registry entry records the first successful real-latent remote
infrastructure run. It does not claim semantic transfer, latent communication
fidelity, model-to-model alignment, or production readiness.

## Data Policy

No artifact zip, checkpoint, `.pt` shard, dataset, generated run directory,
model weight, cache, or generated bundle is committed in this registry entry.
The heavy artifact remains outside git.
