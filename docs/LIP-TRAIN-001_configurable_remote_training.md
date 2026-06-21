# LIP-TRAIN-001 Configurable Remote Bundle Training

## Purpose

LIP-TRAIN-001 adds a configurable remote CPU training workflow for trusted
external latent bundles. It extends the fixed H0-003 smoke path so follow-up
experiments can train for more steps and epochs while preserving bundle
validation, SHA256 verification, safe unzip, and artifact provenance.

## Difference From H0-003

H0-003 remains a two-step smoke workflow for validating ingestion and trainer
execution. LIP-TRAIN-001 is for configurable experiment runs. It accepts
workflow inputs for `experiment_id`, `max_steps`, `epochs`, `batch_size`,
`learning_rate`, and `lambda_mse`, then writes the effective config into the run
artifact.

This workflow does not replace or weaken H0-003.

## Google Drive Bundle Input

For a Google Drive-hosted bundle, trigger
`LIP-TRAIN-001 Configurable Remote Bundle Training` with:

- `experiment_id`: run identifier used for `runs/<experiment_id>`.
- `google_drive_file_id`: Drive file ID for the latent bundle zip.
- `latent_bundle_sha256`: SHA256 digest of the zip.
- `max_steps`, `epochs`, `batch_size`, `learning_rate`, `lambda_mse`: training
  controls.
- `device`: `cpu`.

Do not pass a browser/share URL as `latent_bundle_url` for Drive files. Browser
URLs can return HTML or confirmation pages. Use `google_drive_file_id` for
Drive-hosted bundles.

## Suggested First Run

Use the LIP-DATA-001 8-sample real bundle:

```text
experiment_id: LIP-TRAIN-001-data001-steps32-lambda01
google_drive_file_id: <LIP-DATA-001 bundle file ID>
latent_bundle_sha256: c733242c173ce23837d540f29b1769e668e000009c0eee80e8d173076d44b9e6
max_steps: 32
epochs: 8
batch_size: 2
learning_rate: 0.0001
lambda_mse: 0.1
device: cpu
```

The workflow downloads the bundle, verifies the zip digest, extracts it through
the safe unzip checks, validates the latent bundle, writes
`effective_config.yaml`, trains the adapter, and uploads `runs/<experiment_id>`.

## Expected Run Artifact

The uploaded artifact should contain:

- `latent_bundle_manifest.json`
- `latent_bundle_validation_report.json`
- `effective_config.yaml`
- `metrics.json`
- `train_log.csv`
- `run_summary.md`
- `last_checkpoint.pth`
- `best_model.pth`

The artifact is retained for seven days by default.

## Evaluation

After downloading the artifact and restoring the corresponding latent bundle
locally, evaluate the checkpoint with:

```bash
python -m src.scripts.evaluate_bridge \
  --config config/LIP-EVAL-001_bridge_eval.yaml \
  --checkpoint runs/LIP-TRAIN-001-data001-steps32-lambda01/best_model.pth \
  --bundle-dir datasets/LIP-H0-003/latent_bundle \
  --output-dir runs/LIP-EVAL-TRAIN-001
```

Use the same bundle that was supplied to the training workflow so the evaluation
is traceable to the recorded manifest and shard digest.

## Data Policy

Do not commit generated bundles, shards, datasets, checkpoints, model weights,
zips, caches, or run artifacts. External bundles must be trusted and
content-addressed with SHA256 before workflow execution.

## Scientific Claim Boundary

This workflow adds configurable remote training infrastructure only. It does
not claim semantic transfer, text-level fidelity, model-to-model alignment,
generalization, or production readiness.
