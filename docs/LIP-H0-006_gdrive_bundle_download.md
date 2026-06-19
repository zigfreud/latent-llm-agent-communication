# LIP-H0-006 Google Drive Bundle Download

## Purpose

LIP-H0-006 adds a Google Drive file ID download path to the H0-003 latent bundle
workflow. This lets a manually generated latent bundle zip be stored in Google
Drive and consumed by workflow dispatch while keeping SHA256 verification, safe
unzip, manifest validation, and CPU trainer execution in place.

## Google Drive file ID

Use the file ID from a Google Drive share URL, not the full browser URL.

For a URL like:

```text
https://drive.google.com/file/d/FILE_ID/view?usp=sharing
```

the workflow input is:

```text
FILE_ID
```

For a URL like:

```text
https://drive.google.com/open?id=FILE_ID
```

the workflow input is also:

```text
FILE_ID
```

The file must be shared so anyone with the link can read it. Private files or
files that require interactive browser login cannot be downloaded by the
workflow.

## Workflow dispatch

Trigger `LIP-H0-003 Remote CPU From Latent Bundle` with:

- `google_drive_file_id`: the Google Drive file ID for the latent bundle zip.
- `latent_bundle_sha256`: the SHA256 digest of the local zip file.

Leave `latent_bundle_url` empty when using `google_drive_file_id`. The two
inputs are mutually exclusive.

To compute the digest locally:

```bash
sha256sum LIP-H0-005_real_tiny_latent_bundle.zip
```

On Windows PowerShell:

```powershell
Get-FileHash .\LIP-H0-005_real_tiny_latent_bundle.zip -Algorithm SHA256
```

Browser and share URLs can return HTML, confirmation pages, or virus-scan
interstitials. Do not pass Google Drive browser/share URLs as
`latent_bundle_url`; use `google_drive_file_id` instead.

## Expected outputs

A successful run uploads the H0-003 run artifact containing:

- `latent_bundle_manifest.json`
- `latent_bundle_validation_report.json`
- `metrics.json`
- `train_log.csv`
- `run_summary.md`
- `last_checkpoint.pth`
- `best_model.pth`

## Data policy

No latent bundle, shard, checkpoint, model weight, dataset, cache, or run
artifact is committed by this workflow change. External bundles must remain
outside git and must be content-addressed with `latent_bundle_sha256`.

## Scientific claim status

This PR only adds a Google Drive download path for external latent bundles. It
does not claim semantic transfer, latent communication, model-to-model
alignment, or performance improvement.
