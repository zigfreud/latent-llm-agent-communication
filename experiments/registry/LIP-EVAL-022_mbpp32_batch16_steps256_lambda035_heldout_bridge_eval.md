# LIP-EVAL-022 MBPP32 Batch16 256-Step Lambda 0.35 Held-Out Bridge Evaluation

## Summary

This entry records the held-out latent-space bridge evaluation for
`LIP-TRAIN-011-mbpp64-batch16-steps256-lambda035`. The checkpoint was trained on
the LIP-DATA-004 MBPP64 train bundle with batch size 16 for 256 steps and
evaluated on the separate MBPP32 validation bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-011-mbpp64-batch16-steps256-lambda035`
- Run ID: `28603839197`
- Artifact ID: `8044498853`
- Artifact digest: `sha256:a64b273dbce6ed364e920a976c32336120e84d59c047456597443f1f61557ba0`

Held-out eval bundle:

- Bundle trace: `LIP-DATA-004-MBPP-EVAL`
- Dataset: `google-research-datasets/mbpp`
- Split: `validation`
- Samples: `32`
- Eval shard SHA256: `7e58a9afcb2733ee3a7110eff757b3ffd47ae0391c526639f13f30714d60ce8e`
- Eval zip SHA256: `b1a4c0ff0ec4c7fde3cc1d62b3258f9a7ebc0952c0478f9b9e48d2cb31165c2f`

## Metrics

- Sample count: `32`
- Latent MSE mean: `2.0937340259552`
- Latent RMSE mean: `1.4308502674102783`
- Cosine diagonal mean: `0.8178914785385132`
- Cosine diagonal std: `0.05285530164837837`
- Prediction norm mean: `146.65748596191406`
- Target norm mean: `158.02142333984375`
- Norm ratio mean: `0.9280939102172852`
- Energy drift mean: `11.363945007324219`
- Retrieval top-1: `0.375`
- Retrieval top-k: `{"1": 0.375, "5": 0.75}`
- Off-diagonal cosine mean: `0.728355884552002`
- Diagonal margin mean: `-0.006803490221500397`

## Comparison Against LIP-EVAL-020

Compared with the lambda_mse `0.25` MBPP64 batch16 steps256 held-out result:

- Held-out MSE improved from `2.3901524543762207` to `2.0937340259552`.
- Held-out norm ratio improved from `0.8927412629127502` to `0.9280939102172852`.
- Held-out energy drift improved from `16.948368072509766` to `11.363945007324219`.
- Held-out retrieval top-1 improved from `0.34375` to `0.375`.
- Held-out retrieval top-5 improved from `0.71875` to `0.75`.
- Held-out diagonal margin mean improved slightly from `-0.007311269640922546` to `-0.006803490221500397`, but remains negative.

## Interpretation

Increasing lambda_mse from `0.25` to `0.35` in the MBPP64 batch16 steps256
regime improved reconstruction, norm/energy calibration, and held-out retrieval.
However, held-out diagonal margin remains negative, so pair-level held-out
ranking is improved but still not robust.

This remains latent-space diagnostic evaluation only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, or production readiness claim.
