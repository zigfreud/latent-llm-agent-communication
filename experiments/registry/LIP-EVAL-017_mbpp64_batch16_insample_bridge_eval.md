# LIP-EVAL-017 MBPP64 Batch16 In-Sample Bridge Evaluation

## Summary

This entry records the train-set in-sample latent-space bridge evaluation for
`LIP-TRAIN-009-mbpp64-batch16-steps128-lambda025`. The checkpoint was trained on
the LIP-DATA-004 MBPP64 train bundle with batch size 16 and evaluated on the
same train bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-009-mbpp64-batch16-steps128-lambda025`
- Run ID: `28597364631`
- Artifact ID: `8041794109`
- Artifact digest: `sha256:6cc7fa9e93324a4fba24d8815dd2cc6477688f1a76631e75962b66a044bea6be`

In-sample eval bundle:

- Bundle trace: `LIP-DATA-004-MBPP-TRAIN`
- Dataset: `google-research-datasets/mbpp`
- Split: `train`
- Samples: `64`
- Train shard SHA256: `f1dd527799ad2dd0e99edc253ac71213c9db4db36d9cf6b702e3a6ffc126e811`
- Eval zip SHA256: `532c05156fce451680c1a611fe818ac2577826a3dfb5cd015429f50ea23a839e`

## Metrics

- Sample count: `64`
- Latent MSE mean: `1.1026196479797363`
- Latent RMSE mean: `1.0471668243408203`
- Cosine diagonal mean: `0.9052202701568604`
- Cosine diagonal std: `0.014378687366843224`
- Prediction norm mean: `143.71766662597656`
- Target norm mean: `157.84063720703125`
- Norm ratio mean: `0.9105277061462402`
- Energy drift mean: `14.122961044311523`
- Retrieval top-1: `0.9375`
- Retrieval top-k: `{"1": 0.9375, "5": 1.0}`
- Off-diagonal cosine mean: `0.6292169690132141`
- Diagonal margin mean: `0.10249336063861847`

## Interpretation

In-sample retrieval remains strong at `0.9375`. Compared with the batch size 8
MBPP64 run, reconstruction and norm/energy calibration worsened, likely because
the batch16 run used fewer optimizer updates at matched total sample exposure.

This remains latent-space diagnostic evaluation only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, or production readiness claim.
