# LIP-EVAL-008 LIP-DATA-002 Train-Set In-Sample Bridge Evaluation

## Summary

This entry records the train-set in-sample latent-space evaluation for the same
checkpoint used in LIP-EVAL-007. The checkpoint was trained as
`LIP-TRAIN-004-data002-train16-steps64-lambda025` and is evaluated here on
`LIP-DATA-002-TRAIN` instead of the held-out eval bundle.

## Recorded Evidence

Training run:

- Experiment: `LIP-TRAIN-004-data002-train16-steps64-lambda025`
- Run ID: `28015568225`
- Artifact ID: `7816492548`
- Artifact digest: `sha256:2463c3e7b30176fb1870af8973da53833c7e4b85cd39b83b97224b36781a5c93`

In-sample eval bundle:

- Train bundle trace ID: `LIP-DATA-002-TRAIN`
- Samples: `16`
- Train shard SHA256: `3ee2950e73b7ddf920667aad8a7cb590586b538c377e1afe8e480acfd076efc4`
- Eval zip SHA256: `591c192c849053476399d090b9e97f57ac7977b168e1b00496fd2abe66d36554`

## Metrics

- Sample count: `16`
- Latent MSE mean: `1.1561908721923828`
- Latent RMSE mean: `1.067488431930542`
- Cosine diagonal mean: `0.8997008800506592`
- Cosine diagonal std: `0.02634170837700367`
- Prediction norm mean: `137.71466064453125`
- Target norm mean: `156.91961669921875`
- Norm ratio mean: `0.8774809837341309`
- Energy drift mean: `19.204971313476562`
- Retrieval top-1: `0.4375`
- Retrieval top-k: `{"1": 0.4375, "5": 0.9375}`
- Off-diagonal cosine mean: `0.828340470790863`
- Diagonal margin mean: `0.00045713409781455994`

## Interpretation

The checkpoint performs better on the train bundle than on held-out eval, but
train-set top-1 remains weak. This indicates the issue is not only held-out
generalization; pair-level contrastive separation is still insufficient even
in-sample. Several wrong predictions are attracted to target `4`.

This is a latent-space in-sample diagnostic only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space in-sample diagnostic only. No semantic transfer or generalization
claim.
