# LIP-EVAL-010 Batch8 LIP-DATA-002 Held-Out Bridge Evaluation

## Summary

This entry records the held-out latent-space bridge evaluation for
`LIP-TRAIN-005-data002-train16-batch8-steps64-lambda025`. The checkpoint was
trained on the LIP-DATA-002 train bundle with batch size 8 and evaluated on the
separate LIP-DATA-002 held-out eval bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-005-data002-train16-batch8-steps64-lambda025`
- Run ID: `28018067585`
- Artifact ID: `7817517981`
- Artifact digest: `sha256:fdca94e4991fc141dd10ba47fd42bb2557bd3091dd2f4cbdce6233f7d11ee458`

Held-out eval bundle:

- Bundle trace: `LIP-DATA-002-EVAL`
- Samples: `8`
- Shard SHA256: `8b18416d37312e9fbfde53b6b0404c3e69836b3d15fea7ec4807b565661e9fbd`
- Eval zip SHA256: `9ff206693188e59b477f00ee170485ae735a1ac4ce1a1e4133fe4facace749f3`

## Metrics

- Sample count: `8`
- Latent MSE mean: `2.177349328994751`
- Latent RMSE mean: `1.4710201025009155`
- Cosine diagonal mean: `0.8247871994972229`
- Cosine diagonal std: `0.032499074935913086`
- Prediction norm mean: `160.7887420654297`
- Target norm mean: `157.0806427001953`
- Norm ratio mean: `1.0236396789550781`
- Energy drift mean: `8.508981704711914`
- Retrieval top-1: `0.375`
- Retrieval top-k: `{"1": 0.375, "5": 0.875}`
- Off-diagonal cosine mean: `0.7781047821044922`
- Diagonal margin mean: `-0.00044032931327819824`

## Interpretation

Held-out retrieval_top1 improved from chance to `0.375`, but held-out pair-level
generalization is still not robust. Held-out diagonal margin remains
approximately zero and slightly negative. The bridge transfers some signal to
held-out prompts, but generalization remains the main bottleneck.

This remains latent-space evaluation only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, or production readiness claim.
