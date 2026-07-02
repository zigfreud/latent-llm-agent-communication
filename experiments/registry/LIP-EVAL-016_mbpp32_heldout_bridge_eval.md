# LIP-EVAL-016 MBPP32 Held-Out Bridge Evaluation

## Summary

This entry records the held-out latent-space bridge evaluation for
`LIP-TRAIN-008-mbpp64-batch8-steps256-lambda025`. The checkpoint was trained on
the LIP-DATA-004 MBPP64 train bundle and evaluated on the separate MBPP32
validation bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-008-mbpp64-batch8-steps256-lambda025`
- Run ID: `28028527051`
- Artifact ID: `7821777879`
- Artifact digest: `sha256:ad4624dd539c84930ef938cb8cf56f3af3bc5435833eb71694c505fbfe328ba4`

Held-out eval bundle:

- Bundle trace: `LIP-DATA-004-MBPP-EVAL`
- Dataset: `google-research-datasets/mbpp`
- Split: `validation`
- Samples: `32`
- Eval shard SHA256: `7e58a9afcb2733ee3a7110eff757b3ffd47ae0391c526639f13f30714d60ce8e`
- Eval zip SHA256: `65de7852f6d1a129e82ffddedd258890e1cdc48c3780b5bd6425746fd15203d9`

## Metrics

- Sample count: `32`
- Latent MSE mean: `2.0768864154815674`
- Latent RMSE mean: `1.4221818447113037`
- Cosine diagonal mean: `0.8230685591697693`
- Cosine diagonal std: `0.05485030636191368`
- Prediction norm mean: `151.08436584472656`
- Target norm mean: `158.02142333984375`
- Norm ratio mean: `0.9560889005661011`
- Energy drift mean: `7.2567138671875`
- Retrieval top-1: `0.25`
- Retrieval top-k: `{"1": 0.25, "5": 0.59375}`
- Off-diagonal cosine mean: `0.7400941252708435`
- Diagonal margin mean: `-0.022361302748322487`

## Interpretation

Held-out top1 is `0.25` over 32 candidates, which is above chance, but held-out
diagonal margin remains negative. This suggests the bridge learns useful latent
geometry, while pair-level held-out ranking remains the main bottleneck.

Scaling from MBPP32/16 to MBPP64/32 improved reconstruction, norm ratio, and
energy calibration. This remains latent-space diagnostic evaluation only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, or production readiness claim.
