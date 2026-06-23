# LIP-EVAL-011 MBPP32 In-Sample Bridge Evaluation

## Summary

This entry records the train-set in-sample latent-space bridge evaluation for
`LIP-TRAIN-006-mbpp32-batch8-steps64-lambda025`. The checkpoint was trained on
the LIP-DATA-003 MBPP32 train bundle and evaluated on the same train bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-006-mbpp32-batch8-steps64-lambda025`
- Run ID: `28023800921`
- Artifact ID: `7819833905`
- Artifact digest: `sha256:3333875c0e54a21a63fc805413269c663c3a8d4c60938b3293f3eed1232f585c`

In-sample eval bundle:

- Bundle trace: `LIP-DATA-003-MBPP-TRAIN`
- Dataset: `google-research-datasets/mbpp`
- Split: `train`
- Samples: `32`
- Train shard SHA256: `c0e3eb2c8832be89f61e8033407853e48d402f761d3918b75c3a0291b02373a5`

## Metrics

- Sample count: `32`
- Latent MSE mean: `1.415250539779663`
- Latent RMSE mean: `1.1845561265945435`
- Cosine diagonal mean: `0.8769411444664001`
- Cosine diagonal std: `0.02436354197561741`
- Prediction norm mean: `132.56350708007812`
- Target norm mean: `157.84262084960938`
- Norm ratio mean: `0.8397974967956543`
- Energy drift mean: `25.279125213623047`
- Retrieval top-1: `1.0`
- Retrieval top-k: `{"1": 1.0, "5": 1.0}`
- Off-diagonal cosine mean: `0.6628353595733643`
- Diagonal margin mean: `0.06740360707044601`

## Interpretation

MBPP in-sample retrieval is perfect but with lower margin than the curated
LIP-DATA-002 in-sample run. The batch8 MBPP32 checkpoint clearly learns the
train prompt bundle in latent space, while the lower margin suggests MBPP prompt
diversity makes pair separation harder than the curated 16-prompt train set.

This remains latent-space diagnostic evaluation only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, or production readiness claim.
