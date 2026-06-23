# LIP-EVAL-009 Batch8 LIP-DATA-002 In-Sample Bridge Evaluation

## Summary

This entry records the train-set in-sample latent-space bridge evaluation for
`LIP-TRAIN-005-data002-train16-batch8-steps64-lambda025`. The checkpoint was
trained on the LIP-DATA-002 train bundle with batch size 8 and evaluated on the
same train bundle.

## Recorded Evidence

Training source:

- Training experiment: `LIP-TRAIN-005-data002-train16-batch8-steps64-lambda025`
- Run ID: `28018067585`
- Artifact ID: `7817517981`
- Artifact digest: `sha256:fdca94e4991fc141dd10ba47fd42bb2557bd3091dd2f4cbdce6233f7d11ee458`

In-sample eval bundle:

- Bundle trace: `LIP-DATA-002-TRAIN`
- Samples: `16`
- Shard SHA256: `3ee2950e73b7ddf920667aad8a7cb590586b538c377e1afe8e480acfd076efc4`
- Eval zip SHA256: `2ad2da3373a50331b03ffa39ba714f90f5d45c3d572a39a67de0704313164443`

## Metrics

- Sample count: `16`
- Latent MSE mean: `0.7565261125564575`
- Latent RMSE mean: `0.8679006099700928`
- Cosine diagonal mean: `0.936291515827179`
- Cosine diagonal std: `0.008641578257083893`
- Prediction norm mean: `153.98049926757812`
- Target norm mean: `156.91961669921875`
- Norm ratio mean: `0.9812732934951782`
- Energy drift mean: `4.051018714904785`
- Retrieval top-1: `1.0`
- Retrieval top-k: `{"1": 1.0, "5": 1.0}`
- Off-diagonal cosine mean: `0.673990786075592`
- Diagonal margin mean: `0.15249031782150269`

## Interpretation

Batch size 8 substantially improved in-sample pair separation. In-sample
retrieval_top1 reached `1.0` and diagonal margin became strongly positive. The
bridge now clearly learns the train set.

This remains latent-space evaluation only.

## Data Policy

No generated eval outputs, checkpoints, shards, bundles, datasets, zips, caches,
or run artifacts are committed in this registry entry. This PR records only
lightweight, text-based evidence and interpretation.

## Scientific Claim Status

Latent-space diagnostic evaluation only. No semantic transfer, text-level
fidelity, model-to-model alignment, or production readiness claim.
