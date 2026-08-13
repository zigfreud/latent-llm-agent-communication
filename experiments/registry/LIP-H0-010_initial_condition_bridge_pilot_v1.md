# LIP-H0-010 pilot v1 — numeric feasibility failure

## Outcome

The first unrolled initial-condition pilot completed all 16 required updates on
an NVIDIA L4, with finite losses, nonzero bridge gradients, and only 6.98 GB of
peak allocated VRAM. It nevertheless failed the frozen feasibility gate because
AMP overflowed 52 times, against a maximum of two.

All 52 events occurred before the first accepted optimizer update. The scaler
fell from 65,536 to approximately `1.46e-11`; afterward the full 16-update pilot
completed without another overflow. This is not an OOM or broken-gradient
result.

## Localized cause

The induced trajectory's normalized RMSE was finite (2.54–2.55 on development),
but the inherited relative component-norm loss reached approximately
`3.37e18` on the boundary region. Some teacher boundary trajectories have
near-zero norm, so the ratio `predicted_norm / (target_norm + eps)` is singular
outside the teacher neighborhood. The resulting total losses and bridge
gradient norms were on the order of `1e15–1e17`.

The v1 objective therefore did not provide a valid numeric test of the causal
hypothesis. Its development margins were negative, but model quality is not
interpretable from a pilot dominated by a known singular loss term.

## Revision

Protocol v2 separates the entry-snapshot and induced-trajectory losses. The
entry objective retains the original relative-norm regularizer. The trajectory
objective retains Huber, cosine, symmetric NCE, and margin terms, while setting
only `lambda_norm` to zero. The architecture, data, seed, batch size, number of
accepted updates, receiver dynamics, and feasibility thresholds remain fixed.

The full two-variant, three-seed matrix remains unauthorized until the revised
pilot passes.

## Execution record

- Run commit: `c1a94bd259378d7f89292e67970691400728d46e`.
- L4 peak allocated VRAM: 6.98 GB.
- Wall time: 150.62 seconds.
- Colab usage: 0.12 compute units.
- Artifact: `lip-artifacts/LIP-H0-010/pilot/run_summary.json`.
- SHA-256: `90074e797a63ad31f050910120e793461af4a00d622c5cedb6df010b109fe6ee`.
