# LIP-H0-010 pilot v2 — entry-loss localization

Pilot v2 separated the layer-0 entry and induced-trajectory loss
configurations. It again completed 16 accepted updates but produced 52 AMP
overflows before the first accepted update.

The split metrics localized the singularity unambiguously. At the first update,
the entry relative-norm loss was `8.31e18`; the induced-trajectory relative-norm
loss was `1.01`. At the last update they were respectively `3.37e17` and `2.87`.
The single-layer entry auxiliary is therefore the problem. The original
PROTO-014 penalty aggregated eight layers and does not transfer safely to a
layer-0-only output when the teacher boundary norm is near zero.

Protocol v3 disables and skips the entry relative-norm calculation while
restoring it for the empirically stable induced trajectory. All other losses,
the causal graph, data, architecture, and pilot gate remain fixed. The full
matrix remains unauthorized.

- Run commit: `4db3c68954749fb1f009876ed6afc6ea358ee07c`.
- Wall time: 37.80 seconds on NVIDIA L4.
- Colab usage: 0.13 compute units.
- Artifact: `lip-artifacts/LIP-H0-010/pilot-v2/run_summary.json`.
- SHA-256: `622cad3cb7b7b883cce4c1c6370e8a0575554e194b466f4606a04a95d87fedbc`.
