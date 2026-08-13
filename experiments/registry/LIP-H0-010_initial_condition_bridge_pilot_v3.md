# LIP-H0-010 pilot v3 — feasibility passed

Protocol v3 passed the frozen L4 feasibility gate. All 16 updates completed with
finite losses and nonzero bridge gradients. AMP recorded one overflow against a
maximum of two, and peak allocated VRAM was 6.98 GB against the 22.5 GB ceiling.

Training loss fell from 3.19 to 2.71 and gradient norms stayed between 0.52 and
3.32. These are stability diagnostics, not a quality result. The 16-update
model did not pass the development Holm gate; that gate is intentionally not
used to stop or redesign a feasibility pilot.

The frozen two-variant, three-seed development matrix is now authorized. It
uses only train, development-selection, and development-gate records. No
confirmation tasks may be opened.

- Run commit: `1c7e4da2f7d8259d74ee772912f0caf5323afdf5`.
- Wall time: 30.11 seconds on NVIDIA L4.
- Colab usage: 0.13 compute units.
- Artifact: `lip-artifacts/LIP-H0-010/pilot-v3/run_summary.json`.
- SHA-256: `ada20c333cfed7adf05b2f788cee941cd7c872190e67ed337269e0491306f41d`.
