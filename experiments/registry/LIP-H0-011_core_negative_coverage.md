# LIP-H0-011 — directional success for negative coverage

H0-011 isolated a single change from H0-010: batch size increased from four to
sixteen while the total number of examples remained 2,048. This increased the
in-batch alternatives from three to fifteen without changing architecture,
loss weights, receiver evolution, data, or seed.

The intervention inverted the development-gate core margin from -0.00417 to
+0.00335 and raised core retrieval from 56.25% to 68.75%. Mean retrieval rose
from 65.63% to 73.96%. Joint and name passed their Holm-adjusted tests. Core did
not (`p_Holm=0.373`), so the complete family still failed.

RMSE moved in the opposite direction, from 1.288 to 1.425. That tradeoff is
informative: packet reconstruction and hard-negative identity separation are
not interchangeable objectives. H0-010 found the receiver-dynamics mechanism;
H0-011 shows that contrastive coverage is part of the missing discriminative
property.

The result meets the frozen directional gate but not the strong gate. It
authorizes a three-seed replication of the same intervention. It does not
authorize functional confirmation or PROTO-015.

- Run commit: `bc289d1197165715ef31ff0b3d6bdf038fead4dc`.
- Pilot: 4 updates, numeric gate passed, 8.24 GB peak allocated VRAM.
- Screen: seed 4007, 128 updates, best step 120, 111.57 seconds.
- H0-011 interval: 0.52 visible Colab compute units.
- Pilot SHA-256: `bc90e865d8559861ec5c0bc8f305e919b3d2d7acd6a9d04fd9410af759ebc872`.
- Screen SHA-256: `9fab01ec20b1a691911b24dc2e276eae8ecfa1f498a2f555cacd1bf436bfb08a`.
- Confirmation data used: no.
