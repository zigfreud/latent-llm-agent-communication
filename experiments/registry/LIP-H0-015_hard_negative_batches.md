# LIP-H0-015 — hard-negative coverage passes the strong identity gate

H0-015 changed only train-batch membership relative to H0-013. The frozen
H0-013 candidate bank supplied one global hardest negative for each of 256
train anchors. A balanced partition colocated 224 pairs, giving 87.50%
within-batch coverage versus 5.88% expected for a random partition. Every task
still appeared exactly once per epoch; loss, architecture, receiver evolution,
seed, batch size, update count, example exposure, selection, and held gate were
unchanged. Confirmation data were not used.

The intervention passed both the directional and strong gates. Core margin
rose from `+0.010905` to `+0.017437`, core retrieval from 71.88% to 75.00%,
and mean regional retrieval from 81.25% to 84.38%. Joint, core, and name all
rejected after Holm correction; core reached `p_Holm=0.00553`.

The mechanism is visible in training. Hard batching increased the median core
hinge presented to the optimizer from `0.03781` to `0.05212`, because the
negatives were genuinely harder, but its final value fell from `0.02473` to
`0.01639`. Unlike the core-only scale intervention, this improvement
generalized to held development identity geometry.

There is a real tradeoff. Normalized residual RMSE worsened from `1.41856` to
`1.45571`. H0-015 therefore establishes a robust single-seed identity-geometry
gain, not uniform packet reconstruction improvement.

The earlier 4.30% figure is now explicitly bounded: it measured the static
EVAL-032 diagnostic partition, not cumulative coverage across H0-013's shuffled
epochs. The causal contrast here is 87.50% guaranteed frozen coverage versus
the H0-013 random-batch policy, not 87.50% versus a claimed historical 4.30%.

The result authorizes exact replication on seeds 4001 and 4003 using the same
frozen partition. It does not authorize dynamic mining, functional
confirmation, or PROTO-015.

- Run commit: `06b3d8a2a3399c7a903e329eb994644317bb80d3`.
- Pilot: 4 updates, numeric gate passed, best step 2, 35.28 seconds.
- Screen: 128 updates, best step 120, 116.94 seconds.
- Peak allocated VRAM: 8.24 GB.
- Screen SHA-256: `b0b3a9b87b5825a45bdb3085bded4c4cef31d7310ec667e66dbc0beff40d5f54`.
- Batch-plan SHA-256: `2057fbc1f146058cbf4da4687d1eae7e158d3c48afa788cdad02827172963783`.
