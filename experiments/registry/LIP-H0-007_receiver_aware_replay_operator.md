# LIP-H0-007 — receiver-aware replay operator result

## Outcome

H0-007 is a clean negative result for the naive additive replay operator. It
does not advance to the frozen functional identity pilot.

Replacing an absolute packet with

`h_injected[l] = h_live_incoming[l] + (packet[l] - training_scaffold[l])`

reduced the measured cross-layer intervention jump for every task and every
learned replica. However, it moved the resulting attention and residual states
farther from the oracle replay regime for every task and every replica.

| Variant | Seed | Jump replace | Jump add | Attention replace | Attention add | Residual replace | Residual add |
|---|---:|---:|---:|---:|---:|---:|---:|
| Component contrastive | 4001 | 0.621 | 0.259 | 0.781 | 1.230 | 0.800 | 1.214 |
| Component contrastive | 4003 | 0.603 | 0.271 | 0.773 | 1.241 | 0.802 | 1.240 |
| Component contrastive | 4007 | 0.637 | 0.253 | 0.786 | 1.209 | 0.804 | 1.188 |
| Structured linear | 4001 | 0.554 | 0.264 | 0.710 | 1.082 | 0.705 | 1.096 |
| Structured linear | 4003 | 0.563 | 0.257 | 0.716 | 1.078 | 0.713 | 1.073 |
| Structured linear | 4007 | 0.550 | 0.265 | 0.708 | 1.084 | 0.702 | 1.092 |

For transition jump, additive replay improved 8/8 tasks in all six learned
replicas. For both attention-output and residual-output NRMSE, it improved 0/8
tasks in all six replicas. Consequently, zero of the three primary replicas
passed the preregistered all-metrics gate.

## Oracle falsification

The oracle diagnostic isolates the operator assumption from bridge quality.
Oracle absolute replacement defines the alignment reference and had a mean
transition jump of 0.168. Applying the same task-delta construction to the
oracle packet produced:

- transition jump NRMSE: 0.289;
- query NRMSE: 0.436;
- key NRMSE: 0.465;
- value NRMSE: 0.648;
- attention-output NRMSE: 1.348;
- residual-output NRMSE: 1.713.

Thus, the additive construction fails even when the task packet itself is
exact. The negative result cannot be attributed only to learned bridge error.

## What was falsified

The experiment falsifies the assumption that the mean teacher packet is a
valid affine origin for the receiver's live neutral carrier at every layer.
Subtracting that static scaffold and adding the remainder to a separately
evolved live state combines states that do not share a coordinate origin.

It also shows that cross-layer jump alone is not a sufficient target. The
operator made the intervention numerically smaller while making every observed
downstream state less oracle-like. Future correctors must be judged jointly by
their intervention geometry and the receiver state they actually induce.

H0-007 does **not** falsify the broader receiver-trajectory hypothesis from
EVAL-031. It falsifies one static, untrained realization of it.

## Next experiment

The next smallest causal test is an anchored hybrid:

- layer 0: absolute packet replacement, establishing the task/carrier origin;
- layers 1–7: preserve the live incoming receiver state and add the
  packet-minus-scaffold task residual.

This directly tests whether the naive operator failed primarily because it
treated the carrier-entry boundary like an ordinary trajectory update. It
remains an exploratory operator screen, not PROTO-015.

If anchoring layer 0 does not repair downstream alignment even for the oracle
diagnostic, the next justified step is a trained, live-state-conditioned
corrector rather than another static arithmetic rule.

## Execution record

- Run commit: `cf3411d2778cd0edc397acc38f337a47cc0f4408`.
- Sample: 8 exposed PROTO-014 confirmation tasks, balanced across the two
  tokenizer strata.
- Learned conditions: 2 bridge variants × 3 training seeds × 2 operators.
- Total result rows, including oracle diagnostics: 112.
- Accelerator: NVIDIA L4; peak allocated VRAM 6.16 GB.
- Wall time: 317.46 seconds.
- Observed Colab use: 0.26 compute units.
- Final artifact SHA-256:
  `48b38270c359ccd0ffe529dabd1faa64ad683e4ef025a82ed4d5edabfd0763b6`.

The canonical artifact is
`lip-artifacts/LIP-H0-007/trajectory_gate.json` in Drive.

