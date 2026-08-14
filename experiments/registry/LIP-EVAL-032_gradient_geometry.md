# LIP-EVAL-032 — receiver-unrolled gradient geometry

Status: completed. The predeclared route is **scale-limited**.

## Result

The core identity objective is not fighting the receiver-aware reconstruction
and contrastive objectives. In H0-013, the median cosine between the core-margin
and non-margin gradients was `+0.4093` with a 95% bootstrap interval of
`[+0.3743, +0.4305]`; none of the 16 batches crossed the conflict threshold.

The limiting property is leverage. The median raw core/non-margin gradient-norm
ratio was `0.1143`, but the configured trajectory margin averages joint, core,
and name. The core's actual coefficient is therefore `lambda_margin / 3`,
leaving its median effective ratio at only `0.0380` after the H0-013 tenfold
coefficient. Its 95% bootstrap interval, `[0.0328, 0.0411]`, is entirely below
the predeclared `0.10` scale threshold.

This is not hinge saturation: the median combined active core-hinge fraction
was `0.4844`. Nor did the stronger scalar materially rotate the total update:
the median cosine between the configured-total and non-margin gradients was
`0.9990`.

## Secondary coverage result

The full 256-task candidate bank found the all-train hardest negative inside
the assigned H0-013 batch for only `4.30%` of anchors, 95% bootstrap interval
`[1.95%, 7.03%]`. H0-011 was similar at `3.52%`, and the two checkpoints agreed
on the global-hardest negative identity for `71.09%` of anchors.

Coverage is therefore still poor. The frozen routing rule, however, defines a
coverage-limited result only when scale and conflict conditions are absent.
Because the scale criterion passed cleanly, coverage remains a documented
secondary deficit rather than the authorized H0-014 intervention.

## Decision

LIP-EVAL-032 authorizes one development-only H0-014 intervention in the frozen
family: an explicit core-only objective or an adaptive core-gradient weight.
It does not authorize another blind aggregate scalar increase, multi-seed
replication, functional confirmation, or PROTO-015.

The most identified H0-014 is an explicit core-only contribution with a target
effective norm ratio, while preserving receiver evolution, batch size 16,
examples seen, checkpoint-selection rule, and all data boundaries. Hard-negative
mining should remain unchanged in that experiment so the intervention tests
scale without mixing the secondary coverage mechanism.

## Provenance

- Execution commit: `b334c293899f4afb114537e577fe4cdb0c11619b`.
- Accelerator: NVIDIA L4; CUDA 12.8.
- Peak allocated VRAM: `8,064,463,360` bytes.
- Full evaluation wall time: `68.73` seconds, excluding the earlier cold model download.
- Full run summary SHA-256: `9b08adbd8550bed45fffb3ea0cc8d045da1fbc6f81139bdbaaf2b513d73f9ae7`.
- Batch rows SHA-256: `b0c9d53b668f80d8799f6e9ae0908624d410d96c856d91d8b181de336f7c2fb7`.
- Candidate banks SHA-256: `ca4cb9cd7aec2981b76227265dbe325c77be00dba145014a83f1bf0ca2def7e4`.
