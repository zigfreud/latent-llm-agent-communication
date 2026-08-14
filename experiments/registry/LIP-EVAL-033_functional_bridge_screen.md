# LIP-EVAL-033 — bounded functional bridge screen

Status: completed development-only screen; **functional signal not detected**.

## Result

The full frozen grid completed all 576 cells. Learned matched and learned
shuffled each produced `0/288` hardened functional passes. The task-clustered
mean difference was exactly `0.0`, its 95% bootstrap interval was
`[0.0, 0.0]`, the exact one-sided sign-flip value was `p=1.0`, and none of the
32 task differences was nonzero.

All three bridge-seed point estimates were zero, so the two-of-three positive
seed guardrail also failed. The result is not a weak or seed-sensitive clue;
the frozen endpoint observed no functional contrast at all.

## Failure signature

Generated text often contained parseable Python: `228/288` matched outputs and
`226/288` shuffled outputs passed syntax. Nevertheless, neither condition ever
declared the exact required task entry point. Entry-point declaration and
functional pass were both `0/288` in both arms.

The already-open P014 controls exclude a general receiver or scorer failure.
On the same cohort, text-only achieved `89/96`, oracle teacher matched achieved
`84/96`, and oracle teacher shuffled achieved `0/96`. The learned path is the
failed link under the tested interface.

## Interpretation

H0-015/H0-016 improved held-out identity geometry, but that geometry did not
become symbolic task identity in the receiver's program when injected as a
single static layer-0 initial condition. This rejects sufficiency under the
frozen interface. It does not prove heterogeneous latent transport impossible
or identify one unique internal cause.

A post-result example generated `word_length` where the required name was
`word_len`. That observation motivates, but does not answer, whether target
computation exists under an alias. The cheapest next design is therefore a
non-claim hardened diagnostic that deterministically renames a single declared
function to the required entry point before testing. It uses the existing
outputs and spends no fresh holdout.

## Decision

- Do not open or execute PROTO-015.
- Do not spend a fresh confirmation cohort.
- Return to the bridge/intervention mechanism.
- Design the alias-normalized diagnostic as a separate post-hoc evaluation;
  never use it to upgrade the frozen EVAL-033 result.

## Provenance

- Execution commit: `fbba79892f84f07936e1daf9d227c8865d2f3439`.
- Accelerator: NVIDIA L4; interrupted after 512 atomic rows and resumed for
  the remaining 64 without duplication.
- Generation SHA-256: `071332b5dc93b933d31a9bf11156f7b3387311689afbbef8b568a9136d4ee84d`.
- Metadata SHA-256: `09f17aef707d4ffa6a2e92a9cc0001c716bc2f2d7863bb36ec7a1a52b4d35c1e`.
- Hardened summary SHA-256: `c499a6e7e0713f43f24c837a0d42b1cfad78aec3f85377b409668a2933ef73d9`.
- Scored JSONL SHA-256: `cc5bdb4eeac8f99a5b6a516a58af404ca3a72abcf15e4c4050975de8e2f0c3c`.
- Sandbox report SHA-256: `c481c955fefd72b9d53edea38b78c3bea7395df30f13d1615e21e35322782989`.
