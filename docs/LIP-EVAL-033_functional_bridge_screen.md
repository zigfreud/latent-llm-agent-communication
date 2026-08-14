# LIP-EVAL-033 — bounded functional bridge screen

Status: completed development-only screen; functional signal not detected.

## Question

Does the identity geometry learned by the H0-015/H0-016 frozen
hard-negative-batch system cause task-specific functional behavior in the
receiver, rather than only geometric separation on the development gate?

## Claim boundary

This is a development evaluation on the already-open 32-task P014 functional
cohort. It cannot be treated as independent confirmation, because P014 already
revealed task-level outcomes and the project subsequently adapted the bridge.
A positive result is evidence for the missing functional property and may
authorize a fresh-cohort PROTO-015 design. It is not itself PROTO-015.

## Frozen systems

Evaluate all three checkpoints; do not select only the development-strong
seeds after observing H0-016:

- H0-016 seed 4001, best step 128;
- H0-016 seed 4003, best step 120;
- frozen H0-015 seed 4007, best step 120.

The 4001 checkpoint remains included even though its complete identity family
missed, because excluding it now would change the estimand from robustness of
the training system to performance of retrospectively selected seeds.

## Conditions and budget

Generate only two new conditions per bridge seed:

1. `learned_matched`;
2. `learned_shuffled`, using the frozen same-stratum derangement.

Reuse the P014 neutral, text, oracle-matched, and oracle-shuffled results only
as descriptive calibration anchors; do not regenerate them or include them in
the primary family. Keep the frozen 32 tasks, generation seeds
`[4127, 4241, 4357]`, derangement seed `4513`, receiver revision, prompt,
sampling parameters, maximum new-token budget, scorer, and sandbox.

The new generation budget is therefore:

`3 bridge seeds × 32 tasks × 3 generation seeds × 2 conditions = 576 cells`.

This is 42.86% of the 1,344-cell P014 functional matrix.

## Primary endpoint

For each task, average the binary hardened functional-pass outcome over all
three bridge seeds and all three generation seeds within each condition. The
single confirmatory statistic inside this development eval is the task-level
mean difference:

`D_i = mean(learned_matched_i) - mean(learned_shuffled_i)`.

Report the mean of `D_i`, a task-bootstrap 95% interval, and a one-sided
sign-flip p-value over the 32 task clusters. The implementation enumerates the
test exactly when at most 20 task differences are nonzero and otherwise uses
the frozen-seed 100,000-draw Monte Carlo form. The primary endpoint passes only
if the mean is positive and the one-sided p-value is at most 0.05.

## Robustness guardrail

Report each bridge seed's task-aggregated matched-minus-shuffled point estimate
without adding it to a post-hoc multiplicity family. At least two of the three
fixed seed estimates must be strictly positive. This prevents a pooled result
from being driven by only one initialization.

## Decisions

- Primary endpoint and guardrail pass: authorize design, not execution, of a
  fresh capability-calibrated PROTO-015 cohort.
- Primary passes but guardrail fails: retain a seed-sensitive functional clue;
  do not open PROTO-015.
- Primary fails: identity geometry is insufficient for functional transport
  under this replay interface; do not spend a fresh holdout and return to the
  bridge/intervention mechanism.

Before execution, the exact input artifact hashes, evaluator reuse boundary,
checkpoint loader, derangement map, output paths, and compute estimate must be
frozen in a versioned config and validated on CPU.

## Result

Execution completed on 2026-08-14 from source commit
`fbba79892f84f07936e1daf9d227c8865d2f3439`. The run generated all 576
frozen cells. A Colab runtime replacement interrupted the first process after
512 atomic JSONL rows; `--resume` recovered those rows without duplication and
generated the remaining 64 on an NVIDIA L4. Final metadata reported
`complete=true` and `claim_eligible=false`.

The independent hardened evaluator completed inside the validated private
Linux namespace. Its summary reported
`execution_mode=functional_hardened_namespace`,
`subprocess_is_security_sandbox=true`, and `claim_eligible=false`.

### Primary endpoint

Both conditions had zero functional passes:

| Condition | Functional passes | Rate |
| --- | ---: | ---: |
| `learned_matched` | 0/288 | 0.0000% |
| `learned_shuffled` | 0/288 | 0.0000% |

The task-clustered matched-minus-shuffled mean difference was exactly `0.0`,
with bootstrap interval `[0.0, 0.0]`, one-sided exact sign-flip `p=1.0`, and
zero nonzero task clusters. The primary endpoint failed.

All three bridge-seed estimates were also exactly zero. Thus zero of the three
seeds were positive, below the frozen two-of-three guardrail, and the final
summary reported `development_functional_signal_detected=false`.

### Failure signature

The zero functional rate was not caused primarily by unparsable text. Syntax
pass rates were `228/288` (`79.1667%`) for learned matched and `226/288`
(`78.4722%`) for learned shuffled. However, no output in either condition
declared the exact task entry point: `0/288` matched and `0/288` shuffled.

This is a sharper localization than a merely underpowered functional
contrast. The H0-015/H0-016 identity geometry did not become the symbolic task
identity required by the receiver's generated program under the frozen
layer-0 initial-condition interface. A post-result example produced
`word_length` for a task whose required entry point was `word_len`; this is an
anecdote motivating an alias-normalized diagnostic, not evidence that task
semantics were transported.

The already-open P014 calibration remains important. On the same cohort,
text-only achieved `89/96`, oracle teacher matched achieved `84/96`, and
oracle teacher shuffled achieved `0/96`. The receiver, prompt family, carrier,
and functional scorer can therefore express and detect task identity. P014's
earlier learned matched and learned shuffled conditions each achieved only
`1/288`; the hardened H0-015/H0-016 systems did not convert their stronger
geometric identity gates into functional transport.

### Interpretation and decision

The frozen decision route is the negative branch: learned identity geometry is
insufficient for functional transport under this replay interface. This
development result does not spend a fresh holdout, authorize PROTO-015, prove
that heterogeneous latent transport is impossible, or isolate one unique
mechanism inside the failed learned path.

The cheapest discriminating next design is a non-claim, post-hoc evaluation on
the already-generated outputs. It would deterministically alpha-rename a
single declared function to the required task entry point inside the same
hardened sandbox, then rerun the functional tests. A matched advantage after
alias normalization would localize failure toward symbolic naming/readout; a
continued zero would localize it toward missing core task computation; matched
and shuffled recovery together would indicate generic code priors rather than
task identity. This diagnostic must remain separate from the frozen EVAL-033
endpoint.

### Artifact record

Canonical artifacts are stored under `lip-artifacts/LIP-EVAL-033/full-v1` on
Drive.

- generation JSONL SHA-256: `071332b5dc93b933d31a9bf11156f7b3387311689afbbef8b568a9136d4ee84d`;
- generation metadata SHA-256: `09f17aef707d4ffa6a2e92a9cc0001c716bc2f2d7863bb36ec7a1a52b4d35c1e`;
- scored JSONL SHA-256: `cc5bdb4eeac8f99a5b6a516a58af404ca3a72abcf15e4c4050975de8e2f0c3c`;
- hardened summary SHA-256: `c499a6e7e0713f43f24c837a0d42b1cfad78aec3f85377b409668a2933ef73d9`;
- sandbox report SHA-256: `c481c955fefd72b9d53edea38b78c3bea7395df30f13d1615e21e35322782989`.
