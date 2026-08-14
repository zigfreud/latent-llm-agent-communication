# LIP-EVAL-033 — bounded functional bridge screen

Status: design authorized by H0-016; execution not yet authorized or frozen.

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

Report the mean of `D_i`, a task-bootstrap 95% interval, and a one-sided exact
sign-flip p-value over the 32 task clusters. The primary endpoint passes only
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
