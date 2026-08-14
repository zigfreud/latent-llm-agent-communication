# LIP-EVAL-034 — alias-normalized functional diagnostic

## Status

LIP-EVAL-034 is a post-hoc, development-only diagnostic over the completed
LIP-EVAL-033 outputs. It cannot rescue, upgrade, or reinterpret the frozen
negative EVAL-033 endpoint. It spends no new generation cohort and makes no
confirmatory claim.

## Question

EVAL-033 produced zero exact declarations of the required task entry point in
both learned conditions. LIP-EVAL-034 asks one narrower question:

> Does an EVAL-033 output contain functional target computation under one
> different generated top-level function name?

This distinguishes a conservative name/readout failure from absence of usable
core computation under the frozen interface. It does not test broader program
repair.

## Frozen source

The diagnostic reuses all 576 completed EVAL-033 rows:

- 32 open P014 tasks;
- `learned_matched` and `learned_shuffled`;
- bridge seeds 4001, 4003, and 4007;
- generation seeds 4127, 4241, and 4357;
- nine fixed replicates per task-condition.

The EVAL-033 generation and metadata SHA-256 values are embedded in the
EVAL-034 config and verified before scoring. The source must remain complete,
non-claim-eligible, and contain zero exact entry-point declarations.

## Frozen normalization policy

A candidate is eligible only when its extracted code:

1. parses as Python; and
2. contains exactly one top-level `FunctionDef` or `AsyncFunctionDef`.

For an eligible candidate, the scorer preserves the original code and appends
only:

```python
expected_entry_point = generated_function_name
```

The scorer does not edit the function body, arguments, tests, control flow, or
self-references. Appending a binding instead of renaming the AST preserves
recursion through the generated name. Zero top-level functions, multiple
top-level functions, and invalid syntax are ineligible and count as functional
failures in the all-row diagnostic. Eligibility rates are reported separately.

All functional tests run through the existing validated Linux namespace
sandbox. Direct functional execution remains opt-in and is not an evidentiary
result.

## Exploratory endpoint and routes

The endpoint is `alias_functional_pass`, clustered by task. The scorer reports
the matched-minus-shuffled task mean, a task bootstrap interval, an exploratory
one-sided sign-flip value, and the fixed bridge-seed differences. These numbers
remain descriptive because the hypothesis was chosen after EVAL-033.

The decision routes are frozen before execution:

- zero matched alias passes: `no_alias_normalized_core_recovery`; move to a
  dynamic or closed-loop trajectory bridge;
- positive matched-minus-shuffled task difference with at least two of three
  bridge seeds positive: `matched_specific_alias_recovery_candidate`; design a
  name/readout mechanism branch without upgrading EVAL-033;
- any other recovery: `non_specific_or_seed_sensitive_alias_recovery`; treat it
  as generic prior, ambiguity, or seed sensitivity and make no name-only claim.

Regardless of route, `claim_eligible=false`, fresh holdout spend is not
authorized, and PROTO-015 execution is not authorized by this diagnostic alone.

## Execution

The diagnostic requires CPU only. On the connected Colab runtime, the intended
command is:

```bash
python -m src.scripts.run_hardened_oracle_evaluation \
  --config config/LIP-EVAL-034_alias_normalized_functional_diagnostic.yaml \
  --generations /content/drive/MyDrive/lip-artifacts/LIP-EVAL-033/full-v1/generations.jsonl \
  --output-dir /content/drive/MyDrive/lip-artifacts/LIP-EVAL-034/diagnostic-v1/evaluation \
  --overwrite
```

The hardened wrapper copies the exact source inputs into its read-only
namespace, verifies their hashes, evaluates eligible aliases as the restricted
candidate UID, and binds the sandbox input hashes into the output summary.
