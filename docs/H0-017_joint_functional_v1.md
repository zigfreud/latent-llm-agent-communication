# H0-017 joint functional diagnostic

The user authorized testing generated answers on 2026-09-24 after accepting
core + name as one unit and distinguishing retrieval from functional success.
This is a new exploratory diagnostic. It does not change the failed historical
H0-017 screen gate or authorize confirmation. No model is trained here.

Freeze the regional checkpoint at step 384 and joint-only checkpoint at step
304, both chosen previously by joint retrieval, margin, RMSE, and earliest step.
Their hashes, the existing 32 development-selection task IDs, and all conditions
are in `config/H0-017_joint_functional_v1.json`. These tasks were used for
checkpoint selection; results cannot establish unseen-task generalization.

Generate one greedy answer per task and condition, capped at 256 new tokens,
using the frozen Llama receiver/revision/4-bit loading and native stop tokens.
Conditions: regional matched/shuffled, joint-only matched/shuffled, native
oracle packet matched/shuffled, full original textual prompt, neutral prompt
without a message. A seeded cyclic permutation reuses every donor exactly once
and never assigns a task to itself. All latent arms retain the full joint
packet and exactly the neutral carrier used during training. No forced prefix,
task-specific name, tests, answer, or textual specification enters these arms.
The textual baseline uses the exact token IDs from original target extraction.

At layers 0..7, learned corrections depend on each live prefill state. Complete
all remaining receiver layers and decode using the corrected KV cache. Apply
no additional corrections during single-token decoding. Before generation,
compare the first task's cached prefill states with the original truncated
training/evaluation implementation, atol/rtol 0.005. Failure stops execution.
This check tests integration, not functional performance or checkpoint choice.

Primary: pass all original MBPP tests without repair. Record syntax and exact
entry-point declarations. Secondary: for syntactically valid outputs containing
exactly one top-level function, append an alias to the expected name and rerun
the same tests. This does not change the body, arguments, control flow, or tests;
it cannot count as a primary success. Use the existing probed Linux namespace
sandbox and unprivileged candidate subprocess policy for every execution.

Persist every response before continuing. Preserve task hashes, donor mapping,
checkpoint/statistics hashes, code commit, output token IDs, truncation flags,
hook audit, and timing. Full grid completion is required for the report. Report
paired matched-only and shuffled-only successes with denominators; no new gate
or post-hoc checkpoint selection follows from this diagnostic.

Timing includes source-packet encoding, corrections and receiver generation,
but the source model's extraction is cached. It excludes model loading and
cannot establish end-to-end cost savings. The float32 32x512 code itself uses
65,536 bytes; compression relative to hidden-state packets is not automatically
compression relative to text. Baselines also have different prompt lengths.
