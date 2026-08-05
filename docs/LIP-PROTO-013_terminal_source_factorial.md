# LIP-PROTO-013 constant-capacity terminal-source factorial

Status: completed preregistered execution; confirmatory result reported below.

## Research question

`LIP-PROTO-012` showed that removing the final prompt octet can be more damaging
than its low state energy suggests. That result is informative, but deletion
changes both semantic content and packet capacity. `LIP-PROTO-013` asks the
sharper causal question:

> When all 24 replay positions remain populated, which terminal components must
> carry the target task's identity for latent communication to work?

The intervention keeps the packet size, replay positions, replay layers,
carrier, model, and number of injected scalars constant. It changes only the
task identity from which each component was captured.

This protocol does not test a learned inter-model bridge. It remains an oracle
replay experiment inside the target model. Its role in the larger thesis is to
identify a compact and causally useful message structure before asking a learned
protocol to reproduce it.

## Why the preceding result is counterintuitive

The Euclidean norm of a state vector measures its magnitude:

\[
\lVert h \rVert_2 = \sqrt{h_1^2 + h_2^2 + \cdots + h_d^2}.
\]

Magnitude is not the same as information. A small vector can point in a highly
diagnostic direction, while a large vector can mostly contain task-invariant
structure. A simple analogy is a compass: the needle's displacement is small,
but its direction can determine an entire route.

For task communication, a useful first-order decomposition is

\[
h_{t,p,\ell} = \mu_{p,\ell} + \delta_{t,p,\ell} + \varepsilon_{t,p,\ell},
\]

where:

- \(\mu_{p,\ell}\) is structure shared by tasks at position \(p\) and layer
  \(\ell\);
- \(\delta_{t,p,\ell}\) is the task-specific displacement;
- \(\varepsilon_{t,p,\ell}\) is residual variation.

A position may have a modest total norm but a reliable
\(\delta_{t,p,\ell}\). Conversely, a high-energy position may be dominated by
\(\mu_{p,\ell}\), which is large but not useful for identifying the task.
This is why energy diagnostics and causal interventions answer different
questions.

## Audited task population

The tokenizer audit used the exact generator path:

1. render the shared chat template with `tokenize=False`;
2. tokenize that rendered string with `add_special_tokens=False`.

The audit is bound to:

- source commit `d9f44c3d4526a46f389dd61f8e13ff784f4c145d`;
- target/tokenizer revision
  `53346005fb0ef11d3b6a83b12c895cca40156b6c`;
- preflight SHA-256
  `07e6098f567e21654d04fecbd7d665cd7f24fce467ea97bad93917b78b57a416`.

The MBPP test split contains 500 tasks. Earlier registries account for 242
unique tasks, leaving 258 never materialized by this project. Among those 258,
179 have one of two terminal layouts:

| Stratum | Structural pool | Core | Function name | Boundary |
| --- | ---: | --- | --- | --- |
| two name tokens | 83 | `-24…-9` (16) | `-8,-7` (2) | `-6…-1` (6) |
| three name tokens | 96 | `-24…-10` (15) | `-9,-8,-7` (3) | `-6…-1` (6) |

Each row contains exactly 24 positions. The changing core/name boundary is
necessary: the name occupies two tokens in one stratum and three in the other.
Within a stratum, component positions are identical across tasks.

The final confirmation registry contains 16 text-capable tasks from each
stratum, for 32 tasks total. Screening uses only text generation, so no latent
result influences selection. Candidates are ordered before screening by a
SHA-256 key within each tokenizer stratum.

## The `2 × 2 × 2` source factorial

Let the three components be:

- \(C\): terminal core;
- \(N\): required function-name tokens;
- \(B\): fixed generation-boundary tokens.

For each component, the source has two levels:

- `M` — matched: states captured from the target task;
- `S` — shuffled: states captured from one same-stratum donor task.

Every `S` component in a record comes from the same donor. This preserves the
donor's internal coherence and avoids constructing an artificial packet from
three unrelated tasks. Donors are assigned independently inside each 16-task
stratum by a Sattolo derangement; no task can donate to itself.

The eight K=24 conditions are:

| Code | Core | Name | Boundary | Interpretation |
| --- | --- | --- | --- | --- |
| `MMM` | target | target | target | fully matched K=24 |
| `SMM` | donor | target | target | target terminal tail only |
| `MSM` | target | donor | target | replace name identity |
| `MMS` | target | target | donor | replace boundary identity |
| `SSM` | donor | donor | target | boundary-only target identity |
| `SMS` | donor | target | donor | name-only target identity |
| `MSS` | target | donor | donor | core-only target identity |
| `SSS` | donor | donor | donor | fully shuffled K=24 |

Two K=32 replication controls and the neutral/text controls produce 12 total
conditions. With 32 tasks and three fresh generation seeds, confirmation has

\[
32 \times 12 \times 3 = 1152
\]

records. Screening has

\[
179 \times 2 = 358
\]

text-only records.

## What “constant capacity” means

For all eight factorial conditions:

- 24 prompt-suffix positions are replayed;
- the positions are exactly `-24…-1`;
- eight early decoder-block inputs are intervened on;
- the hidden-state dimensionality is unchanged;
- the number of injected scalars is unchanged.

Vector norms are recorded but are not forcibly equalized. Rescaling a donor
state to match a target norm would itself be a second intervention and could
distort semantic direction. The primary control is therefore equal structural
capacity, not artificial norm equality.

## Confirmatory hypotheses

The analysis has two ordered replication gates.

1. matched K=32 must beat shuffled K=32;
2. if gate 1 passes, `MMM` must beat `SSS` at K=24.

Only if both gates reject does one Holm-adjusted family of seven component
hypotheses open.

### Contribution contrasts

These ask whether replacing one matched component damages performance:

\[
\begin{aligned}
H_C &: Y_{MMM} > Y_{SMM},\\
H_N &: Y_{MMM} > Y_{MSM},\\
H_B &: Y_{MMM} > Y_{MMS}.
\end{aligned}
\]

For example, \(H_B\) isolates boundary identity: core and function name remain
matched on both sides, and only the boundary's source task changes.

### Sufficiency or rescue contrasts

These ask whether returning selected target identity improves a fully donor
packet:

\[
\begin{aligned}
H_{C\text{-only}} &: Y_{MSS} > Y_{SSS},\\
H_{N\text{-only}} &: Y_{SMS} > Y_{SSS},\\
H_{B\text{-only}} &: Y_{SSM} > Y_{SSS},\\
H_{tail} &: Y_{SMM} > Y_{SSS}.
\end{aligned}
\]

All seven hypotheses share one Holm family. They are grouped conceptually in
the paper, but not split statistically to obtain easier thresholds.

## Statistical unit and exact test intuition

Each task is one statistical unit. The three stochastic generations are first
averaged within task. For a contrast \(A-B\), task \(i\) contributes

\[
d_i = \overline{Y}_{i,A} - \overline{Y}_{i,B}.
\]

The sign-flip test asks what mean differences would be possible if the signs of
the \(d_i\) values were exchangeable under the null. With \(n\) nonzero tasks,
there are \(2^n\) sign assignments. If all eight nonzero task effects point in
the predicted direction, the most extreme one-sided probability is

\[
2^{-8} = 0.00390625.
\]

The smallest Holm threshold in a seven-test family is

\[
0.05 / 7 \approx 0.00714.
\]

Thus eight perfectly consistent nonzero task effects can cross the strictest
family threshold. This is a resolution statement, not a promise of power:
effect prevalence and magnitude remain empirical.

Bootstrap intervals resample tasks, not individual generations. This preserves
the cluster structure and prevents the three seeds from being mistaken for
three independent tasks.

## Decision semantics

The protocol supports terminal source attribution only when:

1. the text control has nonzero functional success;
2. both replication gates reject in the predicted direction;
3. at least one of the seven Holm-adjusted component contrasts rejects.

Interpretation is bounded:

- a contribution result says matched identity in that component matters in this
  oracle replay setting;
- a rescue result says that component's target identity can improve an otherwise
  donor packet;
- neither result alone proves that a learned encoder can produce the states;
- failure can mean the chosen decomposition is wrong, identity is distributed,
  the effect is weak, or K=24 is insufficient for the sampled tasks.

## Reproducible execution order

No confirmation GPU work begins before the registry and screening artifacts are
sealed.

```bash
python -m src.scripts.materialize_oracle_terminal_candidates \
  --config config/LIP-PROTO-013_terminal_source_factorial.yaml

python -m src.scripts.run_oracle_terminal_factorial_screen \
  --config config/LIP-PROTO-013_terminal_source_factorial.yaml

python -m src.scripts.run_hardened_oracle_evaluation \
  --config config/LIP-PROTO-013_terminal_source_factorial.yaml \
  --generations runs/LIP-PROTO-013/screening/generations.jsonl \
  --output-dir runs/LIP-PROTO-013/screening/functional-evaluation \
  --overwrite

python -m src.scripts.select_oracle_terminal_factorial_tasks \
  --config config/LIP-PROTO-013_terminal_source_factorial.yaml

python -m src.scripts.run_oracle_memory_functional \
  --config config/LIP-PROTO-013_terminal_source_factorial.yaml

python -m src.scripts.run_hardened_oracle_evaluation \
  --config config/LIP-PROTO-013_terminal_source_factorial.yaml \
  --generations runs/LIP-PROTO-013/generations.jsonl \
  --output-dir runs/LIP-PROTO-013/evaluation \
  --overwrite

python -m src.scripts.plot_oracle_terminal_factorial \
  --summary runs/LIP-PROTO-013/evaluation/summary.json \
  --output-stem paper/figures/LIP-PROTO-013_terminal_source_factorial
```

Use a standard T4. The expected workload is about 1,510 generations and roughly
five hours under the observed `LIP-PROTO-012` throughput. A premium GPU is not
part of the frozen design and is not justified for this run.

## Paper-facing figure

The figure script produces:

1. functional pass rates for all eight K=24 factorial vertices;
2. a forest plot for the seven task-clustered confirmatory contrasts, including
   Holm-adjusted p-values or a closed-gate annotation.

The SVG/PDF outputs are the paper artifacts; PNG is a review convenience.

## Pre-registered interpretation table

| Result pattern | Preferred interpretation |
| --- | --- |
| K=32 gate fails | replication failure; stop component claims |
| K=32 passes, K=24 fails | the terminal K=24 packet does not retain confirmed identity |
| boundary contribution only | fixed visible boundary positions carry task-specific causal signal |
| name contribution/rescue | function identity is locally encoded in its prompt-token states |
| core contribution/rescue | task information is distributed through the terminal instruction context |
| tail-only rescue | name plus boundary can restore information absent from donor core |
| all component tests fail after both gates | identity is redundant, interactive, or too diffuse for main-effect contrasts |
| several contribution and rescue tests pass | mixed distributed code with multiple causally useful subchannels |

Pairwise and three-way factorial interactions may be reported descriptively,
but they are not promoted to confirmatory claims in this protocol.

## Result

Execution completed on 2026-08-05. The structural screen produced the complete
358-record grid for 179 candidates and two generation seeds. Under the frozen
`any_functional_pass_across_screening_seeds` rule, 35 tasks in the two-token
function-name stratum and 32 tasks in the three-token stratum were eligible.
The selector took the first 16 tasks in the frozen hash order from each stratum,
yielding the preregistered balanced 32-task confirmation population. The
selection report passed, and the final registry and manifest hashes were
`01442bfec82a12c3dde6bde4eb72a1c5ae28682a65ff1c35b5e1f09580b5c9f9`
and `d9d1b211b5c1fcf97520e85fa80a5633c4a2511648be02712b8e4918989f38d3`.

Confirmation produced all 1,152 registered records: 32 tasks, 12 conditions,
and three generation seeds. The frozen self-replay checks had zero maximum
absolute logit error for both `full_k32` and `terminal_k24`. Hardened scoring
reported no missing cells, a validated Linux namespace sandbox,
`claim_eligible=true`, `semantic_transport_supported=true`, and a passing
terminal-source semantic gate.

Task-clustered functional results were:

| Condition | Component identity | Passes | Rate | 95% task-bootstrap interval |
| --- | --- | ---: | ---: | ---: |
| Neutral carrier | no latent packet | 0/96 | 0.00% | [0.00%, 0.00%] |
| Task text | textual task prompt | 87/96 | 90.62% | [80.21%, 98.96%] |
| Full `K=32` matched | matched full packet | 85/96 | 88.54% | [78.12%, 96.88%] |
| Full `K=32` shuffled | same-stratum donor packet | 0/96 | 0.00% | [0.00%, 0.00%] |
| `MMM` | matched core, name, boundary | 79/96 | 82.29% | [68.75%, 93.75%] |
| `SMM` | donor core; matched name, boundary | 3/96 | 3.12% | [0.00%, 9.38%] |
| `MSM` | matched core, boundary; donor name | 0/96 | 0.00% | [0.00%, 0.00%] |
| `MMS` | matched core, name; donor boundary | 78/96 | 81.25% | [68.75%, 93.75%] |
| `SSM` | matched boundary only | 0/96 | 0.00% | [0.00%, 0.00%] |
| `SMS` | matched name only | 4/96 | 4.17% | [0.00%, 11.46%] |
| `MSS` | matched core only | 0/96 | 0.00% | [0.00%, 0.00%] |
| `SSS` | donor core, name, boundary | 0/96 | 0.00% | [0.00%, 0.00%] |

Both ordered replication gates rejected. Full-`K=32` matched-minus-shuffled
replay had mean task-level difference `0.885417`, 30 nonzero task clusters,
interval `[0.781250, 0.968750]`, and one-sided Monte Carlo
`p=0.0000099999`. Terminal-`K=24` `MMM`-minus-`SSS` replay had mean difference
`0.822917`, 27 nonzero task clusters, interval `[0.687500, 0.937500]`, and the
same one-sided Monte Carlo p-value. The known full packet therefore replicated,
and the constant-capacity terminal packet independently transmitted task
identity above its all-donor control.

The opened seven-test family produced two Holm-confirmed component claims:

| Contrast | Mean difference | 95% interval | Raw one-sided `p` | Holm `p` | Confirmed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Core contribution: `MMM - SMM` | 0.791667 | [0.645833, 0.916667] | 0.0000099999 | 0.0000699993 | yes |
| Function-name contribution: `MMM - MSM` | 0.822917 | [0.687500, 0.937500] | 0.0000099999 | 0.0000699993 | yes |
| Boundary contribution: `MMM - MMS` | 0.010417 | [0.000000, 0.031250] | 0.5 | 1.0 | no |
| Core-only rescue: `MSS - SSS` | 0.000000 | [0.000000, 0.000000] | 1.0 | 1.0 | no |
| Name-only rescue: `SMS - SSS` | 0.041667 | [0.000000, 0.114583] | 0.25 | 1.0 | no |
| Boundary-only rescue: `SSM - SSS` | 0.000000 | [0.000000, 0.000000] | 1.0 | 1.0 | no |
| Name-plus-boundary rescue: `SMM - SSS` | 0.031250 | [0.000000, 0.093750] | 0.5 | 1.0 | no |

Core and function-name identity are therefore independently useful in the
otherwise matched packet. The observed factorial surface is also strongly
conjunctive. Replacing only the name reduced functional success from 82.29% to
0.00%; replacing only the core reduced it to 3.12%. Conversely, neither the
matched core nor the matched name rescued an otherwise donor packet. Matched
core plus matched name with a donor boundary (`MMS`) retained an observed
81.25% rate, only one success below `MMM`. The protocol did not preregister
equivalence or non-inferiority, so the failed boundary-contribution test and
the near-equal point estimates do not prove that the boundary is universally
irrelevant. They show that this experiment detected no positive boundary
contribution conditional on matched core and name.

The result resolves the apparent terminal-octet paradox from `LIP-PROTO-012`.
That experiment deleted offsets `-8 ... -1` together: two task-dependent
function-name positions and six visible task-invariant generation-boundary
positions. The present constant-capacity intervention separates their source
identities. Performance collapsed when name identity was wrong but remained
nearly unchanged when only boundary identity was wrong. On this new
confirmation population, the earlier deletion effect is therefore best
attributed to removing the name-bearing subchannel, not to strong evidence that
the six fixed boundary positions themselves carry the task identity required
here.

The pattern was stable across generation seeds. Full-`K=32` matched rates were
93.75%, 87.50%, and 84.38% for seeds 1667, 1789, and 1901. Terminal-`K=24`
`MMM` rates were 81.25%, 84.38%, and 81.25%; `MMS` was 81.25% under every
seed. Text scored 87.50%, 93.75%, and 90.62%. Neutral, full-`K=32` shuffled,
`MSM`, `MSS`, `SSM`, and `SSS` remained at 0.00% under every seed.

### Exploratory state geometry

The aggregate state diagnostics are descriptive and do not replace the
matched-source interventions. At the eight replay-layer inputs, the two common
function-name offsets `-8, -7` had mean task-signal fraction `0.7050`, mean
pairwise cosine `0.2737`, effective-rank fraction `0.8323`, and mean residual
norm `2.3281`. The common core offsets `-24 ... -10` had corresponding values
`0.5875`, `0.4027`, `0.4337`, and `2.2025`. The six boundary offsets
`-6 ... -1` had task-signal fraction only `0.0297` and pairwise cosine
`0.9696`, despite a comparable mean norm of `1.9974`. Offset `-9` is a mixed
stratum transition--core for two-token names and name for three-token names--
and is not assigned to either common-region aggregate.

Thus the causal and geometric views agree after resolving the terminal block:
name and core states vary substantially with task identity, while the fixed
boundary states are dominated by shared structure. The similar vector norms
again show why magnitude alone is not an information or causal-capacity score.

The supported claim remains bounded. For this model revision, prompt template,
capability-screened MBPP population, native same-model oracle source, and
replay through the first eight decoder blocks, a 24-position terminal packet
transmitted task identity above an equal-capacity same-stratum donor control.
Within that packet, both core and function-name source identity made confirmed
positive contributions. This is the smallest packet confirmed among the sizes
tested, not a mathematical minimum. The result does not establish arbitrary
task coverage, non-inferiority to text, a learned sender-to-receiver bridge,
cross-model interoperability, or textual replacement in an end-to-end agent
system.

The canonical artifact is stored at `lip-artifacts/LIP-PROTO-013` on Drive.
Its manifest binds 23 scientific payloads: source commit, frozen config,
preflight layout audit, candidate and selected registries, screening records
and hardened scores, confirmation generations and metadata, aggregate state
diagnostics, hardened confirmation scores, and raster/vector/PDF figures.
Operational logs are retained beside the artifact but excluded from the
scientific manifest. All 23 entries passed `sha256sum -c` after final rendering,
and the normalized folder was independently visible through the authenticated
Drive API.

- `SOURCE_COMMIT.txt`: `a288613e6c65c8c4dc9d48d360274d35ea6cd219`
- `SHA256SUMS`: `0e425b037b913d5fa11d26c6f12f5c17280466ab77c339bdde309988b6d138c6`
- `confirmation/generations.jsonl`: `8ba18bf74e112cf6b43adf3b86ccd715540c5f907b0da7ba5c2272603e2bd7b5`
- `confirmation/evaluation/summary.json`: `5970b00550807af336c856427c5c39b1a89278b1f8f7e84eb86cfd02be28a3ef`
- `confirmation/state-diagnostics.json`: `b48ad459500babd12bdc9eb2c06326c4d3e8e99e760b68c35c95384da8a2f456`
- `confirmation/figures/LIP-PROTO-013_terminal_source_factorial.svg`: `a8971ea765e1ef66a54ae76e1638f6843d9437102a0b03b1c3b3d942eb7b7a8e`
- `screening/functional-evaluation/summary.json`: `8c72542d45b219fd3df96ce630a93c9d55c26a6bc4eefd641ede43fc24370eb7`
