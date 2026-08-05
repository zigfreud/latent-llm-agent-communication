# LIP-PROTO-013 constant-capacity terminal-source factorial

Status: frozen pre-data protocol; confirmation results pending.

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
  --functional --overwrite

python -m src.scripts.select_oracle_terminal_factorial_tasks \
  --config config/LIP-PROTO-013_terminal_source_factorial.yaml

python -m src.scripts.run_oracle_memory_functional \
  --config config/LIP-PROTO-013_terminal_source_factorial.yaml

python -m src.scripts.run_hardened_oracle_evaluation \
  --config config/LIP-PROTO-013_terminal_source_factorial.yaml \
  --generations runs/LIP-PROTO-013/generations.jsonl \
  --output-dir runs/LIP-PROTO-013/evaluation \
  --functional --overwrite

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
