# LIP-PROTO-007 functional packet-capacity escalation

## Purpose

`LIP-PROTO-005` found that an exact same-task `K=8` suffix packet recovered
predictive likelihood but never produced the required callable entry point.
`LIP-PROTO-006` localized the apparent discrepancy: the first task-specific
function-name token recovers about 49%/25% of the text advantage at `K=8`, but
86%/89% at `K=16` and 90%/95% at `K=32` on the selection/confirmation splits.

`LIP-PROTO-007` tests the resulting preregistered hypothesis: a larger packet
at the same residual interface crosses a functional decision boundary. This is
still a target-oracle interface experiment. It asks whether the interface can
carry actionable task identity before training a source-to-target bridge.

## Frozen design

The final 16 tasks of the immutable 32-task held-out bundle (Python slice
`16:32`) were not used by `LIP-PROTO-004`, `LIP-PROTO-005`, or
`LIP-PROTO-006`. They are reserved here as one claim-eligible evaluation set.
The layer (`-16`), suffix positions, exact `replace` intervention,
length-controlled neutral carrier, model revision, prompt contract, sampling
parameters, and three generation seeds remain fixed.

The full factorial design has eight conditions:

- shared controls: `neutral_no_lip` and `text_only_no_lip`;
- same-task oracle packets at `K = 8, 16, 32`;
- a Sattolo task derangement at each `K`, preserving capacity and intervention
  form while destroying task identity.

The runner captures the largest 32-vector suffix once per task. Each smaller
condition takes the corresponding suffix of that exact capture, so the
capacity comparison does not change layer, task state, or capture pass. The
same effective random seed is reused across all eight conditions within each
task and replicate. The full design contains `16 × 8 × 3 = 384` generations.

## Causal contrasts and decision rule

For each capacity, the same-task packet is compared with both the neutral
carrier and the equal-capacity task-mismatched packet. A capacity is called
task-specific only when its task-clustered functional pass mean is strictly
higher than both. The protocol gate passes when the task-text control has
nonzero capacity and at least one packet size is task-specific. The summary
also reports the smallest supported capacity.

This gate is a design decision, not a substitute for uncertainty. All nine
registered paired contrasts report task-clustered bootstrap intervals, exact
sign-flip tests for 16 tasks, and Holm correction. Raw task/seed outcomes remain
primary if the mean gate and inferential evidence disagree.

Interpretation is frozen as follows:

- `K=16` or `K=32` beats both controls: actionable task identity exists at this
  single-layer residual interface, and bridge training should target the
  smallest supported capacity;
- all matched packets fail while text succeeds: increasing residual suffix
  capacity is insufficient, motivating a soft-prefix or KV-cache interface;
- matched and mismatched packets move together: the effect is packet energy or
  generation perturbation, not transmitted task identity;
- text fails broadly: the task/model/generation budget is not an informative
  functional test and no interface conclusion is claim-eligible.

## Execution

```bash
python -m src.scripts.run_oracle_packet_functional \
  --config config/LIP-PROTO-007_oracle_packet_functional_capacity.yaml \
  --preflight --overwrite

python -m src.scripts.run_oracle_packet_functional \
  --config config/LIP-PROTO-007_oracle_packet_functional_capacity.yaml \
  --overwrite

python -m src.scripts.evaluate_oracle_packet_semantics \
  --config config/LIP-PROTO-007_oracle_packet_functional_capacity.yaml \
  --generations runs/LIP-PROTO-007/generations.jsonl \
  --overwrite

python -m src.scripts.run_hardened_oracle_evaluation \
  --config config/LIP-PROTO-007_oracle_packet_functional_capacity.yaml \
  --generations runs/LIP-PROTO-007/generations.jsonl \
  --output-dir runs/LIP-PROTO-007/functional-evaluation \
  --overwrite
```

Generation and syntax inspection do not execute candidate code. Functional
evaluation must use the previously probed disposable sandbox: private network
and mount namespaces, Drive and credentials masked, UID/GID `nobody`, empty
environment, `no_new_privs`, and per-candidate CPU/memory/time limits. The
resource-limited Python subprocess alone is explicitly not a security
boundary. The versioned hardened entry point stages immutable inputs, enters
private mount/network/IPC/UTS namespaces, masks host data mounts, and keeps
the evaluator privileged only inside that namespace. Every candidate then
drops to UID/GID `nobody`, an allowlisted environment, zero effective
capabilities, and `no_new_privs`. Candidate processes cannot write the staged
source or the evaluator's result directory. A machine-readable probe report
must pass before any generated program executes.

## Result

The complete hardened evaluation was claim-eligible, but the functional gate
failed. The text control passed 20 of 48 task/seed observations across seven of
16 tasks. Every latent and neutral condition passed 0 of 48 observations. This
includes exact same-task target-oracle states at all three registered
capacities; task shuffling therefore cannot explain the failure.

| Condition | K | Functional passes | Task-clustered mean | 95% task-bootstrap CI |
|---|---:|---:|---:|---:|
| Neutral carrier | -- | 0/48 | 0.00% | [0.00%, 0.00%] |
| Matched target oracle | 8 | 0/48 | 0.00% | [0.00%, 0.00%] |
| Task-shuffled oracle | 8 | 0/48 | 0.00% | [0.00%, 0.00%] |
| Matched target oracle | 16 | 0/48 | 0.00% | [0.00%, 0.00%] |
| Task-shuffled oracle | 16 | 0/48 | 0.00% | [0.00%, 0.00%] |
| Matched target oracle | 32 | 0/48 | 0.00% | [0.00%, 0.00%] |
| Task-shuffled oracle | 32 | 0/48 | 0.00% | [0.00%, 0.00%] |
| Task text | -- | 20/48 | 41.67% | [18.75%, 66.67%] |

All eight registered latent contrasts had mean difference zero, exact
two-sided `p=1`, and Holm-adjusted `p=1`. The task-text contrast against the
neutral carrier had mean difference `0.4167`, interval `[0.1875, 0.6667]`,
exact unadjusted `p=0.015625`, and Holm-adjusted `p=0.140625` across the nine
registered comparisons. The conservative multiplicity correction affects the
text-control inference but not the interface diagnosis: there was no latent
success at the observation or task level.

The failure mode was structural. All 288 matched/shuffled oracle generations
raised `NameError` except for five `K=16` generations that instead had syntax
errors. None declared the required entry point. Text generations declared the
entry point in all 48 cases and passed hidden tests in 20. The syntax-only
metric was therefore misleadingly high for the latent conditions while the
callable program contract remained absent.

This result rejects the registered hypothesis that increasing a residual
suffix packet from `K=8` to `K=16` or `K=32` crosses the free-generation
decision boundary. It does **not** reject latent communication in general.
Together with `LIP-PROTO-006`, it identifies a predictive-to-functional gap:
the exact target-model states recover 69--77% of the task-text advantage over
the first eight reference tokens, yet this one-layer residual replacement does
not reliably bootstrap the discrete function identity during autonomous
decoding. The next oracle protocol should change the carrier to a learned soft
prefix or a multi-layer KV-cache intervention before any new source-to-target
bridge training.

Generate the registered comparison figure from the two immutable summaries:

```bash
python -m src.scripts.plot_oracle_functional_capacity \
  --position-summary runs/LIP-PROTO-006/oracle-token-position/summary.json \
  --functional-summary runs/LIP-PROTO-007/functional-evaluation/summary.json
```

Immutable artifact hashes and the Drive path are recorded after archival.
