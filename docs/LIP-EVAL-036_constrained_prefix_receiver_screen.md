# LIP-EVAL-036 — constrained-prefix receiver screen

## Status

LIP-EVAL-036 is a frozen, sequential, development-only mechanism screen on the
already open P014 cohort. It follows the EVAL-035 oracle-capacity failure and
does not authorize PROTO-015, a fresh holdout, or an upgrade of EVAL-033/034/035.

Execution completed on 2026-08-18 and stopped after the control phase. The
registered route is `constrained_prefix_oracle_capacity_failure`: prefix
realization was 288/288, shuffled oracle was 0/96, no packet was 1/96, and
matched-oracle core recovery was `69/96 = 71.875%`, three rows below the frozen
75% capacity gate. The learned phase was not authorized or executed. See
`experiments/registry/LIP-EVAL-036_constrained_prefix_receiver_screen.md`.

## Question

EVAL-035 placed the constant opaque name `f_0` outside the overwritten prompt
suffix. The no-packet receiver declared that name in 96/96 rows, but the
matched oracle declared it in only 11/96 rows and produced 62 binding gaps.
This shows that positional separation at input does not keep the symbolic and
semantic channels independent through decoding.

LIP-EVAL-036 asks one narrower question:

> If the decoder is constrained to begin with the task-invariant prefix
> `def f_0`, does the frozen receiver regain oracle capacity, and—only if it
> does—does the learned matched signal survive as exact executable behavior?

The prefix contains no task text, argument list, body, test, original function
name, or donor identity. Only the fixed lexical binding is constrained.

## Sequential design

### Phase 1 — controls

Generate exactly 288 rows:

| Condition | Tasks | Generation seeds | Rows |
|---|---:|---:|---:|
| canonical no packet | 32 | 3 | 96 |
| oracle teacher matched | 32 | 3 | 96 |
| oracle teacher shuffled | 32 | 3 | 96 |

The learned phase remains locked unless hardened functional scoring verifies:

- forced prefix realization is 100%;
- matched-oracle core recovery is at least 75%;
- shuffled-oracle core recovery is at most 10%;
- no-packet core recovery is at most 10%.

The control summary cryptographically binds the config, generations, and
metadata. The learned runner refuses to append rows unless that exact hardened
summary passes all gates.

### Phase 2 — learned

If and only if phase 1 passes, append exactly 576 rows:

| Condition | Tasks | Generation seeds | Bridge seeds | Rows |
|---|---:|---:|---:|---:|
| learned matched | 32 | 3 | 3 | 288 |
| learned shuffled | 32 | 3 | 3 | 288 |

No hyperparameter is selected from learned results. Packet sites, checkpoints,
model revisions, donor map, prompts, seeds, sampling, and full-replacement gain
remain frozen from EVAL-035.

## Endpoints and decisions

The exact and core endpoints remain the EVAL-035 endpoints. The primary
contrast is task-clustered learned matched minus learned shuffled, with the
same one-sided exact sign-flip statistic and two-of-three bridge-seed
guardrail.

- Control failure stops the experiment after 288 rows. The next candidate is
  an oracle-only native-to-packet blend/gain sweep.
- Valid controls plus an exact learned signal support a constrained-binding
  learned-transport candidate on the open cohort; the next problem becomes
  coverage through a trajectory extension.
- Valid controls plus core-only recovery trigger an audit of recursive and
  multi-function binding failures.
- Valid controls with no learned signal prioritize a dynamic closed-loop
  trajectory bridge.

All routes remain `claim_eligible=false`.

## Intended Colab execution

Control generation:

```bash
python -m src.scripts.run_constrained_prefix_receiver_screen \
  --config config/LIP-EVAL-036_constrained_prefix_receiver_screen.yaml \
  --artifact-root /content/drive/MyDrive/lip-artifacts \
  --output /content/drive/MyDrive/lip-artifacts/LIP-EVAL-036/screen-v1/generations.jsonl \
  --phase controls \
  --device cuda \
  --prediction-batch-size 4 \
  --resume
```

Hardened control scoring:

```bash
python -m src.scripts.run_hardened_oracle_evaluation \
  --config config/LIP-EVAL-036_constrained_prefix_receiver_screen.yaml \
  --generations /content/drive/MyDrive/lip-artifacts/LIP-EVAL-036/screen-v1/generations.jsonl \
  --output-dir /content/drive/MyDrive/lip-artifacts/LIP-EVAL-036/screen-v1/control-evaluation \
  --allow-incomplete \
  --overwrite
```

Locked learned generation, only after a passing control summary:

```bash
python -m src.scripts.run_constrained_prefix_receiver_screen \
  --config config/LIP-EVAL-036_constrained_prefix_receiver_screen.yaml \
  --artifact-root /content/drive/MyDrive/lip-artifacts \
  --output /content/drive/MyDrive/lip-artifacts/LIP-EVAL-036/screen-v1/generations.jsonl \
  --phase learned \
  --control-lock /content/drive/MyDrive/lip-artifacts/LIP-EVAL-036/screen-v1/control-evaluation/summary.json \
  --device cuda \
  --prediction-batch-size 4 \
  --resume
```

The final hardened evaluation omits `--allow-incomplete` and writes to
`screen-v1/functional-evaluation`.
