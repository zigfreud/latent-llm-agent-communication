# LIP-EVAL-037 — oracle native-to-packet blend screen

## Status

LIP-EVAL-037 is a frozen, sequential, development-only oracle mechanism screen
on the already open P014 cohort. It follows the EVAL-036 oracle-capacity gate
failure. It does not execute a learned bridge, spend a fresh holdout, authorize
PROTO-015, or upgrade any previous result.

Execution completed on 2026-08-21 and stopped after the 192-row screen. No
tested alpha passed the frozen capacity gate: matched core was `22/32` at
`alpha=0.25`, `23/32` at `alpha=0.50`, and `22/32` at `alpha=0.75`; all three
shuffled conditions were `0/32`, and the constrained prefix realized in every
row. The registered route is `oracle_blend_screen_no_candidate`. Confirmation
was not authorized or executed. See
`experiments/registry/LIP-EVAL-037_oracle_native_packet_blend_screen.md`.

## Question

EVAL-036 forced the task-invariant prefix `def f_0` and eliminated the oracle
binding gap, but full packet replacement recovered only `69/96` rows. The
remaining failure was task-structured.

This screen asks:

> Does the receiver need to retain part of its native residual trajectory in
> order to use the task-specific oracle packet?

At every selected position and receiver block, the intervention is:

```text
h_intervened = (1 - alpha) * h_native + alpha * h_oracle_packet
```

`alpha=0` is the native receiver and `alpha=1` is EVAL-036 full replacement.
Both endpoints are reused from cryptographically registered EVAL-036 artifacts.

## Phase 1 — screen

Evaluate `alpha ∈ {0.25, 0.50, 0.75}` with generation seed 4127:

| Condition | Alphas | Tasks | Rows |
|---|---:|---:|---:|
| oracle blend matched | 3 | 32 | 96 |
| oracle blend shuffled | 3 | 32 | 96 |

An alpha is eligible only if the prefix realizes in 100% of its rows, matched
core recovery is at least 75%, and shuffled core recovery is at most 10%.
Among eligible alphas, select the highest matched core rate; ties select the
smallest alpha.

## Phase 2 — locked confirmation

Only a hardened, hash-bound passing screen can authorize confirmation. The
selected alpha is then evaluated on generation seeds 4241 and 4357, producing
64 matched and 64 shuffled rows. The screen seed is excluded from the
confirmation gate.

Confirmation requires the same 100% prefix, 75% matched-capacity, and 10%
shuffled-specificity thresholds. Passing would restore oracle capacity only
for this bounded development interface; it would not establish a learned
bridge.

## Frozen boundaries

- The receiver prompt, constrained prefix, donor map, oracle packets, packet
  sites, model revisions, sampling settings, and task cohort are unchanged.
- Random seeds are paired across matched, shuffled, and alpha conditions.
- No alpha is selected from learned outputs; learned outputs do not exist here.
- A failed screen stops after 192 rows. A failed confirmation stops after 320.
- All routes remain `claim_eligible=false`.

## Intended Colab execution

Screen generation:

```bash
python -m src.scripts.run_oracle_native_packet_blend_screen \
  --config config/LIP-EVAL-037_oracle_native_packet_blend_screen.yaml \
  --artifact-root /content/drive/MyDrive/lip-artifacts \
  --output /content/drive/MyDrive/lip-artifacts/LIP-EVAL-037/screen-v1/generations.jsonl \
  --phase screen \
  --device cuda \
  --resume
```

The hardened screen evaluation writes to `screen-v1/screen-evaluation` with
`--allow-incomplete`:

```bash
python -m src.scripts.run_hardened_oracle_evaluation \
  --config config/LIP-EVAL-037_oracle_native_packet_blend_screen.yaml \
  --generations /content/drive/MyDrive/lip-artifacts/LIP-EVAL-037/screen-v1/generations.jsonl \
  --output-dir /content/drive/MyDrive/lip-artifacts/LIP-EVAL-037/screen-v1/screen-evaluation \
  --allow-incomplete \
  --overwrite
```

Only if that summary selects an eligible alpha, append the locked confirmation:

```bash
python -m src.scripts.run_oracle_native_packet_blend_screen \
  --config config/LIP-EVAL-037_oracle_native_packet_blend_screen.yaml \
  --artifact-root /content/drive/MyDrive/lip-artifacts \
  --output /content/drive/MyDrive/lip-artifacts/LIP-EVAL-037/screen-v1/generations.jsonl \
  --phase confirm \
  --screen-lock /content/drive/MyDrive/lip-artifacts/LIP-EVAL-037/screen-v1/screen-evaluation/summary.json \
  --device cuda \
  --resume
```

The final hardened evaluation omits `--allow-incomplete` and writes to
`screen-v1/confirmation-evaluation`.
