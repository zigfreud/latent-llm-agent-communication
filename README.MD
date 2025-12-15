<div align="center">

# 🧬 LIP — Latent Injection Protocol

### Why speak in JSON when models can think in vectors?

</div>

---

## Status

**Active Research Prototype (Gen6 — stabilization phase)**

This repository contains **experimental research code**.  
Architectures, APIs, and results are subject to change.

The project is currently transitioning from **Gen4 → Gen6**, with recent
breakthroughs in **correct hidden-state injection mechanics** and **deterministic
AB validation**.

---

## 🚀 The Thesis

> *“The field is focused on organizing LLMs via orchestration frameworks.
But little attention is given to teaching models to communicate in an efficient,
native language.”*

Modern Multi-Agent Systems (MAS) suffer from what we call the **Dinner Table Paradox**:
highly capable agents share the same environment, yet are forced to communicate
through **human-centric channels** (text / JSON).

This introduces structural bottlenecks:

- **Latency** — repeated tokenization and decoding waste compute.
- **Bandwidth** — large contexts (e.g., RAG) saturate communication channels.
- **Leakage risk** — semantic intent is flattened into plaintext.
- **Semantic loss** — rich latent structure is discarded at every hop.

---

## ✨ Core Idea — Direct Vector-to-Vector Communication

LIP explores **direct latent communication between heterogeneous LLMs**.

Instead of exchanging tokens, a source model transmits **semantic vectors**
that are translated and injected directly into the internal computation of
a target model.

Example setup:

- **Edge / Source:** TinyLlama or DeepSeek-Coder (2048d)
- **Cloud / Target:** Llama-3-8B (4096d)
- **Bridge:** Asymmetric adapter with a bottleneck (e.g., 2048 → 512 → 4096)

This enables **functional steering** of the target model **without textual exchange**.

---

## ⚡ What Works (Gen6)

- ✅ **Heterogeneous latent translation** (2048d → 4096d)
- ✅ **Contrastive alignment (InfoNCE-style)** outperforms MSE in high-dimensional spaces
- ✅ **Correct hidden-state injection** (not input embeddings)
- ✅ Stable injection via **forward hooks** into the **penultimate transformer layer (-2)**
- ✅ **Deterministic AB testing** confirms non-placebo steering
- ✅ Functional control demonstrated in code-generation tasks
  (e.g., explanation → implementation)

---

## 🔬 Key Technical Decisions (Empirically Validated)

- **Injection target:** Hidden states (never input embeddings)
- **Layer:** Penultimate transformer layer (-2)
- **Spatial focus:** Final token position
- **Temporal scope:** Single injection at initial forward pass
- **Calibration:** Norm-based energy matching relative to target embeddings

These choices were validated through ablations on layer, position, and gain.

---

## ⚠️ Known Limitations

- ❌ Fine-grained intent transfer remains inconsistent  
  (vectors often encode *mode/style* better than precise constraints)
- ❌ Dataset currently aligned to final-layer targets while injection occurs at layer -2
  (being corrected)
- ❌ Evaluation is limited to controlled prompts
- ❌ No standardized benchmarks yet
- ❌ 8B inference is **cloud-only** for stability reasons

These limitations are explicitly acknowledged and align with open problems
identified in related work.

---

## 🧪 Reproducibility

This repository contains a **script-based, config-driven pipeline** (no notebooks):

src/
core/ # models, losses
pipelines/ # extract, train, infer
integrations/ # injection hooks
configs/ # yaml experiment configs
datasets/ # sharded latent pairs
checkpoints/ # adapter weights
scripts/ # CLI entry points


Experiments are designed to be **deterministic and inspectable**.

---

## 🛣️ Roadmap (Near-Term)

- Align dataset mining with **layer-specific injection**
- Add **quantitative retrieval metrics** (top-k, cosine vs baseline)
- Ablations on:
  - injection layer
  - token position
  - gain / blending strength
- Small-scale benchmark tasks (code, reasoning, structured output)

---

## 🔐 Privacy & Security (Research Scope)

LIP currently provides **obfuscation by design**:
vectors are non-human-readable and reduce accidental leakage.

Planned research phases:

1. **Current:** Obfuscation by design ✅  
2. **Near-term:** Secure vector transport (TLS-like protocol for packets)  
3. **Long-term:** Partial homomorphic routing (research-only, limited scope)

No cryptographic security claims are made at this stage.

---

## 🤝 Contribution Policy

This project is **not yet open for general contributions**.

Feedback, inspection, and discussion are welcome.
Targeted collaboration may be enabled once the Gen6 pipeline is fully stabilized.

---

## 📎 Disclaimer

This is **research code**, not a production library.

Expect breaking changes.

---

<div align="center">

_An Independent Research Project by  
[Cristiano Silva](https://www.linkedin.com/in/cristiano-silva-a2a084204/)_

</div>

---

## 📚 Citation

If you reference this work:

Silva, Cristiano. "LIP — Latent Injection Protocol". 2025.
GitHub repository: https://github.com/zigfreud/latent-llm-agent-communication