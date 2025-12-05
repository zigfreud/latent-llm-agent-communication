<div align="center">

# 🧬 LIP: The Latent Injection Protocol

### Why speak in JSON when you can think in Vectors?

</div>

---

### 🚀 The Thesis

> "The whole world is asking how to organize LLMs efficiently. But why is no one talking about teaching them to speak an efficient language?"

Current Multi-Agent Systems (MAS) suffer from the **"Dinner Table Paradox"**: brilliant AI agents sit at the same table but are forced to communicate by passing handwritten notes (text/JSON). This legacy, human-centric interface is slow, insecure, and inefficient.

This process introduces critical bottlenecks:

- **Latency**: Endless tokenization and decoding cycles waste massive GPU/TPU time.
- **Insecurity**: Sensitive data travels as plaintext, vulnerable to interception and leakage.
- **Inefficiency**: High-bandwidth context from techniques like RAG clogs the network.

### ✨ The Solution: Direct Vector-to-Vector Communication

LIP proposes a paradigm shift: **Direct Vector-to-Vector (V2V) Communication**.

By establishing a _semantic bridge_ between different models (e.g., a `TinyLlama-1.1B` at the Edge and a `Llama-3-8B` in the Cloud), we enable direct functional control and code generation without exchanging a single word of natural language. It's like telepathy for LLMs.

---

### ⚡ Key Features

🔌 **Heterogeneous Bridging**: Connects models with different architectures and latent space dimensions (e.g., 2048d → 512d → 4096d).

🔒 **Obfuscation by Design**: Transmits abstract mathematical intent, not human-readable text, making communication inherently private.

⚖️ **Energy Matching**: Uses physics-based calibration to ensure the injected vector is stable and coherent within the target model's latent space.

📉 **Zero-Overhead Receiver**: The server-side model skips the expensive "Reading" (Prefill) phase entirely, receiving pre-digested "thought vectors" ready for processing.

---

### 🛠️ Get Started in 2 Steps

#### 1. Clone the Lab

```bash
git clone https://github.com/zigfreud/latent-llm-agent-communication.git
cd latent-llm-agent-communication
pip install -r requirements.txt
```

#### 2. Run the Telepathy Demo

Watch a 1.1B parameter model control an 8B parameter model using only vectors.

```bash
# Ensure you have the trained adapter located in the datasets/ directory
python src/lip_integration.py
```

---

### 🔬 Reproducibility

This repository contains the full scientific pipeline to reproduce our results.

- `experiments/data_factory_v3.py`: Mines instruction-following vector pairs from the dataset.
- `experiments/train.py`: Trains the Dual-Encoder Adapter that bridges the two models.
- `experiments/benchmarks/`: Contains scripts to generate the convergence and stability graphs.
- `experiments/extract_embeddings.py`: Contain a function to extract the embedding layer in case you wants to use anoter LLM.

For a deep dive into the mathematical proofs and ablation studies, please see the full paper in the `/paper` directory.

---

<div align="center">

_An Independent Research Project by [Cristiano Silva](https://www.linkedin.com/in/cristiano-silva-a2a084204/?locale=en-US)_

</div>

---

### 📚 Citation

If you use LIP in a paper, project, or derivative work, please cite this repository so others can find the original research.

Suggested citation (plain text):

Silva, Cristiano. "LIP: The Latent Injection Protocol". 2025. GitHub repository. https://github.com/zigfreud/latent-llm-agent-communication

BibTeX example:

```bibtex
@misc{silva2025lip,
  title        = {LIP: The Latent Injection Protocol},
  author       = {Silva, Cristiano},
  year         = {2025},
  howpublished = {GitHub repository, \url{https://github.com/zigfreud/latent-llm-agent-communication}},
  note         = {Accessed: 2025-11-27}
}
```

---

### 🛣️ Roadmap & Future Work

LIP is an active research project. Below is a succinct, realistic roadmap capturing short, medium, and long-term goals — technical improvements, packaging, and deployment plans.

Key release goals:

- First protocol release: include YAML configuration templates, best-practice example pipelines, and pre-trained adapter weights for widely used open-source models where licensing allows (e.g., TinyLlama, community LLMs).
- Publish the LIP Python package to PyPI for easy integration and experimentation.

Encryption & privacy roadmap (three phases):

1. Phase 1 — Current (Obfuscation by Design) ✅

   - Vectors are intentionally non-human-readable by design — this reduces accidental leaks and limits casual inspection.
   - This phase provides an immediate privacy and safety improvement with low operational cost.

2. Phase 2 — Near-term (Vector Transport Encryption) 🔒

   - Engineering a secure transport layer for vector packets — a TLS-like mechanism at the LIP packet level.
   - Goals: authenticated sessions, integrity checks, forward secrecy for vector transport, and robust key management for heterogeneous systems.

3. Phase 3 — Research / Long-term (Partial Homomorphic LIP) 🧪

   - Investigate partial homomorphic encryption for LIP, enabling some vector-space operations while data remains encrypted.
   - Realistic research direction: it is imposible to run full Llama-3-sized models under homomorphic encryption. Instead, explore running the semantic router in HE — the router can decide where to forward requests without learning the request content.

Additional notes:

- We will prioritize a pragmatic, staged approach: start with obfuscation and secure transport, then pursue HE for specific components with strong privacy needs.
- Release artifacts will include YAML config templates and example weights (subject to model license). Packaging on PyPI will make it easy for developers and researchers to adopt LIP.
