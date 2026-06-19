# 🧬 LIP — Latent Injection Protocol
### Research Prototype for Heterogeneous LLM Latent-Space Experiments

**Status: Research Prototype / Contrastive Bridge v2 — functional but under active metric and fidelity improvement.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Validation: In Progress](https://img.shields.io/badge/Validation-In%20Progress-yellow.svg)]()

---

## 📖 Abstract

LIP (Latent Injection Protocol) is a research prototype for studying whether
latent vectors from one model can be mapped into a usable target-model vector
space. The current mainline promotes the contrastive bridge and remote CPU smoke
infrastructure used for small, auditable experiments.

The active prototype focuses on a **DeepSeek-Coder-1.3B** source vector shape and
a **Llama-3-8B** target vector shape via a trained adapter
(2048d → hidden_dim → 4096d). This is infrastructure for research iteration,
not evidence that semantic translation is solved.

Current components:
1.  **Contrastive Bridge Prototype:** Adapter training with contrastive and MSE loss terms.
2.  **Latent Bundle Contract:** Small external-bundle ingestion path with manifest validation and safe unzip.
3.  **Remote CPU Smoke Workflows:** H0 workflows for toy, full-dimension, and latent-bundle infrastructure checks.

---

## 🛠️ Installation & Setup

### 1. Environment
This project requires **Python 3.10+**.
We recommend a fresh virtual environment to avoid conflicts with system-wide PyTorch versions.

```bash
# Create and activate environment
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Dependencies
Installation is strict to ensure reproducibility. Note: For Windows users with AMD GPUs, torch-directml is supported.

```
Bash

pip install -r requirements.txt
```

### 3. Verify Installation
Run the integration test suite to ensure the environment is sound before downloading heavy models.

```Bash

pytest tests/test_integration.py
```
_(Expected output: 3 passed)_

## 🚀 Quickstart: Research Prototype

### Step 1: Data Mining (Demo Mode)
Run in `--demo` mode to process a small sample set. This downloads the source
model, extracts vectors, and prepares local shards.

```
Bash
# Runs extraction for 100 samples (cpu/gpu auto-detect)
python src/pipelines/data_factory.py --config configs/data_factory_config.yaml --demo
```
_Output: datasets/Gen6.1/raw_shards/shard_0.pt_

### Step 2: Training the Adapter
Train the LIP Adapter using mined or validated shards.

```
Bash
python src/pipelines/trainer.py --config configs/train_config.yaml
```
_Artifacts will be saved to: checkpoints/Gen6.2/_

### Step 3: Inference & Injection Probe
Run the deterministic injection probe. This injects a mapped latent vector into
the target path and compares output behavior against a baseline.

```
Bash
python src/pipelines/run_infer.py --config configs/infer.yaml
```
_Expected result: output behavior can be inspected against a baseline. Treat any
alignment as preliminary until evaluated on held-out semantic tests._

### Step 4: Eval Syntax Pass
Run the syntax evaluation to check Python validity of generated code.

```
Bash
python src/scripts/eval_syntax_pass1.py --config configs/infer.yaml
```
_Output: Syntax Pass Rate and CSV `results_syntax_eval.csv`_

## 🧠 Preliminary Prototype Examples

The following historical prototype outputs are examples for inspection only.
They are not proof of solved semantic transfer and should be treated as
preliminary until reproduced on held-out semantic evaluations.

| DeepSeek Intent (Hidden Input) | Llama-3 Execution (Generated Code) | Status |
| :--- | :--- | :--- |
| *"Create a simple Python game loop..."* | `class Game: def __init__(self)... while True: command = input...` | Preliminary example |
| *"Read a CSV file and sum columns..."* | `import pandas as pd... df = pd.read_csv('file.csv')... print(df.sum())` | Preliminary example |
| *"Sentiment analysis using NLTK..."* | `from nltk.sentiment import SentimentIntensityAnalyzer... def analyze(text):...` | Preliminary example |
| *"Update SQL table example..."* | `import sqlite3... c.execute('UPDATE stocks SET qty=? WHERE symbol=?', ...)` | Preliminary example |

*Note: Historical alignment metrics are preliminary and training-set-only unless
explicitly labeled otherwise. Held-out semantic fidelity evaluation still needs
to be scaled.*

## 📂 Project Structure
```
├── configs/              # Mainline experiment configurations (YAML)
├── config/               # H0 workflow configs kept for traceability
├── .github/workflows/    # Remote CPU smoke and bundle workflows
├── docs/                 # H0 contracts and runbooks
├── src/
│   ├── core/             # Model definitions (LIPAdapter) & Loss functions
│   ├── integrations/     # PyTorch Hooks for Injection
│   ├── pipelines/        # Data Factory, Trainer, Inference Logic
│   └── scripts/          # Utilities (Graphs, Diagrams)
├── tests/                # Integration tests (pytest)
├── datasets/             # (Created at runtime) Vector shards
└── checkpoints/          # (Created at runtime) Model weights
```

## 🔬 Citation
If you use this code in your research, please cite:

```
@misc{silva2025lip,
  title={LIP: A Universal Latent Injection Protocol for Heterogeneous Model Communication},
  author={Silva, Cristiano},
  year={2025},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{[https://github.com/zigfreud/latent-llm-agent-communication](https://github.com/zigfreud/latent-llm-agent-communication)}}
}
```
