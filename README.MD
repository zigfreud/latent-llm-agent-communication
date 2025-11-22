<div align="center">

🧠 LIP: Latent Injection Protocol (Research Lab)

🧪 Universal Latent Injection Protocol for Heterogeneous Model Communication

"O mundo inteiro está se perguntando como organizar LLMs de forma eficiente. Mas por que ninguém fala sobre ensiná-las a conversar em uma língua eficiente?"

Este projeto explora a comunicação direta via espaço latente, eliminando a serialização de texto intermediária.

</div>

📋 Resumo do Projeto

Este repositório contém a implementação oficial, dados experimentais e código-fonte do paper "LIP: A Universal Latent Injection Protocol for Heterogeneous Model Communication".

A arquitetura atual de Sistemas Multi-Agente (MAS) é ineficiente:
Agente A (Pensa) -> Texto -> Tokenização -> Agente B (Lê)

Nossa Proposta (V2V):
Agente A (Pensa) -> Vetor Latente -> Adaptador LIP -> Agente B (Entende)

O LIP permite que LLMs (mesmo de arquiteturas diferentes, como TinyLlama e Llama-3) se comuniquem diretamente via vetores, possibilitando:

Privacidade: Ovetores são ilegíveis para humanos (obfuscação natural).

Eficiência: Elimina a etapa de decodificação/geração de texto intermediário.

Computação Distribuída: Ideal para arquiteturas Edge-to-Cloud.

🏗️ Arquitetura do Experimento (Gen 4)

Nesta versão (v2), validamos a comunicação Heterogênea. Utilizamos um TinyLlama-1.1B (Cliente/Sender) comandando um Llama-3-8B (Servidor/Receiver).

O pipeline consiste em três estágios:

Sender (TinyLlama): Processa a instrução e exporta o last_hidden_state (o "pensamento vetorial").

Projector (Adaptador LIP): Uma rede neural treinada (Dual-Encoder) que alinha o espaço latente do modelo menor ao do modelo maior.

Receiver (Llama-3): Recebe o vetor projetado diretamente em seus inputs_embeds e executa a ação.

graph TD;
User[User Prompt] -->|Texto| AgentA(Sender: TinyLlama 1.1B);
AgentA -->|Last Hidden State| Adapter{LIP Adapter / Projector};
Adapter -->|Vetor Alinhado| AgentB(Receiver: Llama-3 8B);
AgentB -->|Execução| Output[Código/Resposta];

    style AgentA fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style AgentB fill:#e3f2fd,stroke:#01579b,stroke-width:2px
    style Adapter fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,stroke-dasharray: 5 5

📂 Estrutura do Repositório

Este laboratório está organizado para garantir a reprodutibilidade científica:

.
├── paper/ # O Artigo Científico (PDF, LaTeX, Figuras)
├── src/ # Código Fonte do Protocolo e Demos
│ ├── lip_protocol.py # A definição do Pacote LIP (JSON + Tensores)
│ └── lip_integration.py # Demo Cliente-Servidor (TinyLlama -> Llama-3)
├── experiments/ # Scripts de Reprodução Científica
│ ├── train.py # Script de Treino do Adaptador (Gen 4)
│ ├── dataset_generator.py # Fábrica de Vetores (Filtra Alpaca)
│ └── benchmarks/ # Scripts que geraram os gráficos do paper
├── models/ # Checkpoints dos Adaptadores Treinados (.pth)
└── datasets/ # Datasets de vetores pré-processados (.pt)

🚀 Guia de Início Rápido (Quick Start)

1. Pré-requisitos

Python 3.10+

Hardware: 16GB RAM (CPU-Only) ou GPU com 12GB+ VRAM.

# Clone o repositório

git clone [https://github.com/zigfreud/latent-llm-agent-communication.git](https://github.com/zigfreud/latent-llm-agent-communication.git)
cd latent-llm-agent-communication

# Crie um ambiente virtual

python -m venv .venv
source .venv/bin/activate # Linux/Mac

# ou .venv\Scripts\activate no Windows

# Instale as dependências

pip install torch transformers accelerate datasets matplotlib pandas numpy

2. Executando a Demo (Telepatia IA)

Esta demo carrega o TinyLlama (Sender) e o Llama-3 (Receiver) e demonstra a transferência de instruções via vetor.

Certifique-se de possuir o arquivo de pesos lip_hetero_adapter_final.pth na pasta models/.

python src/lip_integration.py

🔬 Reproduzindo a Ciência

Siga este pipeline para treinar o adaptador do zero e reproduzir os resultados da Gen 4 descritos no paper.

Passo 1: Gerar o Dataset de Vetores

Baixa o dataset Alpaca, filtra instruções de código e extrai/alinha os vetores dos dois modelos (Source e Target).

python experiments/dataset_generator.py

# Saída: Cria datasets/dataset_code_only.pt

Passo 2: Treinar o Adaptador

Treina a rede neural (Dual-Encoder) para alinhar os espaços latentes.

python experiments/train.py

# Saída: Salva models/lip_hetero_adapter_final.pth e logs

📊 Resultados Preliminares (Gen 4)

Comparativo com versões anteriores (PoC):

Métrica

Status

Observação

Viabilidade Técnica

✅ Sucesso

Convergência de Loss < 0.003. O modelo aceita injeção vetorial heterogênea.

Transferência de Sintaxe

✅ Sucesso

O Receiver (Llama-3) gera código Python válido a partir de "pensamentos" do TinyLlama.

Integridade Semântica

✅ Estável

Ao contrário da PoC (que sofria Mode Collapse), a Gen 4 atingiu estabilidade semântica robusta na validação.

📝 Citação / Citation

Se você utilizar o LIP (Latent Injection Protocol) em sua pesquisa ou projeto, por favor cite da seguinte forma:

BibTeX (LaTeX)

@article{silva2025lip,
title={LIP: A Universal Latent Injection Protocol for Heterogeneous Model Communication},
author={Silva, Cristiano},
year={2025},
publisher={GitHub},
url={[https://github.com/zigfreud/latent-llm-agent-communication](https://github.com/zigfreud/latent-llm-agent-communication)}
}
