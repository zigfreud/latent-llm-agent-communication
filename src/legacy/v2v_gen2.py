import os
import sys
import torch
import shutil
import datetime
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 0. CONFIGURAÇÃO E HIPERPARÂMETROS
# ==========================================
ADAPTER_PATH = "v2v_hybrid_adapter.pth" # Novo nome para não misturar
DEVICE = "cpu"
EPOCHS = 30            # Com Loss Híbrida, convergir deve ser mais rápido
BATCH_SIZE = 4
LEARNING_RATE = 1e-3   # AdamW padrão

print(f"🔧 [Dr. Thorne System] Ambiente: {DEVICE.upper()} (Intel Core i7 Optimized)")

# ==========================================
# 1. CARREGAMENTO DO MODELO (LLM CONGELADO)
# ==========================================
model_id = "NousResearch/Meta-Llama-3-8B-Instruct"

print(f"📥 Carregando Tokenizer e Modelo: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Estratégia de Memória: Pesos em bfloat16 (16GB), Computação em float32 (Estabilidade)
try:
    print("   Tentando carregar em bfloat16 (Economia de RAM)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
except Exception as e:
    print(f"⚠️ Falha no bfloat16: {e}. Carregando em float32 (Cuidado com a RAM).")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=DEVICE,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

model.eval() # Congela o Llama
for param in model.parameters():
    param.requires_grad = False

HIDDEN_SIZE = model.config.hidden_size
print(f"✅ Modelo Pronto. Dimensão Latente: {HIDDEN_SIZE}")

# ==========================================
# 2. DATA FACTORY (DATASET REFINADO)
# ==========================================
print("🏭 Gerando Dataset Sintético Focado...")
def generate_data():
    # Pares focados em comandos de programação distintos
    base_data = [
        ("soma python", "def sum(a, b):"),
        ("função multiplicar", "def mult(x, y):"),
        ("hello world", "print('Hello')"),
        ("imprimir texto", "print('Texto')"),
        ("lista de pares", "[x for x in"),
        ("reverter string", "s[::-1]"),
        ("loop for", "for i in range"),
        ("classe vazia", "class MyClass:"),
        ("importar pandas", "import pandas as"),
        ("importar numpy", "import numpy as"),
        ("ler csv", "pd.read_csv("),
        ("try except", "try:"),
        ("dicionario novo", "d = {}"),
        ("json dump", "json.dumps("),
        ("lambda simples", "lambda x:"),
        ("condição if", "if x > 0:")
    ]
    final_data = []
    # Multiplicar para dar volume ao treino (Batching)
    for _ in range(4):
        for inst, code in base_data:
            final_data.append({"instruction": f"Code: {inst}", "target": code})
    return final_data

dataset_raw = generate_data()
train_loader = DataLoader(dataset_raw, batch_size=BATCH_SIZE, shuffle=True)
print(f"📊 Dataset carregado: {len(dataset_raw)} exemplos.")

# ==========================================
# 3. ARQUITETURA DUAL ENCODER (V2)
# ==========================================
class V2V_Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        # Bottleneck Architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU() # Non-linearity crucial
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim) # Estabilidade final
        )

    def forward(self, x):
        # Force float32 para evitar NaN em treino CPU
        x = x.to(torch.float32)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

adapter = V2V_Adapter(HIDDEN_SIZE).to(DEVICE)
optimizer = optim.AdamW(adapter.parameters(), lr=LEARNING_RATE)

# --- SISTEMA DE LOSS HÍBRIDA ---
mse_loss = nn.MSELoss()
cosine_loss = nn.CosineEmbeddingLoss()

# ==========================================
# 4. LOOP DE TREINO (CORRIGIDO & HÍBRIDO)
# ==========================================
start_epoch = 0
if os.path.exists(ADAPTER_PATH):
    print(f"\n💾 Arquivo encontrado: {ADAPTER_PATH}")
    op = input("Opções: [c]ontinuar | [n]ovo treino | [t]estar: ").lower()
    if op == 'c':
        adapter.load_state_dict(torch.load(ADAPTER_PATH))
        print("✅ Pesos carregados.")
    elif op == 't':
        adapter.load_state_dict(torch.load(ADAPTER_PATH))
        start_epoch = EPOCHS # Pula para o final
    else:
        print("⚠️  Iniciando do ZERO (Recomendado para aplicar correções).")

if start_epoch < EPOCHS:
    print(f"\n🚀 Iniciando Treino Híbrido (MSE + Cosine) - {EPOCHS} Épocas...")
    print("ℹ️  Use Ctrl+C para salvar e sair.")

    try:
        for epoch in range(start_epoch, EPOCHS):
            total_loss = 0
            total_mse = 0
            total_cos = 0

            for batch in train_loader:
                instructions = batch['instruction']
                targets = batch['target']

                # Ajuste dinâmico do target para Cosine Loss (Batch final pode ser menor)
                current_bs = len(instructions)
                target_ones = torch.ones(current_bs).to(DEVICE)

                optimizer.zero_grad()

                with torch.no_grad():
                    # INPUT: Instrução completa
                    inp = tokenizer(instructions, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                    out_a = model(**inp, output_hidden_states=True)
                    vec_input = out_a.hidden_states[-1][:, -1, :].to(torch.float32)

                    # TARGET: 1º Token do Código (CORREÇÃO CRÍTICA: add_special_tokens=False)
                    # Isso garante que pegamos 'def' e não '<|begin_of_text|>'
                    tgt = tokenizer(targets, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(DEVICE)
                    vec_target = model.get_input_embeddings()(tgt.input_ids[:, 0]).to(torch.float32)

                # Forward
                vec_predicted, _ = adapter(vec_input)

                # Loss Calculation
                l_mse = mse_loss(vec_predicted, vec_target)
                l_cos = cosine_loss(vec_predicted, vec_target, target_ones)

                # Peso 2.0 para Cosine para forçar a direção semântica
                loss = l_mse + (2.0 * l_cos)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_mse += l_mse.item()
                total_cos += l_cos.item()

            # Médias
            avg_loss = total_loss / len(train_loader)
            avg_mse = total_mse / len(train_loader)
            avg_cos = total_cos / len(train_loader)

            print(f"Época {epoch+1}/{EPOCHS} | Total: {avg_loss:.4f} (MSE: {avg_mse:.4f} | COS: {avg_cos:.4f})")
            # --- LÓGICA DE SALVAMENTO INTELIGENTE ---
            # 1. Salvar o "latest" (para retomar treino rápido)
            torch.save(adapter.state_dict(), "v2v_latest.pth")
            # 2. Salvar Melhores Modelos (Best Loss)
            # Inicialize best_loss = float('inf') fora do loop se quiser rastrear
            # Mas por segurança, vamos salvar checkpoints periódicos com timestamp

            if (epoch+1) % 5 == 0:
                # Cria um nome único: v2v_epoch_10_loss_0.35.pth
                timestamp = datetime.datetime.now().strftime("%H%M")
                safe_name = f"checkpoints/v2v_ep{epoch+1}_loss{avg_loss:.3f}_{timestamp}.pth"
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(adapter.state_dict(), safe_name)
                print(f"   💾 Backup seguro salvo: {safe_name}")


    except KeyboardInterrupt:
        print("\n🛑 Interrompido!")
        if input("Salvar? (s/n): ").lower() == 's':
            torch.save(adapter.state_dict(), ADAPTER_PATH)

    torch.save(adapter.state_dict(), ADAPTER_PATH)
    print("🏁 Treino Finalizado.")


# ==========================================
# 5. DIAGNÓSTICO RAIO-X (TOP-K)
# ==========================================
print("\n" + "="*40)
print("🧪 TESTE DE INJEÇÃO SEMÂNTICA (RAIO-X)")
print("="*40)

prompts_teste = ["soma python", "importar pandas", "hello world"]

adapter.eval()
# Cache da matriz de embeddings (Pesada, mas necessária para o teste)
E = model.get_input_embeddings().weight.to(torch.float32)
E_norm = F.normalize(E, p=2, dim=-1)

for p in prompts_teste:
    print(f"\n📝 Prompt: '{p}'")
    with torch.no_grad():
        inp = tokenizer(p, return_tensors="pt").to(DEVICE)
        vec_thought = model(**inp, output_hidden_states=True).hidden_states[-1][:, -1, :].to(torch.float32)

        # Traduz
        vec_trans, _ = adapter(vec_thought)
        vec_trans_norm = F.normalize(vec_trans, p=2, dim=-1)

        # Busca Top 5
        sims = torch.matmul(vec_trans_norm, E_norm.T).squeeze()
        top_vals, top_idxs = torch.topk(sims, k=5)

        for i, (idx, score) in enumerate(zip(top_idxs, top_vals)):
            tok = tokenizer.decode(idx)
            # Limpa caracteres de controle para leitura fácil
            tok_clean = tok.replace("\n", "\\n").strip()
            print(f"   {i+1}.Token: [{tok_clean}] \t(Sim: {score:.4f})")

print("\n-------------------------------------------------")
print("CRITÉRIO DE SUCESSO DO DR. THORNE:")
print("1. COS Loss deve cair abaixo de 0.1")
print("2. Tokens como 'def', 'import', 'print' devem aparecer no Top 5.")
print("3. '<|begin_of_text|>' deve desaparecer do Top 1.")

# ==========================================
# 6. V2V GENERATION: CALIBRAGEM DE MAGNITUDE (PHYSICS FIX)
# ==========================================
print("\n" + "="*50)
print("🧠 V2V GENERATION: CALIBRAGEM DE ENERGIA")
print("="*50)

prompts_teste = [
    "soma python",
    "importar pandas",
    "hello world",
    "loop for"
]

adapter.eval()

# Âncora (BOS)
bos_token_id = tokenizer.bos_token_id
if bos_token_id is None: bos_token_id = tokenizer.eos_token_id
bos_tensor = torch.tensor([[bos_token_id]], device=DEVICE)

with torch.no_grad():
    # Embeddings originais do modelo
    embedding_layer = model.get_input_embeddings()
    bos_embed = embedding_layer(bos_tensor)

    # CÁLCULO DA ENERGIA MÉDIA DO LLAMA (REFERENCE NORM)
    ref_norm = embedding_layer.weight.norm(p=2, dim=-1).mean().item()
    print(f"⚡ Energia Média do Llama-3: {ref_norm:.4f}")

for prompt in prompts_teste:
    print(f"\n🔴 Input Humano: '{prompt}'")

    with torch.no_grad():
        inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        vec_thought = model(**inp, output_hidden_states=True).hidden_states[-1][:, -1, :].to(torch.float32)
        vec_translated, _ = adapter(vec_thought)

        current_norm = vec_translated.norm(p=2, dim=-1)
        scale_factor = ref_norm / (current_norm + 1e-6)

        vec_scaled = vec_translated * scale_factor
        print(f"   ⚖️  Calibrando vetor: Norm {current_norm.item():.2f} -> {ref_norm:.2f}")
        vec_final = vec_scaled.unsqueeze(1).to(model.dtype)

        # 3. Input Híbrido: [BOS] + [VETOR CALIBRADO]
        input_embeds = torch.cat([bos_embed.to(model.dtype), vec_final], dim=1)
        attention_mask = torch.ones(input_embeds.shape[:2], dtype=torch.long, device=DEVICE)

        generated_ids = model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=35,
            do_sample=True,
            temperature=0.1,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id
        )

        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Verificação do Conceito
        E = embedding_layer.weight.to(torch.float32)
        sims = torch.matmul(F.normalize(vec_translated, p=2), F.normalize(E, p=2).T)
        top_token = tokenizer.decode(torch.argmax(sims))

        print(f"🟢 [Conceito '{top_token}']: {output_text}")

print("\n" + "="*50)