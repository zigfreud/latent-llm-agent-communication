import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER_PATH = "lip_hetero_adapter.pth"
DEVICE = "cpu"
EPOCHS = 20
BATCH_SIZE = 4

print(f"🔧 [Protocolo LIP] Iniciando Ponte Heterogênea...")
source_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"📥 Carregando Source (Tiny): {source_id}...")

tokenizer_src = AutoTokenizer.from_pretrained(source_id)
tokenizer_src.pad_token = tokenizer_src.eos_token

model_src = AutoModelForCausalLM.from_pretrained(
    source_id,
    device_map=DEVICE,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

model_src.eval()
for p in model_src.parameters(): p.requires_grad = False

DIM_SRC = model_src.config.hidden_size # Deve ser 2048
print(f"✅ TinyLlama Pronto. Vetor de Saída: {DIM_SRC}d")
target_id = "NousResearch/Meta-Llama-3-8B-Instruct"
print(f"📥 Carregando Target (Llama-3): {target_id}...")
tokenizer_tgt = AutoTokenizer.from_pretrained(target_id)
tokenizer_tgt.pad_token = tokenizer_tgt.eos_token

try:
    model_tgt = AutoModelForCausalLM.from_pretrained(
        target_id,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
except:
    print("⚠️ Aviso: RAM apertada. Tentando float32 (Risco de OOM)...")
    model_tgt = AutoModelForCausalLM.from_pretrained(
        target_id, device_map=DEVICE, torch_dtype=torch.float32
    )

model_tgt.eval()
for p in model_tgt.parameters(): p.requires_grad = False

DIM_TGT = model_tgt.config.hidden_size # Deve ser 4096
print(f"✅ Llama-3 Pronto. Vetor de Entrada: {DIM_TGT}d")


class HeteroAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        # Compressão (Tiny -> Latent)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        # Expansão (Latent -> Llama)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim) # Normaliza para a escala do Llama
        )

    def forward(self, x):
        x = x.to(torch.float32)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Input: 2048 (Tiny), Output: 4096 (Llama)
adapter = HeteroAdapter(DIM_SRC, DIM_TGT).to(DEVICE)
optimizer = optim.AdamW(adapter.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()
cosine_loss = nn.CosineEmbeddingLoss()
print(f"🌉 Ponte Construída: {DIM_SRC} -> [512] -> {DIM_TGT}")


def generate_data():
    base_data = [
        ("python function sum", "def sum(a, b):"),
        ("create list python", "my_list = []"),
        ("print hello", "print('Hello')"),
        ("import pandas lib", "import pandas as"),
        ("loop ten times", "for i in range(10):"),
        ("class person", "class Person:"),
        ("try catch block", "try:"),
        ("if condition", "if x > 0:"),
        ("return true", "return True"),
        ("import numpy", "import numpy as")
    ]
    final_data = []
    for _ in range(6):
        for inst, code in base_data:
            final_data.append({"instruction": inst, "target": code})
    return final_data


dataset = generate_data()
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"🚀 Iniciando Transferência de Conhecimento ({EPOCHS} épocas)...")

for epoch in range(EPOCHS):
    total_loss = 0

    for batch in train_loader:
        insts = batch['instruction']
        targs = batch['target']
        optimizer.zero_grad()

        with torch.no_grad():
            inp = tokenizer_src(insts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            out_src = model_src(**inp, output_hidden_states=True)
            vec_src = out_src.hidden_states[-1][:, -1, :].to(torch.float32)
            tgt = tokenizer_tgt(targs, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(DEVICE)
            vec_tgt = model_tgt.get_input_embeddings()(tgt.input_ids[:, 0]).to(torch.float32)

        vec_pred = adapter(vec_src) # 2048 -> 4096

        loss_m = mse_loss(vec_pred, vec_tgt)
        loss_c = cosine_loss(vec_pred, vec_tgt, torch.ones(len(insts)).to(DEVICE))
        loss = loss_m + (2.0 * loss_c)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg = total_loss / len(train_loader)
    if (epoch+1) % 2 == 0:
        print(f"Época {epoch+1}/{EPOCHS} | Loss: {avg:.4f}")


torch.save(adapter.state_dict(), ADAPTER_PATH)
print("🏁 Treino Heterogêneo Concluído.")

print("\n" + "="*50)
print("🧠 TESTE: TINY-LLAMA COMANDA LLAMA-3")
print("="*50)

test_prompts = ["python function sum", "import pandas lib", "print hello"]

with torch.no_grad():
    bos_id = tokenizer_tgt.bos_token_id if tokenizer_tgt.bos_token_id else tokenizer_tgt.eos_token_id
    bos_tensor = torch.tensor([[bos_id]], device=DEVICE)
    bos_embed = model_tgt.get_input_embeddings()(bos_tensor).to(model_tgt.dtype)

    ref_norm = model_tgt.get_input_embeddings().weight.norm(p=2, dim=-1).mean().item()

adapter.eval()

for prompt in test_prompts:
    print(f"\n🔴 TinyLlama Pensa: '{prompt}'")

    with torch.no_grad():
        inp = tokenizer_src(prompt, return_tensors="pt").to(DEVICE)
        vec_tiny = model_src(**inp, output_hidden_states=True).hidden_states[-1][:, -1, :].to(torch.float32)
        vec_llama = adapter(vec_tiny)
        curr_norm = vec_llama.norm(p=2, dim=-1)
        scale = ref_norm / (curr_norm + 1e-6)
        vec_scaled = vec_llama * scale
        vec_final = vec_scaled.unsqueeze(1).to(model_tgt.dtype)
        input_embeds = torch.cat([bos_embed, vec_final], dim=1)
        att_mask = torch.ones(input_embeds.shape[:2], dtype=torch.long, device=DEVICE)

        gen_ids = model_tgt.generate(
            inputs_embeds=input_embeds,
            attention_mask=att_mask,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer_tgt.eos_token_id
        )

        out_text = tokenizer_tgt.decode(gen_ids[0], skip_special_tokens=True)
        print(f"🟢 Llama-3 Responde: {out_text}")