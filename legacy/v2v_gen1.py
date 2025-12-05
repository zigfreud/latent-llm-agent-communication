import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cpu"
print(f"🔧 Rodando localmente em: {device.upper()} (Intel i7 13th Gen)")

model_id = "NousResearch/Meta-Llama-3-8B-Instruct"
print(f"📥 Carregando {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
dtype_config = torch.bfloat16 if torch.cuda.is_available() or torch.backends.mps.is_available() else torch.float32


try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
except Exception as e:
    print(f"⚠️ Erro com bfloat16: {e}")
    print("Tentando fallback para float32 (Cuidado com a RAM)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )


model.eval()
HIDDEN_SIZE = model.config.hidden_size
print(f"✅ Modelo Carregado. Dimensão Latente: {HIDDEN_SIZE}")
print("🏭 Gerando Dados...")


def generate_data():
    base_data = [
        ("soma python", "def sum(a, b): return a + b"),
        ("hello world", "print('Hello World')"),
        ("lista par", "[x for x in range(10) if x % 2 == 0]"),
        ("reverter", "s[::-1]"),
        ("loop", "for i in range(5): print(i)"),
        ("classe", "class MyClass: pass"),
        ("importar pandas", "import pandas as pd"),
        ("media lista", "sum(lista) / len(lista)"),
        ("try except", "try: x/0 except: pass"),
        ("dicionario", "d = {'key': 'value'}")
    ]
    final_data = []
    for _ in range(2): # Menos repetição para teste inicial rápido
        for inst, code in base_data:
            final_data.append({"instruction": f"Code: {inst}", "target": code})
    return final_data


dataset_raw = generate_data()
train_loader = DataLoader(dataset_raw, batch_size=4, shuffle=True)


class V2V_Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        # Bottleneck Architecture (4096 -> 512 -> 4096)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


adapter = V2V_Adapter(HIDDEN_SIZE).to(device)
optimizer = optim.AdamW(adapter.parameters(), lr=1e-3)

print("🚀 Iniciando Treino Local...")
epochs = 10
loss_fn = nn.MSELoss()

for epoch in range(epochs):
    total_loss = 0

    for batch in train_loader:
        instructions = batch['instruction']
        targets = batch['target']

        optimizer.zero_grad()

        with torch.no_grad():
            inp = tokenizer(instructions, return_tensors="pt", padding=True, truncation=True).to(device)
            out_a = model(**inp, output_hidden_states=True)
            vec_input = out_a.hidden_states[-1][:, -1, :].to(torch.float32)

            # Target
            tgt = tokenizer(targets, return_tensors="pt", padding=True, truncation=True).to(device)
            vec_target = model.get_input_embeddings()(tgt.input_ids[:, 0]).to(torch.float32)

        vec_predicted, _ = adapter(vec_input)
        loss = loss_fn(vec_predicted, vec_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Época {epoch+1}/{epochs} | Loss: {avg_loss:.5f}")


print("🏁 Treino Concluído.")
print("\n--- 🧪 Teste de Injeção Semântica ---")
test_prompt = "soma python"
print(f"Prompt Original: {test_prompt}")

with torch.no_grad():
    inp = tokenizer(test_prompt, return_tensors="pt").to(device)
    vec_thought = model(**inp, output_hidden_states=True).hidden_states[-1][:, -1, :].to(torch.float32)
    vec_translated, _ = adapter(vec_thought)
    embedding_matrix = model.get_input_embeddings().weight.to(torch.float32)
    sims = F.cosine_similarity(vec_translated.unsqueeze(1), embedding_matrix.unsqueeze(0), dim=-1)
    best_token_id = torch.argmax(sims).item()
    decoded_token = tokenizer.decode(best_token_id)

print(f"O Adaptador traduziu para o token: '{decoded_token}'")