import os
import torch
from transformers import AutoModelForCausalLM

model_id = "NousResearch/Meta-Llama-3-8B-Instruct"
print(f"📥 Carregando Llama-3 para extração cirúrgica...")

try:
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
except:
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True)

print("⛏️  Extraindo Matriz de Embeddings...")
embedding_matrix = model.get_input_embeddings().weight.detach().cpu()
output_file = "datasets/llama3_embeddings.pt"
torch.save(embedding_matrix, output_file)

print(f"✅ Matriz salva em {output_file}")
print(f"   Tamanho original do modelo: ~15 GB")
print(f"   Tamanho do arquivo otimizado: {os.path.getsize(output_file) / (1024*1024):.2f} MB")