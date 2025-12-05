import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import os
import tqdm

DEVICE = "cpu"
BATCH_SIZE = 1
MAX_SAMPLES = 1000
OUTPUT_FILE = "dataset_vectors_1k.pt"

print(f"🏭 [Dr. Thorne Factory] Iniciando Linha de Montagem em {DEVICE}...")
src_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"   Carregando Source: {src_id}...")
tok_src = AutoTokenizer.from_pretrained(src_id)
model_src = AutoModelForCausalLM.from_pretrained(src_id, device_map=DEVICE, torch_dtype=torch.float32, low_cpu_mem_usage=True)
model_src.eval()

tgt_id = "NousResearch/Meta-Llama-3-8B-Instruct"
print(f"   Carregando Target: {tgt_id}...")
tok_tgt = AutoTokenizer.from_pretrained(tgt_id)

try:
    model_tgt = AutoModelForCausalLM.from_pretrained(tgt_id, device_map=DEVICE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
except:
    model_tgt = AutoModelForCausalLM.from_pretrained(tgt_id, device_map=DEVICE, torch_dtype=torch.float32)

model_tgt.eval()

print("   Baixando Dataset Alpaca (yahma/alpaca-cleaned)...")
dataset_source = load_dataset("yahma/alpaca-cleaned", split=f"train[:{MAX_SAMPLES}]")
print(f"✅ Dataset carregado: {len(dataset_source)} amostras.")

processed_data = []

print("🚀 Iniciando extração de vetores...")

for item in tqdm.tqdm(dataset_source):
    instruction = item['instruction']
    input_ctxt = item['input']
    output_text = item['output']

    if input_ctxt:
        full_prompt = f"<|user|>\n{instruction}\nContext: {input_ctxt}</s>\n<|assistant|>\n"
    else:
        full_prompt = f"<|user|>\n{instruction}</s>\n<|assistant|>\n"

    with torch.no_grad():
        inp_src = tok_src(full_prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        vec_src = model_src(**inp_src, output_hidden_states=True).hidden_states[-1][:, -1, :].cpu()
        tgt_tokens = tok_tgt(output_text, return_tensors="pt", add_special_tokens=False).input_ids

        if tgt_tokens.shape[1] > 0:
            first_token = tgt_tokens[:, 0].to(DEVICE)
            vec_tgt = model_tgt.get_input_embeddings()(first_token).cpu()

            processed_data.append({
                "src_vector": vec_src, # [1, 2048]
                "tgt_vector": vec_tgt, # [1, 4096]
                "instruction": instruction, # Meta-dados for debugging
                "target_text": output_text[:50]
            })

print(f"\n💾 Salvando {len(processed_data)} pares de vetores em {OUTPUT_FILE}...")
torch.save(processed_data, OUTPUT_FILE)
print("✅ Fábrica finalizada. Pode iniciar o treino massivo.")