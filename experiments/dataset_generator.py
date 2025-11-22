import os
import tqdm
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cpu"
OUTPUT_FILE = os.path.join("datasets", "dataset_code_only.pt")
KEYWORDS = ["python", "def ", "import ", "class ", "print(", "return", "function", "code"]

os.makedirs("datasets", exist_ok=True)
print(f"🏭 [Factory v2] Mining Code-Vectors on {DEVICE}...")
src_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tgt_id = "NousResearch/Meta-Llama-3-8B-Instruct"
print("   Loading Source...")
tok_src = AutoTokenizer.from_pretrained(src_id)
model_src = AutoModelForCausalLM.from_pretrained(src_id, device_map=DEVICE, torch_dtype=torch.float32, low_cpu_mem_usage=True).eval()
print("   Loading Target...")
tok_tgt = AutoTokenizer.from_pretrained(tgt_id)

try:
    model_tgt = AutoModelForCausalLM.from_pretrained(tgt_id, device_map=DEVICE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
except:
    model_tgt = AutoModelForCausalLM.from_pretrained(tgt_id, device_map=DEVICE, torch_dtype=torch.float32).eval()

print("   Downloading Alpaca dataset...")
dataset_full = load_dataset("yahma/alpaca-cleaned", split="train")

processed_data = []
count = 0
MAX_VECTORS = 1000

print("🚀 Starting Selective Extraction...")
for item in tqdm.tqdm(dataset_full):

    if count >= MAX_VECTORS: break
    text_content = (item['instruction'] + item['output']).lower()

    # Domain Filter
    if any(kw in text_content for kw in KEYWORDS):

        instruction = item['instruction']
        input_ctxt = item['input']
        output_text = item['output']

        if input_ctxt:
            full_prompt = f"<|user|>\n{instruction}\nContext: {input_ctxt}</s>\n<|assistant|>\n"
        else:
            full_prompt = f"<|user|>\n{instruction}</s>\n<|assistant|>\n"

        with torch.no_grad():
            # Tiny Vector (Source)
            inp_src = tok_src(full_prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            vec_src = model_src(**inp_src, output_hidden_states=True).hidden_states[-1][:, -1, :].cpu()

            # Llama Vector (Target) - First Token
            tgt_tokens = tok_tgt(output_text, return_tensors="pt", add_special_tokens=False).input_ids

            if tgt_tokens.shape[1] > 0:
                first_token = tgt_tokens[:, 0].to(DEVICE)
                vec_tgt = model_tgt.get_input_embeddings()(first_token).cpu()

                processed_data.append({
                    "src_vector": vec_src,
                    "tgt_vector": vec_tgt,
                    "instruction": instruction
                })
                count += 1


print(f"\n💾 Saving {len(processed_data)} vectors to {OUTPUT_FILE}...")
torch.save(processed_data, OUTPUT_FILE)