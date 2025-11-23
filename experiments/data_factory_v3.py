import os
import sys
import tqdm
import time
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASET_NAME = "flytech/python-codes-25k"
EMBEDDING_FILE = os.path.join("datasets", "llama3_embeddings.pt")
OUTPUT_DIR = "datasets"


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def get_device_choice():
    print("\n🖥️  HARDWARE SELECTION")
    print("1. CPU")
    print("2. GPU NVIDIA (CUDA)")
    print("3. GPU AMD (DirectML - Requires 'pip install torch-directml')")

    choice = input("Choose (1-3): ")

    if choice == '1': return "cpu"
    if choice == '2': return "cuda"
    if choice == '3':
        try:
            import torch_directml
            return torch_directml.device()
        except ImportError:
            print("❌ Error: 'torch-directml' not installed. Run: pip install torch-directml")
            sys.exit(1)
    return "cpu"


def extract_embeddings_if_needed():
    if os.path.exists(EMBEDDING_FILE):
        print(f"✅ Matriz de Embeddings encontrada em {EMBEDDING_FILE}")
        return

    print("\n⚠️  Llama-3 Embedding Matrix not found! Extracting now (One-time process, may take a while)...")
    model_id = "NousResearch/Meta-Llama-3-8B-Instruct"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    except:
        print("   ...Loading in light CPU mode...")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True)

    emb = model.get_input_embeddings().weight.detach().cpu()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(emb, EMBEDDING_FILE)
    print("✅ Extração concluída e salva.")
    del model


def run_factory():
    clear_screen()
    print("🏭 [LIP DATA FACTORY MASTER] 🏭")
    print("="*40)

    device = get_device_choice()
    print(f"✅ Selected Hardware: {device}")
    extract_embeddings_if_needed()

    print("\n🌐 MODO DE OPERAÇÃO")
    print("1. Single Machine (Processar tudo aqui)")
    print("2. Cluster Mode (Dividir carga entre máquinas)")
    mode = input("Choose (1-2): ")

    shard_id = 0
    num_shards = 1

    if mode == '2':
        num_shards = int(input("Total number of machines? (e.g., 2): "))
        shard_id = int(input(f"ID of this machine? (0 to {num_shards-1}): "))

    if str(device) == 'cpu':
        batch_size = 1
        use_fp16 = False
    else:
        batch_size = int(input("\n🚀 GPU Batch Size (Recommended 32 or 64): ") or "32")
        use_fp16 = True

    print(f"   Dataset: {DATASET_NAME}")
    print(f"   Shard: {shard_id+1}/{num_shards}")
    print(f"   Device: {device} | Batch: {batch_size} | FP16: {use_fp16}") # Changed from "FP16: {use_fp16}" to "FP16: {use_fp16}"

    src_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok_src = AutoTokenizer.from_pretrained(src_id)
    tok_src.pad_token = tok_src.eos_token

    dtype = torch.float16 if use_fp16 else torch.float32
    model_src = AutoModelForCausalLM.from_pretrained(src_id, torch_dtype=dtype).to(device).eval()

    llama_emb = torch.load(EMBEDDING_FILE).cpu()
    tok_tgt = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")
    ds = load_dataset(DATASET_NAME, split="train")
    total_len = len(ds)
    shard_len = total_len // num_shards
    start = shard_id * shard_len
    end = start + shard_len if shard_id < num_shards - 1 else total_len

    ds_shard = ds.select(range(start, end))
    print(f"   📊 Processing {len(ds_shard)} examples (Global Total: {total_len})")


    def collate_fn(batch):
        prompts = []
        targets = []
        raw_insts = []
        for item in batch:
            inst = item.get('instruction') or item.get('input') or ""
            out = item.get('output') or item.get('text') or ""
            prompts.append(f"<|user|>\n{inst}</s>\n<|assistant|>\n")
            targets.append(out)
            raw_insts.append(inst)
        return prompts, targets, raw_insts


    loader = DataLoader(ds_shard, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    processed_data = []
    start_time = time.time()

    for batch_prompts, batch_targets, batch_insts in tqdm.tqdm(loader, desc="Mining Vectors"):
        try:
            with torch.no_grad():
                inputs = tok_src(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                outputs = model_src(**inputs, output_hidden_states=True)

                # Extrair vetores (último token)
                # [Batch, Seq, Dim] -> [Batch, Dim]
                vecs_src = outputs.hidden_states[-1][:, -1, :].float().cpu()

                tgt_enc = tok_tgt(batch_targets, return_tensors="pt", padding=True, add_special_tokens=False)

                for i in range(len(batch_prompts)):
                    tgt_ids = tgt_enc.input_ids[i]
                    if len(tgt_ids) > 0:
                        first_id = tgt_ids[0]
                        vec_tgt = llama_emb[first_id].unsqueeze(0).float()

                        if len(batch_targets[i]) > 5:
                            processed_data.append({
                                "src_vector": vecs_src[i].unsqueeze(0),
                                "tgt_vector": vec_tgt,
                                "instruction": batch_insts[i]
                            })

        except Exception as e:
            print(f"⚠️ Error in batch: {e}")
            continue

    filename = f"dataset_shard_{shard_id}.pt" if num_shards > 1 else "dataset_massive_full.pt"
    out_path = os.path.join(OUTPUT_DIR, filename)
    torch.save(processed_data, out_path)

    elapsed = time.time() - start_time
    print(f"\n✅ Completed in {elapsed:.2f}s.")
    print(f"💾 Saved to: {out_path} ({len(processed_data)} vectors)")

if __name__ == "__main__":
    run_factory()