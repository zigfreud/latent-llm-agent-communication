import re
import os
import sys
import time
import glob
import tqdm
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


SAVE_EVERY_N_BATCHES = 50
OUTPUT_DIR = "datasets/seek-coder-1.3b"
DATASET_NAME = "flytech/python-codes-25k"
SRC_ID = "deepseek-ai/deepseek-coder-1.3b-base"
EMBEDDING_FILE = os.path.join("datasets", "llama3_embeddings.pt")


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
            print("❌ Error: 'torch-directml' not installed.")
            sys.exit(1)

    return "cpu"


def extract_embeddings_if_needed():

    if os.path.exists(EMBEDDING_FILE):
        print(f"✅ Embedding matrix found in {EMBEDDING_FILE}")
        return

    print("\n⚠️  Llama-3 Embedding Matrix not found! Extracting now...")
    model_id = "NousResearch/Meta-Llama-3-8B-Instruct"

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    except:
        print("   ...Loading in light CPU mode...")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True)

    emb = model.get_input_embeddings().weight.detach().cpu()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(emb, EMBEDDING_FILE)
    print("✅ Extraction completed.")

    del model


def get_existing_progress(shard_id, output_dir):
    """
    Verify how many files have been processed in according to the filenames
    and continues after that chunk.
    """
    pattern = os.path.join(output_dir, f"shard_{shard_id}_part_*.pt")
    files = glob.glob(pattern)
    total_processed = 0
    max_part_idx = 0

    if not files:
        return 0, 1

    print(f"🔄 Found {len(files)} partial files.")

    for f in files:
        match = re.search(r'part_(\d+).pt', f)
        if match:
            part_idx = int(match.group(1))
            if part_idx > max_part_idx:
                max_part_idx = part_idx

        try:
            data = torch.load(f)
            total_processed += len(data)
        except:
            print(f"⚠️ Ignoring corrupted file: {f}")

    return total_processed, max_part_idx + 1


def save_checkpoint(data_buffer, shard_id, part_idx, output_dir):

    if not data_buffer:
        return

    filename = f"shard_{shard_id}_part_{part_idx}.pt"
    out_path = os.path.join(output_dir, filename)
    torch.save(data_buffer, out_path)
    print(f"\n💾 New checkpoint {filename} ({len(data_buffer)} items)")


def run_factory():
    clear_screen()
    print("🏭 [LIP DATA FACTORY V3] 🏭")
    print("="*50)

    device = get_device_choice()
    extract_embeddings_if_needed()

    print("\n🌐 OPERATION MODE")
    print("1. Single Machine")
    print("2. Cluster Mode")
    mode = input("Choose (1-2): ")

    shard_id = 0
    num_shards = 1
    if mode == '2':
        num_shards = int(input("Total number of machines?: "))
        shard_id = int(input(f"ID of this machine? (0 to {num_shards-1}): "))

    if str(device) == 'cpu':
        batch_size = 1
        use_fp16 = False

    else:
        batch_size = int(input("\n🚀 GPU Batch Size (Recommended 32 or 64): ") or "32")
        use_fp16 = True


    print(f"\n🧠 Loading {SRC_ID}...")
    tok_src = AutoTokenizer.from_pretrained(SRC_ID)
    tok_src.pad_token = tok_src.eos_token
    dtype = torch.float16 if use_fp16 else torch.float32
    model_src = AutoModelForCausalLM.from_pretrained(SRC_ID, torch_dtype=dtype, use_safetensors=True).to(device).eval()

    print("📚 Loading Llama-3 embeddings...")
    llama_emb = torch.load(EMBEDDING_FILE).cpu()
    tok_tgt = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")
    tok_tgt.pad_token = tok_tgt.eos_token

    ds = load_dataset(DATASET_NAME, split="train")
    total_len = len(ds)
    shard_len = total_len // num_shards

    start_global = shard_id * shard_len
    end_global = start_global + shard_len if shard_id < num_shards - 1 else total_len

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    items_already_done, next_part_idx = get_existing_progress(shard_id, OUTPUT_DIR)
    current_start = start_global + items_already_done

    if current_start >= end_global:
        print("\n✅ This shard has already been fully processed!")
        return

    print(f"\n📊 Work Status:")
    print(f"   Total shards: {end_global - start_global}")
    print(f"   JAlready processed: {items_already_done}")
    print(f"   Remaining: {end_global - current_start}")
    print(f"   Starting from global index: {current_start}")

    ds_shard = ds.select(range(current_start, end_global))


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

    data_buffer = []
    batch_counter = 0
    start_time = time.time()

    print("\n⛏️  Starting Mining... (Press Ctrl+C to pause and save)")


    try:
        for batch_prompts, batch_targets, batch_insts in tqdm.tqdm(loader, desc="Mining"):
            try:
                with torch.no_grad():
                    inputs = tok_src(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                    outputs = model_src(**inputs, output_hidden_states=True)

                    vecs_src = outputs.hidden_states[-1][:, -1, :].float().cpu()
                    tgt_enc = tok_tgt(batch_targets, return_tensors="pt", padding=True, add_special_tokens=False)

                    for i in range(len(batch_prompts)):
                        tgt_ids = tgt_enc.input_ids[i]
                        if len(tgt_ids) > 0:
                            first_id = tgt_ids[0]
                            vec_tgt = llama_emb[first_id].unsqueeze(0).float()

                            if len(batch_targets[i]) > 5:
                                data_buffer.append({
                                    "src_vector": vecs_src[i].unsqueeze(0), # vector source
                                    "tgt_vector": vec_tgt,                  # vector Llama (target)
                                    "instruction": batch_insts[i]
                                })

                batch_counter += 1

                if batch_counter >= SAVE_EVERY_N_BATCHES:
                    save_checkpoint(data_buffer, shard_id, next_part_idx, OUTPUT_DIR)
                    data_buffer = [] # ram cleaner
                    next_part_idx += 1
                    batch_counter = 0

            except Exception as e:
                print(f"⚠️ Error in batch processing: {e}")
                continue

    except KeyboardInterrupt:
        print("\n\n🛑 Manual interruption detected (Ctrl+C)!")
        print("Finishing current batch and saving progress...")


    if data_buffer:
        save_checkpoint(data_buffer, shard_id, next_part_idx, OUTPUT_DIR)

    elapsed = time.time() - start_time
    print(f"\n✅ Session ended in {elapsed:.2f}s.")
    print(f"📁 Data saved in parts within: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_factory()