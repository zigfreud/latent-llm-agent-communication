import os
import gc
import yaml
import glob
import tqdm
import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import torch_directml
    HAS_DML = True
except:
    HAS_DML = False


def get_device(device_str):
    if device_str == "cuda" and torch.cuda.is_available(): return "cuda"
    if device_str == "dml" and HAS_DML: return torch_directml.device()
    if device_str == "cpu": return "cpu"
    if torch.cuda.is_available(): return "cuda"
    if HAS_DML: return torch_directml.device()
    return "cpu"


def load_config(path):
    with open(path, 'r') as f: return yaml.safe_load(f)


def get_model(model_id, device):
    print(f"📥 Loading: {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if str(device) == "cpu" else torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map=device, torch_dtype=dtype, low_cpu_mem_usage=True
        )
    except:
        print("⚠️ Fallback to float32/CPU due to memory/compatibility...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cpu", torch_dtype=torch.float32
        )

    model.eval()
    return model, tokenizer


def run_extraction(config_path):
    cfg = load_config(config_path)
    device = get_device(cfg['device'])
    shard_size = cfg['extraction']['shard_size']
    output_dir = cfg['extraction']['output_dir']

    os.makedirs(output_dir, exist_ok=True)

    print("📚 Loading prompts...")
    ds = load_dataset(cfg['dataset_name'], split=f"train[:{cfg['max_samples']}]")

    existing_shards = glob.glob(os.path.join(output_dir, "shard_*.pt"))
    processed_indices = len(existing_shards) * shard_size
    print(f"🔄 Found {len(existing_shards)} shards. Resuming from index {processed_indices}...")

    if processed_indices >= len(ds):
        print("✅ Job already complete!")
        return

    total_shards = (len(ds) - processed_indices) // shard_size
    print("\n🏭 STARTING PIPELINE...")
    model_src, tok_src = get_model(cfg['models']['source'], device)

    for shard_idx in range(len(existing_shards), (len(ds) // shard_size) + 1):
        start = shard_idx * shard_size
        end = min(start + shard_size, len(ds))
        if start >= len(ds): break

        shard_file_src = os.path.join(output_dir, f"temp_src_{shard_idx}.pt")

        if not os.path.exists(shard_file_src):
            print(f"⛏️  Mining Source Shard {shard_idx} ({start}-{end})...")
            batch_data = []
            subset = ds.select(range(start, end))
            prompts = [f"<|user|>\n{item['input']}\n{item['instruction']}</s>\n<|assistant|>\n" for item in subset]
            infer_batch = cfg['extraction']['batch_size']

            for i in tqdm.tqdm(range(0, len(prompts), infer_batch), leave=False):
                p_batch = prompts[i:i+infer_batch]
                inputs = tok_src(p_batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                with torch.no_grad():
                    out = model_src(**inputs, output_hidden_states=True)
                    vecs = out.hidden_states[-1][:, -1, :].cpu().float()
                    batch_data.append(vecs)

            torch.save(torch.cat(batch_data), shard_file_src)
            print(f"💾 Saved temp source: {shard_file_src}")

    del model_src, tok_src
    gc.collect()
    if device == "cuda": torch.cuda.empty_cache()

    model_tgt, tok_tgt = get_model(cfg['models']['target'], device)

    for shard_idx in range(len(existing_shards), (len(ds) // shard_size) + 1):
        start = shard_idx * shard_size
        end = min(start + shard_size, len(ds))
        if start >= len(ds): break

        shard_file_src = os.path.join(output_dir, f"temp_src_{shard_idx}.pt")
        final_shard_file = os.path.join(output_dir, f"shard_{shard_idx}.pt")

        if os.path.exists(final_shard_file): continue
        if not os.path.exists(shard_file_src):
            print(f"❌ Missing source file for shard {shard_idx}. Skipping.")
            continue

        print(f"🎯 Data Mining Shard {shard_idx}...")
        src_tensor = torch.load(shard_file_src)
        subset = ds.select(range(start, end))
        prompts = [f"<|user|>\n{item['input']}\n{item['instruction']}</s>\n<|assistant|>\n" for item in subset]
        tgt_list = []
        infer_batch = cfg['extraction']['batch_size']

        for i in tqdm.tqdm(range(0, len(prompts), infer_batch), leave=False):
            p_batch = prompts[i:i+infer_batch]
            inputs = tok_tgt(p_batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            with torch.no_grad():
                out = model_tgt(**inputs, output_hidden_states=True)
                vecs = out.hidden_states[-1][:, -1, :].cpu().float()
                tgt_list.append(vecs)

        tgt_tensor = torch.cat(tgt_list)
        combined_data = []
        for k in range(len(src_tensor)):
            combined_data.append({
                "src_vector": src_tensor[k].unsqueeze(0),
                "tgt_vector": tgt_tensor[k].unsqueeze(0),
            })

        torch.save(combined_data, final_shard_file)
        print(f"✅ Final Shard Saved: {final_shard_file}")
        os.remove(shard_file_src)


    print("\n🏁 Process completed..")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data_factory_config.yaml")
    args = parser.parse_args()
    run_extraction(args.config)