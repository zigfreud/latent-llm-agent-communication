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
    if device_str == "dml" and HAS_DML: 
        return torch_directml.device()
    return "cpu"

def load_config(path):
    with open(path, 'r') as f: return yaml.safe_load(f)

def get_model(model_id, device):
    print(f"📥 Loading: {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # --- DEBUG DE ALOCAÇÃO ---
    print(f"   🔧 Target Device: {device}")
    print(f"   🔧 RAM Disponível (aprox): {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3) if hasattr(os, 'sysconf') else 'N/A'} GB")

    try:
        # 1. Carrega na CPU em Float16 (Leve: ~2.6GB)
        print("   1. Carregando na CPU (Float16)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False, # Evita hooks do Accelerate
            use_safetensors=True,
            device_map=None # Proíbe auto-alocação
        )
        
        # 2. Move para DML
        print(f"   2. Movendo para GPU ({device})...")
        model.to(device)
        print("   ✅ SUCESSO! Modelo na GPU.")
        
    except Exception as e:
        print(f"\n❌ ERRO FATAL DE CARREGAMENTO:")
        print(f"   Mensagem: {e}")
        print("   ⚠️ O script vai parar aqui para você não explodir sua RAM em CPU Float32.")
        print("   👉 Se o erro for 'CUDA', desinstale bitsandbytes/accelerate.")
        import sys; sys.exit(1) # Mata o processo antes de travar o PC

    model.eval()
    return model, tokenizer

def run_extraction(config_path):
    cfg = load_config(config_path)
    # Força DML se configurado
    device_str = "dml" if cfg['device'] == "dml" else "cpu"
    device = get_device(device_str)
    
    print(f"⚙️  Pipeline configurado para: {device}")
    
    shard_size = cfg['extraction']['shard_size']
    output_dir = cfg['extraction']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    print("📚 Loading prompts...")
    ds = load_dataset(cfg['dataset_name'], split=f"train[:{cfg['max_samples']}]")
    
    existing_shards = glob.glob(os.path.join(output_dir, "shard_*.pt"))
    processed_indices = len(existing_shards) * shard_size
    
    if processed_indices >= len(ds):
        print("✅ Job already complete!")
        return

    # --- FASE 1: SOURCE ---
    print("\n🏭 INICIANDO SOURCE (DeepSeek)...")
    model_src, tok_src = get_model(cfg['models']['source'], device)
    
    for shard_idx in range(len(existing_shards), (len(ds) // shard_size) + 1):
        start = shard_idx * shard_size
        end = min(start + shard_size, len(ds))
        if start >= len(ds): break

        shard_file_src = os.path.join(output_dir, f"temp_src_{shard_idx}.pt")
        
        if not os.path.exists(shard_file_src):
            print(f"⛏️  Mining Shard {shard_idx}...")
            batch_data = []
            subset = ds.select(range(start, end))
            prompts = [f"<|user|>\n{item['input']}\n{item['instruction']}</s>\n<|assistant|>\n" for item in subset]
            
            # Batch size conservador para GPU de 8GB com YouTube aberto
            infer_batch = 7

            for i in tqdm.tqdm(range(0, len(prompts), infer_batch)):
                p_batch = prompts[i:i+infer_batch]
                # Move inputs manualmente
                inputs = tok_src(p_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                
                # .to(device) seguro
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    out = model_src(**inputs, output_hidden_states=True)
                    vecs = out.hidden_states[-1][:, -1, :].cpu().float()
                    batch_data.append(vecs)
                
                # Limpeza manual
                del inputs, out
                if HAS_DML: torch.cuda.empty_cache()

            if batch_data:
                torch.save(torch.cat(batch_data), shard_file_src)

    print("🧹 Cleaning Source...")
    del model_src, tok_src
    gc.collect()
    if HAS_DML: torch.cuda.empty_cache()

    # --- FASE 2: TARGET ---
    print("\n🏭 INICIANDO TARGET (Llama-3)...")
    # Llama-3 8B é grande. Se der erro aqui, teremos que usar CPU ou Quantização.
    model_tgt, tok_tgt = get_model(cfg['models']['target'], device)

    for shard_idx in range(len(existing_shards), (len(ds) // shard_size) + 1):
        start = shard_idx * shard_size
        end = min(start + shard_size, len(ds))
        if start >= len(ds): break
        
        shard_file_src = os.path.join(output_dir, f"temp_src_{shard_idx}.pt")
        final_shard_file = os.path.join(output_dir, f"shard_{shard_idx}.pt")
        
        if not os.path.exists(shard_file_src): continue
        if os.path.exists(final_shard_file): continue

        print(f"🎯 Target Processing Shard {shard_idx}...")
        src_tensor = torch.load(shard_file_src)
        subset = ds.select(range(start, end))
        prompts = [f"<|user|>\n{item['input']}\n{item['instruction']}</s>\n<|assistant|>\n" for item in subset]
        
        tgt_list = []
        infer_batch = 1

        for i in tqdm.tqdm(range(0, len(prompts), infer_batch)):
            p_batch = prompts[i:i+infer_batch]
            inputs = tok_tgt(p_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model_tgt(**inputs, output_hidden_states=True)
                vecs = out.hidden_states[-1][:, -1, :].cpu().float()
                tgt_list.append(vecs)
            
            del inputs, out
            if HAS_DML: torch.cuda.empty_cache()

        if tgt_list:
            # Salva
            combined = []
            tgt_tensor = torch.cat(tgt_list)
            for k in range(len(src_tensor)):
                combined.append({
                    "src_vector": src_tensor[k].unsqueeze(0),
                    "tgt_vector": tgt_tensor[k].unsqueeze(0)
                })
            torch.save(combined, final_shard_file)
            os.remove(shard_file_src)

    print("🏁 Done.")

def run_extract_reference(config_path):
    cfg = load_config(config_path)
    device_str = "dml" if cfg['device'] == "dml" else "cpu"
    device = get_device(device_str)

    os.makedirs("datasets", exist_ok=True)
    output_file = os.path.join("datasets", "reference_embeddings.pt")

    print("\n🏭 INICIANDO EXTRAÇÃO DE EMBEDDINGS DE REFERÊNCIA (Llama-3)...")
    model_tgt, _ = get_model(cfg['models']['target'], device)

    print("⛏️  Extraindo Matriz de Embeddings...")
    with torch.no_grad():
        embedding_matrix = model_tgt.get_input_embeddings().weight.detach()
        embedding_matrix = embedding_matrix.to(dtype=torch.float16, device="cpu")
    torch.save(embedding_matrix, output_file)

    print(f"✅ Matriz salva em {output_file}")
    if HAS_DML: torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data_factory_config.yaml")
    parser.add_argument("--mode", type=str, choices=["mine_shards", "extract_reference"], default="mine_shards")
    args = parser.parse_args()
    if args.mode == "extract_reference":
        run_extract_reference(args.config)
    else:
        run_extraction(args.config)
