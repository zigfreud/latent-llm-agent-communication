import os
import sys
import argparse
import gc
import torch
import json
from tqdm import tqdm
from human_eval.data import write_jsonl, read_problems
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add project root to sys.path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.core.models import LIPAdapter
from src.integrations.hooks import make_lip_hook

def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/infer.yaml")
    parser.add_argument("--mode", type=str, choices=["lip", "baseline"], required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="lip_humaneval_samples.jsonl")
    args = parser.parse_args()

    # Memory and device setup.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Iniciando HumanEval | Mode: {args.mode} | Device: {device}")

    # 4-bit config (same as benchmark to fit on T4).
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # 1) Load HumanEval dataset.
    problems = read_problems()
    task_ids = list(problems.keys())
    if args.max_samples:
        task_ids = task_ids[:args.max_samples]
    
    print(f"📚 Avaliando {len(task_ids)} problemas...")
    prompts = [problems[task_id]["prompt"] for task_id in task_ids]
    anchor_prompt = "Received instruction via LIP:"

    # 2) Load models.
    # Baseline uses only the target; LIP uses source -> extract -> cleanup -> target.
    target_name = "NousResearch/Meta-Llama-3-8B-Instruct"
    source_name = "deepseek-ai/deepseek-coder-1.3b-base"
    adapter_path = "checkpoints/Gen6.2/adapter_pth"  # Hardcoded for reproducibility.

    samples = []

    if args.mode == "baseline":
        print("Loading Target (Llama-3)...")
        t_tok = AutoTokenizer.from_pretrained(target_name, trust_remote_code=True)
        t_mod = AutoModelForCausalLM.from_pretrained(
            target_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        for task_id, prompt_text in tqdm(zip(task_ids, prompts), total=len(task_ids)):
            inputs = t_tok(prompt_text, return_tensors="pt").to(t_mod.device)
            with torch.no_grad():
                out_ids = t_mod.generate(
                    **inputs,
                    max_new_tokens=512,
                    pad_token_id=t_tok.eos_token_id,
                    do_sample=False  # Greedy for reproducibility.
                )
            code = t_tok.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            samples.append(dict(task_id=task_id, completion=code))
    else:
        print("Loading Source (DeepSeek)...")
        s_tok = AutoTokenizer.from_pretrained(source_name, trust_remote_code=True)
        s_mod = AutoModelForCausalLM.from_pretrained(
            source_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        print("Encoding prompts on source...")
        thought_vectors = []
        for prompt_text in tqdm(prompts, desc="Source encoding"):
            s_inputs = s_tok(prompt_text, return_tensors="pt", truncation=True, max_length=2048).to(s_mod.device)
            with torch.no_grad():
                s_out = s_mod(**s_inputs, output_hidden_states=True)
            vec = s_out.hidden_states[-1][:, -1, :].detach().cpu()
            thought_vectors.append(vec)
            del s_inputs, s_out

        del s_mod, s_tok
        _cleanup_cuda()

        print("Loading Target (Llama-3)...")
        t_tok = AutoTokenizer.from_pretrained(target_name, trust_remote_code=True)
        t_mod = AutoModelForCausalLM.from_pretrained(
            target_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        print("⚡ Energy Matching...")
        with torch.no_grad():
            ref_energy = t_mod.get_input_embeddings().weight.float().norm(p=2, dim=-1).mean().item()

        print("Loading Adapter...")
        adapter = LIPAdapter(input_dim=2048, hidden_dim=1024, output_dim=4096).to(t_mod.device)
        state = torch.load(adapter_path, map_location=t_mod.device, weights_only=False)
        adapter.load_state_dict(state)
        adapter.eval()

        for task_id, prompt_text, vec_cpu in tqdm(zip(task_ids, prompts, thought_vectors), total=len(task_ids)):
            a_device = next(adapter.parameters()).device
            vec_trans = adapter(vec_cpu.to(a_device))

            current_norm = vec_trans.norm(p=2, dim=-1)
            scale = (ref_energy / (current_norm + 1e-6)) * 10.0  # Fixed gain 10.0.
            vec_inj = (vec_trans * scale).to(t_mod.device).to(t_mod.dtype)

            t_inputs = t_tok(anchor_prompt, return_tensors="pt").to(t_mod.device)
            inj_pos = t_inputs.input_ids.shape[1] - 1
            hook = make_lip_hook(vec_injected=vec_inj, inject_pos=inj_pos, enable=True)
            handle = t_mod.model.layers[-2].register_forward_hook(hook)

            with torch.no_grad():
                out_ids = t_mod.generate(
                    **t_inputs,
                    max_new_tokens=512,
                    pad_token_id=t_tok.eos_token_id,
                    do_sample=False
                )
            handle.remove()

            code = t_tok.decode(out_ids[0][t_inputs.input_ids.shape[1]:], skip_special_tokens=True)
            samples.append(dict(task_id=task_id, completion=code))

    # Write JSONL.
    write_jsonl(args.output, samples)
    print(f"✅ Resultados salvos em {args.output}")

if __name__ == "__main__":
    main()
