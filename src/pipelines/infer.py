# src/pipelines/infer.py
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.core.models import LIPAdapter
from src.integrations.hooks import make_lip_hook


@dataclass
class InferenceConfig:
    # models
    source_model: str
    target_model: str
    adapter_ckpt: str

    # runtime
    device_src: str = "cuda"
    device_tgt: str = "auto"
    load_4bit: bool = True

    # lip
    gain: float = 10.0
    layer_idx: int = -2
    inject_pos_mode: str = "last"  # "last" or "last_minus_1"

    # generation
    max_new_tokens: int = 120
    do_sample: bool = False
    temperature: float = 0.0
    repetition_penalty: float = 1.2

    # prompts
    prompts: List[str] = None


def format_prompt_like_dataset(user_text: str) -> str:
    # Close to your mining template: user -> assistant
    return f"<|user|>\n\n{user_text}</s>\n<|assistant|>\n"


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def load_source(model_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,          # "cuda" is fine on most setups
        low_cpu_mem_usage=True,
        use_safetensors=True,
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def load_target(model_id: str, device: str, load_4bit: bool):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if load_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device,        # "auto" recommended when using offload
            use_safetensors=True,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            trust_remote_code=True,
        )

    model.eval()
    return model, tok


def extract_thoughts(
    prompts: List[str],
    src_model,
    src_tok,
    device: str,
) -> List[torch.Tensor]:
    """
    Extract last hidden state vector at last token: hidden_states[-1][:, -1, :]
    Returns list of CPU tensors of shape (1, 2048).
    """
    thought_vectors_cpu: List[torch.Tensor] = []

    for p in prompts:
        p_src = format_prompt_like_dataset(p)
        inputs = src_tok(p_src, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = src_model(**inputs, output_hidden_states=True)
            vec = out.hidden_states[-1][:, -1, :].detach().cpu()  # (1, 2048)
            thought_vectors_cpu.append(vec)

        del inputs, out

    return thought_vectors_cpu


def build_vec_injected(
    vec_cpu: torch.Tensor,
    adapter: LIPAdapter,
    tgt_model,
    gain: float,
) -> torch.Tensor:
    """
    DeepSeek vec (1,2048) -> adapter -> (1,4096) then energy calibration vs target embedding mean norm.
    """
    with torch.no_grad():
        vec_src = vec_cpu.to("cuda")  # adapter lives on cuda (local-first)
        vec_translated = adapter(vec_src)  # (1,4096)

        ref_energy = tgt_model.get_input_embeddings().weight.norm(p=2, dim=-1).mean().item()
        current_norm = vec_translated.norm(p=2, dim=-1)
        scale = (ref_energy / (current_norm + 1e-6)) * float(gain)

        vec_injected = (vec_translated * scale).to(tgt_model.device).to(tgt_model.dtype)  # (1,4096)
        return vec_injected


def run_ab(
    prompt_text: str,
    vec_injected: torch.Tensor,
    tgt_model,
    tgt_tok,
    layer_idx: int,
    inject_pos_mode: str,
    gen_kwargs: Dict,
) -> Dict[bool, str]:
    """
    Runs deterministic AB: enable_lip in {False, True}.
    """
    prompt = format_prompt_like_dataset(prompt_text)
    inputs = tgt_tok(prompt, return_tensors="pt", add_special_tokens=False).to(tgt_model.device)
    input_ids = inputs["input_ids"]
    attn_mask = inputs.get("attention_mask", None)

    if inject_pos_mode == "last":
        inject_pos = input_ids.shape[1] - 1
    elif inject_pos_mode == "last_minus_1":
        inject_pos = input_ids.shape[1] - 2
    else:
        raise ValueError(f"Unknown inject_pos_mode: {inject_pos_mode}")

    results: Dict[bool, str] = {}

    for enable_lip in [False, True]:
        hook_fn = make_lip_hook(vec_injected=vec_injected, inject_pos=inject_pos, enable=enable_lip)
        handle = tgt_model.model.layers[layer_idx].register_forward_hook(hook_fn)

        with torch.no_grad():
            gen = tgt_model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                **gen_kwargs
            )

        handle.remove()

        out = tgt_tok.decode(gen[0], skip_special_tokens=True).replace("</s>", "")
        results[enable_lip] = out

    return results


def run_inference(cfg_dict: Dict):
    # Parse cfg dict -> config object
    cfg = InferenceConfig(
        source_model=cfg_dict["source_model"],
        target_model=cfg_dict["target_model"],
        adapter_ckpt=cfg_dict["adapter_ckpt"],
        device_src=cfg_dict.get("runtime", {}).get("device_src", "cuda"),
        device_tgt=cfg_dict.get("runtime", {}).get("device_tgt", "auto"),
        load_4bit=cfg_dict.get("runtime", {}).get("load_4bit", True),
        gain=float(cfg_dict.get("lip", {}).get("gain", 10.0)),
        layer_idx=int(cfg_dict.get("lip", {}).get("layer_idx", -2)),
        inject_pos_mode=str(cfg_dict.get("lip", {}).get("inject_pos_mode", "last")),
        max_new_tokens=int(cfg_dict.get("generation", {}).get("max_new_tokens", 120)),
        do_sample=bool(cfg_dict.get("generation", {}).get("do_sample", False)),
        temperature=float(cfg_dict.get("generation", {}).get("temperature", 0.0)),
        repetition_penalty=float(cfg_dict.get("generation", {}).get("repetition_penalty", 1.2)),
        prompts=list(cfg_dict.get("prompts", [])),
    )

    if not cfg.prompts:
        raise ValueError("No prompts provided in config under `prompts:`")

    # 0) safety: deterministic AB by default
    if cfg.do_sample and cfg.temperature == 0.0:
        # if user insists sampling, temperature=0 is inconsistent but harmless; keep as is
        pass

    print("🚀 LIP local-first inference")
    print(f"   source: {cfg.source_model}")
    print(f"   target: {cfg.target_model}")
    print(f"   adapter_ckpt: {cfg.adapter_ckpt}")
    print(f"   layer_idx: {cfg.layer_idx} | inject_pos_mode: {cfg.inject_pos_mode} | gain: {cfg.gain}")

    # 1) Load source, extract thoughts, free source
    print("\n🧠 [Phase 1] Loading source...")
    src_model, src_tok = load_source(cfg.source_model, cfg.device_src)
    print("   ✅ source loaded")

    print("   💭 extracting thoughts...")
    thought_vectors_cpu = extract_thoughts(cfg.prompts, src_model, src_tok, cfg.device_src)

    print("   🗑️ freeing source...")
    del src_model, src_tok
    _cleanup_cuda()
    print("   ✅ source freed")

    # 2) Load target + adapter
    print("\n🗣️ [Phase 2] Loading target...")
    tgt_model, tgt_tok = load_target(cfg.target_model, cfg.device_tgt, cfg.load_4bit)
    print(f"   ✅ target loaded on: {tgt_model.device}")

    print("   🔌 loading adapter...")
    adapter = LIPAdapter().to("cuda")
    state = torch.load(cfg.adapter_ckpt, map_location="cuda", weights_only=False)
    adapter.load_state_dict(state)
    adapter.eval()
    print("   ✅ adapter ready")

    gen_kwargs = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        repetition_penalty=cfg.repetition_penalty,
        pad_token_id=tgt_tok.eos_token_id,
    )

    # 3) AB per prompt, using correct vec_injected for each prompt
    print("\n" + "=" * 60)
    print("🧪 Running AB (False/True) per prompt")
    print("=" * 60)

    for p, vec_cpu in zip(cfg.prompts, thought_vectors_cpu):
        vec_injected = build_vec_injected(vec_cpu, adapter, tgt_model, cfg.gain)
        ab = run_ab(
            prompt_text=p,
            vec_injected=vec_injected,
            tgt_model=tgt_model,
            tgt_tok=tgt_tok,
            layer_idx=cfg.layer_idx,
            inject_pos_mode=cfg.inject_pos_mode,
            gen_kwargs=gen_kwargs,
        )

        print("\n" + "-" * 60)
        print("PROMPT:", p)
        print("\n--- LIP=False ---\n", ab[False])
        print("\n--- LIP=True  ---\n", ab[True])

    print("\n🏁 Done.")
