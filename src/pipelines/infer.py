"""Local inference helpers for LIP latent injection.

The legacy entry point remains available for same-task A/B diagnostics.  The
scientific source-only protocol lives in ``src.scripts.run_source_only_probe``
and deliberately never sends the task prompt to the target model.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.core.hidden_states import last_non_padding_indices, select_hidden_vectors
from src.core.models import LIPAdapter
from src.core.prompt_protocol import (
    format_prompt,
    format_prompts,
    tokenizer_add_special_tokens,
)
from src.integrations.hooks import make_lip_hook


@dataclass
class InferenceConfig:
    source_model: str
    target_model: str
    adapter_ckpt: str
    source_revision: Optional[str] = None
    target_revision: Optional[str] = None
    device_src: str = "cuda"
    device_tgt: str = "auto"
    device_adapter: str = "cuda"
    load_4bit: bool = True
    source_load_4bit: bool = False
    gain: float = 10.0
    layer_idx: int = -2
    inject_pos_mode: str = "last_non_padding"
    injection_mode: str = "add"
    source_layer: int = -1
    token_position: str = "last_non_padding"
    input_dim: int = 2048
    hidden_dim: int = 1024
    output_dim: int = 4096
    max_new_tokens: int = 120
    do_sample: bool = False
    temperature: float = 0.0
    repetition_penalty: float = 1.2
    prompts: List[str] = field(default_factory=list)
    prompt_protocol: Dict[str, Any] = field(default_factory=dict)


def format_prompt_like_dataset(user_text: str) -> str:
    """Deprecated compatibility shim; prompt formatting is now configuration-driven."""

    return user_text


def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def _ensure_padding_token(tokenizer):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer needs a pad_token or eos_token")
        tokenizer.pad_token = tokenizer.eos_token


def _verify_loaded_revision(model, tokenizer, revision: Optional[str]):
    if revision is None:
        return
    model_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    tokenizer_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if model_revision != revision or (
        tokenizer_revision is not None and tokenizer_revision != revision
    ):
        raise RuntimeError(
            "loaded model/tokenizer revisions do not match the requested immutable commit"
        )


def model_input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def adapter_device(adapter: LIPAdapter) -> torch.device:
    return next(adapter.parameters()).device


def load_source(
    model_id: str,
    device: str,
    load_4bit: bool = False,
    revision: Optional[str] = None,
):
    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        revision=revision,
    )
    _ensure_padding_token(tok)
    kwargs = {
        "device_map": device,
        "use_safetensors": True,
        "trust_remote_code": True,
        "revision": revision,
    }
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        kwargs["torch_dtype"] = torch.float16
        kwargs["low_cpu_mem_usage"] = True
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    _verify_loaded_revision(model, tok, revision)
    model.eval()
    return model, tok


def load_target(
    model_id: str,
    device: str,
    load_4bit: bool,
    revision: Optional[str] = None,
):
    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        revision=revision,
    )
    _ensure_padding_token(tok)

    if load_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device,
            use_safetensors=True,
            trust_remote_code=True,
            revision=revision,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            trust_remote_code=True,
            revision=revision,
        )

    _verify_loaded_revision(model, tok, revision)
    model.eval()
    return model, tok


def extract_prompt_vectors(
    prompts: List[str],
    model,
    tokenizer,
    device: Optional[str] = None,
    *,
    protocol_config: Optional[Mapping[str, Any]] = None,
    layer_idx: int = -1,
    token_position: str = "last_non_padding",
    max_length: int = 512,
) -> List[torch.Tensor]:
    """Extract one CPU float vector per prompt under the shared protocol."""

    formatted = format_prompts(prompts, tokenizer, protocol_config)
    destination = (
        model_input_device(model)
        if device is None or device == "auto"
        else torch.device(device)
    )
    add_special_tokens = tokenizer_add_special_tokens(protocol_config)
    vectors: List[torch.Tensor] = []

    for prompt in formatted:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        inputs = {key: value.to(destination) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            if outputs.hidden_states is None:
                raise RuntimeError("model did not return hidden states")
            selected = select_hidden_vectors(
                outputs.hidden_states[layer_idx],
                inputs.get("attention_mask"),
                token_position=token_position,
            )
            vectors.append(selected.detach().cpu().float())

        del inputs, outputs

    return vectors


def extract_thoughts(
    prompts: List[str],
    src_model,
    src_tok,
    device: str,
    protocol_config: Optional[Mapping[str, Any]] = None,
    layer_idx: int = -1,
    token_position: str = "last_non_padding",
) -> List[torch.Tensor]:
    """Backward-compatible alias for source prompt-vector extraction."""

    return extract_prompt_vectors(
        prompts,
        src_model,
        src_tok,
        device,
        protocol_config=protocol_config,
        layer_idx=layer_idx,
        token_position=token_position,
    )


def load_adapter_checkpoint_safely(path: str, map_location: str | torch.device):
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "adapter loading requires torch.load(..., weights_only=True) support"
        ) from exc
    if not isinstance(checkpoint, dict):
        raise TypeError("adapter checkpoint must contain a state_dict mapping")
    state = checkpoint.get("model_state", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("checkpoint['model_state'] must be a state_dict mapping")
    return state


def calibrate_for_target(
    vector: torch.Tensor,
    tgt_model,
    gain: float,
    reference_norm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Match target embedding mean norm, then apply the configured intervention gain."""

    embeddings = tgt_model.get_input_embeddings().weight
    destination = embeddings.device
    output_dtype = embeddings.dtype
    vector = vector.to(destination).float()
    if reference_norm is None:
        reference_norm = target_embedding_mean_norm(tgt_model)
    reference_norm = torch.as_tensor(
        reference_norm,
        device=destination,
        dtype=torch.float32,
    )
    current_norm = vector.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-6)
    calibrated = vector * (reference_norm / current_norm) * float(gain)
    return calibrated.to(dtype=output_dtype)


def build_vec_injected(
    vec_cpu: torch.Tensor,
    adapter: LIPAdapter,
    tgt_model,
    gain: float,
    reference_norm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Translate a source vector and calibrate it for the target model."""

    with torch.no_grad():
        translated = adapter(vec_cpu.to(adapter_device(adapter)))
        return calibrate_for_target(
            translated,
            tgt_model,
            gain,
            reference_norm=reference_norm,
        )


def translate_source_vector(
    vec_cpu: torch.Tensor,
    adapter: LIPAdapter,
    gain: float = 1.0,
) -> torch.Tensor:
    """Map a source vector without changing its learned target-space norm."""

    with torch.no_grad():
        return adapter(vec_cpu.to(adapter_device(adapter))).detach() * float(gain)


def target_embedding_mean_norm(tgt_model) -> torch.Tensor:
    """Compute the calibration reference once per loaded target model."""

    embeddings = tgt_model.get_input_embeddings().weight.detach()
    return embeddings.norm(p=2, dim=-1).float().mean()


def resolve_injection_position(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    mode: str,
) -> int:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("latent generation currently requires a single prompt")

    if mode in {"last", "last_non_padding"}:
        if attention_mask is None:
            position = input_ids.shape[1] - 1
        else:
            position = int(last_non_padding_indices(attention_mask)[0].item())
    elif mode in {"last_minus_1", "last_non_padding_minus_1"}:
        if attention_mask is None:
            position = input_ids.shape[1] - 2
        else:
            position = int(last_non_padding_indices(attention_mask)[0].item()) - 1
    else:
        raise ValueError(f"unknown inject_pos_mode: {mode}")

    if position < 0:
        raise ValueError("prompt is too short for the requested injection position")
    return position


def generate_with_optional_injection(
    prompt_text: str,
    vec_injected: Optional[torch.Tensor],
    tgt_model,
    tgt_tok,
    layer_idx: int,
    inject_pos_mode: str,
    gen_kwargs: Dict[str, Any],
    *,
    protocol_config: Optional[Mapping[str, Any]] = None,
    injection_mode: str = "add",
) -> str:
    """Generate only the continuation, optionally injecting one vector once."""

    prompt = format_prompt(prompt_text, tgt_tok, protocol_config)
    inputs = tgt_tok(
        prompt,
        return_tensors="pt",
        add_special_tokens=tokenizer_add_special_tokens(protocol_config),
    )
    destination = model_input_device(tgt_model)
    inputs = {key: value.to(destination) for key, value in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    handle = None

    if vec_injected is not None:
        inject_pos = resolve_injection_position(
            input_ids,
            attention_mask,
            inject_pos_mode,
        )
        hook_fn = make_lip_hook(
            vec_injected=vec_injected,
            inject_pos=inject_pos,
            enable=True,
            mode=injection_mode,
        )
        handle = tgt_model.model.layers[layer_idx].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            generated = tgt_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
    finally:
        if handle is not None:
            handle.remove()

    continuation = generated[0, input_ids.shape[1]:]
    return tgt_tok.decode(continuation, skip_special_tokens=True).replace("</s>", "").strip()


def run_ab(
    prompt_text: str,
    vec_injected: torch.Tensor,
    tgt_model,
    tgt_tok,
    layer_idx: int,
    inject_pos_mode: str,
    gen_kwargs: Dict[str, Any],
    protocol_config: Optional[Mapping[str, Any]] = None,
    injection_mode: str = "add",
) -> Dict[bool, str]:
    """Run a same-task no-injection/injection diagnostic pair."""

    return {
        enabled: generate_with_optional_injection(
            prompt_text,
            vec_injected if enabled else None,
            tgt_model,
            tgt_tok,
            layer_idx,
            inject_pos_mode,
            gen_kwargs,
            protocol_config=protocol_config,
            injection_mode=injection_mode,
        )
        for enabled in (False, True)
    }


def generation_kwargs(cfg: InferenceConfig, tokenizer) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": cfg.do_sample,
        "repetition_penalty": cfg.repetition_penalty,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if cfg.do_sample:
        if cfg.temperature <= 0:
            raise ValueError("temperature must be positive when do_sample=true")
        kwargs["temperature"] = cfg.temperature
    return kwargs


def run_inference(cfg_dict: Dict[str, Any]):
    runtime = cfg_dict.get("runtime", {})
    lip = cfg_dict.get("lip", {})
    extraction = cfg_dict.get("extraction", {})
    model_cfg = cfg_dict.get("adapter", {})
    generation = cfg_dict.get("generation", {})
    cfg = InferenceConfig(
        source_model=cfg_dict["source_model"],
        target_model=cfg_dict["target_model"],
        adapter_ckpt=cfg_dict["adapter_ckpt"],
        source_revision=cfg_dict.get("source_revision"),
        target_revision=cfg_dict.get("target_revision"),
        device_src=runtime.get("device_src", "cuda"),
        device_tgt=runtime.get("device_tgt", "auto"),
        device_adapter=runtime.get("device_adapter", "cuda"),
        load_4bit=bool(runtime.get("load_4bit", True)),
        source_load_4bit=bool(runtime.get("source_load_4bit", False)),
        gain=float(lip.get("gain", 10.0)),
        layer_idx=int(lip.get("layer_idx", -2)),
        inject_pos_mode=str(lip.get("inject_pos_mode", "last_non_padding")),
        injection_mode=str(lip.get("injection_mode", "add")),
        source_layer=int(extraction.get("source_layer", -1)),
        token_position=str(extraction.get("token_position", "last_non_padding")),
        input_dim=int(model_cfg.get("input_dim", 2048)),
        hidden_dim=int(model_cfg.get("hidden_dim", 1024)),
        output_dim=int(model_cfg.get("output_dim", 4096)),
        max_new_tokens=int(generation.get("max_new_tokens", 120)),
        do_sample=bool(generation.get("do_sample", False)),
        temperature=float(generation.get("temperature", 0.0)),
        repetition_penalty=float(generation.get("repetition_penalty", 1.2)),
        prompts=list(cfg_dict.get("prompts", [])),
        prompt_protocol=dict(cfg_dict.get("prompt_protocol", {})),
    )
    if not cfg.prompts:
        raise ValueError("no prompts provided in config under prompts")

    print("LIP same-task diagnostic inference")
    print(f"source: {cfg.source_model}")
    print(f"target: {cfg.target_model}")

    src_model, src_tok = load_source(
        cfg.source_model,
        cfg.device_src,
        cfg.source_load_4bit,
        cfg.source_revision,
    )
    vectors = extract_thoughts(
        cfg.prompts,
        src_model,
        src_tok,
        cfg.device_src,
        protocol_config=cfg.prompt_protocol,
        layer_idx=cfg.source_layer,
        token_position=cfg.token_position,
    )
    del src_model, src_tok
    _cleanup_cuda()

    tgt_model, tgt_tok = load_target(
        cfg.target_model,
        cfg.device_tgt,
        cfg.load_4bit,
        cfg.target_revision,
    )
    adapter = LIPAdapter(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
    ).to(cfg.device_adapter)
    adapter.load_state_dict(
        load_adapter_checkpoint_safely(cfg.adapter_ckpt, cfg.device_adapter)
    )
    adapter.eval()
    gen_kwargs = generation_kwargs(cfg, tgt_tok)

    for prompt, vector in zip(cfg.prompts, vectors):
        injected = build_vec_injected(vector, adapter, tgt_model, cfg.gain)
        results = run_ab(
            prompt,
            injected,
            tgt_model,
            tgt_tok,
            cfg.layer_idx,
            cfg.inject_pos_mode,
            gen_kwargs,
            protocol_config=cfg.prompt_protocol,
            injection_mode=cfg.injection_mode,
        )
        print("\nPROMPT:", prompt)
        print("\n--- LIP=False ---\n", results[False])
        print("\n--- LIP=True ---\n", results[True])

    print("\nDone.")
