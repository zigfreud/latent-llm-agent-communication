import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from src.core.hidden_states import SUPPORTED_TOKEN_POSITIONS, select_hidden_vectors
from src.core.prompt_protocol import (
    format_prompts,
    protocol_metadata,
    tokenizer_add_special_tokens,
)
from src.scripts.package_latent_bundle import package_bundle
from src.scripts.validate_latent_bundle import BundleValidationError, validate_bundle


DEFAULT_CONFIG = Path("config/LIP-H0-005_real_tiny_bundle.yaml")
DEFAULT_LICENSE_NOTES = (
    "Generated from the configured tiny prompt set. Review model, tokenizer, "
    "and dataset licenses before sharing the bundle."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a tiny H0-003-compatible latent bundle for H0-005."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate deterministic mock tensors without loading Hugging Face models.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override extraction.device from config. Use cpu, cuda, or auto.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-zip", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("config must contain a YAML mapping")

    return config


def get_nested(config, *keys):
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"config missing required field: {'.'.join(keys)}")
        value = value[key]
    return value


def positive_int(value, name):
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def config_bool(config_section, key, default):
    value = config_section.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"extraction.{key} must be a boolean")
    return value


def normalize_device_map(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("extraction.device_map must be auto, none, or null")

    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized == "none":
        return None
    raise ValueError("extraction.device_map must be auto, none, or null")


def normalize_cache_dir(value):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("extraction.cache_dir must be null or a non-empty string")
    return value


def selected_prompts(config, max_samples):
    prompts = get_nested(config, "data", "prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("data.prompts must be a non-empty list")
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError("data.prompts entries must be non-empty strings")

    if max_samples is not None:
        positive_int(max_samples, "max_samples")
        prompts = prompts[:max_samples]

    if not prompts:
        raise ValueError("no prompts selected")

    return prompts


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_output_paths(config, output_dir_override, output_zip_override):
    bundle_dir = output_dir_override or Path(get_nested(config, "output", "bundle_dir"))
    output_zip = output_zip_override or Path(get_nested(config, "output", "output_zip"))
    shard_name = get_nested(config, "output", "shard_name")

    if not isinstance(shard_name, str) or not shard_name.endswith(".pt"):
        raise ValueError("output.shard_name must be a .pt filename")
    if "/" in shard_name or "\\" in shard_name:
        raise ValueError("output.shard_name must be a direct file name")

    return Path(bundle_dir), Path(output_zip), shard_name


def resolve_device(config, override):
    requested = override or get_nested(config, "extraction", "device")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu, cuda, or auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda was requested, but CUDA is not available")
    return requested


def resolve_dtype(config, device):
    dtype = get_nested(config, "extraction", "dtype")
    if dtype == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError("extraction.dtype must be auto, float32, float16, or bfloat16")


def make_dry_run_records(num_samples, input_dim, output_dim):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    records = []
    for _ in range(num_samples):
        records.append(
            {
                "src_vector": torch.randn(input_dim, generator=generator),
                "tgt_vector": torch.randn(output_dim, generator=generator),
            }
        )
    return records


def tokenize_batch(tokenizer, prompts, max_length, device, add_special_tokens=True):
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def build_model_load_kwargs(config, dtype):
    extraction_config = get_nested(config, "extraction")
    trust_remote_code = bool(extraction_config.get("trust_remote_code", False))
    low_cpu_mem_usage = config_bool(extraction_config, "low_cpu_mem_usage", False)
    device_map = normalize_device_map(extraction_config.get("device_map"))
    load_in_4bit = config_bool(extraction_config, "load_in_4bit", False)
    use_safetensors = config_bool(extraction_config, "use_safetensors", True)
    cache_dir = normalize_cache_dir(extraction_config.get("cache_dir"))
    local_files_only = config_bool(extraction_config, "local_files_only", False)

    kwargs = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "local_files_only": local_files_only,
        "use_safetensors": use_safetensors,
    }
    tokenizer_kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }

    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
        tokenizer_kwargs["cache_dir"] = cache_dir

    if device_map is not None:
        kwargs["device_map"] = device_map

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
            )
        except Exception as exc:
            raise RuntimeError(
                "load_in_4bit=true requires a transformers version with "
                "BitsAndBytesConfig and a working bitsandbytes installation."
            ) from exc
    else:
        kwargs["torch_dtype"] = dtype

    return kwargs, tokenizer_kwargs, device_map, load_in_4bit


def extract_hidden_vectors(
    model_name,
    prompts,
    layer_index,
    expected_dim,
    config,
    device,
    dtype,
    revision=None,
):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Real extraction requires transformers. Dry-run mode does not import "
            "or load Hugging Face models."
        ) from exc

    extraction_config = get_nested(config, "extraction")
    max_length = positive_int(extraction_config.get("max_length"), "extraction.max_length")
    batch_size = positive_int(extraction_config.get("batch_size"), "extraction.batch_size")
    token_position = extraction_config.get("token_position", "last_non_padding")
    if token_position not in SUPPORTED_TOKEN_POSITIONS:
        allowed = ", ".join(sorted(SUPPORTED_TOKEN_POSITIONS))
        raise ValueError(f"extraction.token_position must be one of: {allowed}")
    model_kwargs, tokenizer_kwargs, device_map, load_in_4bit = build_model_load_kwargs(
        config,
        dtype,
    )
    if revision is not None:
        model_kwargs["revision"] = revision
        tokenizer_kwargs["revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    tokenizer_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if revision is None and tokenizer_revision:
        model_kwargs["revision"] = tokenizer_revision
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    formatted_prompts = format_prompts(
        prompts,
        tokenizer,
        config.get("prompt_protocol"),
    )
    add_special_tokens = tokenizer_add_special_tokens(config.get("prompt_protocol"))

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as exc:
        if load_in_4bit:
            raise RuntimeError(
                f"Failed to load {model_name} with load_in_4bit=true. "
                "Install compatible transformers, accelerate, and bitsandbytes "
                "packages, and run on hardware that supports 4-bit loading."
            ) from exc
        raise

    resolved_revision = (
        getattr(getattr(model, "config", None), "_commit_hash", None)
        or model_kwargs.get("revision")
        or tokenizer_revision
    )
    if tokenizer_revision and resolved_revision and tokenizer_revision != resolved_revision:
        raise RuntimeError(
            f"{model_name} tokenizer/model revisions differ: "
            f"{tokenizer_revision} != {resolved_revision}"
        )

    if device_map is None:
        model.to(device)
    model.eval()

    vectors = []
    try:
        for start in range(0, len(formatted_prompts), batch_size):
            batch_prompts = formatted_prompts[start:start + batch_size]
            inputs = tokenize_batch(
                tokenizer,
                batch_prompts,
                max_length,
                device,
                add_special_tokens=add_special_tokens,
            )
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)

            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError(f"{model_name} did not return hidden states")

            selected = hidden_states[layer_index]
            batch_vectors = select_hidden_vectors(
                selected,
                inputs.get("attention_mask"),
                token_position=token_position,
            ).detach().cpu().float()
            for vector in batch_vectors:
                squeezed = vector.squeeze()
                if tuple(squeezed.shape) != (expected_dim,):
                    raise RuntimeError(
                        f"{model_name} produced vector shape {tuple(squeezed.shape)}; "
                        f"expected ({expected_dim},)"
                    )
                vectors.append(squeezed)
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return vectors, resolved_revision


def make_real_records(prompts, input_dim, output_dim, config, device):
    models_config = get_nested(config, "models")
    source_model = get_nested(models_config, "source_model")
    target_model = get_nested(models_config, "target_model")
    source_revision = models_config.get("source_revision")
    target_revision = models_config.get("target_revision")
    source_layer = get_nested(config, "extraction", "source_layer")
    target_layer = get_nested(config, "extraction", "target_layer")
    dtype = resolve_dtype(config, device)

    src_vectors, resolved_source_revision = extract_hidden_vectors(
        source_model,
        prompts,
        source_layer,
        input_dim,
        config,
        device,
        dtype,
        revision=source_revision,
    )
    tgt_vectors, resolved_target_revision = extract_hidden_vectors(
        target_model,
        prompts,
        target_layer,
        output_dim,
        config,
        device,
        dtype,
        revision=target_revision,
    )
    if len(src_vectors) != len(tgt_vectors):
        raise RuntimeError(
            f"source extracted {len(src_vectors)} vectors, "
            f"target extracted {len(tgt_vectors)} vectors"
        )

    return (
        [
            {"src_vector": src_vector, "tgt_vector": tgt_vector}
            for src_vector, tgt_vector in zip(src_vectors, tgt_vectors)
        ],
        {
            "source_model_revision": resolved_source_revision,
            "target_model_revision": resolved_target_revision,
        },
    )


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_bundle(
    config,
    records,
    bundle_dir,
    output_zip,
    shard_name,
    dry_run,
    extraction_metadata=None,
):
    input_dim = positive_int(get_nested(config, "vectors", "input_dim"), "vectors.input_dim")
    output_dim = positive_int(get_nested(config, "vectors", "output_dim"), "vectors.output_dim")
    if not records:
        raise ValueError("records must be non-empty")

    shards_dir = bundle_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shards_dir / shard_name
    torch.save(records, shard_path)
    shard_digest = sha256_file(shard_path)

    source_layer = get_nested(config, "extraction", "source_layer")
    target_layer = get_nested(config, "extraction", "target_layer")
    manifest = {
        "bundle_format": get_nested(config, "bundle_format"),
        "schema_version": get_nested(config, "schema_version"),
        "trace_id": get_nested(config, "trace_id"),
        "source_model": get_nested(config, "models", "source_model"),
        "target_model": get_nested(config, "models", "target_model"),
        "dataset_origin": get_nested(config, "data", "dataset_origin"),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "num_samples": len(records),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "license_notes": DEFAULT_LICENSE_NOTES,
        "source_layer": str(source_layer),
        "target_layer": str(target_layer),
        "token_position": get_nested(config, "extraction", "token_position"),
        "prompt_protocol": protocol_metadata(config.get("prompt_protocol")),
        "extraction_mode": "dry_run" if dry_run else "real",
        "source_quantization": (
            "bitsandbytes-4bit"
            if get_nested(config, "extraction", "load_in_4bit")
            else "none"
        ),
        "target_quantization": (
            "bitsandbytes-4bit"
            if get_nested(config, "extraction", "load_in_4bit")
            else "none"
        ),
        "quantization_compute_dtype": get_nested(config, "extraction", "dtype"),
        "use_safetensors": bool(
            get_nested(config, "extraction").get("use_safetensors", True)
        ),
        "max_length": get_nested(config, "extraction", "max_length"),
        "prompt_policy": get_nested(config, "data", "prompt_policy"),
        "extraction_notes": (
            "Dry-run deterministic mock tensors; no Hugging Face models were loaded."
            if dry_run
            else "Real hidden-state extraction under the recorded prompt protocol; prompt text and model outputs are not stored in shards."
        ),
        "shards": [
            {
                "path": f"shards/{shard_name}",
                "records": len(records),
                "sha256": shard_digest,
            }
        ],
    }
    data_config = get_nested(config, "data")
    for field in (
        "source_dataset",
        "source_dataset_config",
        "source_split",
        "prompt_field",
        "sampling_seed",
        "sampled_ids",
    ):
        value = data_config.get(field)
        if field == "sampled_ids" and isinstance(value, list):
            value = value[: len(records)]
        if value is not None:
            manifest[field] = value
    sampled_prompts = data_config.get("prompts")
    if sampled_prompts is not None and data_config.get("sampled_ids") is not None:
        manifest["sampled_prompt_sha256"] = [
            hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for prompt in sampled_prompts[: len(records)]
        ]
    for field in ("source_model_revision", "target_model_revision"):
        value = (extraction_metadata or {}).get(field)
        if value:
            manifest[field] = value
    write_json(bundle_dir / "manifest.json", manifest)
    return bundle_dir / "manifest.json", shard_path


def build_bundle(args):
    config = load_config(args.config)
    bundle_dir, output_zip, shard_name = resolve_output_paths(
        config,
        args.output_dir,
        args.output_zip,
    )
    prompts = selected_prompts(config, args.max_samples)
    input_dim = positive_int(get_nested(config, "vectors", "input_dim"), "vectors.input_dim")
    output_dim = positive_int(get_nested(config, "vectors", "output_dim"), "vectors.output_dim")

    if args.dry_run:
        records = make_dry_run_records(len(prompts), input_dim, output_dim)
        extraction_metadata = {}
    else:
        device = resolve_device(config, args.device)
        records, extraction_metadata = make_real_records(
            prompts,
            input_dim,
            output_dim,
            config,
            device,
        )

    manifest_path, shard_path = write_bundle(
        config,
        records,
        bundle_dir,
        output_zip,
        shard_name,
        args.dry_run,
        extraction_metadata=extraction_metadata,
    )
    package_bundle(bundle_dir / "shards", manifest_path, output_zip)

    try:
        report = validate_bundle(bundle_dir)
    except BundleValidationError as exc:
        raise RuntimeError(f"built bundle failed validation: {exc}") from exc

    zip_digest = sha256_file(output_zip)
    return {
        "bundle_dir": bundle_dir,
        "manifest_path": manifest_path,
        "shard_path": shard_path,
        "output_zip": output_zip,
        "zip_sha256": zip_digest,
        "records": report["total_records"],
    }


def main():
    args = parse_args()
    result = build_bundle(args)
    print("Latent bundle build passed")
    print(f"bundle_dir: {result['bundle_dir']}")
    print(f"manifest: {result['manifest_path']}")
    print(f"shard: {result['shard_path']}")
    print(f"output_zip: {result['output_zip']}")
    print(f"records: {result['records']}")
    print(f"sha256: {result['zip_sha256']}")


if __name__ == "__main__":
    main()
