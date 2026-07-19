import argparse
import json
import random
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path("config/LIP-DATA-003_mbpp_sampling.yaml")
DEFAULT_TRAIN_TRACE_ID = "LIP-DATA-003-MBPP-TRAIN"
DEFAULT_EVAL_TRACE_ID = "LIP-DATA-003-MBPP-EVAL"
DEFAULT_TRAIN_CONFIG_NAME = "LIP-DATA-003_train_mbpp_32.yaml"
DEFAULT_EVAL_CONFIG_NAME = "LIP-DATA-003_eval_mbpp_16.yaml"
DEFAULT_TRAIN_BUNDLE_DIR = "datasets/LIP-DATA-003/mbpp_train_bundle_32"
DEFAULT_EVAL_BUNDLE_DIR = "datasets/LIP-DATA-003/mbpp_eval_bundle_16"
DEFAULT_TRAIN_OUTPUT_ZIP = (
    "datasets/LIP-DATA-003/LIP-DATA-003_mbpp_train_latent_bundle_32.zip"
)
DEFAULT_EVAL_OUTPUT_ZIP = (
    "datasets/LIP-DATA-003/LIP-DATA-003_mbpp_eval_latent_bundle_16.zip"
)
DEFAULT_TRAIN_DATASET_ORIGIN = "LIP-DATA-003 MBPP train prompt sample"
DEFAULT_EVAL_DATASET_ORIGIN = "LIP-DATA-003 MBPP held-out validation prompt sample"
DEFAULT_TRAIN_PROMPT_POLICY = (
    "Thirty-two public MBPP train split natural-language prompts; "
    "code, tests, and completions are not included."
)
DEFAULT_EVAL_PROMPT_POLICY = (
    "Sixteen public MBPP validation split natural-language prompts held "
    "out from the train prompt sample; code, tests, and completions are "
    "not included."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Materialize MBPP prompt configs for LIP-DATA-003."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mock-data",
        action="store_true",
        help="Use deterministic mock prompt rows instead of loading Hugging Face datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the generated config output directory.",
    )
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def required_string(config, key):
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def optional_string(config, key, default):
    value = config.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def config_filename(config, key, default):
    value = optional_string(config, key, default)
    if "/" in value or "\\" in value:
        raise ValueError(f"{key} must be a direct filename")
    if not value.endswith((".yaml", ".yml")):
        raise ValueError(f"{key} must be a YAML filename")
    return value


def positive_int(config, key):
    value = config.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def load_split_rows(config, split_name, mock_data):
    prompt_field = required_string(config, "prompt_field")
    if mock_data:
        return make_mock_rows(split_name, prompt_field, 128)

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "MBPP materialization requires the Hugging Face datasets package. "
            "Install datasets or use --mock-data for offline validation."
        ) from exc

    dataset_name = required_string(config, "dataset_name")
    dataset_config = config.get("dataset_config")
    dataset = load_dataset(dataset_name, dataset_config, split=split_name)
    return list(dataset)


def make_mock_rows(split_name, prompt_field, count):
    rows = []
    split_slug = split_name.replace("/", "_").replace("[", "_").replace("]", "_")
    for index in range(count):
        rows.append(
            {
                "task_id": f"mock-{split_slug}-{index:03d}",
                prompt_field: (
                    "Write a Python function for mock MBPP prompt "
                    f"{split_slug}-{index:03d}."
                ),
            }
        )
    return rows


def normalize_row(row, prompt_field, max_prompt_chars, fallback_id):
    if prompt_field not in row:
        raise ValueError(f"row is missing prompt field {prompt_field!r}")

    prompt = row[prompt_field]
    if not isinstance(prompt, str):
        raise ValueError(f"prompt field {prompt_field!r} must contain strings")

    prompt = prompt.strip()
    if not prompt:
        return None
    if len(prompt) > max_prompt_chars:
        return None

    sample_id = str(row.get("task_id", fallback_id))
    tests = row.get("test_list", row.get("tests", []))
    if isinstance(tests, str):
        tests = [tests]
    if not isinstance(tests, list) or any(not isinstance(test, str) for test in tests):
        raise ValueError(f"task {sample_id!r} tests must be text or a list of text")
    entry_point = row.get("entry_point")
    if entry_point is not None and not isinstance(entry_point, str):
        entry_point = str(entry_point)
    return {
        "id": sample_id,
        "prompt": prompt,
        "task": {
            "task_id": sample_id,
            "prompt": prompt,
            "test_list": tests,
            "test_setup_code": str(row.get("test_setup_code", "") or ""),
            "entry_point": entry_point,
        },
    }


def sample_prompts(rows, count, seed, prompt_field, max_prompt_chars):
    candidates = []
    for index, row in enumerate(rows):
        normalized = normalize_row(row, prompt_field, max_prompt_chars, index)
        if normalized is not None:
            candidates.append(normalized)

    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} prompts available after filtering; "
            f"{count} requested."
        )

    sampler = random.Random(seed)
    selected = sampler.sample(candidates, count)
    selected.sort(key=lambda item: item["id"])
    return selected


def build_bundle_config(
    trace_id,
    dataset_origin,
    prompt_policy,
    selected,
    output_bundle_dir,
    output_zip,
    source_split,
    sampling_config,
):
    return {
        "trace_id": trace_id,
        "bundle_format": "lip_latent_bundle",
        "schema_version": 1,
        "models": {
            "source_model": "deepseek-ai/deepseek-coder-1.3b-base",
            "target_model": "NousResearch/Meta-Llama-3-8B-Instruct",
        },
        "prompt_protocol": {
            "version": "lip-prompt-v1",
            "mode": "raw",
            "add_generation_prompt": False,
            "system_prompt": None,
        },
        "extraction": {
            "device": "auto",
            "source_layer": -1,
            "target_layer": int(sampling_config.get("target_layer", -1)),
            "token_position": "last_non_padding",
            "max_length": int(sampling_config.get("max_length", 256)),
            "batch_size": 1,
            "dtype": str(sampling_config.get("dtype", "auto")),
            "trust_remote_code": True,
            "sequential_model_loading": True,
            "low_cpu_mem_usage": True,
            "device_map": "auto",
            "load_in_4bit": True,
            "use_safetensors": True,
            "cache_dir": None,
            "local_files_only": False,
        },
        "vectors": {
            "input_dim": 2048,
            "output_dim": 4096,
        },
        "data": {
            "dataset_origin": dataset_origin,
            "prompt_policy": prompt_policy,
            "source_dataset": required_string(sampling_config, "dataset_name"),
            "source_dataset_config": sampling_config.get("dataset_config"),
            "source_split": source_split,
            "prompt_field": required_string(sampling_config, "prompt_field"),
            "sampling_seed": positive_int(sampling_config, "seed"),
            "sampled_ids": [item["id"] for item in selected],
            "prompts": [item["prompt"] for item in selected],
        },
        "output": {
            "bundle_dir": output_bundle_dir,
            "output_zip": output_zip,
            "shard_name": "shard_0.pt",
        },
    }


def resolve_output_settings(config):
    settings = {
        "train_trace_id": optional_string(
            config,
            "train_trace_id",
            DEFAULT_TRAIN_TRACE_ID,
        ),
        "eval_trace_id": optional_string(
            config,
            "eval_trace_id",
            DEFAULT_EVAL_TRACE_ID,
        ),
        "train_config_name": config_filename(
            config,
            "train_config_name",
            DEFAULT_TRAIN_CONFIG_NAME,
        ),
        "eval_config_name": config_filename(
            config,
            "eval_config_name",
            DEFAULT_EVAL_CONFIG_NAME,
        ),
        "train_bundle_dir": optional_string(
            config,
            "train_bundle_dir",
            DEFAULT_TRAIN_BUNDLE_DIR,
        ),
        "eval_bundle_dir": optional_string(
            config,
            "eval_bundle_dir",
            DEFAULT_EVAL_BUNDLE_DIR,
        ),
        "train_output_zip": optional_string(
            config,
            "train_output_zip",
            DEFAULT_TRAIN_OUTPUT_ZIP,
        ),
        "eval_output_zip": optional_string(
            config,
            "eval_output_zip",
            DEFAULT_EVAL_OUTPUT_ZIP,
        ),
    }
    if settings["train_trace_id"] == DEFAULT_TRAIN_TRACE_ID:
        settings["train_dataset_origin"] = DEFAULT_TRAIN_DATASET_ORIGIN
        settings["train_prompt_policy"] = DEFAULT_TRAIN_PROMPT_POLICY
    else:
        settings["train_dataset_origin"] = (
            f"{settings['train_trace_id']} MBPP train prompt sample"
        )
        settings["train_prompt_policy"] = (
            f"{positive_int(config, 'train_count')} public MBPP train split "
            "natural-language prompts; code, tests, and completions are not included."
        )

    if settings["eval_trace_id"] == DEFAULT_EVAL_TRACE_ID:
        settings["eval_dataset_origin"] = DEFAULT_EVAL_DATASET_ORIGIN
        settings["eval_prompt_policy"] = DEFAULT_EVAL_PROMPT_POLICY
    else:
        settings["eval_dataset_origin"] = (
            f"{settings['eval_trace_id']} MBPP held-out validation prompt sample"
        )
        settings["eval_prompt_policy"] = (
            f"{positive_int(config, 'eval_count')} public MBPP validation split "
            "natural-language prompts held out from the train prompt sample; "
            "code, tests, and completions are not included."
        )

    return settings


def write_yaml(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def materialize_configs(config_path, output_dir_override=None, mock_data=False):
    config = load_yaml(config_path)
    train_split = required_string(config, "train_split")
    eval_split = required_string(config, "eval_split")
    prompt_field = required_string(config, "prompt_field")
    train_count = positive_int(config, "train_count")
    eval_count = positive_int(config, "eval_count")
    seed = positive_int(config, "seed")
    max_prompt_chars = positive_int(config, "max_prompt_chars")
    output_dir = output_dir_override or Path(required_string(config, "output_dir"))
    output_settings = resolve_output_settings(config)

    train_rows = load_split_rows(config, train_split, mock_data)
    eval_rows = load_split_rows(config, eval_split, mock_data)
    train_selected = sample_prompts(
        train_rows,
        train_count,
        seed,
        prompt_field,
        max_prompt_chars,
    )
    eval_selected = sample_prompts(
        eval_rows,
        eval_count,
        seed,
        prompt_field,
        max_prompt_chars,
    )

    train_ids = {item["id"] for item in train_selected}
    eval_ids = {item["id"] for item in eval_selected}
    overlap = sorted(train_ids & eval_ids)
    if overlap:
        raise ValueError(
            "train/eval sampled IDs must be disjoint; overlap: "
            + ", ".join(overlap[:10])
        )

    train_config = build_bundle_config(
        trace_id=output_settings["train_trace_id"],
        dataset_origin=output_settings["train_dataset_origin"],
        prompt_policy=output_settings["train_prompt_policy"],
        selected=train_selected,
        output_bundle_dir=output_settings["train_bundle_dir"],
        output_zip=output_settings["train_output_zip"],
        source_split=train_split,
        sampling_config=config,
    )
    eval_config = build_bundle_config(
        trace_id=output_settings["eval_trace_id"],
        dataset_origin=output_settings["eval_dataset_origin"],
        prompt_policy=output_settings["eval_prompt_policy"],
        selected=eval_selected,
        output_bundle_dir=output_settings["eval_bundle_dir"],
        output_zip=output_settings["eval_output_zip"],
        source_split=eval_split,
        sampling_config=config,
    )

    train_path = output_dir / output_settings["train_config_name"]
    eval_path = output_dir / output_settings["eval_config_name"]
    write_yaml(train_path, train_config)
    write_yaml(eval_path, eval_config)
    tasks_jsonl_value = config.get("tasks_jsonl")
    tasks_jsonl = None
    if tasks_jsonl_value is not None:
        if not isinstance(tasks_jsonl_value, str) or not tasks_jsonl_value.strip():
            raise ValueError("tasks_jsonl must be a non-empty string when provided")
        tasks_jsonl = Path(tasks_jsonl_value)
        write_jsonl(tasks_jsonl, [item["task"] for item in eval_selected])

    return {
        "train_config": train_path,
        "eval_config": eval_path,
        "train_count": len(train_selected),
        "eval_count": len(eval_selected),
        "train_ids": sorted(train_ids),
        "eval_ids": sorted(eval_ids),
        "tasks_jsonl": tasks_jsonl,
        "mock_data": mock_data,
    }


def main():
    args = parse_args()
    result = materialize_configs(args.config, args.output_dir, args.mock_data)
    print("MBPP prompt config materialization passed")
    print(f"mock_data: {result['mock_data']}")
    print(f"train_config: {result['train_config']}")
    print(f"eval_config: {result['eval_config']}")
    print(f"train_prompts: {result['train_count']}")
    print(f"eval_prompts: {result['eval_count']}")
    if result["tasks_jsonl"] is not None:
        print(f"tasks_jsonl: {result['tasks_jsonl']}")
    print("train_eval_sampled_ids_disjoint: true")


if __name__ == "__main__":
    main()
