"""Run the controlled LIP source-only generation experiment.

The primary ``source_latent`` condition extracts the task prompt only with the
source model.  The target receives a fixed neutral prompt plus the translated
latent vector.  Text-visible and vector controls are emitted in the same JSONL.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from src.core.models import LIPAdapter
from src.core.prompt_protocol import format_prompt, protocol_metadata
from src.core.utils import set_seed
from src.evaluation.source_only import (
    SOURCE_ONLY_PROTOCOL_VERSION,
    build_condition_plan,
    design_fingerprint,
    target_prompt_for_condition,
    validate_conditions,
)
from src.pipelines.infer import (
    extract_prompt_vectors,
    generate_with_optional_injection,
    load_adapter_checkpoint_safely,
    load_source,
    load_target,
    translate_source_vector,
)
from src.scripts.validate_latent_bundle import validate_bundle


DEFAULT_CONFIG = Path("config/LIP-PROTO-001_source_only_eval.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the LIP source-only probe.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tasks", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="Append only missing task/condition/seed records to an existing JSONL.",
    )
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generations JSONL.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must contain a YAML mapping")
    return config


def task_from_row(row: Mapping[str, Any], fallback_index: int) -> dict:
    task_id = row.get("task_id", row.get("id", fallback_index))
    prompt = row.get("prompt", row.get("text", row.get("instruction")))
    if prompt is None or not str(prompt).strip():
        raise ValueError(f"task {task_id!r} has no prompt/text/instruction")
    tests = row.get("test_list", row.get("tests", []))
    if isinstance(tests, str):
        tests = [tests]
    if not isinstance(tests, list) or any(not isinstance(test, str) for test in tests):
        raise ValueError(f"task {task_id!r} tests must be text or a list of text")
    return {
        "task_id": str(task_id),
        "prompt": str(prompt).strip(),
        "test_list": tests,
        "test_setup_code": str(row.get("test_setup_code", "") or ""),
        "entry_point": row.get("entry_point"),
    }


def load_jsonl_tasks(path: Path) -> list[dict]:
    tasks = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            tasks.append(task_from_row(row, len(tasks)))
    if not tasks:
        raise ValueError(f"no tasks found in {path}")
    return tasks


def load_dataset_tasks(data_config: Mapping[str, Any]) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("dataset loading requires the datasets package") from exc

    dataset_name = data_config.get("dataset_name")
    split = data_config.get("split")
    if not dataset_name or not split:
        raise ValueError("data.dataset_name and data.split are required")
    dataset = load_dataset(
        dataset_name,
        data_config.get("dataset_config"),
        split=split,
    )
    max_prompt_chars = int(data_config.get("max_prompt_chars", 512))
    candidates = [
        task
        for index, row in enumerate(dataset)
        for task in [task_from_row(row, index)]
        if len(task["prompt"]) <= max_prompt_chars
    ]
    count = int(data_config.get("task_count", len(candidates)))
    if count <= 0 or count > len(candidates):
        raise ValueError("data.task_count must fit the selected dataset split")
    rng = random.Random(int(data_config.get("sampling_seed", 42)))
    tasks = rng.sample(candidates, count)
    tasks.sort(key=lambda task: task["task_id"])
    return tasks


def resolve_tasks(config: Mapping[str, Any], override: Path | None) -> list[dict]:
    data_config = config.get("data", {})
    if override is not None:
        tasks = load_jsonl_tasks(override)
    else:
        configured_path = data_config.get("tasks_jsonl")
        path = Path(configured_path) if configured_path else None
        tasks = load_jsonl_tasks(path) if path is not None and path.is_file() else load_dataset_tasks(data_config)
    ids = [task["task_id"] for task in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("task IDs must be unique")
    return tasks


def random_norm_matched(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    reference_norm = reference.detach().cpu().float().norm(p=2, dim=-1, keepdim=True)
    noise = noise * (reference_norm / noise.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12))
    return noise.to(device=reference.device, dtype=reference.dtype)


def stable_seed(*values: int) -> int:
    payload = ":".join(str(int(value)) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def generation_kwargs(config: Mapping[str, Any], tokenizer) -> dict:
    do_sample = bool(config.get("do_sample", True))
    kwargs = {
        "max_new_tokens": int(config.get("max_new_tokens", 256)),
        "do_sample": do_sample,
        "repetition_penalty": float(config.get("repetition_penalty", 1.0)),
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        temperature = float(config.get("temperature", 0.2))
        if temperature <= 0:
            raise ValueError("generation.temperature must be positive when sampling")
        kwargs["temperature"] = temperature
        kwargs["top_p"] = float(config.get("top_p", 0.95))
    return kwargs


def resolve_adapter_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device_adapter=cuda but CUDA is unavailable")
    if value not in {"cpu", "cuda"}:
        raise ValueError("runtime.device_adapter must be auto, cpu, or cuda")
    return value


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_json_object(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_training_bundle(config: Mapping[str, Any]) -> dict:
    adapter_config = config.get("adapter", {})
    manifest_value = adapter_config.get("training_bundle_manifest")
    if not manifest_value:
        raise ValueError("adapter.training_bundle_manifest is required")
    manifest_path = Path(manifest_value)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"training bundle manifest not found: {manifest_path}")

    report = validate_bundle(manifest_path.parent)
    manifest = read_json_object(manifest_path)
    extraction = config.get("extraction", {})
    models = config.get("models", {})
    adapter_dims = config.get("adapter", {})
    expected = {
        "trace_id": adapter_config.get("training_bundle_trace_id"),
        "extraction_mode": "real",
        "source_quantization": (
            "bitsandbytes-4bit"
            if config.get("runtime", {}).get("source_load_4bit", False)
            else "none"
        ),
        "target_quantization": (
            "bitsandbytes-4bit"
            if config.get("runtime", {}).get("load_4bit", True)
            else "none"
        ),
        "quantization_compute_dtype": config.get("runtime", {}).get(
            "quantization_compute_dtype",
            "float16",
        ),
        "use_safetensors": True,
        "max_length": int(extraction.get("max_length", 512)),
        "source_model": models.get("source_model"),
        "target_model": models.get("target_model"),
        "source_layer": str(int(extraction.get("source_layer", -1))),
        "target_layer": str(int(extraction.get("target_layer", -1))),
        "token_position": str(extraction.get("token_position", "last_non_padding")),
        "prompt_protocol": protocol_metadata(config.get("prompt_protocol")),
        "input_dim": int(adapter_dims.get("input_dim", 2048)),
        "output_dim": int(adapter_dims.get("output_dim", 4096)),
    }
    if not expected["trace_id"]:
        raise ValueError("adapter.training_bundle_trace_id is required")
    mismatches = [
        f"{field}: manifest={manifest.get(field)!r}, expected={value!r}"
        for field, value in expected.items()
        if manifest.get(field) != value
    ]
    if mismatches:
        raise ValueError("training bundle protocol mismatch: " + "; ".join(mismatches))
    revisions = {
        role: manifest.get(f"{role}_model_revision")
        for role in ("source", "target")
    }
    if any(
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision.lower())
        for revision in revisions.values()
    ):
        raise ValueError(
            "training bundle must record immutable 40-character model revisions"
        )
    return {
        "path": str(manifest_path),
        "sha256": sha256_path(manifest_path),
        "trace_id": report["trace_id"],
        "records": report["total_records"],
        "shard_validation": report["validation_status"],
        "source_model_revision": revisions["source"],
        "target_model_revision": revisions["target"],
    }


def verify_checkpoint_run(
    checkpoint_path: Path,
    training_seed: int,
    training_bundle_manifest: Path,
) -> dict:
    metrics_path = checkpoint_path.parent / "metrics.json"
    resolved_config_path = checkpoint_path.parent / "resolved_config.yaml"
    if not metrics_path.is_file() or not resolved_config_path.is_file():
        raise FileNotFoundError(
            f"checkpoint provenance files missing beside {checkpoint_path}; "
            "expected metrics.json and resolved_config.yaml"
        )
    metrics = read_json_object(metrics_path)
    resolved_config = load_yaml(resolved_config_path)
    if int(metrics.get("seed", -1)) != training_seed:
        raise ValueError(
            f"checkpoint {checkpoint_path} seed does not match training_seed={training_seed}"
        )
    expected_shards = (training_bundle_manifest.parent / "shards").resolve()
    actual_shards = Path(str(metrics.get("dataset_path", ""))).resolve()
    resolved_shards = Path(
        str(resolved_config.get("data", {}).get("dataset_path", ""))
    ).resolve()
    if actual_shards != expected_shards or resolved_shards != expected_shards:
        raise ValueError(
            f"checkpoint {checkpoint_path} provenance does not match {expected_shards}"
        )
    if int(resolved_config.get("seed", -1)) != training_seed:
        raise ValueError(
            f"checkpoint {checkpoint_path} resolved config has the wrong seed"
        )
    dataset_manifest = metrics.get("dataset_manifest")
    expected_manifest_sha256 = sha256_path(training_bundle_manifest)
    if not isinstance(dataset_manifest, dict) or (
        dataset_manifest.get("sha256") != expected_manifest_sha256
        or dataset_manifest.get("trace_id")
        != resolved_config.get("data", {}).get("expected_trace_id")
    ):
        raise ValueError(
            f"checkpoint {checkpoint_path} was not trained on the current bundle manifest"
        )
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "metrics": str(metrics_path),
        "resolved_config": str(resolved_config_path),
        "training_seed": training_seed,
        "training_bundle_manifest_sha256": expected_manifest_sha256,
    }


def result_key(record: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(record["task_id"]),
        str(record["condition"]),
        int(record["training_seed"]),
        int(record["generation_seed"]),
    )


def load_existing_results(path: Path) -> tuple[set[tuple[str, str, int, int]], list[dict]]:
    keys = set()
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            key = result_key(row)
            if key in keys:
                raise ValueError(f"duplicate result key in {path}: {key}")
            keys.add(key)
            rows.append(row)
    return keys, rows


def run_probe(
    config: dict,
    tasks: list[dict],
    output_path: Path,
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> dict:
    experiment_id = str(config.get("experiment_id", "LIP-PROTO-001"))
    conditions = validate_conditions(config.get("conditions", []))
    neutral_prompt = str(config.get("neutral_target_prompt", "")).strip()
    if not neutral_prompt:
        raise ValueError("neutral_target_prompt must be non-empty")

    runtime = config.get("runtime", {})
    extraction = config.get("extraction", {})
    protocol = config.get("prompt_protocol", {})
    lip = config.get("lip", {})
    adapter_config = config.get("adapter", {})
    generation_config = config.get("generation", {})
    checkpoints = adapter_config.get("checkpoints", [])
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("adapter.checkpoints must be a non-empty list")
    if any(
        not isinstance(item, dict) or "training_seed" not in item or "path" not in item
        for item in checkpoints
    ):
        raise ValueError("each adapter checkpoint needs training_seed and path")
    generation_seeds = generation_config.get("seeds", [])
    if not isinstance(generation_seeds, list) or not generation_seeds:
        raise ValueError("generation.seeds must be a non-empty list")
    checkpoint_seeds = [int(item["training_seed"]) for item in checkpoints]
    if len(set(checkpoint_seeds)) != len(checkpoint_seeds):
        raise ValueError("adapter checkpoint training_seed values must be unique")
    normalized_generation_seeds = [int(seed) for seed in generation_seeds]
    if len(set(normalized_generation_seeds)) != len(normalized_generation_seeds):
        raise ValueError("generation.seeds values must be unique")
    if int(extraction.get("target_layer", -1)) != int(lip.get("layer_idx", -2)):
        raise ValueError(
            "extraction.target_layer must match lip.layer_idx for intervention alignment"
        )
    if lip.get("injection_mode", "replace") != "replace":
        raise ValueError("LIP-PROTO-001 requires lip.injection_mode=replace")
    if lip.get("vector_scaling", "none") != "none":
        raise ValueError("LIP-PROTO-001 requires lip.vector_scaling=none")
    if runtime.get("quantization_compute_dtype", "float16") != "float16":
        raise ValueError(
            "current model loaders require runtime.quantization_compute_dtype=float16"
        )
    design_sha256 = design_fingerprint(config)

    existing_keys = set()
    existing_rows = []
    if output_path.exists():
        if resume:
            existing_keys, existing_rows = load_existing_results(output_path)
        elif not overwrite:
            raise FileExistsError(
                f"results already exist: {output_path}; use --resume or --overwrite"
            )

    source_prompts = [task["prompt"] for task in tasks]
    task_ids = [task["task_id"] for task in tasks]
    models = config.get("models", {})
    allowed_keys = {
        (task_id, condition, training_seed, generation_seed)
        for task_id in task_ids
        for condition in conditions
        for training_seed in checkpoint_seeds
        for generation_seed in normalized_generation_seeds
    }
    unexpected_existing = existing_keys.difference(allowed_keys)
    if unexpected_existing:
        example = sorted(unexpected_existing)[0]
        raise ValueError(f"existing result does not belong to this run design: {example}")
    if any(
        row.get("protocol_version") != SOURCE_ONLY_PROTOCOL_VERSION
        or row.get("experiment_id") != experiment_id
        or row.get("design_sha256") != design_sha256
        for row in existing_rows
    ):
        raise ValueError("existing results use a different experiment design")
    tasks_by_id = {task["task_id"]: task for task in tasks}
    if any(
        row.get("task_spec") != tasks_by_id[str(row["task_id"])]
        for row in existing_rows
    ):
        raise ValueError("existing results contain a different task specification")

    training_bundle = verify_training_bundle(config)
    if any(
        row.get("training_bundle_manifest_sha256") != training_bundle["sha256"]
        for row in existing_rows
    ):
        raise ValueError("existing results used a different training bundle manifest")
    training_manifest_path = Path(training_bundle["path"])
    checkpoint_provenance_by_seed = {}
    for checkpoint_spec in checkpoints:
        training_seed = int(checkpoint_spec["training_seed"])
        checkpoint_path = Path(checkpoint_spec["path"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"adapter checkpoint not found: {checkpoint_path}")
        checkpoint_provenance_by_seed[training_seed] = verify_checkpoint_run(
            checkpoint_path,
            training_seed,
            training_manifest_path,
        )
    if any(
        row.get("adapter_checkpoint_sha256")
        != checkpoint_provenance_by_seed[int(row["training_seed"])][
            "checkpoint_sha256"
        ]
        for row in existing_rows
    ):
        raise ValueError("existing results were generated with different checkpoints")

    print("Loading source model and extracting source-only task vectors...")
    source_model, source_tokenizer = load_source(
        models["source_model"],
        runtime.get("device_src", "cuda"),
        bool(runtime.get("source_load_4bit", False)),
        training_bundle["source_model_revision"],
    )
    source_vectors = extract_prompt_vectors(
        source_prompts,
        source_model,
        source_tokenizer,
        runtime.get("device_src", "cuda"),
        protocol_config=protocol,
        layer_idx=int(extraction.get("source_layer", -1)),
        token_position=str(extraction.get("token_position", "last_non_padding")),
        max_length=int(extraction.get("max_length", 512)),
    )
    del source_model, source_tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading target model...")
    target_model, target_tokenizer = load_target(
        models["target_model"],
        runtime.get("device_tgt", "auto"),
        bool(runtime.get("load_4bit", True)),
        training_bundle["target_model_revision"],
    )
    target_vectors = None
    if "oracle_target_latent" in conditions:
        print("Extracting target-hidden oracle control vectors...")
        target_vectors = extract_prompt_vectors(
            source_prompts,
            target_model,
            target_tokenizer,
            None,
            protocol_config=protocol,
            layer_idx=int(extraction.get("target_layer", -1)),
            token_position=str(extraction.get("token_position", "last_non_padding")),
            max_length=int(extraction.get("max_length", 512)),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_records = len(existing_rows)
    new_records = 0
    adapter_runs = []
    baseline_cache = {}
    for row in existing_rows:
        if row.get("vector_kind") is None:
            cache_key = (
                str(row["condition"]),
                str(row["task_id"]),
                int(row["generation_seed"]),
            )
            cached = baseline_cache.setdefault(cache_key, row.get("output_text", ""))
            if cached != row.get("output_text", ""):
                raise ValueError("existing no-vector baseline outputs are inconsistent")
    gen_kwargs = generation_kwargs(generation_config, target_tokenizer)
    adapter_location = resolve_adapter_device(runtime.get("device_adapter", "auto"))

    output_mode = "a" if resume and output_path.exists() else "w"
    with open(output_path, output_mode, encoding="utf-8") as output_handle:
        for checkpoint_spec in checkpoints:
            if not isinstance(checkpoint_spec, dict):
                raise ValueError("each adapter checkpoint entry must be a mapping")
            training_seed = int(checkpoint_spec["training_seed"])
            checkpoint_path = Path(checkpoint_spec["path"])
            checkpoint_provenance = checkpoint_provenance_by_seed[training_seed]

            print(f"Loading adapter seed {training_seed}: {checkpoint_path}")
            adapter = LIPAdapter(
                input_dim=int(adapter_config.get("input_dim", 2048)),
                hidden_dim=int(adapter_config.get("hidden_dim", 1024)),
                output_dim=int(adapter_config.get("output_dim", 4096)),
            ).to(adapter_location)
            adapter.load_state_dict(
                load_adapter_checkpoint_safely(checkpoint_path, adapter_location)
            )
            adapter.eval()
            translated = [
                translate_source_vector(
                    vector,
                    adapter,
                    float(lip.get("gain", 1.0)),
                )
                for vector in source_vectors
            ]
            oracle = (
                [
                    vector * float(lip.get("gain", 1.0))
                    for vector in target_vectors
                ]
                if target_vectors is not None
                else None
            )

            adapter_record_count = sum(
                1 for row in existing_rows if int(row["training_seed"]) == training_seed
            )
            for generation_seed in normalized_generation_seeds:
                control_seed = stable_seed(generation_seed, 17)
                plan = build_condition_plan(task_ids, conditions, control_seed)
                random_vectors = [
                    random_norm_matched(
                        translated[index],
                        stable_seed(generation_seed, index, 29),
                    )
                    for index in range(len(tasks))
                ]

                for item in plan:
                    task = tasks[item.task_index]
                    key = (
                        task["task_id"],
                        item.condition,
                        training_seed,
                        generation_seed,
                    )
                    if key in existing_keys:
                        continue
                    target_prompt = target_prompt_for_condition(
                        item.condition,
                        task["prompt"],
                        neutral_prompt,
                    )
                    formatted_target_prompt = format_prompt(
                        target_prompt,
                        target_tokenizer,
                        protocol,
                    )
                    vector = None
                    vector_task_id = None
                    if item.vector_kind == "translated_source":
                        vector = translated[item.vector_index]
                        vector_task_id = tasks[item.vector_index]["task_id"]
                    elif item.vector_kind == "random_norm_matched":
                        vector = random_vectors[item.vector_index]
                    elif item.vector_kind == "target_hidden":
                        vector = oracle[item.vector_index]
                        vector_task_id = tasks[item.vector_index]["task_id"]

                    effective_seed = stable_seed(
                        generation_seed,
                        item.task_index,
                        101,
                    )
                    baseline_key = (
                        item.condition,
                        task["task_id"],
                        generation_seed,
                    )
                    generation_reused = vector is None and baseline_key in baseline_cache
                    if generation_reused:
                        output_text = baseline_cache[baseline_key]
                    else:
                        set_seed(effective_seed)
                        output_text = generate_with_optional_injection(
                            target_prompt,
                            vector,
                            target_model,
                            target_tokenizer,
                            int(lip.get("layer_idx", -2)),
                            str(lip.get("inject_pos_mode", "last_non_padding")),
                            gen_kwargs,
                            protocol_config=protocol,
                            injection_mode=str(lip.get("injection_mode", "replace")),
                        )
                        if vector is None:
                            baseline_cache[baseline_key] = output_text
                    record = {
                        "protocol_version": SOURCE_ONLY_PROTOCOL_VERSION,
                        "design_sha256": design_sha256,
                        "experiment_id": experiment_id,
                        "task_id": task["task_id"],
                        "condition": item.condition,
                        "training_seed": training_seed,
                        "generation_seed": generation_seed,
                        "effective_generation_seed": effective_seed,
                        "target_prompt_kind": item.target_prompt_kind,
                        "target_user_prompt_sha256": hashlib.sha256(
                            target_prompt.encode("utf-8")
                        ).hexdigest(),
                        "target_formatted_prompt_sha256": hashlib.sha256(
                            formatted_target_prompt.encode("utf-8")
                        ).hexdigest(),
                        "vector_kind": item.vector_kind,
                        "vector_task_id": vector_task_id,
                        "injected_vector_norm": (
                            float(vector.detach().float().norm(p=2).cpu().item())
                            if vector is not None
                            else None
                        ),
                        "adapter_checkpoint": str(checkpoint_path),
                        "adapter_checkpoint_sha256": checkpoint_provenance[
                            "checkpoint_sha256"
                        ],
                        "training_bundle_manifest_sha256": training_bundle["sha256"],
                        "output_text": output_text,
                        "generation_reused_across_adapter_seeds": generation_reused,
                        "task_spec": task,
                    }
                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output_handle.flush()
                    total_records += 1
                    new_records += 1
                    adapter_record_count += 1
                    existing_keys.add(key)

            adapter_runs.append(
                {
                    "training_seed": training_seed,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_provenance["checkpoint_sha256"],
                    "records": adapter_record_count,
                }
            )
            vector = None
            del adapter, translated, oracle, random_vectors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metadata = {
        "protocol_version": SOURCE_ONLY_PROTOCOL_VERSION,
        "design_sha256": design_sha256,
        "experiment_id": experiment_id,
        "results_jsonl": str(output_path),
        "task_count": len(tasks),
        "conditions": conditions,
        "generation_seeds": normalized_generation_seeds,
        "adapter_runs": adapter_runs,
        "training_bundle": training_bundle,
        "records": total_records,
        "expected_records": len(allowed_keys),
        "complete": existing_keys == allowed_keys,
        "new_records": new_records,
        "resumed": resume,
        "models": models,
        "prompt_protocol": protocol,
        "extraction": extraction,
        "lip": lip,
        "neutral_target_user_prompt_sha256": hashlib.sha256(
            neutral_prompt.encode("utf-8")
        ).hexdigest(),
        "target_text_policy": {
            "source_latent": "neutral prompt only",
            "text_only_no_lip": "task prompt (oracle textual control)",
        },
    }
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    return metadata


def main():
    args = parse_args()
    config = load_yaml(args.config)
    tasks = resolve_tasks(config, args.tasks)
    if args.max_tasks is not None:
        if args.max_tasks <= 0:
            raise ValueError("--max-tasks must be positive")
        tasks = tasks[: args.max_tasks]
    output_path = args.output or Path(
        config.get("output", {}).get(
            "generations_jsonl",
            "runs/LIP-PROTO-001/generations.jsonl",
        )
    )
    metadata = run_probe(
        config,
        tasks,
        output_path,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print("Source-only probe completed")
    print(f"records: {metadata['records']}")
    print(f"new_records: {metadata['new_records']}")
    print(f"results: {metadata['results_jsonl']}")


if __name__ == "__main__":
    main()
