"""Generate the text-only structural capability screen for LIP-PROTO-013."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch

from src.core.prompt_protocol import protocol_metadata
from src.core.utils import set_seed
from src.evaluation.oracle_functional import stable_seed
from src.evaluation.oracle_terminal_factorial import (
    ORACLE_TERMINAL_CANDIDATE_COUNT,
    ORACLE_TERMINAL_PROTOCOL_VERSION,
    ORACLE_TERMINAL_SCREENING_CONDITION,
    ORACLE_TERMINAL_SCREENING_SCOPE,
    ORACLE_TERMINAL_SCREENING_SEEDS,
    candidate_binding_config,
    design_fingerprint,
)
from src.pipelines.infer import load_target, model_input_device
from src.pipelines.oracle_experiment import (
    bind_tasks_to_manifest,
    generation_kwargs,
    load_tasks,
    load_yaml,
    prompt_sha256,
    sha256_path,
    write_json,
)
from src.pipelines.oracle_transport import encode_prompt, generate_with_optional_packet
from src.scripts.run_oracle_memory_functional import (
    read_existing,
    tensor_sha256,
    validate_config,
)


DEFAULT_CONFIG = Path("config/LIP-PROTO-013_terminal_source_factorial.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--resume", action="store_true")
    output_mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_screening(
    config: dict[str, Any],
    config_path: Path,
    output_path: Path,
    *,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    validate_config(config)
    data = config["data"]
    if int(data["candidate_task_count"]) != ORACLE_TERMINAL_CANDIDATE_COUNT:
        raise ValueError("candidate count does not match the frozen terminal screen")
    all_candidates = load_tasks(Path(str(data["candidate_tasks_jsonl"])))
    candidates, manifest, manifest_path = bind_tasks_to_manifest(
        candidate_binding_config(config),
        all_candidates,
    )
    seeds = [int(seed) for seed in config["screening"]["seeds"]]
    if tuple(seeds) != ORACLE_TERMINAL_SCREENING_SEEDS:
        raise ValueError("screening seeds do not match the frozen terminal screen")
    expected_keys = {
        (str(task["task_id"]), ORACLE_TERMINAL_SCREENING_CONDITION, seed)
        for task in candidates
        for seed in seeds
    }
    design_sha256 = design_fingerprint(config)
    if output_path.exists() and not (resume or overwrite):
        raise FileExistsError(f"output already exists: {output_path}")
    if overwrite and output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys: set[tuple[str, str, int]] = set()
    existing_rows: list[dict] = []
    if resume and output_path.exists():
        existing_keys, existing_rows = read_existing(output_path)
        if existing_keys.difference(expected_keys):
            raise ValueError("existing generations do not belong to this screening")
        if any(
            row.get("protocol_version") != ORACLE_TERMINAL_PROTOCOL_VERSION
            or row.get("design_sha256") != design_sha256
            or row.get("run_scope") != ORACLE_TERMINAL_SCREENING_SCOPE
            for row in existing_rows
        ):
            raise ValueError("existing generations use a different frozen screen")

    target_revision = str(manifest["target_model_revision"])
    print("Loading target model for terminal-layout capability screening...")
    model, tokenizer = load_target(
        str(config["models"]["target_model"]),
        str(config["runtime"].get("device", "auto")),
        bool(config["runtime"]["load_4bit"]),
        revision=target_revision,
    )
    device = model_input_device(model)
    protocol = protocol_metadata(config.get("prompt_protocol"))
    encoded = []
    for task_index, task in enumerate(candidates):
        print(f"Encoding candidate {task_index + 1}/{len(candidates)}: {task['task_id']}")
        formatted, inputs = encode_prompt(task["prompt"], tokenizer, protocol, device)
        encoded.append((formatted, inputs))

    generation_config = config["screening"]["generation"]
    gen_kwargs = generation_kwargs(generation_config, tokenizer)
    output_mode = "a" if resume and output_path.exists() else "w"
    new_records = 0
    with output_path.open(output_mode, encoding="utf-8") as output_handle:
        for generation_seed in seeds:
            for task_index, task in enumerate(candidates):
                task_id = str(task["task_id"])
                key = (task_id, ORACLE_TERMINAL_SCREENING_CONDITION, generation_seed)
                if key in existing_keys:
                    continue
                formatted, inputs = encoded[task_index]
                effective_seed = stable_seed(generation_seed, task_index, 113)
                set_seed(effective_seed)
                print(
                    f"Screening seed={generation_seed} task={task_id} "
                    f"condition={ORACLE_TERMINAL_SCREENING_CONDITION}"
                )
                output_text = generate_with_optional_packet(
                    model,
                    tokenizer,
                    inputs,
                    generation_kwargs=gen_kwargs,
                )
                record = {
                    "protocol_version": ORACLE_TERMINAL_PROTOCOL_VERSION,
                    "design_sha256": design_sha256,
                    "experiment_id": config["experiment_id"],
                    "run_scope": ORACLE_TERMINAL_SCREENING_SCOPE,
                    "claim_eligible": False,
                    "task_id": task_id,
                    "functional_split": ORACLE_TERMINAL_SCREENING_SCOPE,
                    "condition": ORACLE_TERMINAL_SCREENING_CONDITION,
                    "generation_seed": generation_seed,
                    "effective_generation_seed": effective_seed,
                    "target_prompt_kind": "task",
                    "target_user_prompt_sha256": prompt_sha256(task["prompt"]),
                    "target_formatted_prompt_sha256": prompt_sha256(formatted),
                    "target_prompt_token_count": int(inputs["input_ids"].shape[1]),
                    "target_input_ids_sha256": tensor_sha256(inputs["input_ids"]),
                    "target_attention_mask_sha256": tensor_sha256(
                        inputs["attention_mask"]
                    ),
                    "target_model_revision": target_revision,
                    "task_manifest_sha256": sha256_path(manifest_path),
                    "output_text": output_text,
                    "task_spec": task,
                }
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_handle.flush()
                existing_keys.add(key)
                new_records += 1

    complete = existing_keys == expected_keys
    metadata = {
        "protocol_version": ORACLE_TERMINAL_PROTOCOL_VERSION,
        "design_sha256": design_sha256,
        "experiment_id": config["experiment_id"],
        "predecessor_experiment": config["predecessor_experiment"],
        "config": str(config_path),
        "config_sha256": sha256_path(config_path),
        "generations_jsonl": str(output_path),
        "run_scope": ORACLE_TERMINAL_SCREENING_SCOPE,
        "claim_eligible": False,
        "task_ids": [str(task["task_id"]) for task in candidates],
        "task_count": len(candidates),
        "conditions": [ORACLE_TERMINAL_SCREENING_CONDITION],
        "generation_seeds": seeds,
        "expected_records": len(expected_keys),
        "records": len(existing_keys),
        "new_records": new_records,
        "complete": complete,
        "target_model": config["models"]["target_model"],
        "target_model_revision": target_revision,
        "task_manifest": str(manifest_path),
        "task_manifest_sha256": sha256_path(manifest_path),
        "prompt_protocol": protocol,
        "screening": dict(config["screening"]),
    }
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    del encoded, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_path = args.output or Path(
        str(config["output"]["screening_generations_jsonl"])
    )
    metadata = run_screening(
        config,
        args.config,
        output_path,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print("Terminal-layout capability screening completed")
    print(f"records: {metadata['records']}/{metadata['expected_records']}")
    print(f"complete: {metadata['complete']}")
    print(f"generations: {metadata['generations_jsonl']}")


if __name__ == "__main__":
    main()
