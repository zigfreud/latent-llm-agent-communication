"""Generate frozen target-oracle packet controls for semantic evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from src.core.prompt_protocol import protocol_metadata
from src.core.utils import set_seed
from src.evaluation.oracle_functional import (
    build_condition_plan,
    design_fingerprint,
    expected_functional_conditions,
    packet_contract,
    plan_as_dicts,
    protocol_version_for_config,
    stable_seed,
)
from src.evaluation.oracle_transport import normalize_layer_indices
from src.pipelines.infer import load_target, model_input_device
from src.pipelines.oracle_experiment import (
    bind_tasks_to_manifest,
    load_tasks,
    load_yaml,
    prompt_sha256,
    sha256_path,
    write_json,
)
from src.pipelines.oracle_transport import (
    build_neutral_carrier,
    encode_prompt,
    forward_with_packet_capture,
    generate_with_optional_packet,
)


DEFAULT_CONFIG = Path("config/LIP-PROTO-005_oracle_packet_functional.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--preflight", action="store_true")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--resume", action="store_true")
    output_mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.preflight and args.max_tasks is not None:
        parser.error("--preflight and --max-tasks are mutually exclusive")
    return args


def validate_config(config: Mapping[str, Any]) -> None:
    experiment_id = config.get("experiment_id")
    if experiment_id not in {"LIP-PROTO-005", "LIP-PROTO-007"}:
        raise ValueError("experiment_id must be LIP-PROTO-005 or LIP-PROTO-007")

    expected_top_level = {
        "experiment_id",
        "source_protocol_experiment",
        "capacity_selection_experiment",
        "models",
        "prompt_protocol",
        "runtime",
        "data",
        "neutral_target_prompt",
        "carrier",
        "packet",
        "conditions",
        "controls",
        "generation",
        "evaluation",
        "output",
    }
    if experiment_id == "LIP-PROTO-007":
        expected_top_level.update(
            {"functional_anchor_experiment", "position_audit_experiment"}
        )
    unknown = sorted(set(config).difference(expected_top_level))
    if unknown:
        raise ValueError(f"unknown config field(s): {', '.join(unknown)}")
    if config.get("source_protocol_experiment") != "LIP-PROTO-001":
        raise ValueError("source_protocol_experiment must bind LIP-PROTO-001")
    if config.get("capacity_selection_experiment") != "LIP-PROTO-004":
        raise ValueError("capacity_selection_experiment must bind LIP-PROTO-004")
    if experiment_id == "LIP-PROTO-007":
        if config.get("functional_anchor_experiment") != "LIP-PROTO-005":
            raise ValueError("functional_anchor_experiment must bind LIP-PROTO-005")
        if config.get("position_audit_experiment") != "LIP-PROTO-006":
            raise ValueError("position_audit_experiment must bind LIP-PROTO-006")

    protocol = protocol_metadata(config.get("prompt_protocol"))
    if protocol["mode"] != "chat_template" or not protocol["add_generation_prompt"]:
        raise ValueError("functional oracle requires the target generation boundary")
    if not str(config.get("neutral_target_prompt", "")).strip():
        raise ValueError("neutral_target_prompt must be non-empty")
    carrier = config.get("carrier", {})
    if not isinstance(carrier, Mapping) or carrier.get("mode") != (
        "left_pad_masked_to_task_length"
    ):
        raise ValueError(
            f"{experiment_id} requires carrier.mode=left_pad_masked_to_task_length"
        )

    data = config.get("data", {})
    runtime = config.get("runtime", {})
    packet = config.get("packet", {})
    controls = config.get("controls", {})
    generation = config.get("generation", {})
    evaluation = config.get("evaluation", {})
    output = config.get("output", {})
    for name, value in (
        ("data", data),
        ("runtime", runtime),
        ("packet", packet),
        ("controls", controls),
        ("generation", generation),
        ("evaluation", evaluation),
        ("output", output),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping")

    bound_count = int(data.get("task_count", 0))
    start = int(data.get("functional_task_start", -1))
    functional_count = int(data.get("functional_task_count", 0))
    preflight_count = int(data.get("preflight_task_count", 0))
    if bound_count <= 0 or start < 0 or functional_count < 2:
        raise ValueError("functional task slice must be a non-empty part of the bundle")
    if start + functional_count > bound_count:
        raise ValueError("functional task slice exceeds the bound held-out task count")
    if not 2 <= preflight_count <= functional_count:
        raise ValueError("preflight_task_count must fit the functional task slice")
    if experiment_id == "LIP-PROTO-005":
        if data.get("functional_split") != "confirmation":
            raise ValueError("LIP-PROTO-005 freezes the confirmation task split")
        if bound_count != 16 or start != 8 or functional_count != 8:
            raise ValueError(
                "LIP-PROTO-005 freezes the LIP-PROTO-004 final eight tasks"
            )
    else:
        if data.get("functional_split") != "unused_heldout_16_32":
            raise ValueError("LIP-PROTO-007 freezes the unused held-out task split")
        if (
            bound_count != 32
            or start != 16
            or functional_count != 16
            or preflight_count != 2
        ):
            raise ValueError("LIP-PROTO-007 freezes held-out tasks 16:32")

    if int(packet.get("layer_idx", 0)) != -16:
        raise ValueError(f"{experiment_id} freezes layer_idx=-16")
    if packet.get("injection_mode") != "replace":
        raise ValueError(f"{experiment_id} fixes injection_mode=replace")
    if experiment_id == "LIP-PROTO-005":
        if int(packet.get("selected_size", 0)) != 8:
            raise ValueError("LIP-PROTO-005 freezes selected_size=8")
        if int(packet.get("replication_size", 0)) != 1:
            raise ValueError("LIP-PROTO-005 freezes replication_size=1")
    packet_sizes, replication_size = packet_contract(config)
    if experiment_id == "LIP-PROTO-007" and packet_sizes != (8, 16, 32):
        raise ValueError("LIP-PROTO-007 freezes packet.sizes=[8, 16, 32]")
    expected_conditions = expected_functional_conditions(
        packet_sizes,
        replication_size=replication_size,
    )
    if list(config.get("conditions", [])) != list(expected_conditions):
        raise ValueError("conditions must match the frozen oracle functional design")
    if controls != {
        "shuffled_oracle_packet": {
            "permutation": "sattolo_derangement",
            "seed": 1729,
        }
    }:
        raise ValueError("shuffled packet control must use the frozen derangement")

    seeds = generation.get("seeds", [])
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("generation.seeds must be a non-empty unique list")
    if experiment_id == "LIP-PROTO-007" and seeds != [101, 202, 303]:
        raise ValueError("LIP-PROTO-007 freezes generation seeds [101, 202, 303]")
    if int(generation.get("max_new_tokens", 0)) <= 0:
        raise ValueError("generation.max_new_tokens must be positive")
    if not isinstance(generation.get("do_sample"), bool):
        raise ValueError("generation.do_sample must be a boolean")
    if bool(generation["do_sample"]) and float(generation.get("temperature", 0)) <= 0:
        raise ValueError("generation.temperature must be positive when sampling")
    if not isinstance(runtime.get("load_4bit"), bool):
        raise ValueError("runtime.load_4bit must be a boolean")
    if int(evaluation.get("bootstrap_iterations", 0)) <= 0:
        raise ValueError("evaluation.bootstrap_iterations must be positive")
    comparisons = evaluation.get("comparisons", [])
    conditions = set(config.get("conditions", []))
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("evaluation.comparisons must be a non-empty list")
    if any(
        not isinstance(comparison, list)
        or len(comparison) != 2
        or any(condition not in conditions for condition in comparison)
        for comparison in comparisons
    ):
        raise ValueError("every evaluation comparison must name two conditions")
    if not str(output.get("generations_jsonl", "")).strip():
        raise ValueError("output.generations_jsonl must be configured")


def generation_kwargs(config: Mapping[str, Any], tokenizer) -> dict[str, Any]:
    do_sample = bool(config["do_sample"])
    kwargs = {
        "max_new_tokens": int(config["max_new_tokens"]),
        "do_sample": do_sample,
        "repetition_penalty": float(config.get("repetition_penalty", 1.0)),
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs.update(
            {
                "temperature": float(config["temperature"]),
                "top_p": float(config.get("top_p", 1.0)),
            }
        )
    return kwargs


def read_existing(path: Path) -> tuple[set[tuple[str, str, int]], list[dict]]:
    keys = set()
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                str(row["task_id"]),
                str(row["condition"]),
                int(row["generation_seed"]),
            )
            if key in keys:
                raise ValueError(f"duplicate generation at line {line_number}: {key}")
            keys.add(key)
            rows.append(row)
    return keys, rows


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and contiguous value bytes."""

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_generation(
    config: dict[str, Any],
    config_path: Path,
    output_path: Path,
    *,
    preflight: bool,
    max_tasks: int | None,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    validate_config(config)
    all_tasks = load_tasks(Path(str(config["data"]["tasks_jsonl"])))
    bound_tasks, manifest, manifest_path = bind_tasks_to_manifest(config, all_tasks)
    start = int(config["data"]["functional_task_start"])
    configured_count = int(config["data"]["functional_task_count"])
    tasks = bound_tasks[start : start + configured_count]
    generation_seeds = [int(seed) for seed in config["generation"]["seeds"]]
    if preflight:
        tasks = tasks[: int(config["data"]["preflight_task_count"])]
        generation_seeds = generation_seeds[:1]
        run_scope = "preflight"
    elif max_tasks is not None:
        if not 2 <= max_tasks <= len(tasks):
            raise ValueError("--max-tasks must fit the functional task slice")
        tasks = tasks[:max_tasks]
        run_scope = "full" if max_tasks == configured_count else "diagnostic_subset"
    else:
        run_scope = "full"

    if output_path.exists() and not (resume or overwrite):
        raise FileExistsError(f"output already exists: {output_path}")
    if overwrite and output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    protocol = protocol_metadata(config.get("prompt_protocol"))
    protocol_version = protocol_version_for_config(config)
    packet_sizes, replication_size = packet_contract(config)
    design_sha256 = design_fingerprint(config)
    conditions = list(config["conditions"])
    expected_keys = {
        (str(task["task_id"]), condition, seed)
        for task in tasks
        for condition in conditions
        for seed in generation_seeds
    }
    existing_keys: set[tuple[str, str, int]] = set()
    existing_rows: list[dict] = []
    if resume and output_path.exists():
        existing_keys, existing_rows = read_existing(output_path)
        if existing_keys.difference(expected_keys):
            raise ValueError("existing generations do not belong to this run scope")
        if any(
            row.get("protocol_version") != protocol_version
            or row.get("design_sha256") != design_sha256
            or row.get("run_scope") != run_scope
            for row in existing_rows
        ):
            raise ValueError("existing generations use a different frozen design")

    target_revision = str(manifest["target_model_revision"])
    set_seed(int(config["controls"]["shuffled_oracle_packet"]["seed"]))
    print("Loading target model for functional oracle packet generation...")
    model, tokenizer = load_target(
        str(config["models"]["target_model"]),
        str(config["runtime"].get("device", "auto")),
        bool(config["runtime"]["load_4bit"]),
        revision=target_revision,
    )
    device = model_input_device(model)
    layer_idx = normalize_layer_indices(
        [int(config["packet"]["layer_idx"])], len(model.model.layers)
    )[0]
    capture_size = max(packet_sizes)
    neutral_formatted, native_neutral_inputs = encode_prompt(
        str(config["neutral_target_prompt"]), tokenizer, protocol, device
    )
    native_neutral_length = int(native_neutral_inputs["input_ids"].shape[1])
    if capture_size > native_neutral_length:
        raise ValueError("largest packet does not fit visible neutral carrier tokens")

    task_inputs: list[dict[str, torch.Tensor]] = []
    carrier_inputs: list[dict[str, torch.Tensor]] = []
    formatted_task_prompts: list[str] = []
    packets: list[torch.Tensor] = []
    for task_index, task in enumerate(tasks):
        print(f"Capturing packet {task_index + 1}/{len(tasks)}: {task['task_id']}")
        formatted, inputs = encode_prompt(task["prompt"], tokenizer, protocol, device)
        prompt_length = int(inputs["input_ids"].shape[1])
        carrier = build_neutral_carrier(
            native_neutral_inputs,
            task_prompt_length=prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            mode=str(config["carrier"]["mode"]),
        )
        positions = torch.arange(
            prompt_length - capture_size,
            prompt_length,
            device=device,
        )
        if not bool(torch.all(carrier["attention_mask"][0, positions] == 1).item()):
            raise RuntimeError("selected packet overlaps masked carrier positions")
        outputs, packet = forward_with_packet_capture(
            model,
            inputs,
            layer_idx=layer_idx,
            positions=positions,
        )
        del outputs
        task_inputs.append(inputs)
        carrier_inputs.append(carrier)
        formatted_task_prompts.append(formatted)
        packets.append(packet.detach().cpu())

    plan = build_condition_plan(
        [str(task["task_id"]) for task in tasks],
        conditions,
        shuffle_seed=int(config["controls"]["shuffled_oracle_packet"]["seed"]),
        packet_sizes=packet_sizes,
        replication_size=replication_size,
    )
    gen_kwargs = generation_kwargs(config["generation"], tokenizer)
    output_mode = "a" if resume and output_path.exists() else "w"
    new_records = 0
    with output_path.open(output_mode, encoding="utf-8") as output_handle:
        for generation_seed in generation_seeds:
            for item in plan:
                key = (item.task_id, item.condition, generation_seed)
                if key in existing_keys:
                    continue
                task = tasks[item.task_index]
                inputs = (
                    task_inputs[item.task_index]
                    if item.target_prompt_kind == "task"
                    else carrier_inputs[item.task_index]
                )
                positions = None
                vectors = None
                packet_task_id = None
                packet_norm = None
                packet_sha256 = None
                if item.packet_size is not None and item.packet_index is not None:
                    prompt_length = int(inputs["input_ids"].shape[1])
                    positions = torch.arange(
                        prompt_length - item.packet_size,
                        prompt_length,
                        device=device,
                    )
                    vectors = packets[item.packet_index][-item.packet_size :, :]
                    packet_task_id = str(tasks[item.packet_index]["task_id"])
                    packet_norm = float(vectors.float().norm(p=2).item())
                    packet_sha256 = tensor_sha256(vectors)

                effective_seed = stable_seed(generation_seed, item.task_index, 101)
                set_seed(effective_seed)
                print(
                    f"Generating seed={generation_seed} task={item.task_id} "
                    f"condition={item.condition}"
                )
                output_text = generate_with_optional_packet(
                    model,
                    tokenizer,
                    inputs,
                    generation_kwargs=gen_kwargs,
                    layer_idx=layer_idx if vectors is not None else None,
                    positions=positions,
                    vectors=vectors,
                )
                target_formatted = (
                    formatted_task_prompts[item.task_index]
                    if item.target_prompt_kind == "task"
                    else neutral_formatted
                )
                record = {
                    "protocol_version": protocol_version,
                    "design_sha256": design_sha256,
                    "experiment_id": config["experiment_id"],
                    "run_scope": run_scope,
                    "claim_eligible": run_scope == "full",
                    "task_id": item.task_id,
                    "functional_split": config["data"]["functional_split"],
                    "condition": item.condition,
                    "generation_seed": generation_seed,
                    "effective_generation_seed": effective_seed,
                    "target_prompt_kind": item.target_prompt_kind,
                    "target_user_prompt_sha256": prompt_sha256(
                        task["prompt"]
                        if item.target_prompt_kind == "task"
                        else str(config["neutral_target_prompt"])
                    ),
                    "target_formatted_prompt_sha256": prompt_sha256(target_formatted),
                    "task_prompt_token_count": int(
                        task_inputs[item.task_index]["input_ids"].shape[1]
                    ),
                    "target_prompt_token_count": int(inputs["input_ids"].shape[1]),
                    "target_input_ids_sha256": tensor_sha256(inputs["input_ids"]),
                    "target_attention_mask_sha256": tensor_sha256(
                        inputs["attention_mask"]
                    ),
                    "native_neutral_prompt_token_count": native_neutral_length,
                    "carrier_mode": config["carrier"]["mode"],
                    "layer_idx": layer_idx if vectors is not None else None,
                    "injection_mode": "replace" if vectors is not None else None,
                    "packet_size": item.packet_size,
                    "packet_task_id": packet_task_id,
                    "packet_frobenius_norm": packet_norm,
                    "packet_sha256": packet_sha256,
                    "target_model_revision": target_revision,
                    "heldout_bundle_manifest_sha256": sha256_path(manifest_path),
                    "output_text": output_text,
                    "task_spec": task,
                }
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_handle.flush()
                existing_keys.add(key)
                new_records += 1

    complete = existing_keys == expected_keys
    metadata = {
        "protocol_version": protocol_version,
        "design_sha256": design_sha256,
        "experiment_id": config["experiment_id"],
        "source_protocol_experiment": config["source_protocol_experiment"],
        "capacity_selection_experiment": config["capacity_selection_experiment"],
        "config": str(config_path),
        "config_sha256": sha256_path(config_path),
        "generations_jsonl": str(output_path),
        "run_scope": run_scope,
        "claim_eligible": run_scope == "full" and complete,
        "functional_split": config["data"]["functional_split"],
        "task_ids": [str(task["task_id"]) for task in tasks],
        "task_count": len(tasks),
        "conditions": conditions,
        "condition_plan": plan_as_dicts(plan),
        "generation_seeds": generation_seeds,
        "expected_records": len(expected_keys),
        "records": len(existing_keys),
        "new_records": new_records,
        "complete": complete,
        "target_model": config["models"]["target_model"],
        "target_model_revision": target_revision,
        "heldout_bundle_manifest": str(manifest_path),
        "heldout_bundle_manifest_sha256": sha256_path(manifest_path),
        "prompt_protocol": protocol,
        "packet": dict(config["packet"]),
        "captured_packets": [
            {
                "task_id": str(task["task_id"]),
                "packet_size": capture_size,
                "packet_frobenius_norm": float(packet.float().norm(p=2).item()),
                "packet_sha256": tensor_sha256(packet),
            }
            for task, packet in zip(tasks, packets)
        ],
        "carrier": dict(config["carrier"]),
        "generation": dict(config["generation"]),
    }
    for lineage_field in ("functional_anchor_experiment", "position_audit_experiment"):
        if lineage_field in config:
            metadata[lineage_field] = config[lineage_field]
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    del packets, task_inputs, carrier_inputs, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.output is not None:
        output_path = args.output
    elif args.preflight:
        configured_output = Path(str(config["output"]["generations_jsonl"]))
        output_path = configured_output.parent / "preflight" / "generations.jsonl"
    else:
        output_path = Path(str(config["output"]["generations_jsonl"]))
    metadata = run_generation(
        config,
        args.config,
        output_path,
        preflight=args.preflight,
        max_tasks=args.max_tasks,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print("Functional oracle packet generation completed")
    print(f"run_scope: {metadata['run_scope']}")
    print(f"records: {metadata['records']}/{metadata['expected_records']}")
    print(f"complete: {metadata['complete']}")
    print(f"generations: {metadata['generations_jsonl']}")


if __name__ == "__main__":
    main()
