"""Generate frozen multi-layer target-oracle memory controls."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from src.core.prompt_protocol import protocol_metadata
from src.core.utils import set_seed
from src.evaluation.oracle_memory import (
    ORACLE_MEMORY_CONDITIONS,
    ORACLE_MEMORY_LAYER_COUNT,
    ORACLE_MEMORY_PACKET_SIZE,
    ORACLE_MEMORY_PROTOCOL_VERSION,
    build_condition_plan,
    design_fingerprint,
    plan_as_dicts,
    validate_memory_contract,
)
from src.evaluation.oracle_layer_depth import (
    ORACLE_LAYER_DEPTH_CONDITIONS,
    ORACLE_LAYER_DEPTH_LAYER_COUNT,
    ORACLE_LAYER_DEPTH_PACKET_SIZE,
    ORACLE_LAYER_DEPTH_PROTOCOL_VERSION,
    ORACLE_LAYER_DEPTH_SCOPE_ORDER,
    build_condition_plan as build_layer_depth_condition_plan,
    design_fingerprint as layer_depth_design_fingerprint,
    plan_as_dicts as layer_depth_plan_as_dicts,
    primary_fixed_sequence,
    validate_layer_depth_contract,
)
from src.evaluation.oracle_state_diagnostics import (
    validate_state_diagnostics_contract,
    summarize_state_diagnostics,
)
from src.evaluation.oracle_functional import stable_seed
from src.evaluation.oracle_transport import normalize_layer_indices
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
from src.pipelines.oracle_memory import (
    forward_with_layer_input_replay,
    forward_with_layer_state_capture,
    generate_with_layer_input_replay,
)
from src.pipelines.oracle_transport import (
    build_neutral_carrier,
    encode_prompt,
    forward_with_packet_capture,
    forward_with_packet_replacement,
    generate_with_optional_packet,
)


DEFAULT_CONFIG = Path("config/LIP-PROTO-008_oracle_multilayer_memory.yaml")


def experiment_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    experiment_id = str(config.get("experiment_id", ""))
    if experiment_id == "LIP-PROTO-008":
        return {
            "experiment_id": experiment_id,
            "protocol_version": ORACLE_MEMORY_PROTOCOL_VERSION,
            "packet_size": ORACLE_MEMORY_PACKET_SIZE,
            "layer_count": ORACLE_MEMORY_LAYER_COUNT,
            "conditions": ORACLE_MEMORY_CONDITIONS,
            "scope_order": (
                "single_layer_output",
                "late_half_input",
                "all_layer_input",
            ),
            "build_condition_plan": build_condition_plan,
            "design_fingerprint": design_fingerprint,
            "plan_as_dicts": plan_as_dicts,
            "validate_memory_contract": validate_memory_contract,
            "predecessor_field": "functional_capacity_experiment",
        }
    if experiment_id == "LIP-PROTO-009":
        return {
            "experiment_id": experiment_id,
            "protocol_version": ORACLE_LAYER_DEPTH_PROTOCOL_VERSION,
            "packet_size": ORACLE_LAYER_DEPTH_PACKET_SIZE,
            "layer_count": ORACLE_LAYER_DEPTH_LAYER_COUNT,
            "conditions": ORACLE_LAYER_DEPTH_CONDITIONS,
            "scope_order": ORACLE_LAYER_DEPTH_SCOPE_ORDER,
            "build_condition_plan": build_layer_depth_condition_plan,
            "design_fingerprint": layer_depth_design_fingerprint,
            "plan_as_dicts": layer_depth_plan_as_dicts,
            "validate_memory_contract": validate_layer_depth_contract,
            "predecessor_field": "predecessor_experiment",
        }
    raise ValueError("unsupported oracle memory experiment_id")


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


def expected_comparisons() -> list[list[str]]:
    comparisons = []
    for scope in ("single_layer_output", "late_half_input", "all_layer_input"):
        matched = f"oracle_{scope}_k{ORACLE_MEMORY_PACKET_SIZE}"
        comparisons.extend(
            [
                [matched, "neutral_no_lip"],
                [
                    matched,
                    f"shuffled_oracle_{scope}_k{ORACLE_MEMORY_PACKET_SIZE}",
                ],
            ]
        )
    comparisons.extend(
        [
            ["oracle_late_half_input_k32", "oracle_single_layer_output_k32"],
            ["oracle_all_layer_input_k32", "oracle_late_half_input_k32"],
            ["text_only_no_lip", "neutral_no_lip"],
        ]
    )
    return comparisons


def _validate_proto008_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != "LIP-PROTO-008":
        raise ValueError("experiment_id must be LIP-PROTO-008")
    expected_top_level = {
        "experiment_id",
        "functional_capacity_experiment",
        "models",
        "prompt_protocol",
        "runtime",
        "data",
        "neutral_target_prompt",
        "carrier",
        "memory",
        "diagnostics",
        "conditions",
        "controls",
        "generation",
        "evaluation",
        "output",
    }
    unknown = sorted(set(config).difference(expected_top_level))
    if unknown:
        raise ValueError(f"unknown config field(s): {', '.join(unknown)}")
    if config.get("functional_capacity_experiment") != "LIP-PROTO-007":
        raise ValueError("functional_capacity_experiment must bind LIP-PROTO-007")
    protocol = protocol_metadata(config.get("prompt_protocol"))
    if protocol["mode"] != "chat_template" or not protocol["add_generation_prompt"]:
        raise ValueError("oracle memory replay requires the target generation boundary")
    if not str(config.get("neutral_target_prompt", "")).strip():
        raise ValueError("neutral_target_prompt must be non-empty")
    if config.get("carrier") != {"mode": "left_pad_masked_to_task_length"}:
        raise ValueError("LIP-PROTO-008 freezes the length-controlled carrier")

    data = config.get("data", {})
    runtime = config.get("runtime", {})
    controls = config.get("controls", {})
    generation = config.get("generation", {})
    evaluation = config.get("evaluation", {})
    output = config.get("output", {})
    diagnostics = config.get("diagnostics", {})
    for name, value in (
        ("data", data),
        ("runtime", runtime),
        ("controls", controls),
        ("generation", generation),
        ("evaluation", evaluation),
        ("output", output),
        ("diagnostics", diagnostics),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping")
    required_paths = ("tasks_jsonl", "task_manifest")
    if any(not str(data.get(field, "")).strip() for field in required_paths):
        raise ValueError("data must configure tasks_jsonl and task_manifest")
    frozen_data = {
        "task_count": 32,
        "preflight_task_start": 0,
        "preflight_task_count": 2,
        "functional_task_start": 16,
        "functional_task_count": 16,
        "functional_split": "mbpp_test_16_32",
    }
    if any(data.get(field) != value for field, value in frozen_data.items()):
        raise ValueError("LIP-PROTO-008 freezes disjoint MBPP-test task slices")
    if not isinstance(runtime.get("load_4bit"), bool):
        raise ValueError("runtime.load_4bit must be a boolean")
    validate_memory_contract(config.get("memory", {}))
    validate_state_diagnostics_contract(diagnostics)
    if list(config.get("conditions", [])) != list(ORACLE_MEMORY_CONDITIONS):
        raise ValueError("conditions must match the frozen oracle memory design")
    if controls != {
        "shuffled_oracle_memory": {
            "permutation": "sattolo_derangement",
            "seed": 1729,
        }
    }:
        raise ValueError("shuffled memory control must use the frozen derangement")
    if generation.get("seeds") != [101, 202, 303]:
        raise ValueError("LIP-PROTO-008 freezes generation seeds [101, 202, 303]")
    if int(generation.get("max_new_tokens", 0)) != 256:
        raise ValueError("LIP-PROTO-008 freezes max_new_tokens=256")
    if not isinstance(generation.get("do_sample"), bool):
        raise ValueError("generation.do_sample must be a boolean")
    if bool(generation["do_sample"]) and float(generation.get("temperature", 0)) <= 0:
        raise ValueError("temperature must be positive when sampling")
    if evaluation.get("comparisons") != expected_comparisons():
        raise ValueError("evaluation.comparisons must match the frozen contrasts")
    if int(evaluation.get("bootstrap_iterations", 0)) <= 0:
        raise ValueError("evaluation.bootstrap_iterations must be positive")
    if any(not str(output.get(field, "")).strip() for field in (
        "generations_jsonl",
        "evaluation_dir",
        "state_diagnostics_json",
    )):
        raise ValueError("output paths must be configured")


def expected_layer_depth_comparisons() -> list[list[str]]:
    comparisons = []
    for scope in ORACLE_LAYER_DEPTH_SCOPE_ORDER:
        matched = f"oracle_{scope}_k{ORACLE_LAYER_DEPTH_PACKET_SIZE}"
        comparisons.extend(
            [
                [matched, "neutral_no_lip"],
                [
                    matched,
                    f"shuffled_oracle_{scope}_k{ORACLE_LAYER_DEPTH_PACKET_SIZE}",
                ],
            ]
        )
    comparisons.extend(
        [
            ["oracle_early_half_input_k32", "oracle_early_quarter_input_k32"],
            [
                "oracle_early_three_quarters_input_k32",
                "oracle_early_half_input_k32",
            ],
            [
                "oracle_all_layer_input_k32",
                "oracle_early_three_quarters_input_k32",
            ],
            ["text_only_no_lip", "neutral_no_lip"],
        ]
    )
    return comparisons


def _validate_proto009_config(config: Mapping[str, Any]) -> None:
    expected_top_level = {
        "experiment_id",
        "predecessor_experiment",
        "models",
        "prompt_protocol",
        "runtime",
        "data",
        "neutral_target_prompt",
        "carrier",
        "memory",
        "diagnostics",
        "conditions",
        "controls",
        "generation",
        "evaluation",
        "output",
    }
    unknown = sorted(set(config).difference(expected_top_level))
    if unknown:
        raise ValueError(f"unknown config field(s): {', '.join(unknown)}")
    if config.get("predecessor_experiment") != "LIP-PROTO-008":
        raise ValueError("predecessor_experiment must bind LIP-PROTO-008")
    protocol = protocol_metadata(config.get("prompt_protocol"))
    if protocol["mode"] != "chat_template" or not protocol["add_generation_prompt"]:
        raise ValueError("layer-depth replay requires the target generation boundary")
    if not str(config.get("neutral_target_prompt", "")).strip():
        raise ValueError("neutral_target_prompt must be non-empty")
    if config.get("carrier") != {"mode": "left_pad_masked_to_task_length"}:
        raise ValueError("LIP-PROTO-009 freezes the length-controlled carrier")

    data = config.get("data", {})
    runtime = config.get("runtime", {})
    controls = config.get("controls", {})
    generation = config.get("generation", {})
    evaluation = config.get("evaluation", {})
    output = config.get("output", {})
    diagnostics = config.get("diagnostics", {})
    for name, value in (
        ("data", data),
        ("runtime", runtime),
        ("controls", controls),
        ("generation", generation),
        ("evaluation", evaluation),
        ("output", output),
        ("diagnostics", diagnostics),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping")
    if any(
        not str(data.get(field, "")).strip()
        for field in ("tasks_jsonl", "task_manifest")
    ):
        raise ValueError("data must configure tasks_jsonl and task_manifest")
    frozen_data = {
        "task_count": 18,
        "preflight_task_start": 0,
        "preflight_task_count": 2,
        "functional_task_start": 2,
        "functional_task_count": 16,
        "functional_split": "mbpp_test_fresh_2_18",
    }
    if any(data.get(field) != value for field, value in frozen_data.items()):
        raise ValueError("LIP-PROTO-009 freezes disjoint preflight/functional slices")
    if not isinstance(runtime.get("load_4bit"), bool):
        raise ValueError("runtime.load_4bit must be a boolean")
    validate_layer_depth_contract(config.get("memory", {}))
    validate_state_diagnostics_contract(diagnostics)
    if list(config.get("conditions", [])) != list(ORACLE_LAYER_DEPTH_CONDITIONS):
        raise ValueError("conditions must match the frozen layer-depth design")
    if controls != {
        "shuffled_oracle_memory": {
            "permutation": "sattolo_derangement",
            "seed": 1729,
        }
    }:
        raise ValueError("shuffled memory control must use the frozen derangement")
    if generation.get("seeds") != [101, 202, 303]:
        raise ValueError("LIP-PROTO-009 freezes generation seeds [101, 202, 303]")
    if int(generation.get("max_new_tokens", 0)) != 256:
        raise ValueError("LIP-PROTO-009 freezes max_new_tokens=256")
    if not isinstance(generation.get("do_sample"), bool):
        raise ValueError("generation.do_sample must be a boolean")
    if bool(generation["do_sample"]) and float(generation.get("temperature", 0)) <= 0:
        raise ValueError("temperature must be positive when sampling")
    if evaluation.get("comparisons") != expected_layer_depth_comparisons():
        raise ValueError("evaluation.comparisons must match frozen secondary contrasts")
    primary = evaluation.get("primary_testing")
    if primary != {
        "method": "fixed_sequence_gatekeeping",
        "alternative": "greater",
        "alpha": 0.05,
        "sequence": [list(pair) for pair in primary_fixed_sequence()],
    }:
        raise ValueError("evaluation.primary_testing must match the frozen depth order")
    if int(evaluation.get("bootstrap_iterations", 0)) <= 0:
        raise ValueError("evaluation.bootstrap_iterations must be positive")
    if any(
        not str(output.get(field, "")).strip()
        for field in (
            "generations_jsonl",
            "evaluation_dir",
            "state_diagnostics_json",
        )
    ):
        raise ValueError("output paths must be configured")


def validate_config(config: Mapping[str, Any]) -> None:
    experiment_id = config.get("experiment_id")
    if experiment_id == "LIP-PROTO-008":
        _validate_proto008_config(config)
        return
    if experiment_id == "LIP-PROTO-009":
        _validate_proto009_config(config)
        return
    raise ValueError("unsupported oracle memory experiment_id")


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
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_bundle_sha256(layer_packets: Mapping[int, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for layer_idx, tensor in sorted(layer_packets.items()):
        digest.update(str(int(layer_idx)).encode("ascii"))
        digest.update(tensor_sha256(tensor).encode("ascii"))
    return digest.hexdigest()


def state_bundle_norm(layer_packets: Mapping[int, torch.Tensor]) -> float:
    squared = sum(
        float(tensor.float().pow(2).sum().item()) for tensor in layer_packets.values()
    )
    return math.sqrt(squared)


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
    contract = experiment_contract(config)
    experiment_id = contract["experiment_id"]
    protocol_version = contract["protocol_version"]
    packet_size = int(contract["packet_size"])
    layer_count = int(contract["layer_count"])
    all_tasks = load_tasks(Path(str(config["data"]["tasks_jsonl"])))
    bound_tasks, manifest, manifest_path = bind_tasks_to_manifest(config, all_tasks)
    if preflight:
        start = int(config["data"]["preflight_task_start"])
        count = int(config["data"]["preflight_task_count"])
        generation_seeds = [int(config["generation"]["seeds"][0])]
        run_scope = "preflight"
    else:
        start = int(config["data"]["functional_task_start"])
        count = int(config["data"]["functional_task_count"])
        generation_seeds = [int(seed) for seed in config["generation"]["seeds"]]
        run_scope = "full"
    tasks = bound_tasks[start : start + count]
    if max_tasks is not None:
        if not 2 <= max_tasks <= len(tasks):
            raise ValueError("--max-tasks must fit the selected task slice")
        tasks = tasks[:max_tasks]
        run_scope = (
            "full" if max_tasks == count and not preflight else "diagnostic_subset"
        )

    if output_path.exists() and not (resume or overwrite):
        raise FileExistsError(f"output already exists: {output_path}")
    if overwrite and output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    design_sha256 = contract["design_fingerprint"](config)
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
    print("Loading target model for multi-layer oracle memory replay...")
    model, tokenizer = load_target(
        str(config["models"]["target_model"]),
        str(config["runtime"].get("device", "auto")),
        bool(config["runtime"]["load_4bit"]),
        revision=target_revision,
    )
    device = model_input_device(model)
    if len(model.model.layers) != layer_count:
        raise ValueError("target model does not have the frozen 32 decoder layers")
    scope_specs = contract["validate_memory_contract"](config["memory"])
    normalized_scopes = {
        scope["name"]: {
            **scope,
            "normalized_layers": normalize_layer_indices(
                scope["layers"], len(model.model.layers)
            ),
        }
        for scope in scope_specs
    }
    output_scope_names = [
        name
        for name in contract["scope_order"]
        if normalized_scopes[name]["boundary"] == "block_output"
    ]
    if len(output_scope_names) > 1:
        raise ValueError("only one block-output anchor scope is supported")
    anchor_scope_name = output_scope_names[0] if output_scope_names else None
    anchor_layer = (
        normalized_scopes[anchor_scope_name]["normalized_layers"][0]
        if anchor_scope_name is not None
        else None
    )
    all_input_layers = normalized_scopes["all_layer_input"]["normalized_layers"]
    protocol = protocol_metadata(config.get("prompt_protocol"))
    neutral_formatted, native_neutral_inputs = encode_prompt(
        str(config["neutral_target_prompt"]), tokenizer, protocol, device
    )
    native_neutral_length = int(native_neutral_inputs["input_ids"].shape[1])
    if packet_size > native_neutral_length:
        raise ValueError("K=32 does not fit visible neutral carrier tokens")

    task_inputs = []
    carrier_inputs = []
    formatted_task_prompts = []
    anchor_packets = []
    state_memories = []
    self_checks = []
    for task_index, task in enumerate(tasks):
        print(f"Capturing memory {task_index + 1}/{len(tasks)}: {task['task_id']}")
        formatted, inputs = encode_prompt(task["prompt"], tokenizer, protocol, device)
        prompt_length = int(inputs["input_ids"].shape[1])
        carrier = build_neutral_carrier(
            native_neutral_inputs,
            task_prompt_length=prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            mode=str(config["carrier"]["mode"]),
        )
        positions = torch.arange(
            prompt_length - packet_size,
            prompt_length,
            device=device,
        )
        if not bool(torch.all(carrier["attention_mask"][0, positions] == 1).item()):
            raise RuntimeError("selected memory overlaps masked carrier positions")
        outputs = None
        anchor = None
        if anchor_layer is not None:
            outputs, anchor = forward_with_packet_capture(
                model,
                inputs,
                layer_idx=anchor_layer,
                positions=positions,
            )
        memory_outputs, captured_states = forward_with_layer_state_capture(
            model,
            inputs,
            layer_indices=all_input_layers,
            positions=positions,
        )
        memory = captured_states["residual_input"]
        if task_index < int(config["memory"]["self_check_tasks"]):
            threshold = float(config["memory"]["maximum_self_logit_delta"])
            if anchor_layer is not None:
                anchor_replayed = forward_with_packet_replacement(
                    model,
                    inputs,
                    layer_idx=anchor_layer,
                    positions=positions,
                    vectors=anchor,
                )
                anchor_delta = float(
                    (anchor_replayed.logits - outputs.logits).abs().max().item()
                )
                self_checks.append(
                    {
                        "task_id": str(task["task_id"]),
                        "scope": anchor_scope_name,
                        "maximum_absolute_logit_delta": anchor_delta,
                    }
                )
                if anchor_delta > threshold:
                    raise RuntimeError("block-output self-replay check failed")
                del anchor_replayed
            for scope_name in contract["scope_order"]:
                if normalized_scopes[scope_name]["boundary"] != "block_input":
                    continue
                scope_layers = normalized_scopes[scope_name]["normalized_layers"]
                replayed = forward_with_layer_input_replay(
                    model,
                    inputs,
                    positions=positions,
                    layer_packets={
                        layer_idx: memory[layer_idx] for layer_idx in scope_layers
                    },
                )
                delta = float(
                    (replayed.logits - memory_outputs.logits).abs().max().item()
                )
                self_checks.append(
                    {
                        "task_id": str(task["task_id"]),
                        "scope": scope_name,
                        "maximum_absolute_logit_delta": delta,
                    }
                )
                if delta > threshold:
                    raise RuntimeError(f"{scope_name} self-replay check failed")
                del replayed
        task_inputs.append(inputs)
        carrier_inputs.append(carrier)
        formatted_task_prompts.append(formatted)
        if anchor is not None:
            anchor_packets.append(anchor.detach().cpu())
        state_memories.append(
            {
                state_type: {
                    layer_idx: value.detach().cpu()
                    for layer_idx, value in layer_states.items()
                }
                for state_type, layer_states in captured_states.items()
            }
        )
        del anchor, memory, captured_states, outputs, memory_outputs

    diagnostics_path = output_path.parent / Path(
        str(config["output"]["state_diagnostics_json"])
    ).name
    state_diagnostics = summarize_state_diagnostics(
        state_memories,
        task_ids=[str(task["task_id"]) for task in tasks],
        layer_indices=all_input_layers,
        packet_size=packet_size,
        run_scope=run_scope,
    )
    state_diagnostics.update(
        {
            "experiment_id": experiment_id,
            "design_sha256": design_sha256,
            "target_model": config["models"]["target_model"],
            "target_model_revision": target_revision,
            "task_manifest_sha256": sha256_path(manifest_path),
        }
    )
    write_json(diagnostics_path, state_diagnostics)

    plan = contract["build_condition_plan"](
        [str(task["task_id"]) for task in tasks],
        conditions,
        shuffle_seed=int(config["controls"]["shuffled_oracle_memory"]["seed"]),
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
                prompt_length = int(inputs["input_ids"].shape[1])
                positions = torch.arange(
                    prompt_length - packet_size,
                    prompt_length,
                    device=device,
                )
                scope = normalized_scopes.get(item.scope_name)
                oracle_task_id = None
                injected_layers: list[int] = []
                boundary = None
                bundle_sha256 = None
                bundle_norm = None
                scalar_count = None
                effective_seed = stable_seed(generation_seed, item.task_index, 108)
                set_seed(effective_seed)
                print(
                    f"Generating seed={generation_seed} task={item.task_id} "
                    f"condition={item.condition}"
                )
                if scope is None:
                    output_text = generate_with_optional_packet(
                        model,
                        tokenizer,
                        inputs,
                        generation_kwargs=gen_kwargs,
                    )
                elif scope["boundary"] == "block_output":
                    oracle_task_id = str(tasks[item.oracle_index]["task_id"])
                    packet = anchor_packets[item.oracle_index]
                    injected_layers = list(scope["normalized_layers"])
                    replay_layer = injected_layers[0]
                    boundary = "block_output"
                    bundle_sha256 = state_bundle_sha256({replay_layer: packet})
                    bundle_norm = state_bundle_norm({replay_layer: packet})
                    scalar_count = int(packet.numel())
                    output_text = generate_with_optional_packet(
                        model,
                        tokenizer,
                        inputs,
                        generation_kwargs=gen_kwargs,
                        layer_idx=replay_layer,
                        positions=positions,
                        vectors=packet,
                    )
                else:
                    oracle_task_id = str(tasks[item.oracle_index]["task_id"])
                    injected_layers = list(scope["normalized_layers"])
                    packets = {
                        layer_idx: state_memories[item.oracle_index][
                            "residual_input"
                        ][layer_idx]
                        for layer_idx in injected_layers
                    }
                    boundary = "block_input"
                    bundle_sha256 = state_bundle_sha256(packets)
                    bundle_norm = state_bundle_norm(packets)
                    scalar_count = sum(
                        int(packet.numel()) for packet in packets.values()
                    )
                    output_text = generate_with_layer_input_replay(
                        model,
                        tokenizer,
                        inputs,
                        generation_kwargs=gen_kwargs,
                        positions=positions,
                        layer_packets=packets,
                    )
                target_formatted = (
                    formatted_task_prompts[item.task_index]
                    if item.target_prompt_kind == "task"
                    else neutral_formatted
                )
                record = {
                    "protocol_version": protocol_version,
                    "design_sha256": design_sha256,
                    "experiment_id": experiment_id,
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
                    "target_prompt_token_count": prompt_length,
                    "target_input_ids_sha256": tensor_sha256(inputs["input_ids"]),
                    "target_attention_mask_sha256": tensor_sha256(
                        inputs["attention_mask"]
                    ),
                    "native_neutral_prompt_token_count": native_neutral_length,
                    "carrier_mode": config["carrier"]["mode"],
                    "memory_scope": item.scope_name,
                    "memory_boundary": boundary,
                    "memory_layer_indices": injected_layers,
                    "memory_layer_count": len(injected_layers),
                    "memory_packet_size": (
                        packet_size if scope is not None else None
                    ),
                    "memory_scalar_count": scalar_count,
                    "oracle_task_id": oracle_task_id,
                    "memory_frobenius_norm": bundle_norm,
                    "memory_sha256": bundle_sha256,
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
    predecessor_field = contract["predecessor_field"]
    metadata = {
        "protocol_version": protocol_version,
        "design_sha256": design_sha256,
        "experiment_id": experiment_id,
        predecessor_field: config[predecessor_field],
        "config": str(config_path),
        "config_sha256": sha256_path(config_path),
        "generations_jsonl": str(output_path),
        "run_scope": run_scope,
        "claim_eligible": run_scope == "full" and complete,
        "functional_split": config["data"]["functional_split"],
        "task_ids": [str(task["task_id"]) for task in tasks],
        "task_count": len(tasks),
        "conditions": conditions,
        "condition_plan": contract["plan_as_dicts"](plan),
        "generation_seeds": generation_seeds,
        "expected_records": len(expected_keys),
        "records": len(existing_keys),
        "new_records": new_records,
        "complete": complete,
        "target_model": config["models"]["target_model"],
        "target_model_revision": target_revision,
        "task_manifest": str(manifest_path),
        "task_manifest_sha256": sha256_path(manifest_path),
        "prompt_protocol": protocol,
        "memory": dict(config["memory"]),
        "normalized_scopes": normalized_scopes,
        "self_checks": self_checks,
        "state_diagnostics": str(diagnostics_path),
        "state_diagnostics_protocol_version": state_diagnostics[
            "protocol_version"
        ],
        "carrier": dict(config["carrier"]),
        "generation": dict(config["generation"]),
    }
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    del anchor_packets, state_memories, task_inputs, carrier_inputs, model
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
        configured = Path(str(config["output"]["generations_jsonl"]))
        output_path = configured.parent / "preflight" / "generations.jsonl"
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
    print("Oracle memory generation completed")
    print(f"run_scope: {metadata['run_scope']}")
    print(f"records: {metadata['records']}/{metadata['expected_records']}")
    print(f"complete: {metadata['complete']}")
    print(f"generations: {metadata['generations_jsonl']}")


if __name__ == "__main__":
    main()
