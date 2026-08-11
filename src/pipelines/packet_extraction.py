"""Sequential source/teacher extraction for LIP packet bridge bundles."""

from __future__ import annotations

import gc
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from src.core.packet_bundle import (
    PACKET_SPLITS,
    TORCH_DTYPES,
    safe_load_packet_shard,
    sha256_file,
    sha256_json,
    validate_packet_bundle,
)
from src.core.prompt_protocol import (
    format_prompt,
    protocol_pair_metadata,
    tokenizer_add_special_tokens,
)
from src.pipelines.infer import load_source, load_target, model_input_device
from src.pipelines.oracle_experiment import (
    load_json_object,
    load_tasks,
    load_yaml,
    prompt_sha256,
    task_sha256,
)
from src.pipelines.oracle_memory import (
    forward_with_layer_input_capture,
    forward_with_layer_input_replay,
)


TRAINING_SPLITS = ("train", "development_selection", "development_gate")


def _packet_contract(config: Mapping, endpoint: str) -> dict:
    contract = config["packets"][endpoint]
    layers = [int(value) for value in contract["layer_indices"]]
    offsets = [int(value) for value in contract["offsets"]]
    width = int(contract["width"])
    dtype_name = str(contract["dtype"])
    if not layers or len(set(layers)) != len(layers):
        raise ValueError(f"{endpoint} packet layers must be non-empty and unique")
    if not offsets or len(set(offsets)) != len(offsets):
        raise ValueError(f"{endpoint} packet offsets must be non-empty and unique")
    if offsets != list(range(-len(offsets), 0)):
        raise ValueError(f"{endpoint} packet must use one contiguous prompt suffix")
    if width <= 0 or dtype_name not in TORCH_DTYPES:
        raise ValueError(f"{endpoint} packet width/dtype is invalid")
    if contract.get("state_type") != "residual_input":
        raise ValueError(f"{endpoint} packet state_type must be residual_input")
    return {
        "shape": [len(layers), len(offsets), width],
        "layer_indices": layers,
        "offsets": offsets,
        "state_type": "residual_input",
        "dtype": dtype_name,
    }


def load_bound_packet_tasks(config: Mapping) -> tuple[list[dict], dict, Path]:
    """Load the three training-side registries and verify their frozen manifest."""

    data = config["data"]
    manifest_path = Path(str(data["registry_manifest"]))
    manifest = load_json_object(manifest_path)
    if (
        manifest.get("manifest_kind") != "lip_packet_bridge_task_registry"
        or manifest.get("schema_version") != 1
        or manifest.get("experiment_id") != config.get("experiment_id")
    ):
        raise ValueError("packet task registry manifest violates the protocol")

    tasks = []
    for split in TRAINING_SPLITS:
        path = Path(str(data["registries"][split]))
        split_manifest = manifest.get("splits", {}).get(split, {})
        if split_manifest.get("tasks_jsonl_sha256") != sha256_file(path):
            raise ValueError(f"{split} task registry hash changed")
        split_tasks = load_tasks(path)
        expected_hashes = split_manifest.get("task_sha256")
        if expected_hashes != [task_sha256(task) for task in split_tasks]:
            raise ValueError(f"{split} task hashes changed")
        if split_manifest.get("count") != len(split_tasks):
            raise ValueError(f"{split} task count changed")
        tasks.extend({**task, "split": split} for task in split_tasks)

    task_keys = [(task["split"], task["task_id"]) for task in tasks]
    if len(task_keys) != len(set(task_keys)):
        raise ValueError("packet task registries are not disjoint")
    if manifest.get("task_keys_sha256") != sha256_json(task_keys):
        raise ValueError("packet task registry order hash changed")
    return tasks, manifest, manifest_path


def _suffix_positions(
    attention_mask: torch.Tensor,
    offsets: Sequence[int],
) -> torch.Tensor:
    if attention_mask.ndim != 2 or attention_mask.shape[0] != 1:
        raise ValueError("packet extraction requires one unbatched prompt")
    prompt_length = int(attention_mask.shape[1])
    positions = torch.tensor(
        [prompt_length + int(offset) for offset in offsets],
        dtype=torch.long,
        device=attention_mask.device,
    )
    if int(positions.min()) < 0 or int(positions.max()) >= prompt_length:
        raise ValueError("prompt is shorter than the configured packet suffix")
    if not bool(torch.all(attention_mask[0, positions] == 1).item()):
        raise ValueError("packet suffix overlaps masked prompt positions")
    return positions


def _tokenize_prompt(
    task: Mapping,
    tokenizer,
    protocol: Mapping,
    *,
    max_length: int,
    return_offsets: bool,
) -> tuple[str, dict[str, torch.Tensor], dict]:
    formatted = format_prompt(str(task["prompt"]), tokenizer, protocol)
    encoded = tokenizer(
        formatted,
        return_tensors="pt",
        add_special_tokens=tokenizer_add_special_tokens(protocol),
        return_attention_mask=True,
        return_offsets_mapping=return_offsets,
        truncation=False,
    )
    offsets = encoded.pop("offset_mapping", None)
    token_count = int(encoded["attention_mask"].sum().item())
    if encoded["input_ids"].shape[1] > max_length:
        raise ValueError(
            f"task {task['task_id']} has {encoded['input_ids'].shape[1]} tokens; "
            f"maximum is {max_length}"
        )
    input_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    attention_mask = [
        int(value) for value in encoded["attention_mask"][0].tolist()
    ]
    metadata = {
        "formatted_prompt_sha256": prompt_sha256(formatted),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "input_ids_sha256": sha256_json(input_ids),
        "attention_mask_sha256": sha256_json(attention_mask),
        "token_count": token_count,
    }
    if offsets is not None:
        metadata["offset_mapping"] = [
            [int(start), int(stop)] for start, stop in offsets[0].tolist()
        ]
    return formatted, encoded, metadata


def classify_target_terminal_layout(
    task: Mapping,
    formatted_prompt: str,
    token_metadata: Mapping,
    *,
    packet_positions: int,
    boundary_positions: int,
) -> dict:
    """Prove that the task name occupies the causal terminal-name region."""

    entry_point = str(task.get("entry_point", "")).strip()
    if not entry_point:
        raise ValueError(f"task {task['task_id']} has no entry point")
    name_start = formatted_prompt.rfind(entry_point)
    if name_start < 0:
        raise ValueError("formatted target prompt does not contain its entry point")
    name_stop = name_start + len(entry_point)
    offsets = token_metadata.get("offset_mapping")
    if not isinstance(offsets, list) or len(offsets) != len(
        token_metadata["input_ids"]
    ):
        raise ValueError("target tokenizer did not return aligned character offsets")
    indices = [
        index
        for index, (start, stop) in enumerate(offsets)
        if stop > name_start and start < name_stop
    ]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("entry-point token span must be non-empty and contiguous")
    name_count = len(indices)
    if name_count >= packet_positions - boundary_positions:
        raise ValueError("entry point leaves no task-core position in target packet")
    token_count = int(token_metadata["token_count"])
    relative = [index - token_count for index in indices]
    expected = list(
        range(-(boundary_positions + name_count), -boundary_positions)
    )
    if relative != expected:
        raise ValueError(
            "entry-point tokens do not immediately precede the frozen boundary"
        )
    if token_count < packet_positions:
        raise ValueError("target prompt is shorter than the receiver packet")
    return {
        "name_token_count": name_count,
        "name_offsets": relative,
        "boundary_offsets": list(range(-boundary_positions, 0)),
        "core_offsets": list(range(-packet_positions, relative[0])),
        "name_token_ids": [token_metadata["input_ids"][index] for index in indices],
        "boundary_token_ids": token_metadata["input_ids"][-boundary_positions:],
    }


def _capture_packet(
    model,
    inputs: Mapping[str, torch.Tensor],
    contract: Mapping,
) -> tuple[object, dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    positions = _suffix_positions(inputs["attention_mask"], contract["offsets"])
    outputs, captured = forward_with_layer_input_capture(
        model,
        inputs,
        layer_indices=contract["layer_indices"],
        positions=positions,
    )
    packet = torch.stack(
        [captured[layer] for layer in contract["layer_indices"]], dim=0
    )
    if tuple(packet.shape) != tuple(contract["shape"]):
        raise RuntimeError(
            f"captured packet shape {tuple(packet.shape)} differs from "
            f"{tuple(contract['shape'])}"
        )
    packet = packet.detach().to(TORCH_DTYPES[contract["dtype"]]).cpu()
    return outputs, captured, positions, packet


def _stage_path(staging_dir: Path, endpoint: str, index: int) -> Path:
    return staging_dir / endpoint / f"record_{index:04d}.pt"


def _load_stage(path: Path) -> dict:
    payload = safe_load_packet_shard(path)
    if not isinstance(payload, dict):
        raise ValueError(f"staged packet must contain a mapping: {path}")
    return payload


def _extract_real_endpoint(
    tasks: Sequence[Mapping],
    *,
    endpoint: str,
    model,
    tokenizer,
    protocol: Mapping,
    contract: Mapping,
    staging_dir: Path,
    max_length: int,
    boundary_positions: int,
    self_check_tasks: int,
    maximum_self_logit_delta: float,
    resume: bool,
) -> list[dict]:
    reports = []
    destination = model_input_device(model)
    for index, task in enumerate(tasks):
        stage_path = _stage_path(staging_dir, endpoint, index)
        expected_task_hash = task_sha256(
            {key: value for key, value in task.items() if key != "split"}
        )
        if resume and stage_path.is_file():
            staged = _load_stage(stage_path)
            if (
                staged.get("task_id") == task["task_id"]
                and staged.get("task_sha256") == expected_task_hash
                and tuple(staged["packet"].shape) == tuple(contract["shape"])
            ):
                continue

        formatted, inputs, metadata = _tokenize_prompt(
            task,
            tokenizer,
            protocol,
            max_length=max_length,
            return_offsets=endpoint == "target",
        )
        layout = None
        if endpoint == "target":
            layout = classify_target_terminal_layout(
                task,
                formatted,
                metadata,
                packet_positions=contract["shape"][1],
                boundary_positions=boundary_positions,
            )
        model_inputs = {key: value.to(destination) for key, value in inputs.items()}
        outputs, captured, positions, packet = _capture_packet(
            model,
            model_inputs,
            contract,
        )
        self_check = None
        if endpoint == "target" and index < self_check_tasks:
            replayed = forward_with_layer_input_replay(
                model,
                model_inputs,
                positions=positions,
                layer_packets=captured,
            )
            delta = float((replayed.logits - outputs.logits).abs().max().item())
            self_check = {
                "task_id": str(task["task_id"]),
                "maximum_absolute_logit_delta": delta,
            }
            reports.append(self_check)
            if delta > maximum_self_logit_delta:
                raise RuntimeError("target packet self-replay check failed")
            del replayed
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "task_id": str(task["task_id"]),
                "task_sha256": expected_task_hash,
                "packet": packet,
                "formatted_prompt_sha256": metadata["formatted_prompt_sha256"],
                "input_ids": metadata["input_ids"],
                "attention_mask": metadata["attention_mask"],
                "input_ids_sha256": metadata["input_ids_sha256"],
                "attention_mask_sha256": metadata["attention_mask_sha256"],
                "token_count": metadata["token_count"],
                "terminal_layout": layout,
                "self_check": self_check,
            },
            stage_path,
        )
        del outputs, captured, positions, packet, model_inputs, inputs
    return reports


def _dry_record(
    task: Mapping,
    *,
    source_contract: Mapping,
    target_contract: Mapping,
    boundary_positions: int,
) -> dict:
    canonical_task = {key: value for key, value in task.items() if key != "split"}
    digest = task_sha256(canonical_task)
    seed = int(digest[:16], 16) % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source_packet = torch.randn(
        source_contract["shape"],
        generator=generator,
        dtype=TORCH_DTYPES[source_contract["dtype"]],
    )
    target_packet = torch.randn(
        target_contract["shape"],
        generator=generator,
        dtype=TORCH_DTYPES[target_contract["dtype"]],
    )
    name_count = min(2, target_contract["shape"][1] - boundary_positions - 1)
    source_count = max(source_contract["shape"][1], 40)
    target_count = max(target_contract["shape"][1], 32)
    source_ids = [int((seed + index) % 32000) for index in range(source_count)]
    target_ids = [int((seed + 1000 + index) % 32000) for index in range(target_count)]
    source_mask = [1] * source_count
    target_mask = [1] * target_count
    return {
        "task_id": str(task["task_id"]),
        "split": str(task["split"]),
        "task_sha256": digest,
        "prompt_sha256": prompt_sha256(str(task["prompt"])),
        "source_prompt_sha256": prompt_sha256("dry-source\0" + str(task["prompt"])),
        "target_prompt_sha256": prompt_sha256("dry-target\0" + str(task["prompt"])),
        "source_input_ids": source_ids,
        "source_attention_mask": source_mask,
        "source_input_ids_sha256": sha256_json(source_ids),
        "source_attention_mask_sha256": sha256_json(source_mask),
        "source_token_count": source_count,
        "target_input_ids": target_ids,
        "target_attention_mask": target_mask,
        "target_input_ids_sha256": sha256_json(target_ids),
        "target_attention_mask_sha256": sha256_json(target_mask),
        "target_token_count": target_count,
        "name_token_count": name_count,
        "source_packet": source_packet,
        "target_packet": target_packet,
    }


def _combine_staged_record(task: Mapping, source: Mapping, target: Mapping) -> dict:
    canonical_task = {key: value for key, value in task.items() if key != "split"}
    expected_hash = task_sha256(canonical_task)
    if any(
        stage.get("task_id") != task["task_id"]
        or stage.get("task_sha256") != expected_hash
        for stage in (source, target)
    ):
        raise ValueError("source/target staged packet identity mismatch")
    layout = target.get("terminal_layout")
    if not isinstance(layout, Mapping):
        raise ValueError("target staged packet lacks terminal-layout evidence")
    return {
        "task_id": str(task["task_id"]),
        "split": str(task["split"]),
        "task_sha256": expected_hash,
        "prompt_sha256": prompt_sha256(str(task["prompt"])),
        "source_prompt_sha256": source["formatted_prompt_sha256"],
        "target_prompt_sha256": target["formatted_prompt_sha256"],
        "source_input_ids": source["input_ids"],
        "source_attention_mask": source["attention_mask"],
        "source_input_ids_sha256": source["input_ids_sha256"],
        "source_attention_mask_sha256": source["attention_mask_sha256"],
        "source_token_count": source["token_count"],
        "target_input_ids": target["input_ids"],
        "target_attention_mask": target["attention_mask"],
        "target_input_ids_sha256": target["input_ids_sha256"],
        "target_attention_mask_sha256": target["attention_mask_sha256"],
        "target_token_count": target["token_count"],
        "name_token_count": int(layout["name_token_count"]),
        "source_packet": source["packet"],
        "target_packet": target["packet"],
    }


def _cleanup_accelerator() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def materialize_packet_bundle(
    config_path: Path | str,
    *,
    bundle_dir: Path | str | None = None,
    dry_run: bool = False,
    resume: bool = False,
    overwrite: bool = False,
    keep_staging: bool = False,
    preflight_tasks_per_split: int | None = None,
) -> dict:
    """Build and validate a confirmation-free packet bundle."""

    config_path = Path(config_path)
    config = load_yaml(config_path)
    extraction = config["extraction"]
    extraction_scope = (
        "preflight" if preflight_tasks_per_split is not None else "full"
    )
    if (
        preflight_tasks_per_split is not None
        and preflight_tasks_per_split < 2
    ):
        raise ValueError("preflight_tasks_per_split must be at least two")
    bundle_dir = Path(bundle_dir or extraction["default_bundle_dir"])
    if bundle_dir.exists() and (bundle_dir / "manifest.json").is_file():
        if resume and not overwrite:
            validation = validate_packet_bundle(
                bundle_dir,
                require_real=not dry_run and extraction_scope == "full",
            )
            if validation["extraction_scope"] != extraction_scope:
                raise ValueError("existing bundle uses a different extraction scope")
            return {
                **validation,
                "manifest": str(bundle_dir / "manifest.json"),
            }
        if not overwrite:
            raise FileExistsError(f"packet bundle already exists: {bundle_dir}")
    if bundle_dir.exists() and overwrite:
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = bundle_dir / "staging"
    shard_dir = bundle_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    tasks, registry_manifest, registry_path = load_bound_packet_tasks(config)
    if preflight_tasks_per_split is not None:
        tasks = [
            task
            for split in TRAINING_SPLITS
            for task in [
                item for item in tasks if item["split"] == split
            ][:preflight_tasks_per_split]
        ]
    source_contract = _packet_contract(config, "source")
    target_contract = _packet_contract(config, "target")
    protocols = protocol_pair_metadata(config["prompt_protocols"])
    boundary_positions = int(config["packets"]["target"]["boundary_positions"])
    max_length = int(extraction["max_length"])
    self_checks = []

    if not dry_run:
        source_spec = config["models"]["source"]
        source_model, source_tokenizer = load_source(
            source_spec["model_id"],
            str(extraction.get("device", "auto")),
            load_4bit=bool(extraction.get("source_load_4bit", False)),
            revision=source_spec["revision"],
            use_safetensors=bool(source_spec.get("use_safetensors", True)),
        )
        _extract_real_endpoint(
            tasks,
            endpoint="source",
            model=source_model,
            tokenizer=source_tokenizer,
            protocol=protocols["source"],
            contract=source_contract,
            staging_dir=staging_dir,
            max_length=max_length,
            boundary_positions=boundary_positions,
            self_check_tasks=0,
            maximum_self_logit_delta=0.0,
            resume=resume,
        )
        del source_model, source_tokenizer
        _cleanup_accelerator()

        target_spec = config["models"]["target"]
        target_model, target_tokenizer = load_target(
            target_spec["model_id"],
            str(extraction.get("device", "auto")),
            bool(extraction.get("target_load_4bit", True)),
            revision=target_spec["revision"],
        )
        self_checks = _extract_real_endpoint(
            tasks,
            endpoint="target",
            model=target_model,
            tokenizer=target_tokenizer,
            protocol=protocols["target"],
            contract=target_contract,
            staging_dir=staging_dir,
            max_length=max_length,
            boundary_positions=boundary_positions,
            self_check_tasks=int(extraction.get("self_check_tasks", 1)),
            maximum_self_logit_delta=float(
                extraction.get("maximum_self_logit_delta", 1e-4)
            ),
            resume=resume,
        )
        del target_model, target_tokenizer
        _cleanup_accelerator()

    records = []
    for index, task in enumerate(tasks):
        if dry_run:
            record = _dry_record(
                task,
                source_contract=source_contract,
                target_contract=target_contract,
                boundary_positions=boundary_positions,
            )
        else:
            record = _combine_staged_record(
                task,
                _load_stage(_stage_path(staging_dir, "source", index)),
                _load_stage(_stage_path(staging_dir, "target", index)),
            )
        records.append(record)

    shard_size = int(extraction.get("shard_size", 8))
    if shard_size <= 0:
        raise ValueError("extraction.shard_size must be positive")
    shard_entries = []
    for shard_index, start in enumerate(range(0, len(records), shard_size)):
        shard_records = records[start : start + shard_size]
        path = shard_dir / f"shard_{shard_index:04d}.pt"
        torch.save(shard_records, path)
        shard_entries.append(
            {
                "path": f"shards/{path.name}",
                "records": len(shard_records),
                "sha256": sha256_file(path),
            }
        )

    source_spec = config["models"]["source"]
    target_spec = config["models"]["target"]
    dataset = config["data"]["dataset"]
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in PACKET_SPLITS
    }
    task_keys = [(record["split"], record["task_id"]) for record in records]
    trace_payload = {
        "config_sha256": sha256_file(config_path),
        "registry_sha256": sha256_file(registry_path),
        "extraction_mode": "dry_run" if dry_run else "real",
        "extraction_scope": extraction_scope,
    }
    trace_id = "LIP-PROTO-014-" + sha256_json(trace_payload)[:16]
    manifest = {
        "bundle_format": "lip_packet_bundle",
        "schema_version": 1,
        "trace_id": trace_id,
        "extraction_mode": "dry_run" if dry_run else "real",
        "extraction_scope": extraction_scope,
        "config_sha256": sha256_file(config_path),
        "source": {
            "model_id": source_spec["model_id"],
            "revision": source_spec["revision"],
            "use_safetensors": bool(source_spec.get("use_safetensors", True)),
            "prompt_protocol": protocols["source"],
        },
        "target": {
            "model_id": target_spec["model_id"],
            "revision": target_spec["revision"],
            "use_safetensors": bool(target_spec.get("use_safetensors", True)),
            "prompt_protocol": protocols["target"],
        },
        "dataset": {
            "dataset_id": dataset["dataset_id"],
            "dataset_config": dataset["dataset_config"],
            "revision": dataset["revision"],
        },
        "registry": {
            "manifest": str(registry_path),
            "manifest_sha256": sha256_file(registry_path),
            "task_keys_sha256": registry_manifest["task_keys_sha256"],
        },
        "source_packet": source_contract,
        "target_packet": target_contract,
        "predecessor": {
            "protocol": config["predecessor"]["protocol"],
            "sha256sums_sha256": config["predecessor"]["sha256sums_sha256"],
        },
        "splits": split_counts,
        "task_keys_sha256": sha256_json(task_keys),
        "shards": shard_entries,
        "extraction": {
            "sequential_model_loading": True,
            "source_load_4bit": bool(extraction.get("source_load_4bit", False)),
            "target_load_4bit": bool(extraction.get("target_load_4bit", True)),
            "max_length": max_length,
            "self_checks": self_checks,
        },
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = validate_packet_bundle(
        bundle_dir,
        require_real=not dry_run and extraction_scope == "full",
    )
    if not keep_staging and staging_dir.exists():
        shutil.rmtree(staging_dir)
    return {**validation, "manifest": str(bundle_dir / "manifest.json")}
