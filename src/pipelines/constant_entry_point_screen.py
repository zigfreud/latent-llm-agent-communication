"""Generate the frozen development-only LIP-EVAL-035 grid."""

from __future__ import annotations

import gc
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from src.core.packet_bundle import sha256_file
from src.core.prompt_protocol import (
    format_prompt,
    protocol_pair_metadata,
    tokenizer_add_special_tokens,
)
from src.core.utils import set_seed
from src.evaluation.constant_entry_point_screen import (
    CONSTANT_ENTRY_POINT_EXPERIMENT_ID,
    CONSTANT_ENTRY_POINT_PROTOCOL_VERSION,
    CONSTANT_ENTRY_POINT_REPLICA_CONDITIONS,
    CONSTANT_ENTRY_POINT_SHARED_CONDITIONS,
    canonicalize_task,
    constant_entry_point_design_fingerprint,
    expected_constant_entry_point_keys,
    validate_constant_entry_point_contract,
)
from src.evaluation.functional_bridge_screen import (
    FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS,
    FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS,
)
from src.evaluation.oracle_functional import stable_seed
from src.evaluation.packet_bridge_confirmation import packet_layer_norms
from src.pipelines.functional_bridge_screen import (
    _load_p014_cohort,
    _load_replica_specs,
    _predict_entry_packets,
    _read_jsonl,
    _repo_path,
    validate_functional_bridge_screen_runtime_contract,
)
from src.pipelines.infer import load_target, model_input_device
from src.pipelines.oracle_experiment import (
    generation_kwargs,
    load_json_object,
    load_yaml,
    prompt_sha256,
    write_json,
)
from src.pipelines.oracle_memory import generate_with_layer_input_replay
from src.pipelines.packet_confirmation import _suffix_positions, tensor_sha256
from src.pipelines.receiver_aware_replay import _lf_sha256_file


def validate_constant_entry_point_runtime_contract(
    config: Mapping, *, config_path: Path
) -> tuple[dict, dict, dict]:
    validate_constant_entry_point_contract(config)
    predecessor_path = _repo_path(config_path, config["predecessor"]["registry"])
    if config["predecessor"]["registry_sha256"] != _lf_sha256_file(
        predecessor_path
    ):
        raise ValueError("EVAL-034 registry differs from the frozen EVAL-035 design")
    predecessor = load_json_object(predecessor_path)
    if (
        predecessor.get("experiment_id") != "LIP-EVAL-034"
        or predecessor.get("decision", {}).get(
            config["predecessor"]["required_decision"]
        )
        is not True
        or predecessor.get("decision", {}).get("diagnostic_route")
        != config["predecessor"]["required_route"]
    ):
        raise ValueError("EVAL-034 did not authorize the EVAL-035 design")

    source_config_path = _repo_path(config_path, config["source_screen"]["config"])
    if config["source_screen"]["config_sha256"] != _lf_sha256_file(
        source_config_path
    ):
        raise ValueError("EVAL-033 config differs from the frozen EVAL-035 design")
    source_config = load_yaml(source_config_path)
    validate_functional_bridge_screen_runtime_contract(
        source_config, config_path=source_config_path
    )
    source_registry_path = _repo_path(
        config_path, config["source_screen"]["registry"]
    )
    if config["source_screen"]["registry_sha256"] != _lf_sha256_file(
        source_registry_path
    ):
        raise ValueError("EVAL-033 registry differs from the frozen EVAL-035 design")
    source_registry = load_json_object(source_registry_path)
    if (
        source_registry.get("experiment_id") != "LIP-EVAL-033"
        or source_registry.get("execution", {}).get("complete") is not True
        or source_registry.get("frozen_system", {}).get("claim_eligible") is not False
        or source_registry.get("outcomes", {})
        .get("learned_matched", {})
        .get("functional_passes")
        != 0
        or source_registry.get("outcomes", {})
        .get("learned_shuffled", {})
        .get("functional_passes")
        != 0
    ):
        raise ValueError("EVAL-033 registry is not the frozen negative source screen")
    return source_config, predecessor, source_registry


def _key(row: Mapping) -> tuple[str, str, int, int | None]:
    training_seed = row.get("training_seed")
    return (
        str(row.get("task_id", "")),
        str(row.get("condition", "")),
        int(row.get("generation_seed", -1)),
        None if training_seed is None else int(training_seed),
    )


def _constant_inputs(
    config: Mapping,
    source_config: Mapping,
    tokenizer,
    device: torch.device,
) -> tuple[str, dict, dict]:
    interface = config["receiver_interface"]
    prompt = str(interface["prompt"])
    entry_point = str(interface["entry_point"])
    protocol = protocol_pair_metadata(source_config["prompt_protocols"])["target"]
    formatted = format_prompt(prompt, tokenizer, protocol)
    if formatted.count(entry_point) != 1:
        raise ValueError("formatted constant prompt must contain one opaque symbol")
    try:
        encoded = tokenizer(
            formatted,
            return_tensors="pt",
            add_special_tokens=tokenizer_add_special_tokens(protocol),
            return_attention_mask=True,
            return_offsets_mapping=True,
            truncation=False,
        )
    except (NotImplementedError, TypeError) as exc:
        raise RuntimeError(
            "EVAL-035 requires a fast tokenizer for positional leakage audit"
        ) from exc
    offset_mapping = encoded.pop("offset_mapping")[0].tolist()
    symbol_start = formatted.index(entry_point)
    symbol_end = symbol_start + len(entry_point)
    symbol_positions = [
        index
        for index, (start, end) in enumerate(offset_mapping)
        if end > symbol_start and start < symbol_end
    ]
    if not symbol_positions:
        raise ValueError("opaque symbol has no tokenizer positions")
    inputs = {key: value.to(device) for key, value in encoded.items()}
    positions = _suffix_positions(inputs, config["packets"]["offsets"])
    if max(symbol_positions) >= int(positions.min().item()):
        raise ValueError("opaque symbol overlaps the intervention suffix")
    tail_token_count = int(inputs["input_ids"].shape[1]) - 1 - max(
        symbol_positions
    )
    if tail_token_count < int(interface["minimum_tokens_after_entry_point"]):
        raise ValueError("opaque symbol lacks its frozen positional separation")
    audit = {
        "entry_point": entry_point,
        "entry_point_token_positions": symbol_positions,
        "intervention_position_min": int(positions.min().item()),
        "intervention_position_max": int(positions.max().item()),
        "tokens_after_entry_point": tail_token_count,
        "positionally_separated": True,
    }
    return formatted, inputs, audit


def _metadata(
    *,
    config: Mapping,
    config_path: Path,
    output_path: Path,
    task_ids: Sequence[str],
    donor_task_ids: Mapping[str, str],
    record_count: int,
    new_records: int,
    provenance: Mapping,
    complete: bool,
    last_error: str | None = None,
) -> dict:
    payload = {
        "experiment_id": CONSTANT_ENTRY_POINT_EXPERIMENT_ID,
        "protocol_version": CONSTANT_ENTRY_POINT_PROTOCOL_VERSION,
        "design_sha256": constant_entry_point_design_fingerprint(config),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "generations_jsonl": str(output_path),
        "run_scope": "development_only_reused_open_P014_cohort",
        "claim_eligible": False,
        "task_ids": list(task_ids),
        "task_count": len(task_ids),
        "donor_task_ids": dict(donor_task_ids),
        "shared_conditions": list(CONSTANT_ENTRY_POINT_SHARED_CONDITIONS),
        "replica_conditions": list(CONSTANT_ENTRY_POINT_REPLICA_CONDITIONS),
        "generation_seeds": list(FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS),
        "training_seeds": list(FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS),
        "expected_records": len(expected_constant_entry_point_keys(task_ids)),
        "records": record_count,
        "new_records": new_records,
        "complete": complete,
        **dict(provenance),
    }
    if last_error is not None:
        payload["last_error"] = last_error
    return payload


def run_constant_entry_point_screen(
    config_path: Path | str,
    *,
    artifact_root: Path | str,
    output_path: Path | str,
    device: str = "auto",
    prediction_batch_size: int = 4,
    resume: bool = False,
    overwrite: bool = False,
    max_new_records: int | None = None,
) -> dict:
    config_path = Path(config_path)
    artifact_root = Path(artifact_root)
    output_path = Path(output_path)
    config = load_yaml(config_path)
    source_config, predecessor, source_registry = (
        validate_constant_entry_point_runtime_contract(
            config, config_path=config_path
        )
    )
    if max_new_records is not None and max_new_records <= 0:
        raise ValueError("max_new_records must be positive")
    source = source_config["cohort"]["source_artifacts"]
    confirmation_bundle_dir = artifact_root / source["confirmation_bundle"]
    tasks, p014_metadata, packet_records, bundle_validation = _load_p014_cohort(
        source_config,
        artifact_root=artifact_root,
        confirmation_bundle_dir=confirmation_bundle_dir,
    )
    task_ids = [str(task["task_id"]) for task in tasks]
    canonical_tasks = [
        canonicalize_task(task, config["receiver_interface"]["entry_point"])
        for task in tasks
    ]
    donor_task_ids = {
        str(target): str(donor)
        for target, donor in p014_metadata["donor_task_ids"].items()
    }
    specs = _load_replica_specs(source_config, artifact_root=artifact_root)
    expected = expected_constant_entry_point_keys(task_ids)
    if len(expected) != int(config["conditions"]["expected_records"]):
        raise ValueError("EVAL-035 expected-record count drifted")
    design_sha = constant_entry_point_design_fingerprint(config)
    config_sha = sha256_file(config_path)
    metadata_path = output_path.with_suffix(".metadata.json")
    if overwrite:
        for path in (output_path, metadata_path):
            if path.exists():
                path.unlink()
    elif output_path.exists() and not resume:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if resume and output_path.exists():
        for row in _read_jsonl(output_path):
            key = _key(row)
            if key in existing or key not in expected:
                raise ValueError("resumed EVAL-035 grid has duplicate/unexpected row")
            if (
                row.get("design_sha256") != design_sha
                or row.get("config_sha256") != config_sha
                or row.get("run_scope")
                != "development_only_reused_open_P014_cohort"
                or row.get("claim_eligible") is not False
            ):
                raise ValueError("resumed EVAL-035 row differs from frozen design")
            existing.add(key)
    if existing == expected:
        metadata = load_json_object(metadata_path)
        metadata.update(
            {"records": len(existing), "new_records": 0, "complete": True}
        )
        metadata.pop("last_error", None)
        write_json(metadata_path, metadata)
        return metadata

    predictions, scaffold, site_scale = _predict_entry_packets(
        source_config,
        specs,
        packet_records,
        device=device,
        batch_size=prediction_batch_size,
    )
    provenance = {
        "predecessor_registry_sha256": config["predecessor"]["registry_sha256"],
        "predecessor_diagnostic_route": predecessor["decision"][
            "diagnostic_route"
        ],
        "source_screen_config_sha256": config["source_screen"]["config_sha256"],
        "source_screen_registry_sha256": config["source_screen"][
            "registry_sha256"
        ],
        "source_screen_run_commit": source_registry["run_commit"],
        "P014_generations_sha256": source["generations_sha256"],
        "P014_metadata_sha256": source["metadata_sha256"],
        "confirmation_bundle_manifest_sha256": source[
            "confirmation_bundle_manifest_sha256"
        ],
        "confirmation_bundle_validation": bundle_validation,
        "primary_replicas": specs,
        "training_scaffold_sha256": tensor_sha256(scaffold),
        "training_site_scale_sha256": tensor_sha256(site_scale),
        "source_model_revision": source_config["models"]["source"]["revision"],
        "target_model_revision": source_config["models"]["target"]["revision"],
        "prompt_protocols": protocol_pair_metadata(
            source_config["prompt_protocols"]
        ),
    }
    target = source_config["models"]["target"]
    model, tokenizer = load_target(
        target["model_id"],
        device,
        bool(target["load_4bit"]),
        revision=target["revision"],
    )
    target_device = model_input_device(model)
    formatted_prompt, constant_inputs, position_audit = _constant_inputs(
        config, source_config, tokenizer, target_device
    )
    positions = _suffix_positions(constant_inputs, config["packets"]["offsets"])
    generation_config = dict(config["generation"])
    gen_kwargs = generation_kwargs(generation_config, tokenizer)
    provenance.update(
        {
            "receiver_user_prompt": config["receiver_interface"]["prompt"],
            "receiver_user_prompt_sha256": prompt_sha256(
                config["receiver_interface"]["prompt"]
            ),
            "receiver_formatted_prompt_sha256": prompt_sha256(formatted_prompt),
            "receiver_input_ids_sha256": tensor_sha256(
                constant_inputs["input_ids"]
            ),
            "receiver_attention_mask_sha256": tensor_sha256(
                constant_inputs["attention_mask"]
            ),
            "receiver_prompt_token_count": int(
                constant_inputs["input_ids"].shape[1]
            ),
            "receiver_position_audit": position_audit,
            "canonical_entry_point": config["receiver_interface"]["entry_point"],
            "generation": {
                key: generation_config[key]
                for key in (
                    "max_new_tokens",
                    "do_sample",
                    "temperature",
                    "top_p",
                    "repetition_penalty",
                )
            },
        }
    )
    write_json(
        metadata_path,
        _metadata(
            config=config,
            config_path=config_path,
            output_path=output_path,
            task_ids=task_ids,
            donor_task_ids=donor_task_ids,
            record_count=len(existing),
            new_records=0,
            provenance=provenance,
            complete=False,
        ),
    )
    by_id = {task_id: index for index, task_id in enumerate(task_ids)}
    new_records = 0
    output_mode = "a" if output_path.exists() else "w"
    try:
        with output_path.open(output_mode, encoding="utf-8") as handle:
            stop = False
            for task_index, task in enumerate(canonical_tasks):
                if stop:
                    break
                task_id = task_ids[task_index]
                donor_id = donor_task_ids[task_id]
                donor_index = by_id[donor_id]
                for generation_seed in FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS:
                    cells = [
                        (condition, None)
                        for condition in CONSTANT_ENTRY_POINT_SHARED_CONDITIONS
                    ] + [
                        (condition, training_seed)
                        for training_seed in FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS
                        for condition in CONSTANT_ENTRY_POINT_REPLICA_CONDITIONS
                    ]
                    for condition, training_seed in cells:
                        key = (
                            task_id,
                            condition,
                            int(generation_seed),
                            training_seed,
                        )
                        if key in existing:
                            continue
                        source_index = None
                        packet = None
                        layer_indices: list[int] = []
                        if condition == "oracle_teacher_matched":
                            source_index = task_index
                            packet = packet_records[source_index][
                                "target_packet"
                            ].float()
                            layer_indices = list(
                                config["packets"]["oracle_layer_indices"]
                            )
                        elif condition == "oracle_teacher_shuffled":
                            source_index = donor_index
                            packet = packet_records[source_index][
                                "target_packet"
                            ].float()
                            layer_indices = list(
                                config["packets"]["oracle_layer_indices"]
                            )
                        elif condition == "learned_matched":
                            assert training_seed is not None
                            source_index = task_index
                            packet = predictions[training_seed][source_index]
                            layer_indices = list(
                                config["packets"]["learned_layer_indices"]
                            )
                        elif condition == "learned_shuffled":
                            assert training_seed is not None
                            source_index = donor_index
                            packet = predictions[training_seed][source_index]
                            layer_indices = list(
                                config["packets"]["learned_layer_indices"]
                            )
                        elif condition != "canonical_no_packet":
                            raise ValueError(f"unknown EVAL-035 condition: {condition}")

                        packet_hash = None
                        packet_norm = None
                        packet_norms = None
                        residual_norms = None
                        layer_packets = None
                        if packet is not None:
                            packet = packet.detach().float().cpu()
                            if packet.shape[0] != len(layer_indices):
                                raise ValueError("packet depth differs from its layers")
                            packet_hash = tensor_sha256(packet)
                            packet_norm = float(
                                torch.linalg.vector_norm(packet.flatten()).item()
                            )
                            if not math.isfinite(packet_norm):
                                raise FloatingPointError("packet norm is non-finite")
                            packet_norms = packet_layer_norms(packet)
                            if len(layer_indices) == 1:
                                residual_norms = packet_layer_norms(packet - scaffold)
                            layer_packets = {
                                layer: packet[index]
                                for index, layer in enumerate(layer_indices)
                            }
                        effective_seed = stable_seed(
                            int(generation_seed), int(task_index), 14014
                        )
                        set_seed(effective_seed)
                        output_text = generate_with_layer_input_replay(
                            model,
                            tokenizer,
                            constant_inputs,
                            generation_kwargs=gen_kwargs,
                            positions=positions if packet is not None else None,
                            layer_packets=layer_packets,
                        )
                        record = {
                            "experiment_id": CONSTANT_ENTRY_POINT_EXPERIMENT_ID,
                            "protocol_version": CONSTANT_ENTRY_POINT_PROTOCOL_VERSION,
                            "design_sha256": design_sha,
                            "config_sha256": config_sha,
                            "run_scope": "development_only_reused_open_P014_cohort",
                            "claim_eligible": False,
                            "task_id": task_id,
                            "condition": condition,
                            "generation_seed": int(generation_seed),
                            "effective_generation_seed": effective_seed,
                            "training_seed": training_seed,
                            "target_prompt_kind": "constant_opaque_entry_point",
                            "target_user_prompt_sha256": provenance[
                                "receiver_user_prompt_sha256"
                            ],
                            "target_formatted_prompt_sha256": provenance[
                                "receiver_formatted_prompt_sha256"
                            ],
                            "target_input_ids_sha256": provenance[
                                "receiver_input_ids_sha256"
                            ],
                            "target_attention_mask_sha256": provenance[
                                "receiver_attention_mask_sha256"
                            ],
                            "target_prompt_token_count": provenance[
                                "receiver_prompt_token_count"
                            ],
                            "canonical_entry_point": config[
                                "receiver_interface"
                            ]["entry_point"],
                            "canonical_entry_point_positionally_separated": True,
                            "packet_present": packet is not None,
                            "packet_kind": condition if packet is not None else None,
                            "packet_layer_indices": layer_indices,
                            "packet_offsets": (
                                list(config["packets"]["offsets"])
                                if packet is not None
                                else []
                            ),
                            "packet_sha256": packet_hash,
                            "packet_frobenius_norm": packet_norm,
                            "packet_layer_norms": packet_norms,
                            "packet_residual_layer_norms": residual_norms,
                            "source_task_id": (
                                task_ids[source_index]
                                if source_index is not None
                                else None
                            ),
                            "donor_task_id": (
                                donor_id if "shuffled" in condition else None
                            ),
                            "P014_generations_sha256": source[
                                "generations_sha256"
                            ],
                            "confirmation_bundle_manifest_sha256": source[
                                "confirmation_bundle_manifest_sha256"
                            ],
                            "target_model_revision": target["revision"],
                            "output_text": output_text,
                            "task_spec": task,
                        }
                        handle.write(
                            json.dumps(record, ensure_ascii=False, allow_nan=False)
                            + "\n"
                        )
                        handle.flush()
                        existing.add(key)
                        new_records += 1
                        if new_records % 8 == 0:
                            write_json(
                                metadata_path,
                                _metadata(
                                    config=config,
                                    config_path=config_path,
                                    output_path=output_path,
                                    task_ids=task_ids,
                                    donor_task_ids=donor_task_ids,
                                    record_count=len(existing),
                                    new_records=new_records,
                                    provenance=provenance,
                                    complete=False,
                                ),
                            )
                        if (
                            max_new_records is not None
                            and new_records >= max_new_records
                        ):
                            stop = True
                            break
                    if stop:
                        break
    except Exception as exc:
        write_json(
            metadata_path,
            _metadata(
                config=config,
                config_path=config_path,
                output_path=output_path,
                task_ids=task_ids,
                donor_task_ids=donor_task_ids,
                record_count=len(existing),
                new_records=new_records,
                provenance=provenance,
                complete=False,
                last_error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    complete = existing == expected
    metadata = _metadata(
        config=config,
        config_path=config_path,
        output_path=output_path,
        task_ids=task_ids,
        donor_task_ids=donor_task_ids,
        record_count=len(existing),
        new_records=new_records,
        provenance=provenance,
        complete=complete,
    )
    write_json(metadata_path, metadata)
    return metadata
