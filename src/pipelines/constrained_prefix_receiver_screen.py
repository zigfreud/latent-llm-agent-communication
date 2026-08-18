"""Generate the sequential LIP-EVAL-036 constrained-prefix screen."""

from __future__ import annotations

import gc
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from src.core.packet_bundle import sha256_file
from src.core.prompt_protocol import protocol_pair_metadata
from src.core.utils import set_seed
from src.evaluation.constrained_prefix_receiver_screen import (
    CONSTRAINED_PREFIX_CONTROL_CONDITIONS,
    CONSTRAINED_PREFIX_EXPERIMENT_ID,
    CONSTRAINED_PREFIX_LEARNED_CONDITIONS,
    CONSTRAINED_PREFIX_PROTOCOL_VERSION,
    constrained_prefix_design_fingerprint,
    expected_constrained_prefix_keys,
    validate_constrained_prefix_contract,
)
from src.evaluation.constant_entry_point_screen import canonicalize_task
from src.evaluation.functional_bridge_screen import (
    FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS,
    FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS,
)
from src.evaluation.oracle_functional import stable_seed
from src.evaluation.packet_bridge_confirmation import packet_layer_norms
from src.pipelines.constant_entry_point_screen import (
    _constant_inputs,
    validate_constant_entry_point_runtime_contract,
)
from src.pipelines.functional_bridge_screen import (
    _load_p014_cohort,
    _load_replica_specs,
    _predict_entry_packets,
    _read_jsonl,
    _repo_path,
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


def validate_constrained_prefix_runtime_contract(
    config: Mapping, *, config_path: Path
) -> tuple[dict, dict, dict]:
    validate_constrained_prefix_contract(config)
    predecessor_path = _repo_path(config_path, config["predecessor"]["registry"])
    if config["predecessor"]["registry_sha256"] != _lf_sha256_file(
        predecessor_path
    ):
        raise ValueError("EVAL-035 registry differs from the frozen EVAL-036 design")
    predecessor = load_json_object(predecessor_path)
    if (
        predecessor.get("experiment_id") != "LIP-EVAL-035"
        or predecessor.get("decision", {}).get("diagnostic_route")
        != config["predecessor"]["required_route"]
        or predecessor.get("execution", {}).get("record_count") != 864
        or predecessor.get("execution", {}).get("claim_eligible") is not False
    ):
        raise ValueError("EVAL-035 is not the frozen capacity-failure predecessor")

    source_path = _repo_path(config_path, config["source_screen"]["config"])
    if config["source_screen"]["config_sha256"] != _lf_sha256_file(source_path):
        raise ValueError("EVAL-035 config differs from the frozen EVAL-036 design")
    source_config = load_yaml(source_path)
    p014_config, _, source_registry = validate_constant_entry_point_runtime_contract(
        source_config, config_path=source_path
    )
    return p014_config, predecessor, source_registry


def _key(row: Mapping) -> tuple[str, str, int, int | None]:
    training_seed = row.get("training_seed")
    return (
        str(row.get("task_id", "")),
        str(row.get("condition", "")),
        int(row.get("generation_seed", -1)),
        None if training_seed is None else int(training_seed),
    )


def _prefix_token_ids(tokenizer, prefix: str) -> tuple[list[int], str]:
    encoded = tokenizer(prefix, add_special_tokens=False)
    token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    token_ids = [int(value) for value in token_ids]
    if not token_ids:
        raise ValueError("forced completion prefix has no tokenizer IDs")
    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    if decoded != prefix:
        raise ValueError(
            f"forced completion prefix tokenization drifted: {decoded!r} != {prefix!r}"
        )
    return token_ids, decoded


def _control_prefix_sha256(path: Path, count: int) -> str:
    digest = hashlib.sha256()
    observed = 0
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("condition") not in CONSTRAINED_PREFIX_CONTROL_CONDITIONS:
                break
            digest.update(line)
            observed += 1
    if observed != count:
        raise ValueError("control prefix rows are not the frozen 288-row phase")
    return digest.hexdigest()


def _validate_control_lock(
    lock_path: Path,
    *,
    config_path: Path,
    output_path: Path,
    metadata_path: Path,
    metadata: Mapping,
    learned_rows_exist: bool,
) -> dict:
    lock = load_json_object(lock_path)
    if (
        lock.get("experiment_id") != CONSTRAINED_PREFIX_EXPERIMENT_ID
        or lock.get("protocol_version") != CONSTRAINED_PREFIX_PROTOCOL_VERSION
        or lock.get("diagnostic_route") != "constrained_prefix_controls_passed"
        or lock.get("inference", {}).get("controls_passed") is not True
        or lock.get("subprocess_is_security_sandbox") is not True
    ):
        raise ValueError("control lock did not pass the frozen EVAL-036 gates")
    input_hashes = lock.get("sandbox", {}).get("input_sha256", {})
    if input_hashes.get("config") != sha256_file(config_path):
        raise ValueError("control lock config hash differs")
    if learned_rows_exist:
        if metadata.get("control_phase_generations_sha256") != input_hashes.get(
            "generations"
        ):
            raise ValueError("resumed learned phase lost its control generation hash")
        if _control_prefix_sha256(output_path, 288) != input_hashes.get(
            "generations"
        ):
            raise ValueError("control rows changed after the learned phase started")
    else:
        if input_hashes.get("generations") != sha256_file(output_path):
            raise ValueError("control lock generation hash differs")
        if input_hashes.get("metadata") != sha256_file(metadata_path):
            raise ValueError("control lock metadata hash differs")
    return lock


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
    active_phase: str,
    control_phase_complete: bool,
    learned_phase_complete: bool,
    control_lock: Mapping | None = None,
    last_error: str | None = None,
) -> dict:
    payload = {
        "experiment_id": CONSTRAINED_PREFIX_EXPERIMENT_ID,
        "protocol_version": CONSTRAINED_PREFIX_PROTOCOL_VERSION,
        "design_sha256": constrained_prefix_design_fingerprint(config),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "generations_jsonl": str(output_path),
        "run_scope": "development_only_reused_open_P014_cohort",
        "claim_eligible": False,
        "task_ids": list(task_ids),
        "task_count": len(task_ids),
        "donor_task_ids": dict(donor_task_ids),
        "control_conditions": list(CONSTRAINED_PREFIX_CONTROL_CONDITIONS),
        "learned_conditions": list(CONSTRAINED_PREFIX_LEARNED_CONDITIONS),
        "generation_seeds": list(FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS),
        "training_seeds": list(FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS),
        "control_expected_records": len(
            expected_constrained_prefix_keys(task_ids, "controls")
        ),
        "learned_expected_records": len(
            expected_constrained_prefix_keys(task_ids, "learned")
        ),
        "expected_records": len(expected_constrained_prefix_keys(task_ids)),
        "records": record_count,
        "new_records": new_records,
        "active_phase": active_phase,
        "control_phase_complete": control_phase_complete,
        "learned_phase_complete": learned_phase_complete,
        "complete": control_phase_complete and learned_phase_complete,
        **dict(provenance),
    }
    if control_lock is not None:
        payload.update(dict(control_lock))
    if last_error is not None:
        payload["last_error"] = last_error
    return payload


def run_constrained_prefix_receiver_screen(
    config_path: Path | str,
    *,
    artifact_root: Path | str,
    output_path: Path | str,
    phase: str,
    control_lock_path: Path | str | None = None,
    device: str = "auto",
    prediction_batch_size: int = 4,
    resume: bool = False,
    overwrite: bool = False,
    max_new_records: int | None = None,
) -> dict:
    if phase not in {"controls", "learned"}:
        raise ValueError("phase must be controls or learned")
    config_path = Path(config_path)
    artifact_root = Path(artifact_root)
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(".metadata.json")
    config = load_yaml(config_path)
    source_config, predecessor, source_registry = (
        validate_constrained_prefix_runtime_contract(config, config_path=config_path)
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
    expected_all = expected_constrained_prefix_keys(task_ids)
    expected_phase = expected_constrained_prefix_keys(task_ids, phase)
    design_sha = constrained_prefix_design_fingerprint(config)
    config_sha = sha256_file(config_path)
    if overwrite:
        if phase != "controls":
            raise ValueError("overwrite is permitted only when starting controls")
        for path in (output_path, metadata_path):
            if path.exists():
                path.unlink()
    elif output_path.exists() and not resume:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing: set[tuple[str, str, int, int | None]] = set()
    if resume and output_path.exists():
        for row in _read_jsonl(output_path):
            key = _key(row)
            if key in existing or key not in expected_all:
                raise ValueError("resumed EVAL-036 grid has duplicate/unexpected row")
            if (
                row.get("design_sha256") != design_sha
                or row.get("config_sha256") != config_sha
                or row.get("claim_eligible") is not False
                or row.get("forced_completion_prefix")
                != config["decoding_interface"]["prefix"]
                or row.get("forced_prefix_realized") is not True
            ):
                raise ValueError("resumed EVAL-036 row differs from frozen design")
            existing.add(key)
    control_keys = expected_constrained_prefix_keys(task_ids, "controls")
    learned_keys = expected_constrained_prefix_keys(task_ids, "learned")
    if phase == "learned" and not control_keys.issubset(existing):
        raise ValueError("learned phase requires all 288 control rows")
    if expected_phase.issubset(existing):
        return load_json_object(metadata_path)

    prior_metadata = (
        load_json_object(metadata_path) if metadata_path.is_file() else {}
    )
    learned_rows_exist = bool(existing.intersection(learned_keys))
    control_lock_metadata = None
    if phase == "learned":
        if control_lock_path is None:
            raise ValueError("learned phase requires --control-lock")
        lock_path = Path(control_lock_path)
        lock = _validate_control_lock(
            lock_path,
            config_path=config_path,
            output_path=output_path,
            metadata_path=metadata_path,
            metadata=prior_metadata,
            learned_rows_exist=learned_rows_exist,
        )
        input_hashes = lock["sandbox"]["input_sha256"]
        control_lock_metadata = {
            "control_lock_path": str(lock_path),
            "control_lock_sha256": sha256_file(lock_path),
            "control_phase_generations_sha256": input_hashes["generations"],
            "control_phase_metadata_sha256": input_hashes["metadata"],
        }

    specs = _load_replica_specs(source_config, artifact_root=artifact_root)
    predictions = None
    scaffold = None
    site_scale = None
    if phase == "learned":
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
        "source_screen_run_commit": source_registry["run_commit"],
        "P014_generations_sha256": source["generations_sha256"],
        "P014_metadata_sha256": source["metadata_sha256"],
        "confirmation_bundle_manifest_sha256": source[
            "confirmation_bundle_manifest_sha256"
        ],
        "confirmation_bundle_validation": bundle_validation,
        "primary_replicas": specs,
        "source_model_revision": source_config["models"]["source"]["revision"],
        "target_model_revision": source_config["models"]["target"]["revision"],
        "prompt_protocols": protocol_pair_metadata(
            source_config["prompt_protocols"]
        ),
    }
    if scaffold is not None and site_scale is not None:
        provenance.update(
            {
                "training_scaffold_sha256": tensor_sha256(scaffold),
                "training_site_scale_sha256": tensor_sha256(site_scale),
            }
        )
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
    prefix = str(config["decoding_interface"]["prefix"])
    prefix_token_ids, decoded_prefix = _prefix_token_ids(tokenizer, prefix)
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
            "forced_completion_prefix": prefix,
            "forced_completion_prefix_sha256": prompt_sha256(prefix),
            "forced_completion_prefix_token_ids": prefix_token_ids,
            "forced_completion_prefix_token_ids_sha256": tensor_sha256(
                torch.tensor(prefix_token_ids, dtype=torch.long)
            ),
            "forced_completion_prefix_decoded": decoded_prefix,
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
            active_phase=phase,
            control_phase_complete=control_keys.issubset(existing),
            learned_phase_complete=learned_keys.issubset(existing),
            control_lock=control_lock_metadata,
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
                    if phase == "controls":
                        cells = [
                            (condition, None)
                            for condition in CONSTRAINED_PREFIX_CONTROL_CONDITIONS
                        ]
                    else:
                        cells = [
                            (condition, training_seed)
                            for training_seed in FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS
                            for condition in CONSTRAINED_PREFIX_LEARNED_CONDITIONS
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
                            packet = packet_records[source_index]["target_packet"].float()
                            layer_indices = list(config["packets"]["oracle_layer_indices"])
                        elif condition == "oracle_teacher_shuffled":
                            source_index = donor_index
                            packet = packet_records[source_index]["target_packet"].float()
                            layer_indices = list(config["packets"]["oracle_layer_indices"])
                        elif condition == "learned_matched":
                            assert training_seed is not None and predictions is not None
                            source_index = task_index
                            packet = predictions[training_seed][source_index]
                            layer_indices = list(config["packets"]["learned_layer_indices"])
                        elif condition == "learned_shuffled":
                            assert training_seed is not None and predictions is not None
                            source_index = donor_index
                            packet = predictions[training_seed][source_index]
                            layer_indices = list(config["packets"]["learned_layer_indices"])
                        elif condition != "canonical_no_packet":
                            raise ValueError(f"unknown EVAL-036 condition: {condition}")

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
                            if len(layer_indices) == 1 and scaffold is not None:
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
                            replay_mode=config["packets"]["replay_mode"],
                            forced_completion_prefix_token_ids=prefix_token_ids,
                        )
                        if not output_text.startswith(prefix):
                            raise RuntimeError("forced prefix was not realized exactly")
                        record = {
                            "experiment_id": CONSTRAINED_PREFIX_EXPERIMENT_ID,
                            "protocol_version": CONSTRAINED_PREFIX_PROTOCOL_VERSION,
                            "design_sha256": design_sha,
                            "config_sha256": config_sha,
                            "run_scope": "development_only_reused_open_P014_cohort",
                            "claim_eligible": False,
                            "phase": phase,
                            "task_id": task_id,
                            "condition": condition,
                            "generation_seed": int(generation_seed),
                            "effective_generation_seed": effective_seed,
                            "training_seed": training_seed,
                            "target_prompt_kind": config["receiver_interface"][
                                "prompt_kind"
                            ],
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
                            "canonical_entry_point": config["receiver_interface"][
                                "entry_point"
                            ],
                            "canonical_entry_point_positionally_separated": True,
                            "forced_completion_prefix": prefix,
                            "forced_completion_prefix_sha256": provenance[
                                "forced_completion_prefix_sha256"
                            ],
                            "forced_completion_prefix_token_ids": prefix_token_ids,
                            "forced_completion_prefix_token_ids_sha256": provenance[
                                "forced_completion_prefix_token_ids_sha256"
                            ],
                            "forced_prefix_realized": True,
                            "packet_present": packet is not None,
                            "packet_kind": condition if packet is not None else None,
                            "packet_layer_indices": layer_indices,
                            "packet_offsets": (
                                list(config["packets"]["offsets"])
                                if packet is not None
                                else []
                            ),
                            "packet_replay_mode": (
                                config["packets"]["replay_mode"]
                                if packet is not None
                                else None
                            ),
                            "packet_replay_gain": (
                                float(config["packets"]["replay_gain"])
                                if packet is not None
                                else None
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
                                    active_phase=phase,
                                    control_phase_complete=control_keys.issubset(existing),
                                    learned_phase_complete=learned_keys.issubset(existing),
                                    control_lock=control_lock_metadata,
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
                active_phase=phase,
                control_phase_complete=control_keys.issubset(existing),
                learned_phase_complete=learned_keys.issubset(existing),
                control_lock=control_lock_metadata,
                last_error=f"{type(exc).__name__}: {exc}",
            ),
        )
        raise
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metadata = _metadata(
        config=config,
        config_path=config_path,
        output_path=output_path,
        task_ids=task_ids,
        donor_task_ids=donor_task_ids,
        record_count=len(existing),
        new_records=new_records,
        provenance=provenance,
        active_phase=phase,
        control_phase_complete=control_keys.issubset(existing),
        learned_phase_complete=learned_keys.issubset(existing),
        control_lock=control_lock_metadata,
    )
    write_json(metadata_path, metadata)
    return metadata
