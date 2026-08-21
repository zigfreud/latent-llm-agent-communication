"""Generate one sequential phase of the frozen LIP-EVAL-037 blend screen."""

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
from src.evaluation.constant_entry_point_screen import canonicalize_task
from src.evaluation.oracle_functional import stable_seed
from src.evaluation.oracle_native_packet_blend_screen import (
    ORACLE_BLEND_CONDITIONS,
    ORACLE_BLEND_CONFIRMATION_GENERATION_SEEDS,
    ORACLE_BLEND_EXPERIMENT_ID,
    ORACLE_BLEND_PROTOCOL_VERSION,
    ORACLE_BLEND_SCREEN_ALPHAS,
    ORACLE_BLEND_SCREEN_GENERATION_SEED,
    expected_oracle_blend_keys,
    oracle_blend_design_fingerprint,
    validate_oracle_blend_contract,
)
from src.evaluation.packet_bridge_confirmation import packet_layer_norms
from src.pipelines.constrained_prefix_receiver_screen import (
    _prefix_token_ids,
    validate_constrained_prefix_runtime_contract,
)
from src.pipelines.constant_entry_point_screen import _constant_inputs
from src.pipelines.functional_bridge_screen import (
    _load_p014_cohort,
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


def validate_oracle_blend_runtime_contract(
    config: Mapping, *, config_path: Path
) -> tuple[dict, dict, dict]:
    """Bind EVAL-037 to the registered EVAL-036 failure and P014 cohort."""

    validate_oracle_blend_contract(config)
    predecessor_path = _repo_path(config_path, config["predecessor"]["registry"])
    if config["predecessor"]["registry_sha256"] != _lf_sha256_file(
        predecessor_path
    ):
        raise ValueError("EVAL-036 registry differs from the frozen EVAL-037 design")
    predecessor = load_json_object(predecessor_path)
    if (
        predecessor.get("experiment_id") != "LIP-EVAL-036"
        or predecessor.get("decision", {}).get("diagnostic_route")
        != config["predecessor"]["required_route"]
        or predecessor.get("execution", {}).get("record_count") != 288
        or predecessor.get("execution", {}).get("claim_eligible") is not False
        or predecessor.get("decision", {}).get("learned_phase_executed") is not False
    ):
        raise ValueError("EVAL-036 is not the frozen oracle-capacity predecessor")

    source_path = _repo_path(config_path, config["source_screen"]["config"])
    if config["source_screen"]["config_sha256"] != _lf_sha256_file(source_path):
        raise ValueError("EVAL-036 config differs from the frozen EVAL-037 design")
    source_config = load_yaml(source_path)
    p014_config, _, source_registry = validate_constrained_prefix_runtime_contract(
        source_config, config_path=source_path
    )
    return p014_config, predecessor, source_registry


def _key(row: Mapping) -> tuple[str, str, int, float]:
    return (
        str(row.get("task_id", "")),
        str(row.get("condition", "")),
        int(row.get("generation_seed", -1)),
        float(row.get("blend_alpha", -1.0)),
    )


def _screen_prefix_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    observed = 0
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("phase") != "screen":
                break
            digest.update(line)
            observed += 1
    if observed != 192:
        raise ValueError("screen prefix rows are not the frozen 192-row phase")
    return digest.hexdigest()


def _validate_screen_lock(
    lock_path: Path,
    *,
    config_path: Path,
    output_path: Path,
    metadata_path: Path,
    metadata: Mapping,
    confirmation_rows_exist: bool,
) -> dict:
    lock = load_json_object(lock_path)
    inference = lock.get("inference", {})
    if (
        lock.get("experiment_id") != ORACLE_BLEND_EXPERIMENT_ID
        or lock.get("protocol_version") != ORACLE_BLEND_PROTOCOL_VERSION
        or lock.get("diagnostic_route")
        != "oracle_blend_screen_candidate_selected"
        or inference.get("screen_passed") is not True
        or inference.get("confirmation_authorized_by_frozen_gate") is not True
        or lock.get("subprocess_is_security_sandbox") is not True
    ):
        raise ValueError("screen lock did not pass the frozen EVAL-037 gates")
    selected_alpha = inference.get("selected_alpha")
    if selected_alpha not in ORACLE_BLEND_SCREEN_ALPHAS:
        raise ValueError("screen lock has no valid selected alpha")
    input_hashes = lock.get("sandbox", {}).get("input_sha256", {})
    if input_hashes.get("config") != sha256_file(config_path):
        raise ValueError("screen lock config hash differs")
    if confirmation_rows_exist:
        if metadata.get("screen_phase_generations_sha256") != input_hashes.get(
            "generations"
        ):
            raise ValueError("resumed confirmation lost its screen generation hash")
        if _screen_prefix_sha256(output_path) != input_hashes.get("generations"):
            raise ValueError("screen rows changed after confirmation started")
    else:
        if input_hashes.get("generations") != sha256_file(output_path):
            raise ValueError("screen lock generation hash differs")
        if input_hashes.get("metadata") != sha256_file(metadata_path):
            raise ValueError("screen lock metadata hash differs")
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
    screen_phase_complete: bool,
    confirmation_phase_complete: bool,
    screen_lock: Mapping | None = None,
    last_error: str | None = None,
) -> dict:
    screen_expected = len(expected_oracle_blend_keys(task_ids, "screen"))
    payload = {
        "experiment_id": ORACLE_BLEND_EXPERIMENT_ID,
        "protocol_version": ORACLE_BLEND_PROTOCOL_VERSION,
        "design_sha256": oracle_blend_design_fingerprint(config),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "generations_jsonl": str(output_path),
        "run_scope": "development_only_reused_open_P014_cohort",
        "claim_eligible": False,
        "task_ids": list(task_ids),
        "task_count": len(task_ids),
        "donor_task_ids": dict(donor_task_ids),
        "conditions": list(ORACLE_BLEND_CONDITIONS),
        "screen_alphas": list(ORACLE_BLEND_SCREEN_ALPHAS),
        "screen_generation_seed": ORACLE_BLEND_SCREEN_GENERATION_SEED,
        "confirmation_generation_seeds": list(
            ORACLE_BLEND_CONFIRMATION_GENERATION_SEEDS
        ),
        "screen_expected_records": screen_expected,
        "confirmation_expected_records": 128,
        "expected_records": 320,
        "records": record_count,
        "new_records": new_records,
        "active_phase": active_phase,
        "screen_phase_complete": screen_phase_complete,
        "confirmation_phase_complete": confirmation_phase_complete,
        "complete": screen_phase_complete and confirmation_phase_complete,
        **dict(provenance),
    }
    if screen_lock is not None:
        payload.update(dict(screen_lock))
    if last_error is not None:
        payload["last_error"] = last_error
    return payload


def run_oracle_native_packet_blend_screen(
    config_path: Path | str,
    *,
    artifact_root: Path | str,
    output_path: Path | str,
    phase: str,
    screen_lock_path: Path | str | None = None,
    device: str = "auto",
    resume: bool = False,
    overwrite: bool = False,
    max_new_records: int | None = None,
) -> dict:
    if phase not in {"screen", "confirm"}:
        raise ValueError("phase must be screen or confirm")
    config_path = Path(config_path)
    artifact_root = Path(artifact_root)
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(".metadata.json")
    config = load_yaml(config_path)
    source_config, predecessor, source_registry = validate_oracle_blend_runtime_contract(
        config, config_path=config_path
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
    screen_keys = expected_oracle_blend_keys(task_ids, "screen")
    design_sha = oracle_blend_design_fingerprint(config)
    config_sha = sha256_file(config_path)

    if overwrite:
        if phase != "screen":
            raise ValueError("overwrite is permitted only when starting the screen")
        for path in (output_path, metadata_path):
            if path.exists():
                path.unlink()
    elif output_path.exists() and not resume:
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(output_path) if resume and output_path.exists() else []
    selected_alpha: float | None = None
    prior_metadata = load_json_object(metadata_path) if metadata_path.is_file() else {}
    if phase == "confirm":
        if not screen_keys.issubset({_key(row) for row in rows}):
            raise ValueError("confirmation requires all 192 screen rows")
        if screen_lock_path is None:
            raise ValueError("confirmation requires --screen-lock")
        confirmation_rows_exist = any(row.get("phase") == "confirm" for row in rows)
        lock = _validate_screen_lock(
            Path(screen_lock_path),
            config_path=config_path,
            output_path=output_path,
            metadata_path=metadata_path,
            metadata=prior_metadata,
            confirmation_rows_exist=confirmation_rows_exist,
        )
        selected_alpha = float(lock["inference"]["selected_alpha"])
        lock_hashes = lock["sandbox"]["input_sha256"]
        screen_lock_metadata = {
            "screen_lock_path": str(screen_lock_path),
            "screen_lock_sha256": sha256_file(Path(screen_lock_path)),
            "selected_alpha": selected_alpha,
            "screen_phase_generations_sha256": lock_hashes["generations"],
            "screen_phase_metadata_sha256": lock_hashes["metadata"],
        }
    else:
        screen_lock_metadata = None

    confirmation_keys = (
        expected_oracle_blend_keys(task_ids, "confirm", selected_alpha=selected_alpha)
        if selected_alpha is not None
        else set()
    )
    expected_known = screen_keys | confirmation_keys
    existing: set[tuple[str, str, int, float]] = set()
    for row in rows:
        key = _key(row)
        if key in existing or key not in expected_known:
            raise ValueError("resumed EVAL-037 grid has duplicate/unexpected row")
        if (
            row.get("design_sha256") != design_sha
            or row.get("config_sha256") != config_sha
            or row.get("claim_eligible") is not False
            or row.get("forced_completion_prefix")
            != config["decoding_interface"]["prefix"]
            or row.get("forced_prefix_realized") is not True
        ):
            raise ValueError("resumed EVAL-037 row differs from frozen design")
        existing.add(key)
    expected_phase = screen_keys if phase == "screen" else confirmation_keys
    if expected_phase.issubset(existing):
        return load_json_object(metadata_path)

    provenance = {
        "predecessor_registry_sha256": config["predecessor"]["registry_sha256"],
        "predecessor_diagnostic_route": predecessor["decision"]["diagnostic_route"],
        "source_screen_config_sha256": config["source_screen"]["config_sha256"],
        "source_screen_run_commit": predecessor["run_commit"],
        "P014_source_registry_run_commit": source_registry["run_commit"],
        "P014_generations_sha256": source["generations_sha256"],
        "P014_metadata_sha256": source["metadata_sha256"],
        "confirmation_bundle_manifest_sha256": source[
            "confirmation_bundle_manifest_sha256"
        ],
        "confirmation_bundle_validation": bundle_validation,
        "source_model_revision": source_config["models"]["source"]["revision"],
        "target_model_revision": source_config["models"]["target"]["revision"],
        "prompt_protocols": protocol_pair_metadata(source_config["prompt_protocols"]),
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
            "receiver_input_ids_sha256": tensor_sha256(constant_inputs["input_ids"]),
            "receiver_attention_mask_sha256": tensor_sha256(
                constant_inputs["attention_mask"]
            ),
            "receiver_prompt_token_count": int(constant_inputs["input_ids"].shape[1]),
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

    def metadata_payload(*, last_error: str | None = None, new_records: int = 0):
        return _metadata(
            config=config,
            config_path=config_path,
            output_path=output_path,
            task_ids=task_ids,
            donor_task_ids=donor_task_ids,
            record_count=len(existing),
            new_records=new_records,
            provenance=provenance,
            active_phase=phase,
            screen_phase_complete=screen_keys.issubset(existing),
            confirmation_phase_complete=bool(confirmation_keys)
            and confirmation_keys.issubset(existing),
            screen_lock=screen_lock_metadata,
            last_error=last_error,
        )

    write_json(metadata_path, metadata_payload())
    by_id = {task_id: index for index, task_id in enumerate(task_ids)}
    alphas = ORACLE_BLEND_SCREEN_ALPHAS if phase == "screen" else (selected_alpha,)
    seeds = (
        (ORACLE_BLEND_SCREEN_GENERATION_SEED,)
        if phase == "screen"
        else ORACLE_BLEND_CONFIRMATION_GENERATION_SEEDS
    )
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
                for generation_seed in seeds:
                    for alpha_value in alphas:
                        assert alpha_value is not None
                        alpha = float(alpha_value)
                        for condition in ORACLE_BLEND_CONDITIONS:
                            key = (task_id, condition, int(generation_seed), alpha)
                            if key in existing:
                                continue
                            source_index = (
                                task_index
                                if condition == "oracle_blend_matched"
                                else donor_index
                            )
                            packet = packet_records[source_index][
                                "target_packet"
                            ].detach().float().cpu()
                            layer_indices = list(config["packets"]["oracle_layer_indices"])
                            if packet.shape[0] != len(layer_indices):
                                raise ValueError("oracle packet depth differs from its layers")
                            packet_norm = float(
                                torch.linalg.vector_norm(packet.flatten()).item()
                            )
                            if not math.isfinite(packet_norm):
                                raise FloatingPointError("oracle packet norm is non-finite")
                            layer_packets = {
                                layer: packet[index]
                                for index, layer in enumerate(layer_indices)
                            }
                            effective_seed = stable_seed(
                                int(generation_seed),
                                int(task_index),
                                int(config["conditions"]["effective_generation_seed_salt"]),
                            )
                            set_seed(effective_seed)
                            output_text = generate_with_layer_input_replay(
                                model,
                                tokenizer,
                                constant_inputs,
                                generation_kwargs=gen_kwargs,
                                positions=positions,
                                layer_packets=layer_packets,
                                replay_mode="blend",
                                replay_alpha=alpha,
                                forced_completion_prefix_token_ids=prefix_token_ids,
                            )
                            if not output_text.startswith(prefix):
                                raise RuntimeError("forced prefix was not realized exactly")
                            record = {
                                "experiment_id": ORACLE_BLEND_EXPERIMENT_ID,
                                "protocol_version": ORACLE_BLEND_PROTOCOL_VERSION,
                                "design_sha256": design_sha,
                                "config_sha256": config_sha,
                                "run_scope": "development_only_reused_open_P014_cohort",
                                "claim_eligible": False,
                                "phase": phase,
                                "task_id": task_id,
                                "condition": condition,
                                "generation_seed": int(generation_seed),
                                "effective_generation_seed": effective_seed,
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
                                "packet_present": True,
                                "packet_kind": condition,
                                "packet_layer_indices": layer_indices,
                                "packet_offsets": list(config["packets"]["offsets"]),
                                "packet_replay_mode": "blend",
                                "blend_alpha": alpha,
                                "native_weight": 1.0 - alpha,
                                "packet_weight": alpha,
                                "packet_sha256": tensor_sha256(packet),
                                "packet_frobenius_norm": packet_norm,
                                "packet_layer_norms": packet_layer_norms(packet),
                                "source_task_id": task_ids[source_index],
                                "donor_task_id": (
                                    donor_id if condition == "oracle_blend_shuffled" else None
                                ),
                                "P014_generations_sha256": source["generations_sha256"],
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
                                    metadata_payload(new_records=new_records),
                                )
                            if max_new_records is not None and new_records >= max_new_records:
                                stop = True
                                break
                        if stop:
                            break
                    if stop:
                        break
    except Exception as exc:
        write_json(
            metadata_path,
            metadata_payload(
                last_error=f"{type(exc).__name__}: {exc}",
                new_records=new_records,
            ),
        )
        raise
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    metadata = metadata_payload(new_records=new_records)
    write_json(metadata_path, metadata)
    return metadata
