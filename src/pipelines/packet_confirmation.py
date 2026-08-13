"""Post-gate functional generation for learned LIP packet bridges."""

from __future__ import annotations

import gc
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from src.core.packet_bridge import reconstruct_target_packet
from src.core.packet_bundle import (
    load_packet_records,
    sha256_file,
    validate_packet_bundle,
)
from src.core.prompt_protocol import (
    format_prompt,
    protocol_pair_metadata,
    tokenizer_add_special_tokens,
)
from src.core.utils import set_seed
from src.evaluation.oracle_functional import stable_seed
from src.evaluation.packet_bridge_confirmation import (
    PACKET_CONFIRMATION_CONDITIONS,
    PACKET_CONFIRMATION_EVALUATION_POLICY,
    PACKET_CONFIRMATION_EXPERIMENT_ID,
    PACKET_CONFIRMATION_GENERATION_SEEDS,
    PACKET_CONFIRMATION_PROTOCOL_VERSION,
    PACKET_CONFIRMATION_REPLICA_CONDITIONS,
    PACKET_CONFIRMATION_SHARED_CONDITIONS,
    PACKET_CONFIRMATION_TRAINING_SEEDS,
    expected_confirmation_generation_keys,
    isotropic_residual_with_matched_layer_norms,
    packet_confirmation_design_fingerprint,
    packet_layer_norms,
    stratified_confirmation_donors,
    validate_packet_confirmation_contract,
)
from src.pipelines.infer import load_target, model_input_device
from src.pipelines.oracle_experiment import (
    generation_kwargs,
    load_json_object,
    load_yaml,
    prompt_sha256,
    task_sha256,
    write_json,
)
from src.pipelines.oracle_memory import generate_with_layer_input_replay
from src.pipelines.packet_bridge import build_packet_bridge, resolve_packet_device
from src.pipelines.packet_extraction import load_bound_confirmation_tasks
from src.pipelines.packet_matrix import build_replica_config


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor semantics and contiguous CPU bytes without pickle."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor_sha256 requires a tensor")
    value = tensor.detach().contiguous().cpu()
    header = json.dumps(
        {"dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(header)
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _finite_tensor(value: torch.Tensor, *, label: str) -> None:
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{label} contains non-finite values")


def _primary_model_config(config: Mapping) -> dict:
    primary = str(config["objectives"]["primary"])
    variant = config["objectives"]["variants"][primary]
    model = {"kind": str(variant["model_kind"])}
    if model["kind"] == "query_conditioned":
        model.update(dict(config["bridge"]))
    return model


def load_primary_replica_specs(
    config_path: Path | str,
    matrix_summary_path: Path | str,
    training_bundle_dir: Path | str,
) -> tuple[dict, list[dict], dict]:
    """Validate the aggregate gate and bind all registered primary replicas."""

    config_path = Path(config_path)
    matrix_summary_path = Path(matrix_summary_path)
    training_bundle_dir = Path(training_bundle_dir)
    config = load_yaml(config_path)
    validate_packet_confirmation_contract(config)
    matrix = load_json_object(matrix_summary_path)
    training_validation = validate_packet_bundle(training_bundle_dir, require_real=True)
    training_manifest_hash = sha256_file(training_bundle_dir / "manifest.json")
    primary = str(config["objectives"]["primary"])
    seeds = [int(seed) for seed in config["training"]["seeds"]]
    primary_gate = matrix.get("development_gates", {}).get(primary, {})
    checks = {
        "experiment": matrix.get("experiment_id")
        == PACKET_CONFIRMATION_EXPERIMENT_ID,
        "protocol": matrix.get("protocol_version")
        == PACKET_CONFIRMATION_PROTOCOL_VERSION,
        "config_hash": matrix.get("contract_config_sha256")
        == sha256_file(config_path),
        "bundle_hash": matrix.get("bundle_manifest_sha256")
        == training_manifest_hash,
        "full_matrix": matrix.get("full_registered_matrix") is True,
        "ready": matrix.get("ready_for_confirmation") is True,
        "primary": matrix.get("primary_variant") == primary,
        "registered_seeds": matrix.get("registered_seeds") == seeds,
        "primary_complete": primary_gate.get("complete") is True,
        "primary_passed": primary_gate.get("passed") is True,
        "minimum_replicas": primary_gate.get("minimum_passing_replicas")
        == int(config["development_gate"]["minimum_passing_replicas"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "confirmation remains blocked by matrix provenance: "
            + ", ".join(failed)
        )
    if training_validation["split_counts"]["confirmation"] != 0:
        raise ValueError("training bundle contains confirmation records")

    run_entries = matrix.get("runs", {}).get(primary, [])
    if [int(entry.get("seed", -1)) for entry in run_entries] != seeds:
        raise ValueError("primary matrix run order differs from registered seeds")
    expected_model = _primary_model_config(config)
    specs = []
    for entry in run_entries:
        seed = int(entry["seed"])
        summary_path = Path(str(entry["summary"]))
        summary = load_json_object(summary_path)
        checkpoint_path = Path(str(summary.get("checkpoint", "")))
        statistics_path = summary_path.parent / "target_statistics.pt"
        resolved_path = summary_path.parent / "resolved_config.yaml"
        for path in (checkpoint_path, statistics_path, resolved_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        resolved = load_yaml(resolved_path)
        expected_resolved = build_replica_config(
            config,
            bundle_dir=training_bundle_dir,
            output_dir=summary_path.parent,
            variant_name=primary,
            seed=seed,
        )
        run_checks = {
            "summary_seed": summary.get("seed") == seed,
            "summary_model": summary.get("model_kind") == expected_model["kind"],
            "summary_updates": summary.get("updates_completed")
            == int(config["training"]["max_updates"]),
            "summary_bundle": summary.get("bundle_manifest_sha256")
            == training_manifest_hash,
            "summary_gate": bool(summary.get("development_gate", {}).get("passed"))
            == bool(entry.get("development_gate_passed")),
            "resolved_config": resolved == expected_resolved,
            "statistics_hash": summary.get("target_statistics_sha256")
            == sha256_file(statistics_path),
        }
        failed_run = [name for name, passed in run_checks.items() if not passed]
        if failed_run:
            raise ValueError(
                f"primary replica {seed} failed provenance: "
                + ", ".join(failed_run)
            )
        specs.append(
            {
                "seed": seed,
                "development_gate_passed": bool(
                    summary["development_gate"]["passed"]
                ),
                "best_step": int(summary["best_step"]),
                "summary": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "target_statistics": str(statistics_path),
                "target_statistics_sha256": sha256_file(statistics_path),
                "resolved_config": str(resolved_path),
                "resolved_config_sha256": sha256_file(resolved_path),
            }
        )
    return matrix, specs, training_validation


def predict_primary_confirmation_packets(
    config: Mapping,
    replica_specs: Sequence[Mapping],
    confirmation_records: Sequence[Mapping],
    *,
    source_shape: Sequence[int],
    target_shape: Sequence[int],
    device: str = "auto",
    batch_size: int = 4,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Run each small bridge sequentially and reconstruct raw receiver packets."""

    if batch_size <= 0:
        raise ValueError("prediction batch_size must be positive")
    prediction_device = resolve_packet_device(device)
    source = torch.stack(
        [record["source_packet"].float() for record in confirmation_records]
    )
    _finite_tensor(source, label="confirmation source packets")
    expected_model = _primary_model_config(config)
    predictions: dict[int, torch.Tensor] = {}
    scaffold = None
    site_scale = None
    for spec in replica_specs:
        statistics = torch.load(
            Path(str(spec["target_statistics"])),
            map_location="cpu",
            weights_only=True,
        )
        current_scaffold = statistics.get("scaffold")
        current_scale = statistics.get("site_scale")
        _finite_tensor(current_scaffold, label="training scaffold")
        _finite_tensor(current_scale, label="training site scale")
        if tuple(current_scaffold.shape) != tuple(target_shape) or tuple(
            current_scale.shape
        ) != tuple(target_shape[:2]):
            raise ValueError("target statistics shape differs from receiver packet")
        if not bool(torch.all(current_scale > 0.0)):
            raise ValueError("target site scale must be strictly positive")
        if scaffold is None:
            scaffold = current_scaffold.float().cpu()
            site_scale = current_scale.float().cpu()
        elif not torch.equal(scaffold, current_scaffold.float().cpu()) or not torch.equal(
            site_scale, current_scale.float().cpu()
        ):
            raise ValueError("target statistics tensors differ across replicas")

        checkpoint = torch.load(
            Path(str(spec["checkpoint"])),
            map_location="cpu",
            weights_only=True,
        )
        if (
            checkpoint.get("model_config") != expected_model
            or checkpoint.get("source_shape") != list(source_shape)
            or checkpoint.get("target_shape") != list(target_shape)
            or int(checkpoint.get("step", -1)) != int(spec["best_step"])
        ):
            raise ValueError(f"checkpoint contract changed for seed {spec['seed']}")
        model = build_packet_bridge(
            expected_model,
            tuple(source_shape),
            tuple(target_shape),
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(prediction_device)
        model.eval()
        normalized_rows = []
        with torch.inference_mode():
            for start in range(0, len(source), batch_size):
                normalized_rows.append(
                    model(source[start : start + batch_size].to(prediction_device))
                    .float()
                    .cpu()
                )
        normalized = torch.cat(normalized_rows, dim=0)
        _finite_tensor(normalized, label=f"seed {spec['seed']} normalized prediction")
        raw = reconstruct_target_packet(normalized, scaffold, site_scale).cpu()
        _finite_tensor(raw, label=f"seed {spec['seed']} receiver packet")
        predictions[int(spec["seed"])] = raw
        del checkpoint, model, normalized, raw
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert scaffold is not None and site_scale is not None
    return predictions, scaffold, site_scale


def _strict_json_object(line: str, *, line_number: int) -> dict:
    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON constant {value} at line {line_number}")

    row = json.loads(line, parse_constant=reject_constant)
    if not isinstance(row, dict):
        raise ValueError(f"generation row {line_number} must be an object")
    return row


def _generation_key(row: Mapping) -> tuple[str, str, int, int | None]:
    training_seed = row.get("training_seed")
    return (
        str(row.get("task_id", "")),
        str(row.get("condition", "")),
        int(row.get("generation_seed")),
        None if training_seed is None else int(training_seed),
    )


def _read_existing_generations(
    path: Path,
    *,
    expected_keys: set[tuple[str, str, int, int | None]],
    design_sha256: str,
    config_sha256: str,
    confirmation_manifest_sha256: str,
    matrix_summary_sha256: str,
) -> tuple[set[tuple[str, str, int, int | None]], list[dict]]:
    keys = set()
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = _strict_json_object(line, line_number=line_number)
            key = _generation_key(row)
            if key in keys:
                raise ValueError(f"duplicate confirmation generation key: {key}")
            if key not in expected_keys:
                raise ValueError(f"unexpected confirmation generation key: {key}")
            provenance_ok = (
                row.get("protocol_version")
                == PACKET_CONFIRMATION_PROTOCOL_VERSION
                and row.get("design_sha256") == design_sha256
                and row.get("config_sha256") == config_sha256
                and row.get("confirmation_manifest_sha256")
                == confirmation_manifest_sha256
                and row.get("matrix_summary_sha256") == matrix_summary_sha256
                and row.get("run_scope") == "confirmation"
            )
            if not provenance_ok:
                raise ValueError("existing confirmation row uses different provenance")
            keys.add(key)
            rows.append(row)
    return keys, rows


def _validate_existing_metadata(
    path: Path,
    *,
    design_sha256: str,
    config_sha256: str,
    task_ids: Sequence[str],
    existing_keys: set[tuple[str, str, int, int | None]],
    expected_keys: set[tuple[str, str, int, int | None]],
    confirmation_manifest_sha256: str,
    confirmation_bundle_manifest_sha256: str,
    training_bundle_manifest_sha256: str,
    matrix_summary_sha256: str,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            "resumed confirmation generations lack their metadata sidecar"
        )
    metadata = load_json_object(path)
    complete = existing_keys == expected_keys
    recorded_count = metadata.get("records")
    checks = {
        "experiment": metadata.get("experiment_id")
        == PACKET_CONFIRMATION_EXPERIMENT_ID,
        "protocol": metadata.get("protocol_version")
        == PACKET_CONFIRMATION_PROTOCOL_VERSION,
        "design": metadata.get("design_sha256") == design_sha256,
        "config": metadata.get("config_sha256") == config_sha256,
        "scope": metadata.get("run_scope") == "confirmation",
        "tasks": metadata.get("task_ids") == list(task_ids),
        "conditions": metadata.get("conditions")
        == list(PACKET_CONFIRMATION_CONDITIONS),
        "generation_seeds": metadata.get("generation_seeds")
        == list(PACKET_CONFIRMATION_GENERATION_SEEDS),
        "training_seeds": metadata.get("training_seeds")
        == list(PACKET_CONFIRMATION_TRAINING_SEEDS),
        "expected_records": metadata.get("expected_records")
        == len(expected_keys),
        "record_count": isinstance(recorded_count, int)
        and 0 <= recorded_count <= len(existing_keys),
        "confirmation_manifest": metadata.get("task_manifest_sha256")
        == confirmation_manifest_sha256,
        "confirmation_bundle": metadata.get(
            "confirmation_bundle_manifest_sha256"
        )
        == confirmation_bundle_manifest_sha256,
        "training_bundle": metadata.get("training_bundle_manifest_sha256")
        == training_bundle_manifest_sha256,
        "matrix": metadata.get("matrix_summary_sha256")
        == matrix_summary_sha256,
        "scaffold_hash": isinstance(
            metadata.get("training_scaffold_sha256"), str
        )
        and len(metadata["training_scaffold_sha256"]) == 64,
        "site_scale_hash": isinstance(
            metadata.get("training_site_scale_sha256"), str
        )
        and len(metadata["training_site_scale_sha256"]) == 64,
        "complete": not bool(metadata.get("complete")) or complete,
        "claim_flag": bool(metadata.get("claim_eligible"))
        == bool(metadata.get("complete")),
    }
    if not complete:
        checks["incomplete_flag"] = metadata.get("complete") is False
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "resumed confirmation metadata changed: " + ", ".join(failed)
        )
    return metadata


def _validate_confirmation_self_checks(
    manifest: Mapping,
    records: Sequence[Mapping],
    config: Mapping,
) -> list[dict]:
    extraction = manifest.get("extraction", {})
    expected_count = min(
        int(config["extraction"].get("self_check_tasks", 1)), len(records)
    )
    reports = extraction.get("self_checks")
    if not isinstance(reports, list) or len(reports) != expected_count:
        raise ValueError("confirmation bundle lacks the frozen self-replay checks")
    maximum = float(
        config["extraction"].get("maximum_self_logit_delta", 1e-4)
    )
    normalized = []
    for index, report in enumerate(reports):
        if not isinstance(report, Mapping):
            raise ValueError("confirmation self-replay report must be an object")
        delta = float(report.get("maximum_absolute_logit_delta", math.inf))
        if (
            report.get("task_id") != str(records[index]["task_id"])
            or not math.isfinite(delta)
            or delta < 0.0
            or delta > maximum
        ):
            raise ValueError("confirmation bundle failed its self-replay contract")
        normalized.append(
            {
                "task_id": str(report["task_id"]),
                "maximum_absolute_logit_delta": delta,
            }
        )
    return normalized


def _validate_confirmation_bundle_binding(
    manifest: Mapping,
    config: Mapping,
    *,
    config_sha256: str,
    confirmation_manifest_sha256: str,
) -> None:
    protocols = protocol_pair_metadata(config["prompt_protocols"])

    def packet_contract(endpoint: str) -> dict:
        registered = config["packets"][endpoint]
        layers = [int(value) for value in registered["layer_indices"]]
        offsets = [int(value) for value in registered["offsets"]]
        return {
            "shape": [len(layers), len(offsets), int(registered["width"])],
            "layer_indices": layers,
            "offsets": offsets,
            "state_type": str(registered["state_type"]),
            "dtype": str(registered["dtype"]),
        }

    source = config["models"]["source"]
    target = config["models"]["target"]
    dataset = config["data"]["dataset"]
    extraction = config["extraction"]
    checks = {
        "config": manifest.get("config_sha256") == config_sha256,
        "registry": manifest.get("registry", {}).get("manifest_sha256")
        == confirmation_manifest_sha256,
        "source": manifest.get("source")
        == {
            "model_id": source["model_id"],
            "revision": source["revision"],
            "use_safetensors": bool(source.get("use_safetensors", True)),
            "prompt_protocol": protocols["source"],
        },
        "target": manifest.get("target")
        == {
            "model_id": target["model_id"],
            "revision": target["revision"],
            "use_safetensors": bool(target.get("use_safetensors", True)),
            "prompt_protocol": protocols["target"],
        },
        "dataset": manifest.get("dataset")
        == {
            "dataset_id": dataset["dataset_id"],
            "dataset_config": dataset["dataset_config"],
            "revision": dataset["revision"],
        },
        "source_packet": manifest.get("source_packet")
        == packet_contract("source"),
        "target_packet": manifest.get("target_packet")
        == packet_contract("target"),
        "predecessor": manifest.get("predecessor")
        == {
            "protocol": config["predecessor"]["protocol"],
            "sha256sums_sha256": config["predecessor"]["sha256sums_sha256"],
        },
        "sequential": manifest.get("extraction", {}).get(
            "sequential_model_loading"
        )
        is True,
        "source_quantization": manifest.get("extraction", {}).get(
            "source_load_4bit"
        )
        == bool(extraction.get("source_load_4bit", False)),
        "target_quantization": manifest.get("extraction", {}).get(
            "target_load_4bit"
        )
        == bool(extraction.get("target_load_4bit", True)),
        "max_length": manifest.get("extraction", {}).get("max_length")
        == int(extraction["max_length"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "confirmation bundle drifted from the frozen config: "
            + ", ".join(failed)
        )


def _target_inputs_from_record(record: Mapping, device: torch.device) -> dict:
    return {
        "input_ids": torch.tensor(
            [record["target_input_ids"]], dtype=torch.long, device=device
        ),
        "attention_mask": torch.tensor(
            [record["target_attention_mask"]], dtype=torch.long, device=device
        ),
    }


def _neutral_inputs(config: Mapping, tokenizer, device: torch.device) -> tuple[str, dict]:
    protocol = protocol_pair_metadata(config["prompt_protocols"])["target"]
    neutral_prompt = str(config["packets"]["target"]["neutral_prompt"])
    formatted = format_prompt(neutral_prompt, tokenizer, protocol)
    encoded = tokenizer(
        formatted,
        return_tensors="pt",
        add_special_tokens=tokenizer_add_special_tokens(protocol),
        return_attention_mask=True,
        truncation=False,
    )
    return formatted, {key: value.to(device) for key, value in encoded.items()}


def _suffix_positions(inputs: Mapping[str, torch.Tensor], offsets: Sequence[int]) -> torch.Tensor:
    length = int(inputs["input_ids"].shape[1])
    positions = torch.tensor(
        [length + int(offset) for offset in offsets],
        dtype=torch.long,
        device=inputs["input_ids"].device,
    )
    if int(positions.min()) < 0 or int(positions.max()) >= length:
        raise ValueError("neutral carrier is shorter than the receiver packet")
    if not bool(torch.all(inputs["attention_mask"][0, positions] == 1).item()):
        raise ValueError("receiver packet overlaps masked neutral-carrier positions")
    return positions


def _packet_for_condition(
    condition: str,
    *,
    task_index: int,
    donor_index: int,
    training_seed: int | None,
    generation_seed: int,
    records: Sequence[Mapping],
    predictions: Mapping[int, torch.Tensor],
    scaffold: torch.Tensor,
) -> tuple[torch.Tensor | None, int | None, int | None, dict | None]:
    source_index = None
    noise_seed = None
    random_audit = None
    if condition in {"neutral_no_lip", "text_only_no_lip"}:
        packet = None
    elif condition == "oracle_teacher_matched":
        source_index = task_index
        packet = records[source_index]["target_packet"].float()
    elif condition == "oracle_teacher_shuffled":
        source_index = donor_index
        packet = records[source_index]["target_packet"].float()
    elif condition == "mean_scaffold":
        packet = scaffold
    elif condition == "learned_matched":
        if training_seed is None:
            raise ValueError("learned_matched requires a bridge replica")
        source_index = task_index
        packet = predictions[training_seed][source_index]
    elif condition == "learned_shuffled":
        if training_seed is None:
            raise ValueError("learned_shuffled requires a bridge replica")
        source_index = donor_index
        packet = predictions[training_seed][source_index]
    elif condition == "random_residual_norm_matched":
        if training_seed is None:
            raise ValueError("random residual requires a bridge replica")
        source_index = task_index
        reference = predictions[training_seed][task_index] - scaffold
        noise_seed = stable_seed(
            int(generation_seed), int(task_index), int(training_seed), 14017
        )
        generator = torch.Generator(device="cpu").manual_seed(noise_seed)
        random_residual = isotropic_residual_with_matched_layer_norms(
            reference,
            generator=generator,
        )
        packet = scaffold + random_residual
        reference_norms = packet_layer_norms(reference)
        observed_norms = packet_layer_norms(packet - scaffold)
        absolute_deltas = [
            abs(observed - expected)
            for observed, expected in zip(observed_norms, reference_norms)
        ]
        relative_deltas = [
            delta / max(1.0, abs(expected))
            for delta, expected in zip(absolute_deltas, reference_norms)
        ]
        random_audit = {
            "reference_residual_layer_norms": reference_norms,
            "maximum_absolute_layer_norm_delta": max(absolute_deltas),
            "maximum_relative_layer_norm_delta": max(relative_deltas),
        }
        if random_audit["maximum_relative_layer_norm_delta"] > 5e-6:
            raise FloatingPointError(
                "norm-matched random residual exceeded numerical tolerance"
            )
    else:
        raise ValueError(f"unknown confirmation condition: {condition}")
    if packet is not None:
        packet = packet.detach().float().cpu()
        _finite_tensor(packet, label=f"{condition} packet")
    return packet, source_index, noise_seed, random_audit


def _metadata_payload(
    *,
    config: Mapping,
    config_path: Path,
    output_path: Path,
    task_ids: Sequence[str],
    donors: Mapping[int, int],
    expected_record_count: int,
    record_count: int,
    new_records: int,
    artifact_provenance: Mapping,
    complete: bool,
    last_error: str | None = None,
) -> dict:
    payload = {
        "protocol_version": PACKET_CONFIRMATION_PROTOCOL_VERSION,
        "design_sha256": packet_confirmation_design_fingerprint(config),
        "experiment_id": PACKET_CONFIRMATION_EXPERIMENT_ID,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "generations_jsonl": str(output_path),
        "run_scope": "confirmation",
        "claim_eligible": bool(complete),
        "task_ids": list(task_ids),
        "task_count": len(task_ids),
        "conditions": list(PACKET_CONFIRMATION_CONDITIONS),
        "shared_conditions": list(PACKET_CONFIRMATION_SHARED_CONDITIONS),
        "replica_conditions": list(PACKET_CONFIRMATION_REPLICA_CONDITIONS),
        "generation_seeds": list(PACKET_CONFIRMATION_GENERATION_SEEDS),
        "training_seeds": list(PACKET_CONFIRMATION_TRAINING_SEEDS),
        "condition_replication": {
            "shared_controls": "generation_seed_only",
            "bridge_dependent": "generation_seed_x_training_seed",
        },
        "donor_task_ids": {
            str(task_ids[target]): str(task_ids[source])
            for target, source in donors.items()
        },
        "expected_records": expected_record_count,
        "records": record_count,
        "new_records": new_records,
        "complete": bool(complete),
        "evaluation_policy": dict(PACKET_CONFIRMATION_EVALUATION_POLICY),
        **dict(artifact_provenance),
    }
    if last_error is not None:
        payload["last_error"] = str(last_error)
    return payload


def run_packet_bridge_confirmation(
    config_path: Path | str,
    *,
    training_bundle_dir: Path | str,
    confirmation_bundle_dir: Path | str,
    matrix_summary_path: Path | str,
    output_path: Path | str,
    device: str = "auto",
    prediction_batch_size: int = 4,
    resume: bool = False,
    overwrite: bool = False,
    max_new_records: int | None = None,
) -> dict:
    """Generate the frozen 014 confirmation grid with safe row-level resume."""

    config_path = Path(config_path)
    training_bundle_dir = Path(training_bundle_dir)
    confirmation_bundle_dir = Path(confirmation_bundle_dir)
    matrix_summary_path = Path(matrix_summary_path)
    output_path = Path(output_path)
    if max_new_records is not None and max_new_records <= 0:
        raise ValueError("max_new_records must be positive")
    config = load_yaml(config_path)
    validate_packet_confirmation_contract(config)
    design_sha256 = packet_confirmation_design_fingerprint(config)
    config_sha256 = sha256_file(config_path)

    tasks_with_split, confirmation_manifest, confirmation_manifest_path = (
        load_bound_confirmation_tasks(config)
    )
    tasks = [
        {key: value for key, value in task.items() if key != "split"}
        for task in tasks_with_split
    ]
    task_ids = [str(task["task_id"]) for task in tasks]
    donors = stratified_confirmation_donors(
        tasks,
        seed=int(config["confirmation"]["derangement_seed"]),
    )
    confirmation_validation = validate_packet_bundle(confirmation_bundle_dir)
    if (
        confirmation_validation["extraction_mode"] != "real"
        or confirmation_validation["extraction_scope"] != "confirmation"
        or confirmation_validation["split_counts"]["confirmation"] != len(tasks)
        or any(
            confirmation_validation["split_counts"][split] != 0
            for split in ("train", "development_selection", "development_gate")
        )
    ):
        raise ValueError("confirmation bundle is not the real sealed cohort")
    confirmation_bundle_manifest = load_json_object(
        confirmation_bundle_dir / "manifest.json"
    )
    _validate_confirmation_bundle_binding(
        confirmation_bundle_manifest,
        config,
        config_sha256=config_sha256,
        confirmation_manifest_sha256=sha256_file(confirmation_manifest_path),
    )
    records = load_packet_records(confirmation_bundle_dir)
    if [str(record["task_id"]) for record in records] != task_ids:
        raise ValueError("confirmation bundle task order differs from the cohort")
    for task, record in zip(tasks, records):
        if record["task_sha256"] != task_sha256(task):
            raise ValueError(f"confirmation task hash changed: {task['task_id']}")
    self_checks = _validate_confirmation_self_checks(
        confirmation_bundle_manifest,
        records,
        config,
    )

    matrix, replica_specs, training_validation = load_primary_replica_specs(
        config_path,
        matrix_summary_path,
        training_bundle_dir,
    )
    expected_keys = expected_confirmation_generation_keys(task_ids)
    confirmation_manifest_hash = sha256_file(confirmation_manifest_path)
    matrix_summary_hash = sha256_file(matrix_summary_path)
    if (
        confirmation_manifest.get("matrix_summary_sha256")
        != matrix_summary_hash
    ):
        raise ValueError(
            "confirmation cohort and primary matrix use different gate evidence"
        )
    confirmation_bundle_hash = sha256_file(confirmation_bundle_dir / "manifest.json")
    training_bundle_hash = sha256_file(training_bundle_dir / "manifest.json")
    artifact_provenance = {
        "task_manifest": str(confirmation_manifest_path),
        "task_manifest_sha256": confirmation_manifest_hash,
        "selection_report": str(config["confirmation"]["selection_report"]),
        "selection_report_sha256": sha256_file(
            Path(str(config["confirmation"]["selection_report"]))
        ),
        "confirmation_bundle": str(confirmation_bundle_dir),
        "confirmation_bundle_manifest_sha256": confirmation_bundle_hash,
        "confirmation_bundle_validation": confirmation_validation,
        "training_bundle": str(training_bundle_dir),
        "training_bundle_manifest_sha256": training_bundle_hash,
        "training_bundle_validation": training_validation,
        "matrix_summary": str(matrix_summary_path),
        "matrix_summary_sha256": matrix_summary_hash,
        "matrix_primary_gate": matrix["development_gates"][
            config["objectives"]["primary"]
        ],
        "primary_variant": str(config["objectives"]["primary"]),
        "primary_replicas": replica_specs,
        "source_model": config["models"]["source"]["model_id"],
        "source_model_revision": config["models"]["source"]["revision"],
        "target_model": config["models"]["target"]["model_id"],
        "target_model_revision": config["models"]["target"]["revision"],
        "prompt_protocols": protocol_pair_metadata(config["prompt_protocols"]),
        "confirmation_self_checks": self_checks,
    }

    metadata_path = output_path.with_suffix(".metadata.json")
    if overwrite:
        for path in (output_path, metadata_path):
            if path.exists():
                path.unlink()
    elif output_path.exists() and not resume:
        raise FileExistsError(f"confirmation output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys: set[tuple[str, str, int, int | None]] = set()
    existing_metadata = None
    if resume and output_path.exists():
        existing_keys, _ = _read_existing_generations(
            output_path,
            expected_keys=expected_keys,
            design_sha256=design_sha256,
            config_sha256=config_sha256,
            confirmation_manifest_sha256=confirmation_manifest_hash,
            matrix_summary_sha256=matrix_summary_hash,
        )
        existing_metadata = _validate_existing_metadata(
            metadata_path,
            design_sha256=design_sha256,
            config_sha256=config_sha256,
            task_ids=task_ids,
            existing_keys=existing_keys,
            expected_keys=expected_keys,
            confirmation_manifest_sha256=confirmation_manifest_hash,
            confirmation_bundle_manifest_sha256=confirmation_bundle_hash,
            training_bundle_manifest_sha256=training_bundle_hash,
            matrix_summary_sha256=matrix_summary_hash,
        )
    complete = existing_keys == expected_keys
    if complete:
        assert existing_metadata is not None
        existing_metadata = {
            **existing_metadata,
            "records": len(existing_keys),
            "new_records": 0,
            "complete": True,
            "claim_eligible": True,
        }
        existing_metadata.pop("last_error", None)
        write_json(metadata_path, existing_metadata)
        return existing_metadata

    predictions, scaffold, site_scale = predict_primary_confirmation_packets(
        config,
        replica_specs,
        records,
        source_shape=confirmation_validation["source_shape"],
        target_shape=confirmation_validation["target_shape"],
        device=device,
        batch_size=prediction_batch_size,
    )
    artifact_provenance["training_scaffold_sha256"] = tensor_sha256(scaffold)
    artifact_provenance["training_site_scale_sha256"] = tensor_sha256(site_scale)
    if existing_metadata is not None and any(
        existing_metadata.get(field) != artifact_provenance[field]
        for field in (
            "training_scaffold_sha256",
            "training_site_scale_sha256",
        )
    ):
        raise ValueError("resumed confirmation uses different training statistics")

    target_spec = config["models"]["target"]
    model, tokenizer = load_target(
        target_spec["model_id"],
        device,
        bool(config["extraction"].get("target_load_4bit", True)),
        revision=target_spec["revision"],
    )
    target_device = model_input_device(model)
    neutral_formatted, neutral_inputs = _neutral_inputs(
        config, tokenizer, target_device
    )
    offsets = [int(value) for value in config["packets"]["target"]["offsets"]]
    neutral_positions = _suffix_positions(neutral_inputs, offsets)
    target_layers = [
        int(value) for value in config["packets"]["target"]["layer_indices"]
    ]
    generation_config = {
        key: config["confirmation"][key]
        for key in (
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "repetition_penalty",
        )
    }
    gen_kwargs = generation_kwargs(generation_config, tokenizer)
    artifact_provenance.update(
        {
            "neutral_prompt": config["packets"]["target"]["neutral_prompt"],
            "neutral_formatted_prompt_sha256": prompt_sha256(neutral_formatted),
            "neutral_input_ids_sha256": tensor_sha256(neutral_inputs["input_ids"]),
            "neutral_attention_mask_sha256": tensor_sha256(
                neutral_inputs["attention_mask"]
            ),
            "neutral_token_count": int(neutral_inputs["input_ids"].shape[1]),
            "generation": generation_config,
        }
    )
    if existing_metadata is not None and any(
        existing_metadata.get(field) != artifact_provenance[field]
        for field in (
            "neutral_formatted_prompt_sha256",
            "neutral_input_ids_sha256",
            "neutral_attention_mask_sha256",
            "neutral_token_count",
            "generation",
        )
    ):
        raise ValueError("resumed confirmation uses different receiver inputs")
    write_json(
        metadata_path,
        _metadata_payload(
            config=config,
            config_path=config_path,
            output_path=output_path,
            task_ids=task_ids,
            donors=donors,
            expected_record_count=len(expected_keys),
            record_count=len(existing_keys),
            new_records=0,
            artifact_provenance=artifact_provenance,
            complete=False,
        ),
    )

    new_records = 0
    output_mode = "a" if output_path.exists() else "w"
    try:
        with output_path.open(output_mode, encoding="utf-8") as output_handle:
            stop = False
            for task_index, task in enumerate(tasks):
                if stop:
                    break
                task_id = task_ids[task_index]
                donor_index = donors[task_index]
                task_inputs = _target_inputs_from_record(
                    records[task_index], target_device
                )
                for generation_seed in PACKET_CONFIRMATION_GENERATION_SEEDS:
                    cells = [
                        (condition, None)
                        for condition in PACKET_CONFIRMATION_SHARED_CONDITIONS
                    ] + [
                        (condition, training_seed)
                        for training_seed in PACKET_CONFIRMATION_TRAINING_SEEDS
                        for condition in PACKET_CONFIRMATION_REPLICA_CONDITIONS
                    ]
                    for condition, training_seed in cells:
                        key = (
                            task_id,
                            condition,
                            int(generation_seed),
                            training_seed,
                        )
                        if key in existing_keys:
                            continue
                        packet, source_index, noise_seed, random_audit = (
                            _packet_for_condition(
                                condition,
                                task_index=task_index,
                                donor_index=donor_index,
                                training_seed=training_seed,
                                generation_seed=int(generation_seed),
                                records=records,
                                predictions=predictions,
                                scaffold=scaffold,
                            )
                        )
                        text_control = condition == "text_only_no_lip"
                        inputs = task_inputs if text_control else neutral_inputs
                        positions = None
                        layer_packets = None
                        packet_hash = None
                        packet_norm = None
                        packet_norms = None
                        residual_norms = None
                        if packet is not None:
                            positions = (
                                _suffix_positions(inputs, offsets)
                                if text_control
                                else neutral_positions
                            )
                            layer_packets = {
                                layer: packet[index]
                                for index, layer in enumerate(target_layers)
                            }
                            packet_hash = tensor_sha256(packet)
                            packet_norm = float(
                                torch.linalg.vector_norm(packet.flatten()).item()
                            )
                            packet_norms = packet_layer_norms(packet)
                            residual_norms = packet_layer_norms(packet - scaffold)
                            if not math.isfinite(packet_norm):
                                raise FloatingPointError("packet norm is non-finite")
                        effective_seed = stable_seed(
                            int(generation_seed), int(task_index), 14014
                        )
                        set_seed(effective_seed)
                        output_text = generate_with_layer_input_replay(
                            model,
                            tokenizer,
                            inputs,
                            generation_kwargs=gen_kwargs,
                            positions=positions,
                            layer_packets=layer_packets,
                        )
                        source_task_id = (
                            task_ids[source_index]
                            if source_index is not None
                            else None
                        )
                        record = {
                            "protocol_version": PACKET_CONFIRMATION_PROTOCOL_VERSION,
                            "design_sha256": design_sha256,
                            "experiment_id": PACKET_CONFIRMATION_EXPERIMENT_ID,
                            "config_sha256": config_sha256,
                            "run_scope": "confirmation",
                            "claim_eligible": True,
                            "task_id": task_id,
                            "condition": condition,
                            "generation_seed": int(generation_seed),
                            "effective_generation_seed": effective_seed,
                            "training_seed": training_seed,
                            "target_prompt_kind": "task" if text_control else "neutral",
                            "target_user_prompt_sha256": prompt_sha256(
                                str(task["prompt"])
                                if text_control
                                else str(config["packets"]["target"]["neutral_prompt"])
                            ),
                            "target_formatted_prompt_sha256": (
                                records[task_index]["target_prompt_sha256"]
                                if text_control
                                else artifact_provenance[
                                    "neutral_formatted_prompt_sha256"
                                ]
                            ),
                            "target_input_ids_sha256": tensor_sha256(
                                inputs["input_ids"]
                            ),
                            "target_attention_mask_sha256": tensor_sha256(
                                inputs["attention_mask"]
                            ),
                            "target_prompt_token_count": int(
                                inputs["input_ids"].shape[1]
                            ),
                            "packet_present": packet is not None,
                            "packet_kind": condition if packet is not None else None,
                            "packet_layer_indices": target_layers if packet is not None else [],
                            "packet_offsets": offsets if packet is not None else [],
                            "packet_sha256": packet_hash,
                            "packet_frobenius_norm": packet_norm,
                            "packet_layer_norms": packet_norms,
                            "packet_residual_layer_norms": residual_norms,
                            "source_task_id": source_task_id,
                            "donor_task_id": task_ids[donor_index]
                            if "shuffled" in condition
                            else None,
                            "random_residual_seed": noise_seed,
                            "random_reference_residual_layer_norms": (
                                random_audit["reference_residual_layer_norms"]
                                if random_audit is not None
                                else None
                            ),
                            "random_norm_match_maximum_absolute_delta": (
                                random_audit[
                                    "maximum_absolute_layer_norm_delta"
                                ]
                                if random_audit is not None
                                else None
                            ),
                            "random_norm_match_maximum_relative_delta": (
                                random_audit[
                                    "maximum_relative_layer_norm_delta"
                                ]
                                if random_audit is not None
                                else None
                            ),
                            "confirmation_manifest_sha256": confirmation_manifest_hash,
                            "confirmation_bundle_manifest_sha256": confirmation_bundle_hash,
                            "training_bundle_manifest_sha256": training_bundle_hash,
                            "matrix_summary_sha256": matrix_summary_hash,
                            "target_model_revision": target_spec["revision"],
                            "output_text": output_text,
                            "task_spec": task,
                        }
                        output_handle.write(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        output_handle.flush()
                        existing_keys.add(key)
                        new_records += 1
                        if new_records % 8 == 0:
                            write_json(
                                metadata_path,
                                _metadata_payload(
                                    config=config,
                                    config_path=config_path,
                                    output_path=output_path,
                                    task_ids=task_ids,
                                    donors=donors,
                                    expected_record_count=len(expected_keys),
                                    record_count=len(existing_keys),
                                    new_records=new_records,
                                    artifact_provenance=artifact_provenance,
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
            _metadata_payload(
                config=config,
                config_path=config_path,
                output_path=output_path,
                task_ids=task_ids,
                donors=donors,
                expected_record_count=len(expected_keys),
                record_count=len(existing_keys),
                new_records=new_records,
                artifact_provenance=artifact_provenance,
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

    complete = existing_keys == expected_keys
    metadata = _metadata_payload(
        config=config,
        config_path=config_path,
        output_path=output_path,
        task_ids=task_ids,
        donors=donors,
        expected_record_count=len(expected_keys),
        record_count=len(existing_keys),
        new_records=new_records,
        artifact_provenance=artifact_provenance,
        complete=complete,
    )
    write_json(metadata_path, metadata)
    return metadata
