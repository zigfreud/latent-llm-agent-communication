"""Generate the bounded development-only LIP-EVAL-033 functional grid."""

from __future__ import annotations

import gc
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from src.core.packet_bridge import reconstruct_target_packet
from src.core.packet_bundle import load_packet_records, sha256_file, validate_packet_bundle
from src.core.prompt_protocol import protocol_pair_metadata
from src.core.utils import set_seed
from src.evaluation.functional_bridge_screen import (
    FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS,
    FUNCTIONAL_BRIDGE_SCREEN_EXPERIMENT_ID,
    FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS,
    FUNCTIONAL_BRIDGE_SCREEN_PROTOCOL_VERSION,
    FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS,
    expected_functional_bridge_screen_keys,
    functional_bridge_screen_design_fingerprint,
    validate_functional_bridge_screen_contract,
)
from src.evaluation.oracle_functional import stable_seed
from src.evaluation.oracle_terminal_factorial import validate_terminal_layout
from src.evaluation.packet_bridge_confirmation import packet_layer_norms
from src.pipelines.infer import load_target, model_input_device
from src.pipelines.oracle_experiment import (
    generation_kwargs,
    load_json_object,
    load_yaml,
    prompt_sha256,
    write_json,
)
from src.pipelines.oracle_memory import generate_with_layer_input_replay
from src.pipelines.packet_bridge import build_packet_bridge, resolve_packet_device
from src.pipelines.packet_confirmation import (
    _neutral_inputs,
    _suffix_positions,
    tensor_sha256,
)
from src.pipelines.receiver_aware_replay import _lf_sha256_file


def _repo_path(config_path: Path, relative: str) -> Path:
    return config_path.resolve().parents[1] / relative


def _read_jsonl(path: Path) -> list[dict]:
    rows = []

    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON constant {value}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line, parse_constant=reject_constant)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            rows.append(row)
    return rows


def validate_functional_bridge_screen_runtime_contract(
    config: Mapping, *, config_path: Path
) -> tuple[dict, dict, dict]:
    validate_functional_bridge_screen_contract(config)
    predecessor_path = _repo_path(config_path, config["predecessor"]["registry"])
    if config["predecessor"]["registry_sha256"] != _lf_sha256_file(
        predecessor_path
    ):
        raise ValueError("LIP-H0-016 registry differs from the frozen screen")
    predecessor = load_json_object(predecessor_path)
    if predecessor.get("experiment_id") != "LIP-H0-016" or predecessor.get(
        "decision", {}
    ).get(config["predecessor"]["required_decision"]) is not True:
        raise ValueError("LIP-H0-016 did not authorize the EVAL-033 design")
    if predecessor.get("decision", {}).get("LIP_EVAL_033_claim_scope") != (
        "development_only_reused_open_P014_functional_cohort"
    ):
        raise ValueError("LIP-H0-016 claim boundary changed")

    h015_path = _repo_path(config_path, config["references"]["H0_015_registry"])
    if config["references"]["H0_015_registry_sha256"] != _lf_sha256_file(
        h015_path
    ):
        raise ValueError("LIP-H0-015 registry differs from the frozen screen")
    h015 = load_json_object(h015_path)

    p014_path = _repo_path(config_path, config["cohort"]["config"])
    if config["cohort"]["config_sha256"] != _lf_sha256_file(p014_path):
        raise ValueError("P014 config differs from the frozen screen")
    p014 = load_yaml(p014_path)
    if p014.get("experiment_id") != "LIP-PROTO-014":
        raise ValueError("EVAL-033 cohort must be P014")
    if dict(config["models"]["source"]) != {
        "model_id": p014["models"]["source"]["model_id"],
        "revision": p014["models"]["source"]["revision"],
    }:
        raise ValueError("source model drifted from P014")
    if (
        config["models"]["target"]["model_id"]
        != p014["models"]["target"]["model_id"]
        or config["models"]["target"]["revision"]
        != p014["models"]["target"]["revision"]
    ):
        raise ValueError("target model drifted from P014")
    if dict(config["prompt_protocols"]) != dict(p014["prompt_protocols"]):
        raise ValueError("prompt protocol drifted from P014")

    checkpoints = config["systems"]["checkpoints"]
    h016_artifacts = predecessor["artifacts"]
    for seed in (4001, 4003):
        frozen = h016_artifacts[f"seed_{seed}"]
        current = checkpoints[seed]
        if (
            current["run_summary_sha256"] != frozen["run_summary"]["sha256"]
            or current["checkpoint_sha256"] != frozen["best_checkpoint"]["sha256"]
            or int(current["best_step"])
            != int(predecessor["cells"][str(seed)]["best_step"])
        ):
            raise ValueError(f"seed {seed} artifact drifted from H0-016")
    seed4007 = checkpoints[4007]
    if (
        seed4007["run_summary_sha256"]
        != h015["artifacts"]["screen"]["run_summary"]["sha256"]
        or seed4007["checkpoint_sha256"]
        != h015["artifacts"]["screen"]["best_checkpoint"]["sha256"]
        or int(seed4007["best_step"]) != int(h015["screen"]["best_step"])
    ):
        raise ValueError("seed 4007 artifact drifted from H0-015")
    return predecessor, h015, p014


def _load_p014_cohort(
    config: Mapping, *, artifact_root: Path, confirmation_bundle_dir: Path
) -> tuple[list[dict], dict, list[dict], dict]:
    source = config["cohort"]["source_artifacts"]
    generations_path = artifact_root / source["generations"]
    metadata_path = artifact_root / source["metadata"]
    functional_summary_path = artifact_root / source["functional_summary"]
    for path, expected in (
        (generations_path, source["generations_sha256"]),
        (metadata_path, source["metadata_sha256"]),
        (functional_summary_path, source["functional_summary_sha256"]),
        (
            confirmation_bundle_dir / "manifest.json",
            source["confirmation_bundle_manifest_sha256"],
        ),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen P014 artifact differs: {path}")
    metadata = load_json_object(metadata_path)
    if (
        metadata.get("experiment_id") != "LIP-PROTO-014"
        or metadata.get("complete") is not True
        or metadata.get("claim_eligible") is not True
        or metadata.get("task_count") != int(config["cohort"]["task_count"])
        or metadata.get("generation_seeds")
        != list(FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS)
    ):
        raise ValueError("P014 generation metadata is not its complete frozen grid")
    task_ids = [str(value) for value in metadata["task_ids"]]
    donor_task_ids = {
        str(target): str(donor)
        for target, donor in metadata["donor_task_ids"].items()
    }
    if set(task_ids) != set(donor_task_ids) or any(
        target == donor or donor not in task_ids
        for target, donor in donor_task_ids.items()
    ):
        raise ValueError("P014 donor map is invalid")

    rows = _read_jsonl(generations_path)
    tasks_by_id = {}
    for row in rows:
        task = row.get("task_spec")
        if not isinstance(task, Mapping):
            raise ValueError("P014 generation lacks task_spec")
        task_id = str(row.get("task_id", ""))
        if str(task.get("task_id", "")) != task_id:
            raise ValueError("P014 task identity changed")
        previous = tasks_by_id.setdefault(task_id, dict(task))
        if previous != dict(task):
            raise ValueError("P014 task specification varies across rows")
    if set(tasks_by_id) != set(task_ids):
        raise ValueError("P014 generation task set differs from metadata")
    tasks = [tasks_by_id[task_id] for task_id in task_ids]
    strata = [validate_terminal_layout(task["terminal_layout"]) for task in tasks]
    if strata.count(2) != 16 or strata.count(3) != 16:
        raise ValueError("P014 task strata drifted")
    by_id = {task_id: index for index, task_id in enumerate(task_ids)}
    for target, donor in donor_task_ids.items():
        if strata[by_id[target]] != strata[by_id[donor]]:
            raise ValueError("P014 donor crosses tokenizer strata")

    validation = validate_packet_bundle(confirmation_bundle_dir, require_real=True)
    if (
        validation["extraction_scope"] != "confirmation"
        or validation["split_counts"]["confirmation"] != len(task_ids)
        or tuple(validation["source_shape"]) != tuple(config["packets"]["source_shape"])
        or tuple(validation["target_shape"])
        != tuple(config["packets"]["receiver_target_shape"])
    ):
        raise ValueError("P014 confirmation bundle shape or scope drifted")
    packet_records = load_packet_records(confirmation_bundle_dir)
    if [str(row["task_id"]) for row in packet_records] != task_ids:
        raise ValueError("P014 confirmation bundle task order drifted")
    return tasks, metadata, packet_records, validation


def _load_replica_specs(config: Mapping, *, artifact_root: Path) -> list[dict]:
    specs = []
    statistics_hash = config["systems"]["target_statistics_sha256"]
    for seed in FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS:
        frozen = config["systems"]["checkpoints"][seed]
        directory = artifact_root / frozen["directory"]
        summary_path = directory / "run_summary.json"
        checkpoint_path = directory / "best_checkpoint.pt"
        statistics_path = directory / "target_statistics.pt"
        for path, expected in (
            (summary_path, frozen["run_summary_sha256"]),
            (checkpoint_path, frozen["checkpoint_sha256"]),
            (statistics_path, statistics_hash),
        ):
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"seed {seed} runtime artifact differs: {path}")
        summary = load_json_object(summary_path)
        if (
            int(summary.get("seed", -1)) != seed
            or int(summary.get("training", {}).get("best_step", -1))
            != int(frozen["best_step"])
            or summary.get("provenance", {}).get("target_statistics_sha256")
            != statistics_hash
        ):
            raise ValueError(f"seed {seed} run summary drifted")
        specs.append(
            {
                "seed": seed,
                "best_step": int(frozen["best_step"]),
                "strong_identity_gate": bool(frozen["strong_identity_gate"]),
                "summary": str(summary_path),
                "summary_sha256": frozen["run_summary_sha256"],
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": frozen["checkpoint_sha256"],
                "target_statistics": str(statistics_path),
                "target_statistics_sha256": statistics_hash,
            }
        )
    return specs


def _predict_entry_packets(
    config: Mapping,
    specs: Sequence[Mapping],
    records: Sequence[Mapping],
    *,
    device: str,
    batch_size: int,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    prediction_device = resolve_packet_device(device)
    source = torch.stack([record["source_packet"].float() for record in records])
    expected_model = dict(config["bridge"])
    source_shape = tuple(config["packets"]["source_shape"])
    bridge_target_shape = tuple(config["packets"]["bridge_target_shape"])
    receiver_target_shape = tuple(config["packets"]["receiver_target_shape"])
    predictions = {}
    shared_scaffold = None
    shared_scale = None
    for spec in specs:
        statistics = torch.load(
            Path(spec["target_statistics"]), map_location="cpu", weights_only=True
        )
        scaffold = statistics["scaffold"].float().cpu()
        scale = statistics["site_scale"].float().cpu()
        if tuple(scaffold.shape) != receiver_target_shape or tuple(scale.shape) != (
            receiver_target_shape[0],
            receiver_target_shape[1],
        ):
            raise ValueError("target statistics shape drifted")
        if shared_scaffold is None:
            shared_scaffold, shared_scale = scaffold, scale
        elif not torch.equal(shared_scaffold, scaffold) or not torch.equal(
            shared_scale, scale
        ):
            raise ValueError("target statistics differ across bridge seeds")
        checkpoint = torch.load(
            Path(spec["checkpoint"]), map_location="cpu", weights_only=True
        )
        if (
            checkpoint.get("model_config") != expected_model
            or checkpoint.get("source_shape") != list(source_shape)
            or checkpoint.get("bridge_target_shape") != list(bridge_target_shape)
            or checkpoint.get("receiver_target_shape") != list(receiver_target_shape)
            or int(checkpoint.get("step", -1)) != int(spec["best_step"])
            or checkpoint.get("variant") != "hard_negative_batches_unrolled"
        ):
            raise ValueError(f"seed {spec['seed']} checkpoint contract drifted")
        model = build_packet_bridge(expected_model, source_shape, bridge_target_shape)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(prediction_device).eval()
        normalized_rows = []
        with torch.inference_mode():
            for start in range(0, len(source), batch_size):
                normalized_rows.append(
                    model(source[start : start + batch_size].to(prediction_device))
                    .float()
                    .cpu()
                )
        normalized = torch.cat(normalized_rows)
        raw = reconstruct_target_packet(
            normalized, shared_scaffold[:1], shared_scale[:1]
        ).cpu()
        if tuple(raw.shape[1:]) != bridge_target_shape or not torch.isfinite(raw).all():
            raise ValueError("predicted entry packet is invalid")
        predictions[int(spec["seed"])] = raw
        del checkpoint, model, normalized, raw
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert shared_scaffold is not None and shared_scale is not None
    return predictions, shared_scaffold[:1], shared_scale[:1]


def _key(row: Mapping) -> tuple[str, str, int, int]:
    return (
        str(row["task_id"]),
        str(row["condition"]),
        int(row["generation_seed"]),
        int(row["training_seed"]),
    )


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
        "experiment_id": FUNCTIONAL_BRIDGE_SCREEN_EXPERIMENT_ID,
        "protocol_version": FUNCTIONAL_BRIDGE_SCREEN_PROTOCOL_VERSION,
        "design_sha256": functional_bridge_screen_design_fingerprint(config),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "generations_jsonl": str(output_path),
        "run_scope": "development_only_reused_open_P014_cohort",
        "claim_eligible": False,
        "task_ids": list(task_ids),
        "task_count": len(task_ids),
        "donor_task_ids": dict(donor_task_ids),
        "conditions": list(FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS),
        "generation_seeds": list(FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS),
        "training_seeds": list(FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS),
        "expected_records": len(expected_functional_bridge_screen_keys(task_ids)),
        "records": record_count,
        "new_records": new_records,
        "complete": complete,
        **dict(provenance),
    }
    if last_error is not None:
        payload["last_error"] = last_error
    return payload


def run_functional_bridge_screen(
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
    validate_functional_bridge_screen_runtime_contract(config, config_path=config_path)
    if max_new_records is not None and max_new_records <= 0:
        raise ValueError("max_new_records must be positive")
    source = config["cohort"]["source_artifacts"]
    confirmation_bundle_dir = artifact_root / source["confirmation_bundle"]
    tasks, p014_metadata, packet_records, bundle_validation = _load_p014_cohort(
        config,
        artifact_root=artifact_root,
        confirmation_bundle_dir=confirmation_bundle_dir,
    )
    task_ids = [str(task["task_id"]) for task in tasks]
    donor_task_ids = {
        str(target): str(donor)
        for target, donor in p014_metadata["donor_task_ids"].items()
    }
    specs = _load_replica_specs(config, artifact_root=artifact_root)
    expected = expected_functional_bridge_screen_keys(task_ids)
    design_sha = functional_bridge_screen_design_fingerprint(config)
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
                raise ValueError("resumed EVAL-033 grid contains duplicate/unexpected row")
            if (
                row.get("design_sha256") != design_sha
                or row.get("config_sha256") != config_sha
                or row.get("run_scope")
                != "development_only_reused_open_P014_cohort"
                or row.get("claim_eligible") is not False
            ):
                raise ValueError("resumed EVAL-033 row differs from frozen design")
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
        config,
        specs,
        packet_records,
        device=device,
        batch_size=prediction_batch_size,
    )
    p014_generations_path = artifact_root / source["generations"]
    p014_metadata_path = artifact_root / source["metadata"]
    p014_summary_path = artifact_root / source["functional_summary"]
    provenance = {
        "P014_generations": str(p014_generations_path),
        "P014_generations_sha256": source["generations_sha256"],
        "P014_metadata": str(p014_metadata_path),
        "P014_metadata_sha256": source["metadata_sha256"],
        "P014_functional_summary": str(p014_summary_path),
        "P014_functional_summary_sha256": source["functional_summary_sha256"],
        "confirmation_bundle": str(confirmation_bundle_dir),
        "confirmation_bundle_manifest_sha256": source[
            "confirmation_bundle_manifest_sha256"
        ],
        "confirmation_bundle_validation": bundle_validation,
        "primary_replicas": specs,
        "training_scaffold_sha256": tensor_sha256(scaffold),
        "training_site_scale_sha256": tensor_sha256(site_scale),
        "source_model_revision": config["models"]["source"]["revision"],
        "target_model_revision": config["models"]["target"]["revision"],
        "prompt_protocols": protocol_pair_metadata(config["prompt_protocols"]),
    }
    target = config["models"]["target"]
    model, tokenizer = load_target(
        target["model_id"],
        device,
        bool(target["load_4bit"]),
        revision=target["revision"],
    )
    target_device = model_input_device(model)
    neutral_formatted, neutral_inputs = _neutral_inputs(
        config, tokenizer, target_device
    )
    offsets = [int(value) for value in config["packets"]["target"]["offsets"]]
    positions = _suffix_positions(neutral_inputs, offsets)
    generation_config = dict(config["generation"])
    gen_kwargs = generation_kwargs(generation_config, tokenizer)
    provenance.update(
        {
            "neutral_prompt": config["packets"]["target"]["neutral_prompt"],
            "neutral_formatted_prompt_sha256": prompt_sha256(neutral_formatted),
            "neutral_input_ids_sha256": tensor_sha256(neutral_inputs["input_ids"]),
            "neutral_attention_mask_sha256": tensor_sha256(
                neutral_inputs["attention_mask"]
            ),
            "neutral_token_count": int(neutral_inputs["input_ids"].shape[1]),
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
            for task_index, task in enumerate(tasks):
                if stop:
                    break
                task_id = task_ids[task_index]
                donor_id = donor_task_ids[task_id]
                donor_index = by_id[donor_id]
                for generation_seed in FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS:
                    for training_seed in FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS:
                        for condition in FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS:
                            key = (
                                task_id,
                                condition,
                                generation_seed,
                                training_seed,
                            )
                            if key in existing:
                                continue
                            source_index = (
                                task_index
                                if condition == "learned_matched"
                                else donor_index
                            )
                            packet = predictions[training_seed][source_index]
                            packet_hash = tensor_sha256(packet)
                            packet_norm = float(
                                torch.linalg.vector_norm(packet.flatten()).item()
                            )
                            if not math.isfinite(packet_norm):
                                raise FloatingPointError("entry packet norm is non-finite")
                            effective_seed = stable_seed(
                                generation_seed, task_index, 14014
                            )
                            set_seed(effective_seed)
                            output_text = generate_with_layer_input_replay(
                                model,
                                tokenizer,
                                neutral_inputs,
                                generation_kwargs=gen_kwargs,
                                positions=positions,
                                layer_packets={0: packet[0]},
                            )
                            record = {
                                "experiment_id": FUNCTIONAL_BRIDGE_SCREEN_EXPERIMENT_ID,
                                "protocol_version": FUNCTIONAL_BRIDGE_SCREEN_PROTOCOL_VERSION,
                                "design_sha256": design_sha,
                                "config_sha256": config_sha,
                                "run_scope": "development_only_reused_open_P014_cohort",
                                "claim_eligible": False,
                                "task_id": task_id,
                                "condition": condition,
                                "generation_seed": generation_seed,
                                "effective_generation_seed": effective_seed,
                                "training_seed": training_seed,
                                "target_prompt_kind": "neutral",
                                "target_formatted_prompt_sha256": provenance[
                                    "neutral_formatted_prompt_sha256"
                                ],
                                "target_input_ids_sha256": provenance[
                                    "neutral_input_ids_sha256"
                                ],
                                "target_attention_mask_sha256": provenance[
                                    "neutral_attention_mask_sha256"
                                ],
                                "target_prompt_token_count": provenance[
                                    "neutral_token_count"
                                ],
                                "packet_present": True,
                                "packet_kind": condition,
                                "packet_layer_indices": [0],
                                "packet_offsets": offsets,
                                "packet_sha256": packet_hash,
                                "packet_frobenius_norm": packet_norm,
                                "packet_layer_norms": packet_layer_norms(packet),
                                "packet_residual_layer_norms": packet_layer_norms(
                                    packet - scaffold
                                ),
                                "source_task_id": task_ids[source_index],
                                "donor_task_id": donor_id
                                if condition == "learned_shuffled"
                                else None,
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
