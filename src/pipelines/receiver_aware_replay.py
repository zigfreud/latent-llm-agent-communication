"""Run the frozen H0-007 receiver-aware replay trajectory gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from src.core.packet_bundle import load_packet_records, sha256_file, validate_packet_bundle
from src.evaluation.packet_trajectory import (
    next_token_distribution_alignment,
    summarize_native_alignment,
    summarize_replay_discontinuity,
)
from src.evaluation.receiver_aware_replay import (
    RECEIVER_AWARE_REPLAY_PROTOCOL_VERSION,
    summarize_receiver_aware_gate,
)
from src.pipelines.infer import load_target, model_input_device
from src.pipelines.oracle_experiment import load_json_object, load_yaml
from src.pipelines.oracle_memory import forward_with_packet_trajectory_capture
from src.pipelines.packet_confirmation import (
    _neutral_inputs,
    _suffix_positions,
    load_variant_replica_specs,
    predict_variant_confirmation_packets,
)
from src.pipelines.packet_trajectory import (
    _atomic_json,
    _layer_packet,
    _select_specs,
    _validate_confirmation_artifact,
)


def _lf_sha256_file(path: Path) -> str:
    """Hash tracked text with Git's LF representation on every host."""

    normalized = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _validate_contract(
    experiment: Mapping,
    parent: Mapping,
    *,
    experiment_path: Path,
    parent_path: Path,
    predecessor_registry_path: Path,
) -> None:
    if experiment.get("experiment_id") != "LIP-H0-007":
        raise ValueError("unexpected receiver-aware experiment_id")
    if experiment.get("protocol_version") != RECEIVER_AWARE_REPLAY_PROTOCOL_VERSION:
        raise ValueError("unexpected receiver-aware protocol_version")
    if experiment.get("claim_status") != "exploratory_causal_operator_screen_only":
        raise ValueError("H0-007 must remain an exploratory operator screen")
    if parent.get("experiment_id") != "LIP-PROTO-014":
        raise ValueError("H0-007 parent must be LIP-PROTO-014")
    if experiment["parent"]["config_sha256"] != _lf_sha256_file(parent_path):
        raise ValueError("parent config hash differs from the frozen contract")
    predecessor = load_json_object(predecessor_registry_path)
    if predecessor.get("experiment_id") != "LIP-EVAL-031":
        raise ValueError("predecessor registry is not LIP-EVAL-031")
    expected_artifact = experiment["predecessor"]["full_artifact_sha256"]
    observed_artifact = predecessor.get("artifacts", {}).get("full_metrics", {}).get(
        "sha256"
    )
    if observed_artifact != expected_artifact:
        raise ValueError("EVAL-031 artifact hash differs from the frozen contract")

    layers = [int(value) for value in experiment["capture"]["target_layer_indices"]]
    offsets = [int(value) for value in experiment["capture"]["packet_offsets"]]
    if layers != [int(value) for value in parent["packets"]["target"]["layer_indices"]]:
        raise ValueError("target layers drifted from PROTO-014")
    if offsets != [int(value) for value in parent["packets"]["target"]["offsets"]]:
        raise ValueError("packet offsets drifted from PROTO-014")
    if list(experiment["capture"]["transition_layers"]) != layers[1:]:
        raise ValueError("transition layers must exclude only layer 0")
    task_indices = [int(value) for value in experiment["sample"]["task_indices"]]
    if len(task_indices) != int(experiment["sample"]["task_count"]):
        raise ValueError("task count differs from the frozen indices")
    if len(set(task_indices)) != len(task_indices):
        raise ValueError("task indices must be unique")

    variants = list(experiment["conditions"]["variants"])
    if variants != ["component_contrastive", "structured_linear_regression"]:
        raise ValueError("bridge variants differ from the frozen operator screen")
    seeds = [int(value) for value in experiment["conditions"]["training_seeds"]]
    if seeds != [int(value) for value in parent["training"]["seeds"]]:
        raise ValueError("training seeds drifted from PROTO-014")
    operators = experiment["conditions"]["operators"]
    if set(operators) != {"absolute_replace", "live_task_delta_add"}:
        raise ValueError("operator set differs from the frozen contract")
    if operators["absolute_replace"]["replay_mode"] != "replace":
        raise ValueError("absolute comparator must use replacement")
    if operators["live_task_delta_add"]["replay_mode"] != "add":
        raise ValueError("receiver-aware operator must use additive replay")
    gate = experiment["gate"]
    if gate["decision_variant"] != experiment["conditions"]["primary_variant"]:
        raise ValueError("gate decision variant differs from the primary variant")
    if experiment["compute"]["preferred_accelerator"] != "L4":
        raise ValueError("H0-007 is frozen for an L4")
    if experiment["compute"]["allow_silent_fallback"] is not False:
        raise ValueError("silent accelerator fallback is prohibited")
    if not experiment_path.is_file():
        raise ValueError("experiment contract path does not exist")


def _gpu_telemetry() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("LIP-H0-007 requires the registered L4 accelerator")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    telemetry = {
        "gpu_name": torch.cuda.get_device_name(device),
        "total_vram_bytes": int(properties.total_memory),
        "cuda_runtime": str(torch.version.cuda),
    }
    if "L4" not in telemetry["gpu_name"].upper():
        raise RuntimeError(
            f"registered accelerator is L4, observed {telemetry['gpu_name']!r}"
        )
    return telemetry


def _result_key(row: Mapping) -> tuple[str, str, int | None, str]:
    seed = row.get("training_seed")
    return (
        str(row["task_id"]),
        str(row["variant"]),
        None if seed is None else int(seed),
        str(row["operator"]),
    )


def _condition_packets(
    record: Mapping,
    *,
    row_index: int,
    scaffold: torch.Tensor,
    variant_predictions: Mapping[str, Mapping[int, torch.Tensor]],
) -> list[dict]:
    oracle = record["target_packet"].float()
    conditions = [
        {
            "condition": "oracle_absolute_replace",
            "variant": "oracle",
            "training_seed": None,
            "operator": "absolute_replace",
            "replay_mode": "replace",
            "packet": oracle,
        },
        {
            "condition": "oracle_live_task_delta_add",
            "variant": "oracle",
            "training_seed": None,
            "operator": "live_task_delta_add",
            "replay_mode": "add",
            "packet": oracle - scaffold,
        },
    ]
    for variant, predictions in variant_predictions.items():
        for seed, packets in predictions.items():
            absolute = packets[row_index]
            conditions.extend(
                (
                    {
                        "condition": f"{variant}_absolute_replace",
                        "variant": variant,
                        "training_seed": int(seed),
                        "operator": "absolute_replace",
                        "replay_mode": "replace",
                        "packet": absolute,
                    },
                    {
                        "condition": f"{variant}_live_task_delta_add",
                        "variant": variant,
                        "training_seed": int(seed),
                        "operator": "live_task_delta_add",
                        "replay_mode": "add",
                        "packet": absolute - scaffold,
                    },
                )
            )
    return conditions


def run_receiver_aware_trajectory_gate(
    *,
    experiment_path: Path,
    parent_path: Path,
    predecessor_registry_path: Path,
    matrix_summary_path: Path,
    training_bundle_dir: Path,
    confirmation_bundle_dir: Path,
    output_path: Path,
    target_device: str,
    bridge_device: str,
    colab_compute_units_before: float | None,
) -> dict:
    experiment = load_yaml(experiment_path)
    parent = load_yaml(parent_path)
    _validate_contract(
        experiment,
        parent,
        experiment_path=experiment_path,
        parent_path=parent_path,
        predecessor_registry_path=predecessor_registry_path,
    )
    telemetry = _gpu_telemetry()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    training_validation = validate_packet_bundle(training_bundle_dir, require_real=True)
    confirmation_validation = validate_packet_bundle(
        confirmation_bundle_dir, require_real=False
    )
    confirmation_manifest = load_json_object(confirmation_bundle_dir / "manifest.json")
    _validate_confirmation_artifact(
        confirmation_validation,
        confirmation_manifest,
        parent_config_sha256=sha256_file(parent_path),
        expected_task_count=int(parent["confirmation"]["task_count"]),
    )
    if training_validation["source_shape"] != confirmation_validation["source_shape"]:
        raise ValueError("source packet shape differs across bundle scopes")
    if training_validation["target_shape"] != confirmation_validation["target_shape"]:
        raise ValueError("target packet shape differs across bundle scopes")

    all_records = load_packet_records(confirmation_bundle_dir, split="confirmation")
    task_indices = [int(value) for value in experiment["sample"]["task_indices"]]
    if any(index < 0 or index >= len(all_records) for index in task_indices):
        raise ValueError("configured task index is outside the confirmation bundle")
    records = [all_records[index] for index in task_indices]
    variants = [str(value) for value in experiment["conditions"]["variants"]]
    seeds = [int(value) for value in experiment["conditions"]["training_seeds"]]

    shared_scaffold = None
    variant_predictions = {}
    variant_provenance = {}
    for variant in variants:
        matrix, specs, _ = load_variant_replica_specs(
            parent_path,
            matrix_summary_path,
            training_bundle_dir,
            variant_name=variant,
        )
        selected_specs = _select_specs(specs, seeds)
        predictions, scaffold, site_scale = predict_variant_confirmation_packets(
            parent,
            selected_specs,
            records,
            variant_name=variant,
            source_shape=training_validation["source_shape"],
            target_shape=training_validation["target_shape"],
            device=bridge_device,
            batch_size=len(records),
        )
        if shared_scaffold is None:
            shared_scaffold = scaffold
        elif not torch.equal(shared_scaffold, scaffold):
            raise ValueError("training scaffold differs across bridge variants")
        variant_predictions[variant] = predictions
        variant_provenance[variant] = {
            "checkpoints": {
                str(spec["seed"]): spec["checkpoint_sha256"] for spec in selected_specs
            },
            "target_statistics_sha256": selected_specs[0][
                "target_statistics_sha256"
            ],
            "development_gate": matrix["development_gates"][variant],
            "site_scale_minimum": float(site_scale.min().item()),
            "site_scale_maximum": float(site_scale.max().item()),
        }
    assert shared_scaffold is not None

    provenance = {
        "experiment_config_sha256": sha256_file(experiment_path),
        "parent_config_sha256": sha256_file(parent_path),
        "predecessor_registry_sha256": sha256_file(predecessor_registry_path),
        "matrix_summary_sha256": sha256_file(matrix_summary_path),
        "training_bundle_manifest_sha256": sha256_file(
            training_bundle_dir / "manifest.json"
        ),
        "confirmation_bundle_manifest_sha256": sha256_file(
            confirmation_bundle_dir / "manifest.json"
        ),
        "variant_provenance": variant_provenance,
    }
    payload = {
        "experiment_id": "LIP-H0-007",
        "protocol_version": RECEIVER_AWARE_REPLAY_PROTOCOL_VERSION,
        "claim_status": experiment["claim_status"],
        "stage": "trajectory_gate",
        "task_indices": task_indices,
        "task_ids": [str(record["task_id"]) for record in records],
        "training_seeds": seeds,
        "provenance": provenance,
        "telemetry": {
            **telemetry,
            "colab_compute_units_before": colab_compute_units_before,
            "colab_compute_units_after": None,
            "colab_compute_units_consumed": None,
        },
        "results": [],
        "gate": None,
        "complete": False,
    }
    if output_path.is_file():
        existing = load_json_object(output_path)
        binding_keys = (
            "experiment_id",
            "protocol_version",
            "stage",
            "task_indices",
            "task_ids",
            "training_seeds",
            "provenance",
        )
        failed = [key for key in binding_keys if existing.get(key) != payload.get(key)]
        if failed:
            raise ValueError("existing output provenance differs: " + ", ".join(failed))
        payload = existing
        if payload.get("complete") is True:
            return payload

    model_config = parent["models"]["target"]
    model, tokenizer = load_target(
        str(model_config["model_id"]),
        target_device,
        bool(experiment["compute"]["target_load_4bit"]),
        revision=str(model_config["revision"]),
    )
    target_input_device = model_input_device(model)
    _, neutral_inputs = _neutral_inputs(parent, tokenizer, target_input_device)
    layers = [int(value) for value in experiment["capture"]["target_layer_indices"]]
    offsets = [int(value) for value in experiment["capture"]["packet_offsets"]]
    positions = _suffix_positions(neutral_inputs, offsets)
    state_types = tuple(str(value) for value in experiment["capture"]["state_types"])
    completed_keys = {_result_key(row) for row in payload["results"]}

    for row_index, record in enumerate(records):
        conditions = _condition_packets(
            record,
            row_index=row_index,
            scaffold=shared_scaffold,
            variant_predictions=variant_predictions,
        )
        oracle_condition = conditions[0]
        oracle_packets = _layer_packet(oracle_condition["packet"], layers)
        oracle_output, oracle_states = forward_with_packet_trajectory_capture(
            model,
            neutral_inputs,
            layer_indices=layers,
            positions=positions,
            layer_packets=oracle_packets,
            replay_mode="replace",
        )
        oracle_logits = oracle_output.logits[:, -1:, :].detach().float().cpu()
        del oracle_output

        for condition in conditions:
            key = (
                str(record["task_id"]),
                str(condition["variant"]),
                condition["training_seed"],
                str(condition["operator"]),
            )
            if key in completed_keys:
                continue
            condition_started = time.perf_counter()
            packet = condition.pop("packet")
            replay_mode = str(condition.pop("replay_mode"))
            scheduled = _layer_packet(packet, layers)
            candidate_output, candidate_states = forward_with_packet_trajectory_capture(
                model,
                neutral_inputs,
                layer_indices=layers,
                positions=positions,
                layer_packets=scheduled,
                replay_mode=replay_mode,
            )
            candidate_logits = candidate_output.logits[:, -1:, :].detach().float().cpu()
            row = {
                "task_id": str(record["task_id"]),
                "task_index": task_indices[row_index],
                **condition,
                "intervention_jump": summarize_replay_discontinuity(
                    candidate_states["incoming_before_replay"],
                    candidate_states["residual_input"],
                    layer_indices=layers,
                ),
                "oracle_replay_alignment": summarize_native_alignment(
                    oracle_states,
                    candidate_states,
                    state_types=state_types,
                    layer_indices=layers,
                ),
                "next_token_alignment_to_oracle_replay": (
                    next_token_distribution_alignment(oracle_logits, candidate_logits)
                ),
                "wall_seconds": float(time.perf_counter() - condition_started),
            }
            payload["results"].append(row)
            completed_keys.add(key)
            payload["telemetry"].update(
                {
                    "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
                    "wall_seconds": float(time.perf_counter() - started),
                }
            )
            _atomic_json(output_path, payload)
            del candidate_output, candidate_logits, candidate_states, scheduled
            gc.collect()
        del oracle_logits, oracle_states, oracle_packets

    gate = experiment["gate"]
    payload["gate"] = summarize_receiver_aware_gate(
        payload["results"],
        variants=variants,
        seeds=seeds,
        primary_variant=str(experiment["conditions"]["primary_variant"]),
        minimum_taskwise_improvements=int(gate["minimum_taskwise_improvements"]),
        minimum_passing_replicas=int(gate["minimum_passing_replicas"]),
    )
    payload["complete"] = True
    total_seconds = float(time.perf_counter() - started)
    payload["telemetry"].update(
        {
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
            "wall_seconds": total_seconds,
            "task_conditions_per_hour": len(payload["results"])
            * 3600.0
            / max(total_seconds, 1e-12),
        }
    )
    _atomic_json(output_path, payload)
    return payload


def finalize_compute_units(results_path: Path, after: float) -> dict:
    payload = load_json_object(results_path)
    telemetry = payload.setdefault("telemetry", {})
    before = telemetry.get("colab_compute_units_before")
    if before is None:
        raise ValueError("compute-unit finalization requires a recorded before value")
    telemetry["colab_compute_units_after"] = float(after)
    telemetry["colab_compute_units_consumed"] = max(0.0, float(before) - float(after))
    _atomic_json(results_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("config/LIP-H0-007_receiver_aware_replay_operator.yaml"),
    )
    parser.add_argument(
        "--parent-config",
        type=Path,
        default=Path("config/LIP-PROTO-014_source_conditioned_residual_packet.yaml"),
    )
    parser.add_argument(
        "--predecessor-registry",
        type=Path,
        default=Path("experiments/registry/LIP-EVAL-031_receiver_trajectory_coherence.json"),
    )
    parser.add_argument("--matrix-summary", type=Path)
    parser.add_argument("--training-bundle", type=Path)
    parser.add_argument("--confirmation-bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-device", default="auto")
    parser.add_argument("--bridge-device", default="cpu")
    parser.add_argument("--colab-compute-units-before", type=float)
    parser.add_argument("--finalize-compute-units-after", type=float)
    parser.add_argument("--dry-run-contract", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment = load_yaml(args.experiment_config)
    parent = load_yaml(args.parent_config)
    _validate_contract(
        experiment,
        parent,
        experiment_path=args.experiment_config,
        parent_path=args.parent_config,
        predecessor_registry_path=args.predecessor_registry,
    )
    if args.dry_run_contract:
        print("LIP-H0-007 contract validated")
        return
    if args.finalize_compute_units_after is not None:
        if args.output is None:
            raise ValueError("--output is required to finalize compute units")
        finalize_compute_units(args.output, args.finalize_compute_units_after)
        print(args.output)
        return
    required = {
        "matrix_summary": args.matrix_summary,
        "training_bundle": args.training_bundle,
        "confirmation_bundle": args.confirmation_bundle,
        "output": args.output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("missing required paths: " + ", ".join(missing))
    result = run_receiver_aware_trajectory_gate(
        experiment_path=args.experiment_config,
        parent_path=args.parent_config,
        predecessor_registry_path=args.predecessor_registry,
        matrix_summary_path=args.matrix_summary,
        training_bundle_dir=args.training_bundle,
        confirmation_bundle_dir=args.confirmation_bundle,
        output_path=args.output,
        target_device=args.target_device,
        bridge_device=args.bridge_device,
        colab_compute_units_before=args.colab_compute_units_before,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "complete": result["complete"],
                "advance_to_functional_identity_test": result["gate"][
                    "advance_to_functional_identity_test"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
