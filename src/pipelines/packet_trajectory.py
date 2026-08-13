"""Exploratory receiver-trajectory diagnostics for learned packet replay."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from src.core.packet_bundle import (
    load_packet_records,
    sha256_file,
    validate_packet_bundle,
)
from src.evaluation.packet_trajectory import (
    PACKET_TRAJECTORY_PROTOCOL_VERSION,
    next_token_distribution_alignment,
    summarize_native_alignment,
    summarize_replay_discontinuity,
)
from src.pipelines.infer import load_target, model_input_device
from src.pipelines.oracle_experiment import load_json_object, load_yaml
from src.pipelines.oracle_memory import forward_with_packet_trajectory_capture
from src.pipelines.packet_confirmation import (
    _neutral_inputs,
    _suffix_positions,
    _target_inputs_from_record,
    load_variant_replica_specs,
    predict_variant_confirmation_packets,
)


EVALUATED_STATE_TYPES = (
    "residual_input",
    "query_pre_rope",
    "key_pre_rope",
    "value_pre_cache",
    "attention_output",
    "residual_output",
)


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _validate_contract(eval_config: Mapping, parent_config: Mapping) -> None:
    if eval_config.get("experiment_id") != "LIP-EVAL-031":
        raise ValueError("unexpected trajectory evaluation experiment_id")
    if eval_config.get("protocol_version") != PACKET_TRAJECTORY_PROTOCOL_VERSION:
        raise ValueError("unexpected trajectory evaluation protocol_version")
    if eval_config.get("claim_status") != "exploratory_hypothesis_generation_only":
        raise ValueError("trajectory evaluation must remain exploratory")
    if eval_config.get("predecessor", {}).get("protocol") != parent_config.get(
        "experiment_id"
    ):
        raise ValueError("trajectory predecessor differs from the parent config")
    configured_layers = [
        int(value) for value in eval_config["capture"]["target_layer_indices"]
    ]
    parent_layers = [
        int(value) for value in parent_config["packets"]["target"]["layer_indices"]
    ]
    configured_offsets = [int(value) for value in eval_config["capture"]["packet_offsets"]]
    parent_offsets = [
        int(value) for value in parent_config["packets"]["target"]["offsets"]
    ]
    if configured_layers != parent_layers or configured_offsets != parent_offsets:
        raise ValueError("trajectory capture sites drifted from PROTO-014")
    if list(eval_config["metrics"]["transition_layers"]) != configured_layers[1:]:
        raise ValueError("transition layers must exclude only the entry layer")
    if tuple(eval_config["capture"]["state_types"])[1:] != EVALUATED_STATE_TYPES:
        raise ValueError("registered receiver-state capture contract changed")
    if eval_config["compute"].get("preferred_accelerator") != "L4":
        raise ValueError("EVAL-031 is registered for an L4 pilot")
    if eval_config["compute"].get("allow_silent_fallback") is not False:
        raise ValueError("silent accelerator fallback is prohibited")


def _select_specs(specs: Sequence[Mapping], seeds: Sequence[int]) -> list[dict]:
    by_seed = {int(spec["seed"]): dict(spec) for spec in specs}
    selected = []
    for seed in seeds:
        if int(seed) not in by_seed:
            raise ValueError(f"registered checkpoint is missing seed {seed}")
        selected.append(by_seed[int(seed)])
    return selected


def _layer_packet(packet: torch.Tensor, layers: Sequence[int]) -> dict[int, torch.Tensor]:
    value = packet.detach().float().cpu()
    if value.ndim != 3 or value.shape[0] != len(layers):
        raise ValueError("receiver packet shape differs from configured layers")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("receiver packet contains non-finite values")
    return {int(layer): value[index] for index, layer in enumerate(layers)}


def _gpu_telemetry() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("EVAL-031 requires the registered L4 accelerator")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "total_vram_bytes": int(properties.total_memory),
        "cuda_runtime": str(torch.version.cuda),
    }


def _validate_confirmation_artifact(
    validation: Mapping,
    manifest: Mapping,
    *,
    parent_config_sha256: str,
    expected_task_count: int,
) -> None:
    checks = {
        "real_extraction": validation.get("extraction_mode") == "real",
        "confirmation_scope": validation.get("extraction_scope") == "confirmation",
        "parent_config": manifest.get("config_sha256") == parent_config_sha256,
        "confirmation_count": validation.get("split_counts", {}).get("confirmation")
        == int(expected_task_count),
        "no_training_rows": all(
            validation.get("split_counts", {}).get(split, 0) == 0
            for split in ("train", "development_selection", "development_gate")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "confirmation artifact differs from the frozen contract: "
            + ", ".join(failed)
        )


def _condition_packets(
    record: Mapping,
    *,
    row_index: int,
    scaffold: torch.Tensor,
    variant_predictions: Mapping[str, Mapping[int, torch.Tensor]],
) -> list[dict]:
    conditions = [
        {
            "condition": "oracle_teacher_matched",
            "variant": "oracle",
            "training_seed": None,
            "packet": record["target_packet"].float(),
        },
        {
            "condition": "mean_scaffold",
            "variant": "mean_scaffold",
            "training_seed": None,
            "packet": scaffold,
        },
    ]
    for variant, predictions in variant_predictions.items():
        for seed, packets in predictions.items():
            conditions.append(
                {
                    "condition": variant,
                    "variant": variant,
                    "training_seed": int(seed),
                    "packet": packets[row_index],
                }
            )
    return conditions


def _result_key(row: Mapping) -> tuple[str, str, int | None]:
    seed = row.get("training_seed")
    return (
        str(row["task_id"]),
        str(row["condition"]),
        None if seed is None else int(seed),
    )


def finalize_compute_units(results_path: Path, after: float) -> dict:
    payload = load_json_object(results_path)
    telemetry = payload.setdefault("telemetry", {})
    if telemetry.get("colab_compute_units_before") is None:
        raise ValueError("compute-unit finalization requires a recorded before value")
    telemetry["colab_compute_units_after"] = float(after)
    telemetry["colab_compute_units_consumed"] = max(
        0.0, float(telemetry["colab_compute_units_before"]) - float(after)
    )
    _atomic_json(results_path, payload)
    return payload


def run_packet_trajectory_evaluation(
    *,
    eval_config_path: Path,
    parent_config_path: Path,
    matrix_summary_path: Path,
    training_bundle_dir: Path,
    confirmation_bundle_dir: Path,
    output_path: Path,
    stage: str,
    target_device: str,
    bridge_device: str,
    colab_compute_units_before: float | None,
) -> dict:
    eval_config = load_yaml(eval_config_path)
    parent_config = load_yaml(parent_config_path)
    _validate_contract(eval_config, parent_config)
    if stage not in {"pilot", "full"}:
        raise ValueError("stage must be pilot or full")

    telemetry = _gpu_telemetry()
    if "L4" not in telemetry["gpu_name"].upper():
        raise RuntimeError(
            f"registered accelerator is L4, observed {telemetry['gpu_name']!r}"
        )
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
        parent_config_sha256=sha256_file(parent_config_path),
        expected_task_count=int(parent_config["confirmation"]["task_count"]),
    )
    if training_validation["source_shape"] != confirmation_validation["source_shape"]:
        raise ValueError("source packet shape differs across training/confirmation")
    if training_validation["target_shape"] != confirmation_validation["target_shape"]:
        raise ValueError("target packet shape differs across training/confirmation")

    all_records = load_packet_records(confirmation_bundle_dir, split="confirmation")
    if stage == "pilot":
        task_indices = [int(value) for value in eval_config["sample"]["pilot_task_indices"]]
        seeds = [int(value) for value in eval_config["conditions"]["pilot_training_seeds"]]
    else:
        task_indices = list(range(int(eval_config["sample"]["full_task_count"])))
        seeds = [int(value) for value in eval_config["conditions"]["full_training_seeds"]]
    if any(index < 0 or index >= len(all_records) for index in task_indices):
        raise ValueError("configured task index is outside the confirmation bundle")
    records = [all_records[index] for index in task_indices]

    variants = ("component_contrastive", "structured_linear_regression")
    variant_predictions = {}
    variant_provenance = {}
    shared_scaffold = None
    for variant in variants:
        matrix, specs, _ = load_variant_replica_specs(
            parent_config_path,
            matrix_summary_path,
            training_bundle_dir,
            variant_name=variant,
        )
        selected_specs = _select_specs(specs, seeds)
        predictions, scaffold, site_scale = predict_variant_confirmation_packets(
            parent_config,
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
            "seeds": seeds,
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
        "eval_config_sha256": sha256_file(eval_config_path),
        "parent_config_sha256": sha256_file(parent_config_path),
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
        "experiment_id": "LIP-EVAL-031",
        "protocol_version": PACKET_TRAJECTORY_PROTOCOL_VERSION,
        "claim_status": "exploratory_hypothesis_generation_only",
        "stage": stage,
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
        "complete": False,
    }
    if output_path.is_file():
        existing = load_json_object(output_path)
        binding = {
            key: existing.get(key) == payload.get(key)
            for key in (
                "experiment_id",
                "protocol_version",
                "stage",
                "task_indices",
                "task_ids",
                "training_seeds",
                "provenance",
            )
        }
        if not all(binding.values()):
            failed = [key for key, passed in binding.items() if not passed]
            raise ValueError("existing output provenance differs: " + ", ".join(failed))
        payload = existing
        if payload.get("complete") is True:
            return payload

    model_config = parent_config["models"]["target"]
    model, tokenizer = load_target(
        str(model_config["model_id"]),
        target_device,
        bool(eval_config["compute"]["target_load_4bit"]),
        revision=str(model_config["revision"]),
    )
    target_input_device = model_input_device(model)
    _, neutral_inputs = _neutral_inputs(parent_config, tokenizer, target_input_device)
    layers = [int(value) for value in eval_config["capture"]["target_layer_indices"]]
    offsets = [int(value) for value in eval_config["capture"]["packet_offsets"]]
    neutral_positions = _suffix_positions(neutral_inputs, offsets)
    completed_keys = {_result_key(row) for row in payload["results"]}

    for row_index, record in enumerate(records):
        task_inputs = _target_inputs_from_record(record, target_input_device)
        task_positions = _suffix_positions(task_inputs, offsets)
        native_output, native_states = forward_with_packet_trajectory_capture(
            model,
            task_inputs,
            layer_indices=layers,
            positions=task_positions,
        )
        native_logits = native_output.logits[:, -1:, :].detach().float().cpu()
        del native_output

        conditions = _condition_packets(
            record,
            row_index=row_index,
            scaffold=shared_scaffold,
            variant_predictions=variant_predictions,
        )
        for condition in conditions:
            key = (
                str(record["task_id"]),
                str(condition["condition"]),
                condition["training_seed"],
            )
            if key in completed_keys:
                continue
            condition_started = time.perf_counter()
            scheduled = _layer_packet(condition.pop("packet"), layers)
            candidate_output, candidate_states = forward_with_packet_trajectory_capture(
                model,
                neutral_inputs,
                layer_indices=layers,
                positions=neutral_positions,
                layer_packets=scheduled,
            )
            candidate_logits = (
                candidate_output.logits[:, -1:, :].detach().float().cpu()
            )
            row = {
                "task_id": str(record["task_id"]),
                "task_index": task_indices[row_index],
                **condition,
                "replay_discontinuity": summarize_replay_discontinuity(
                    candidate_states["incoming_before_replay"],
                    scheduled,
                    layer_indices=layers,
                ),
                "native_alignment": summarize_native_alignment(
                    native_states,
                    candidate_states,
                    state_types=EVALUATED_STATE_TYPES,
                    layer_indices=layers,
                ),
                "next_token_alignment": next_token_distribution_alignment(
                    native_logits,
                    candidate_logits,
                ),
                "wall_seconds": float(time.perf_counter() - condition_started),
            }
            payload["results"].append(row)
            completed_keys.add(key)
            payload["telemetry"].update(
                {
                    "peak_allocated_vram_bytes": int(
                        torch.cuda.max_memory_allocated()
                    ),
                    "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
                    "wall_seconds": float(time.perf_counter() - started),
                }
            )
            _atomic_json(output_path, payload)
            del candidate_output, candidate_logits, candidate_states, scheduled
            gc.collect()
        del native_logits, native_states, task_inputs

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=Path("config/LIP-EVAL-031_receiver_trajectory_coherence.yaml"),
    )
    parser.add_argument(
        "--parent-config",
        type=Path,
        default=Path("config/LIP-PROTO-014_source_conditioned_residual_packet.yaml"),
    )
    parser.add_argument("--matrix-summary", type=Path)
    parser.add_argument("--training-bundle", type=Path)
    parser.add_argument("--confirmation-bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--target-device", default="auto")
    parser.add_argument("--bridge-device", default="cpu")
    parser.add_argument("--colab-compute-units-before", type=float)
    parser.add_argument("--finalize-compute-units-after", type=float)
    parser.add_argument("--dry-run-contract", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.finalize_compute_units_after is not None:
        if args.output is None:
            raise ValueError("--output is required to finalize compute units")
        finalize_compute_units(args.output, args.finalize_compute_units_after)
        print(args.output)
        return
    eval_config = load_yaml(args.eval_config)
    parent_config = load_yaml(args.parent_config)
    _validate_contract(eval_config, parent_config)
    if args.dry_run_contract:
        print("LIP-EVAL-031 contract validated")
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
    result = run_packet_trajectory_evaluation(
        eval_config_path=args.eval_config,
        parent_config_path=args.parent_config,
        matrix_summary_path=args.matrix_summary,
        training_bundle_dir=args.training_bundle,
        confirmation_bundle_dir=args.confirmation_bundle,
        output_path=args.output,
        stage=args.stage,
        target_device=args.target_device,
        bridge_device=args.bridge_device,
        colab_compute_units_before=args.colab_compute_units_before,
    )
    print(json.dumps({"output": str(args.output), "complete": result["complete"]}))


if __name__ == "__main__":
    main()
