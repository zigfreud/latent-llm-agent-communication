"""Run the frozen H0-009 entry-seed free-evolution trajectory gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from src.evaluation.entry_seed_free_evolution import (
    ABSOLUTE_OPERATOR,
    ENTRY_SEED_PROTOCOL_VERSION,
    FREE_EVOLUTION_OPERATOR,
    summarize_entry_seed_gate,
)
from src.pipelines.anchored_receiver_aware_replay import (
    _layer_modes,
    run_anchored_trajectory_gate,
)
from src.pipelines.oracle_experiment import load_json_object, load_yaml
from src.pipelines.receiver_aware_replay import (
    _lf_sha256_file,
    finalize_compute_units,
)


def _validate_contract(
    experiment: Mapping,
    parent: Mapping,
    *,
    experiment_path: Path,
    parent_path: Path,
    predecessor_registry_path: Path,
) -> None:
    if experiment.get("experiment_id") != "LIP-H0-009":
        raise ValueError("unexpected entry-seed experiment_id")
    if experiment.get("protocol_version") != ENTRY_SEED_PROTOCOL_VERSION:
        raise ValueError("unexpected entry-seed protocol_version")
    if experiment.get("claim_status") != "exploratory_causal_operator_screen_only":
        raise ValueError("H0-009 must remain an exploratory operator screen")
    if parent.get("experiment_id") != "LIP-PROTO-014":
        raise ValueError("H0-009 parent must be LIP-PROTO-014")
    if experiment["parent"]["config_sha256"] != _lf_sha256_file(parent_path):
        raise ValueError("parent config hash differs from the frozen contract")

    predecessor = load_json_object(predecessor_registry_path)
    if predecessor.get("experiment_id") != "LIP-H0-008":
        raise ValueError("predecessor registry is not LIP-H0-008")
    expected_artifact = experiment["predecessor"]["trajectory_gate_sha256"]
    observed_artifact = predecessor.get("artifacts", {}).get(
        "trajectory_gate", {}
    ).get("sha256")
    if observed_artifact != expected_artifact:
        raise ValueError("H0-008 artifact hash differs from the frozen contract")
    frozen_reference = experiment["gate"]["oracle_free_evolution"][
        "anchored_delta_means"
    ]
    observed_metrics = predecessor["oracle_origin_metrics"]
    for metric, expected in frozen_reference.items():
        if float(expected) != float(observed_metrics[metric]["anchored"]):
            raise ValueError("oracle anchored-delta reference differs from H0-008")

    layers = [int(value) for value in experiment["capture"]["target_layer_indices"]]
    offsets = [int(value) for value in experiment["capture"]["packet_offsets"]]
    if layers != [int(value) for value in parent["packets"]["target"]["layer_indices"]]:
        raise ValueError("target layers drifted from PROTO-014")
    if layers[0] != 0 or list(experiment["capture"]["transition_layers"]) != layers[1:]:
        raise ValueError("entry-seed replay requires layer 0 followed by layers 1–7")
    if offsets != [int(value) for value in parent["packets"]["target"]["offsets"]]:
        raise ValueError("packet offsets drifted from PROTO-014")
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
    if set(operators) != {ABSOLUTE_OPERATOR, FREE_EVOLUTION_OPERATOR}:
        raise ValueError("operator set differs from the frozen entry-seed contract")
    absolute_modes = _layer_modes(operators[ABSOLUTE_OPERATOR], layers)
    free_modes = _layer_modes(operators[FREE_EVOLUTION_OPERATOR], layers)
    if any(mode != "replace" for mode in absolute_modes.values()):
        raise ValueError("absolute comparator must replace every target layer")
    if free_modes[layers[0]] != "replace" or any(
        free_modes[layer] != "add" for layer in layers[1:]
    ):
        raise ValueError("free evolution must replace layer 0 and add zero thereafter")
    learned_gate = experiment["gate"]["learned_operator"]
    if learned_gate["decision_variant"] != experiment["conditions"]["primary_variant"]:
        raise ValueError("gate decision variant differs from the primary variant")
    if experiment["compute"]["preferred_accelerator"] != "L4":
        raise ValueError("H0-009 is frozen for an L4")
    if experiment["compute"]["allow_silent_fallback"] is not False:
        raise ValueError("silent accelerator fallback is prohibited")
    if not experiment_path.is_file():
        raise ValueError("experiment contract path does not exist")


def _entry_seed_packet(absolute: torch.Tensor) -> torch.Tensor:
    if absolute.ndim != 3 or absolute.shape[0] < 2:
        raise ValueError("entry seed requires a multi-layer packet tensor")
    packet = torch.zeros_like(absolute)
    packet[0] = absolute[0]
    return packet


def _condition_packets(
    experiment: Mapping,
    layers: list[int],
    record: Mapping,
    *,
    row_index: int,
    scaffold: torch.Tensor,
    variant_predictions: Mapping[str, Mapping[int, torch.Tensor]],
) -> list[dict]:
    del scaffold
    operators = experiment["conditions"]["operators"]
    absolute_modes = _layer_modes(operators[ABSOLUTE_OPERATOR], layers)
    free_modes = _layer_modes(operators[FREE_EVOLUTION_OPERATOR], layers)
    oracle = record["target_packet"].float()
    conditions = [
        {
            "condition": "oracle_absolute_replace",
            "variant": "oracle",
            "training_seed": None,
            "operator": ABSOLUTE_OPERATOR,
            "replay_mode": absolute_modes,
            "packet": oracle,
        },
        {
            "condition": "oracle_entry_seed_then_free_evolution",
            "variant": "oracle",
            "training_seed": None,
            "operator": FREE_EVOLUTION_OPERATOR,
            "replay_mode": free_modes,
            "packet": _entry_seed_packet(oracle),
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
                        "operator": ABSOLUTE_OPERATOR,
                        "replay_mode": dict(absolute_modes),
                        "packet": absolute,
                    },
                    {
                        "condition": f"{variant}_{FREE_EVOLUTION_OPERATOR}",
                        "variant": variant,
                        "training_seed": int(seed),
                        "operator": FREE_EVOLUTION_OPERATOR,
                        "replay_mode": dict(free_modes),
                        "packet": _entry_seed_packet(absolute),
                    },
                )
            )
    return conditions


def _summarize_gate(
    experiment: Mapping,
    rows,
    variants: list[str],
    seeds: list[int],
) -> dict:
    gate = experiment["gate"]
    learned = gate["learned_operator"]
    oracle = gate["oracle_free_evolution"]
    return summarize_entry_seed_gate(
        rows,
        variants=variants,
        seeds=seeds,
        primary_variant=str(experiment["conditions"]["primary_variant"]),
        minimum_taskwise_improvements=int(learned["minimum_taskwise_improvements"]),
        minimum_passing_replicas=int(learned["minimum_passing_replicas"]),
        oracle_anchored_delta_reference=oracle["anchored_delta_means"],
        oracle_maximum_fraction=float(
            oracle["maximum_fraction_of_anchored_delta"]
        ),
    )


def run_entry_seed_trajectory_gate(**kwargs) -> dict:
    return run_anchored_trajectory_gate(
        **kwargs,
        _contract_validator=_validate_contract,
        _condition_builder=_condition_packets,
        _gate_summarizer=_summarize_gate,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("config/LIP-H0-009_entry_seed_free_evolution.yaml"),
    )
    parser.add_argument(
        "--parent-config",
        type=Path,
        default=Path("config/LIP-PROTO-014_source_conditioned_residual_packet.yaml"),
    )
    parser.add_argument(
        "--predecessor-registry",
        type=Path,
        default=Path(
            "experiments/registry/LIP-H0-008_anchored_receiver_aware_replay.json"
        ),
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
        print("LIP-H0-009 contract validated")
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
    result = run_entry_seed_trajectory_gate(
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
