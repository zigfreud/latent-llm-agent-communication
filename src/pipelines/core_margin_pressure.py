"""Validate and run the H0-013 trajectory margin-pressure screen."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from src.pipelines.initial_condition_bridge import run_initial_condition_training
from src.pipelines.oracle_experiment import load_json_object
from src.pipelines.receiver_aware_replay import _lf_sha256_file


CORE_MARGIN_PRESSURE_PROTOCOL_VERSION = "lip-core-margin-pressure-v1"


def _validate_loss(loss: Mapping, *, margin_weight: float, norm_weight: float) -> None:
    expected_scalars = {
        "temperature": 0.07,
        "margin_target": 0.05,
        "lambda_huber": 1.0,
        "lambda_cosine": 0.25,
        "lambda_symmetric_nce": 1.0,
        "lambda_margin": margin_weight,
        "lambda_norm": norm_weight,
    }
    for key, expected in expected_scalars.items():
        if float(loss[key]) != expected:
            raise ValueError(f"{key} drifted from the frozen H0-013 intervention")
    if dict(loss["component_weights"]) != {
        "core": 0.45,
        "name": 0.45,
        "boundary": 0.10,
    }:
        raise ValueError("component weights drifted from H0-011")


def validate_core_margin_pressure_contract(
    experiment: Mapping,
    parent: Mapping,
    *,
    experiment_path: Path,
    parent_path: Path,
    predecessor_registry_path: Path,
) -> None:
    if experiment.get("experiment_id") != "LIP-H0-013":
        raise ValueError("unexpected core-margin-pressure experiment_id")
    if experiment.get("protocol_version") != CORE_MARGIN_PRESSURE_PROTOCOL_VERSION:
        raise ValueError("unexpected core-margin-pressure protocol_version")
    if experiment.get("claim_status") != "development_only_core_margin_pressure_screen":
        raise ValueError("H0-013 must remain a development-only screen")
    if parent.get("experiment_id") != "LIP-PROTO-014":
        raise ValueError("H0-013 parent must be LIP-PROTO-014")
    if experiment["parent"]["config_sha256"] != _lf_sha256_file(parent_path):
        raise ValueError("parent config hash differs from the frozen contract")
    if experiment["predecessor"]["registry_sha256"] != _lf_sha256_file(
        predecessor_registry_path
    ):
        raise ValueError("H0-012 registry differs from the frozen contract")

    predecessor = load_json_object(predecessor_registry_path)
    if predecessor.get("experiment_id") != "LIP-H0-012":
        raise ValueError("H0-013 predecessor must be LIP-H0-012")
    aggregate = predecessor.get("aggregate_gate", {})
    if aggregate.get("directional_replication_passed") is not True:
        raise ValueError("H0-012 did not establish the directional property")
    if aggregate.get("strong_replication_passed") is not False:
        raise ValueError("H0-013 requires the negative H0-012 strong gate")
    if predecessor.get("decision", {}).get("core_pressure_screen_authorized") is not True:
        raise ValueError("H0-012 did not authorize a core-pressure screen")

    expected_counts = experiment["data"]["expected_counts"]
    parent_counts = parent["data"]["selection"]
    bindings = {
        "train": "train_count",
        "development_selection": "development_selection_count",
        "development_gate": "development_gate_count",
    }
    for split, parent_key in bindings.items():
        if int(expected_counts[split]) != int(parent_counts[parent_key]):
            raise ValueError(f"{split} count drifted from PROTO-014")
    if list(experiment["data"]["prohibited_splits"]) != ["confirmation"]:
        raise ValueError("H0-013 must prohibit confirmation")

    receiver = experiment["receiver"]
    parent_target = parent["models"]["target"]
    if (receiver["model_id"], receiver["revision"]) != (
        parent_target["model_id"],
        parent_target["revision"],
    ):
        raise ValueError("receiver endpoint drifted from PROTO-014")
    if [int(x) for x in receiver["evolved_layer_indices"]] != [
        int(x) for x in parent["packets"]["target"]["layer_indices"]
    ]:
        raise ValueError("receiver layer prefix drifted from PROTO-014")
    if [int(x) for x in receiver["packet_offsets"]] != [
        int(x) for x in parent["packets"]["target"]["offsets"]
    ]:
        raise ValueError("receiver offsets drifted from PROTO-014")
    if receiver["freeze_all_parameters"] is not True:
        raise ValueError("receiver parameters must remain frozen")

    systems = experiment["variants"]["systems"]
    if set(systems) != {"core_margin_pressure_unrolled"}:
        raise ValueError("H0-013 must contain only its frozen intervention")
    variant = systems["core_margin_pressure_unrolled"]
    if float(variant["lambda_entry_snapshot"]) != 0.25 or float(
        variant["lambda_induced_trajectory"]
    ) != 1.0:
        raise ValueError("unrolled objective drifted from H0-011")
    _validate_loss(
        experiment["loss"]["entry_snapshot"],
        margin_weight=0.10,
        norm_weight=0.0,
    )
    _validate_loss(
        experiment["loss"]["induced_trajectory"],
        margin_weight=1.00,
        norm_weight=0.05,
    )

    pilot = experiment["training"]["pilot"]
    screen = experiment["training"]["full_matrix"]
    if pilot["variant"] != "core_margin_pressure_unrolled":
        raise ValueError("pilot variant drifted")
    if int(pilot["seed"]) != 4007 or int(pilot["max_updates"]) != 4:
        raise ValueError("pilot cell drifted")
    if int(screen["batch_size"]) != 16 or int(screen["max_updates"]) != 128:
        raise ValueError("H0-013 must retain the H0-011 screen budget")
    if [int(seed) for seed in experiment["training"]["seeds"]] != [4007]:
        raise ValueError("H0-013 screen seed drifted")
    gate = experiment["development_gate"]
    if float(gate["reference_core_margin"]) != 0.003350274:
        raise ValueError("H0-011 core-margin reference drifted")
    if float(gate["reference_core_retrieval_top1"]) != 0.6875:
        raise ValueError("H0-011 core-retrieval reference drifted")
    if experiment["confirmation"]["status"] != "prohibited_in_H0-013":
        raise ValueError("confirmation must remain prohibited")
    if experiment["compute"]["preferred_accelerator"] != "L4":
        raise ValueError("H0-013 is frozen for an L4")
    if experiment["compute"]["allow_silent_fallback"] is not False:
        raise ValueError("silent accelerator fallback is prohibited")
    if not experiment_path.is_file():
        raise ValueError("experiment contract path does not exist")


def run_core_margin_pressure_training(**kwargs) -> dict:
    return run_initial_condition_training(
        **kwargs,
        contract_validator=validate_core_margin_pressure_contract,
        result_experiment_id="LIP-H0-013",
        result_protocol_version=CORE_MARGIN_PRESSURE_PROTOCOL_VERSION,
    )
