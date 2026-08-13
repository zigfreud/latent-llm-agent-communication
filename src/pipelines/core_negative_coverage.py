"""Validate and run the H0-011 core negative-coverage screen."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from src.pipelines.initial_condition_bridge import run_initial_condition_training
from src.pipelines.oracle_experiment import load_json_object
from src.pipelines.receiver_aware_replay import _lf_sha256_file


CORE_NEGATIVE_COVERAGE_PROTOCOL_VERSION = "lip-core-negative-coverage-v1"


def validate_core_negative_coverage_contract(
    experiment: Mapping,
    parent: Mapping,
    *,
    experiment_path: Path,
    parent_path: Path,
    predecessor_registry_path: Path,
) -> None:
    if experiment.get("experiment_id") != "LIP-H0-011":
        raise ValueError("unexpected core-negative-coverage experiment_id")
    if experiment.get("protocol_version") != CORE_NEGATIVE_COVERAGE_PROTOCOL_VERSION:
        raise ValueError("unexpected core-negative-coverage protocol_version")
    if experiment.get("claim_status") != "development_only_negative_coverage_screen":
        raise ValueError("H0-011 must remain a development-only screen")
    if parent.get("experiment_id") != "LIP-PROTO-014":
        raise ValueError("H0-011 parent must be LIP-PROTO-014")
    if experiment["parent"]["config_sha256"] != _lf_sha256_file(parent_path):
        raise ValueError("parent config hash differs from the frozen contract")

    predecessor = load_json_object(predecessor_registry_path)
    if predecessor.get("experiment_id") != "LIP-H0-010":
        raise ValueError("H0-011 predecessor must be LIP-H0-010")
    observed_summary = predecessor.get("artifacts", {}).get("matrix_summary", {}).get(
        "sha256"
    )
    if observed_summary != experiment["predecessor"]["matrix_summary_sha256"]:
        raise ValueError("H0-010 matrix artifact differs from the frozen contract")
    if predecessor.get("development_gate", {}).get("passed") is not False:
        raise ValueError("H0-011 requires the negative H0-010 development gate")

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
        raise ValueError("H0-011 must prohibit confirmation")

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
    if set(systems) != {"large_negative_batch_unrolled"}:
        raise ValueError("H0-011 must contain only its frozen intervention")
    variant = systems["large_negative_batch_unrolled"]
    if float(variant["lambda_entry_snapshot"]) != 0.25:
        raise ValueError("entry regularizer drifted from H0-010")
    if float(variant["lambda_induced_trajectory"]) != 1.0:
        raise ValueError("trajectory objective drifted from H0-010")
    for scope in ("entry_snapshot", "induced_trajectory"):
        loss = experiment["loss"][scope]
        if float(loss["lambda_margin"]) != 0.10:
            raise ValueError("margin pressure must remain unchanged in H0-011")
        if dict(loss["component_weights"]) != {
            "core": 0.45,
            "name": 0.45,
            "boundary": 0.10,
        }:
            raise ValueError("component weights must remain unchanged in H0-011")

    pilot = experiment["training"]["pilot"]
    screen = experiment["training"]["full_matrix"]
    if int(pilot["batch_size"]) != 16 or int(screen["batch_size"]) != 16:
        raise ValueError("H0-011 intervention requires batch size 16")
    if int(screen["max_updates"]) != 128:
        raise ValueError("H0-011 must preserve the 2,048-example comparison budget")
    if int(screen["batch_size"]) * int(screen["max_updates"]) != 2048:
        raise ValueError("H0-011 example budget drifted")
    if [int(seed) for seed in experiment["training"]["seeds"]] != [4007]:
        raise ValueError("H0-011 is frozen to the 4007 development screen")
    if experiment["confirmation"]["status"] != "prohibited_in_H0-011":
        raise ValueError("confirmation must remain prohibited")
    if experiment["compute"]["preferred_accelerator"] != "L4":
        raise ValueError("H0-011 is frozen for an L4")
    if experiment["compute"]["allow_silent_fallback"] is not False:
        raise ValueError("silent accelerator fallback is prohibited")
    if not experiment_path.is_file():
        raise ValueError("experiment contract path does not exist")


def run_core_negative_coverage_training(**kwargs) -> dict:
    return run_initial_condition_training(
        **kwargs,
        contract_validator=validate_core_negative_coverage_contract,
        result_experiment_id="LIP-H0-011",
        result_protocol_version=CORE_NEGATIVE_COVERAGE_PROTOCOL_VERSION,
    )
