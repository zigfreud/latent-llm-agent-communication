"""Validate and run an H0-012 negative-coverage replication cell."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from src.pipelines.initial_condition_bridge import run_initial_condition_training
from src.pipelines.oracle_experiment import load_json_object
from src.pipelines.receiver_aware_replay import _lf_sha256_file


CORE_NEGATIVE_REPLICATION_PROTOCOL_VERSION = (
    "lip-core-negative-coverage-replication-v1"
)


def validate_core_negative_replication_contract(
    experiment: Mapping,
    parent: Mapping,
    *,
    experiment_path: Path,
    parent_path: Path,
    predecessor_registry_path: Path,
) -> None:
    if experiment.get("experiment_id") != "LIP-H0-012":
        raise ValueError("unexpected core-negative replication experiment_id")
    if experiment.get("protocol_version") != CORE_NEGATIVE_REPLICATION_PROTOCOL_VERSION:
        raise ValueError("unexpected core-negative replication protocol_version")
    if experiment.get("claim_status") != "development_only_replication_extension":
        raise ValueError("H0-012 must remain a development-only replication")
    if parent.get("experiment_id") != "LIP-PROTO-014":
        raise ValueError("H0-012 parent must be LIP-PROTO-014")
    if experiment["parent"]["config_sha256"] != _lf_sha256_file(parent_path):
        raise ValueError("parent config hash differs from the frozen contract")

    predecessor = load_json_object(predecessor_registry_path)
    if predecessor.get("experiment_id") != "LIP-H0-011":
        raise ValueError("H0-012 predecessor must be LIP-H0-011")
    observed = predecessor.get("artifacts", {}).get("screen", {}).get("sha256")
    if observed != experiment["predecessor"]["screen_sha256"]:
        raise ValueError("H0-011 screen artifact differs from the frozen contract")
    if predecessor.get("decision", {}).get("replication_authorized") is not True:
        raise ValueError("H0-011 did not authorize replication")

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
        raise ValueError("H0-012 must prohibit confirmation")

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

    variant = experiment["variants"]["systems"].get(
        "large_negative_batch_unrolled"
    )
    if variant is None or set(experiment["variants"]["systems"]) != {
        "large_negative_batch_unrolled"
    }:
        raise ValueError("H0-012 must contain only its frozen intervention")
    if float(variant["lambda_entry_snapshot"]) != 0.25 or float(
        variant["lambda_induced_trajectory"]
    ) != 1.0:
        raise ValueError("unrolled objective drifted from H0-011")
    for scope in ("entry_snapshot", "induced_trajectory"):
        loss = experiment["loss"][scope]
        if float(loss["lambda_margin"]) != 0.10:
            raise ValueError("margin pressure must remain unchanged")
        if dict(loss["component_weights"]) != {
            "core": 0.45,
            "name": 0.45,
            "boundary": 0.10,
        }:
            raise ValueError("component weights must remain unchanged")

    stage = experiment["training"]["full_matrix"]
    if int(stage["batch_size"]) != 16 or int(stage["max_updates"]) != 128:
        raise ValueError("H0-012 must reproduce the H0-011 training budget")
    if [int(seed) for seed in experiment["training"]["seeds"]] != [4001, 4003]:
        raise ValueError("H0-012 extension seeds drifted")
    gate = experiment["development_gate"]
    if int(gate["minimum_directional_replicas"]) != 2 or int(
        gate["minimum_strong_replicas"]
    ) != 2:
        raise ValueError("replication thresholds drifted")
    if experiment["confirmation"]["status"] != "prohibited_in_H0-012":
        raise ValueError("confirmation must remain prohibited")
    if experiment["compute"]["preferred_accelerator"] != "L4":
        raise ValueError("H0-012 is frozen for an L4")
    if experiment["compute"]["allow_silent_fallback"] is not False:
        raise ValueError("silent accelerator fallback is prohibited")
    if not experiment_path.is_file():
        raise ValueError("experiment contract path does not exist")


def run_core_negative_replication_training(**kwargs) -> dict:
    return run_initial_condition_training(
        **kwargs,
        contract_validator=validate_core_negative_replication_contract,
        result_experiment_id="LIP-H0-012",
        result_protocol_version=CORE_NEGATIVE_REPLICATION_PROTOCOL_VERSION,
    )
