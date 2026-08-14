"""Validate, run, and aggregate H0-016 hard-negative replication cells."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from src.pipelines.hard_negative_coverage import (
    hard_negative_train_loader_builder,
)
from src.pipelines.initial_condition_bridge import run_initial_condition_training
from src.pipelines.oracle_experiment import load_json_object, load_yaml
from src.pipelines.receiver_aware_replay import _lf_sha256_file


HARD_NEGATIVE_REPLICATION_PROTOCOL_VERSION = "lip-hard-negative-replication-v1"


def _repo_path(experiment_path: Path, relative: str) -> Path:
    return experiment_path.resolve().parents[1] / relative


def validate_hard_negative_replication_contract(
    experiment: Mapping,
    parent: Mapping,
    *,
    experiment_path: Path,
    parent_path: Path,
    predecessor_registry_path: Path,
) -> None:
    if experiment.get("experiment_id") != "LIP-H0-016":
        raise ValueError("unexpected hard-negative replication experiment_id")
    if experiment.get("protocol_version") != HARD_NEGATIVE_REPLICATION_PROTOCOL_VERSION:
        raise ValueError("unexpected hard-negative replication protocol_version")
    if experiment.get("claim_status") != "development_only_hard_negative_replication":
        raise ValueError("H0-016 must remain a development-only replication")
    if parent.get("experiment_id") != "LIP-PROTO-014":
        raise ValueError("H0-016 parent must be LIP-PROTO-014")
    if experiment["parent"]["config_sha256"] != _lf_sha256_file(parent_path):
        raise ValueError("parent config hash differs from the frozen contract")

    if experiment["predecessor"]["registry_sha256"] != _lf_sha256_file(
        predecessor_registry_path
    ):
        raise ValueError("LIP-H0-015 registry differs from the frozen contract")
    predecessor = load_json_object(predecessor_registry_path)
    if predecessor.get("experiment_id") != "LIP-H0-015":
        raise ValueError("H0-016 predecessor must be LIP-H0-015")
    if predecessor.get("artifacts", {}).get("screen", {}).get(
        "run_summary", {}
    ).get("sha256") != experiment["predecessor"]["screen_sha256"]:
        raise ValueError("LIP-H0-015 screen artifact differs from the frozen contract")
    decision = predecessor.get("decision", {})
    if decision.get("H0_016_replication_authorized") is not True:
        raise ValueError("LIP-H0-015 did not authorize H0-016")
    if decision.get("replication_seeds") != [4001, 4003]:
        raise ValueError("LIP-H0-015 replication seeds differ")
    if predecessor.get("screen", {}).get("holm_family_passed") is not True:
        raise ValueError("H0-016 requires the strong H0-015 gate")

    reference_path = _repo_path(experiment_path, experiment["reference"]["config"])
    if experiment["reference"]["config_sha256"] != _lf_sha256_file(reference_path):
        raise ValueError("LIP-H0-015 reference config differs from the frozen contract")
    reference = load_yaml(reference_path)
    if reference.get("experiment_id") != "LIP-H0-015":
        raise ValueError("H0-016 reference must be H0-015")

    for section in ("data", "receiver", "bridge", "loss", "diagnostic_source"):
        if dict(experiment[section]) != dict(reference[section]):
            raise ValueError(f"{section} drifted from H0-015")
    if set(experiment["variants"]["systems"]) != {
        "hard_negative_batches_unrolled"
    }:
        raise ValueError("H0-016 must contain only the frozen H0-015 system")
    if experiment["variants"]["primary"] != reference["variants"]["primary"]:
        raise ValueError("primary variant drifted from H0-015")
    current_variant = experiment["variants"]["systems"][
        "hard_negative_batches_unrolled"
    ]
    frozen_variant = reference["variants"]["systems"][
        "hard_negative_batches_unrolled"
    ]
    for key in (
        "training_loss_scope",
        "lambda_entry_snapshot",
        "lambda_induced_trajectory",
    ):
        if current_variant[key] != frozen_variant[key]:
            raise ValueError(f"variant.{key} drifted from H0-015")

    current_training = experiment["training"]
    frozen_training = reference["training"]
    if [int(seed) for seed in current_training["seeds"]] != [4001, 4003]:
        raise ValueError("H0-016 replication seeds drifted")
    for key in (
        "learning_rate",
        "weight_decay",
        "gradient_clip",
        "fp16_autocast",
        "num_workers",
        "batch_policy",
    ):
        if current_training[key] != frozen_training[key]:
            raise ValueError(f"training.{key} drifted from H0-015")
    for key in (
        "batch_size",
        "gradient_accumulation_steps",
        "max_updates",
        "validation_interval",
    ):
        if current_training["full_matrix"][key] != frozen_training["full_matrix"][key]:
            raise ValueError(f"training.full_matrix.{key} drifted from H0-015")
    if dict(current_training["pilot"]) != dict(frozen_training["pilot"]):
        raise ValueError("training.pilot drifted from H0-015")
    if dict(experiment["development_selection"]) != dict(
        reference["development_selection"]
    ):
        raise ValueError("development-selection rule drifted from H0-015")

    gate = experiment["development_gate"]
    if (float(gate["alpha"]), int(gate["statistics_seed"])) != (
        float(reference["development_gate"]["alpha"]),
        int(reference["development_gate"]["statistics_seed"]),
    ):
        raise ValueError("per-cell statistical gate drifted from H0-015")
    if [int(seed) for seed in gate["new_seeds"]] != [4001, 4003]:
        raise ValueError("H0-016 gate seeds drifted")
    if int(gate["frozen_seed"]) != 4007:
        raise ValueError("H0-016 frozen seed drifted")
    if int(gate["minimum_strong_total_seeds"]) != 2:
        raise ValueError("H0-016 total replication threshold drifted")
    if int(gate["minimum_strong_new_seeds"]) != 1:
        raise ValueError("H0-016 new-seed replication threshold drifted")
    if experiment["confirmation"]["status"] != "prohibited_in_H0-016":
        raise ValueError("confirmation must remain prohibited during H0-016")
    if experiment["compute"]["preferred_accelerator"] != "L4":
        raise ValueError("H0-016 is frozen for an L4")
    if experiment["compute"]["allow_silent_fallback"] is not False:
        raise ValueError("silent accelerator fallback is prohibited")
    if not experiment_path.is_file():
        raise ValueError("experiment contract path does not exist")


def _cell_result(summary: Mapping) -> dict:
    metrics = summary["development_gate_metrics"]["induced_trajectory"]
    family = {row["region"]: row for row in summary["development_gate"]["family"]}
    regions = {}
    for region in ("joint", "core", "name"):
        regions[region] = {
            "retrieval_top1": float(metrics["regions"][region]["retrieval_top1"]),
            "mean_diagonal_margin": float(family[region]["mean_diagonal_margin"]),
            "p_value_holm": float(family[region]["p_value_holm"]),
            "rejected": bool(family[region]["rejected"]),
        }
    return {
        "best_step": int(summary["training"]["best_step"]),
        "normalized_residual_rmse": float(metrics["normalized_residual_rmse"]),
        "mean_retrieval": sum(
            regions[region]["retrieval_top1"] for region in regions
        )
        / 3,
        "regions": regions,
        "strong_gate": bool(summary["development_gate"]["passed"]),
        "training_batch_plan_sha256": summary["provenance"][
            "training_batch_plan_sha256"
        ],
    }


def aggregate_hard_negative_replication(
    experiment: Mapping,
    predecessor: Mapping,
    new_summaries: Mapping[int, Mapping],
) -> dict:
    expected_seeds = [4001, 4003]
    if sorted(int(seed) for seed in new_summaries) != expected_seeds:
        raise ValueError("H0-016 aggregation requires seeds 4001 and 4003")
    results = {}
    expected_plan_hash = predecessor["artifacts"]["screen"][
        "training_batch_plan"
    ]["sha256"]
    for seed in expected_seeds:
        summary = new_summaries[seed]
        if summary.get("experiment_id") != "LIP-H0-016" or int(
            summary.get("seed", -1)
        ) != seed:
            raise ValueError(f"replication summary identity differs for seed {seed}")
        cell = _cell_result(summary)
        if cell["training_batch_plan_sha256"] != expected_plan_hash:
            raise ValueError("replication batch plan differs from frozen H0-015")
        results[str(seed)] = cell

    frozen = predecessor["screen"]
    results["4007"] = {
        "source": "frozen_H0_015_screen",
        "best_step": int(frozen["best_step"]),
        "normalized_residual_rmse": float(frozen["normalized_residual_rmse"]),
        "mean_retrieval": float(frozen["mean_retrieval"]),
        "regions": dict(frozen["regions"]),
        "strong_gate": bool(frozen["holm_family_passed"]),
        "training_batch_plan_sha256": expected_plan_hash,
    }
    strong_new = sum(results[str(seed)]["strong_gate"] for seed in expected_seeds)
    strong_total = strong_new + int(results["4007"]["strong_gate"])
    gate = experiment["development_gate"]
    passed = bool(
        strong_new >= int(gate["minimum_strong_new_seeds"])
        and strong_total >= int(gate["minimum_strong_total_seeds"])
    )
    all_new_passed = bool(strong_new == len(expected_seeds))
    return {
        "results": results,
        "aggregate_gate": {
            "strong_passing_new_seeds": strong_new,
            "strong_new_seeds_required": int(gate["minimum_strong_new_seeds"]),
            "strong_passing_total_seeds": strong_total,
            "strong_total_seeds_required": int(gate["minimum_strong_total_seeds"]),
            "strong_replication_passed": passed,
            "all_new_seeds_passed": all_new_passed,
        },
        "decision": {
            "functional_confirmation_design_authorized": passed,
            "functional_confirmation_execution_authorized": False,
            "claim_scope": (
                "cross_seed_system_replication"
                if passed
                else "single_seed_strong_result_not_replicated"
            ),
            "paired_causal_effect_replicated_across_seeds": False,
            "proto_015_status": "premature",
        },
    }


def run_hard_negative_replication_training(
    *, candidate_bank_path: Path, **kwargs
) -> dict:
    experiment = load_yaml(kwargs["experiment_path"])
    return run_initial_condition_training(
        **kwargs,
        contract_validator=validate_hard_negative_replication_contract,
        result_experiment_id="LIP-H0-016",
        result_protocol_version=HARD_NEGATIVE_REPLICATION_PROTOCOL_VERSION,
        train_loader_builder=hard_negative_train_loader_builder(
            experiment, candidate_bank_path
        ),
    )
