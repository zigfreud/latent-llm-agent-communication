"""Aggregate the frozen paired H0-017 state-conditioning screen."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from src.evaluation.packet_bridge import RETRIEVAL_REGIONS
from src.evaluation.statistics import (
    bootstrap_mean_ci,
    holm_adjust,
    mean,
    sign_flip_p_value,
)
from src.pipelines.receiver_aware_replay import _lf_sha256_file


AGGREGATION_PROTOCOL_VERSION = "lip-closed-loop-paired-screen-aggregation-v1"
SOURCE_PROTOCOL_VERSION = "lip-closed-loop-trajectory-corrector-v1"


def validate_closed_loop_screen_contract(
    aggregation: Mapping,
    experiment: Mapping,
    *,
    aggregation_path: Path,
    experiment_path: Path,
) -> None:
    if aggregation.get("experiment_id") != "LIP-H0-017":
        raise ValueError("unexpected aggregation experiment_id")
    if aggregation.get("aggregation_protocol_version") != AGGREGATION_PROTOCOL_VERSION:
        raise ValueError("unexpected aggregation protocol_version")
    if aggregation.get("claim_status") != (
        "development_only_paired_state_conditioning_screen"
    ):
        raise ValueError("unexpected aggregation claim_status")
    source = aggregation["source_experiment"]
    if source.get("protocol_version") != SOURCE_PROTOCOL_VERSION:
        raise ValueError("aggregation source protocol drifted")
    if source.get("config_sha256") != _lf_sha256_file(experiment_path):
        raise ValueError("aggregation source experiment hash drifted")
    if experiment.get("experiment_id") != "LIP-H0-017" or experiment.get(
        "protocol_version"
    ) != SOURCE_PROTOCOL_VERSION:
        raise ValueError("unexpected source experiment identity")

    paired = aggregation["paired_cells"]
    frozen_stage = experiment["training"]["paired_screen"]
    if (
        paired.get("control"),
        paired.get("treatment"),
        paired.get("required_stage"),
        int(paired.get("seed", -1)),
        int(paired.get("batch_size", -1)),
        int(paired.get("updates", -1)),
    ) != (
        "open_loop_zero_live",
        "closed_loop_live",
        "paired_screen_cell",
        int(experiment["training"]["seed"]),
        int(frozen_stage["batch_size"]),
        int(frozen_stage["max_updates"]),
    ):
        raise ValueError("paired cell contract drifted")
    if paired.get("require_identical_run_commit") is not True:
        raise ValueError("paired cells must share one run commit")
    if paired.get("require_identical_training_batch_plan") is not True:
        raise ValueError("paired cells must share one training batch plan")
    if paired.get("require_identical_shared_provenance") is not True:
        raise ValueError("paired cells must share frozen provenance")

    inference = aggregation["inference"]
    if (
        inference.get("split"),
        int(inference.get("task_count", -1)),
        inference.get("trajectory"),
        inference.get("paired_margin_metric"),
        list(inference.get("regions", [])),
        inference.get("alternative"),
        float(inference.get("alpha", -1.0)),
        inference.get("multiplicity"),
    ) != (
        "development_gate",
        32,
        "incoming_trajectory",
        "diagonal_margin",
        list(RETRIEVAL_REGIONS),
        "greater",
        0.05,
        "one_Holm_family_across_joint_core_name",
    ):
        raise ValueError("paired inference contract drifted")
    if inference.get("treatment_minus_control") is not True:
        raise ValueError("paired effect direction drifted")
    if int(inference.get("monte_carlo_samples", 0)) != 100_000:
        raise ValueError("paired randomization budget drifted")
    if int(inference.get("bootstrap_iterations", 0)) != 10_000:
        raise ValueError("paired bootstrap budget drifted")

    gate = aggregation["gate"]
    source_gate = experiment["paired_screen_gate"]
    if (
        gate.get("primary_metric"),
        float(gate.get("minimum_relative_rmse_reduction_vs_control", -1.0)),
        gate.get("require_core_retrieval_not_lower"),
        gate.get("require_mean_retrieval_not_lower"),
        gate.get("require_all_Holm_adjusted_margin_tests"),
    ) != (
        source_gate["primary_metric"],
        float(source_gate["minimum_relative_rmse_reduction_vs_state_blind"]),
        source_gate["require_core_retrieval_not_lower"],
        source_gate["require_mean_retrieval_not_lower"],
        source_gate["require_complete_joint_core_name_holm_family"],
    ):
        raise ValueError("paired aggregate gate drifted")
    boundary = aggregation["decision_boundary"]
    if any(
        boundary.get(key) is not False
        for key in (
            "eval_038_execution_authorized",
            "functional_transport_supported",
            "confirmation_used",
        )
    ):
        raise ValueError("H0-017 cannot authorize execution or a functional claim")
    if boundary.get("proto_015_status") != "premature":
        raise ValueError("PROTO-015 boundary drifted")
    if not aggregation_path.is_file() or not experiment_path.is_file():
        raise ValueError("aggregation contract path does not exist")


def _finite(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _validate_pilot(aggregation: Mapping, pilot: Mapping) -> None:
    frozen = aggregation["pilot_authorization"]
    if (
        pilot.get("experiment_id"),
        pilot.get("protocol_version"),
        pilot.get("stage"),
        pilot.get("variant"),
        pilot.get("run_commit"),
        pilot.get("complete"),
        pilot.get("pilot_gate", {}).get("passed"),
    ) != (
        "LIP-H0-017",
        SOURCE_PROTOCOL_VERSION,
        frozen["required_stage"],
        frozen["required_variant"],
        frozen["run_commit"],
        True,
        frozen["required_passed"],
    ):
        raise ValueError("pilot does not authorize the paired screen")
    if pilot.get("provenance", {}).get("experiment_config_sha256") != aggregation[
        "source_experiment"
    ]["config_sha256"]:
        raise ValueError("pilot source experiment hash drifted")


def _validate_cell(
    aggregation: Mapping,
    summary: Mapping,
    *,
    variant: str,
) -> None:
    paired = aggregation["paired_cells"]
    if (
        summary.get("experiment_id"),
        summary.get("protocol_version"),
        summary.get("stage"),
        summary.get("variant"),
        int(summary.get("seed", -1)),
        summary.get("complete"),
    ) != (
        "LIP-H0-017",
        SOURCE_PROTOCOL_VERSION,
        paired["required_stage"],
        variant,
        int(paired["seed"]),
        True,
    ):
        raise ValueError(f"paired cell identity differs for {variant}")
    training = summary.get("training", {})
    stage = training.get("resolved_stage", {})
    if (
        int(training.get("updates_completed", -1)),
        int(training.get("batch_size", -1)),
        int(stage.get("max_updates", -1)),
        int(stage.get("batch_size", -1)),
        stage.get("variant"),
    ) != (
        int(paired["updates"]),
        int(paired["batch_size"]),
        int(paired["updates"]),
        int(paired["batch_size"]),
        variant,
    ):
        raise ValueError(f"paired training stage differs for {variant}")
    if summary.get("pilot_gate") is not None:
        raise ValueError("paired cell unexpectedly contains a pilot gate")
    if summary.get("provenance", {}).get("experiment_config_sha256") != aggregation[
        "source_experiment"
    ]["config_sha256"]:
        raise ValueError(f"paired source experiment hash differs for {variant}")


def _task_margin_map(metrics: Mapping, region: str, expected_count: int) -> dict[str, float]:
    rows = metrics["regions"][region]["tasks"]
    values: dict[str, float] = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if not task_id or task_id in values:
            raise ValueError(f"{region} margins require unique task identities")
        values[task_id] = _finite(row["diagonal_margin"], f"{region} margin")
    if len(values) != expected_count:
        raise ValueError(f"{region} paired task count differs from the contract")
    return values


def _paired_margin_family(
    aggregation: Mapping,
    treatment_metrics: Mapping,
    control_metrics: Mapping,
) -> dict:
    inference = aggregation["inference"]
    task_count = int(inference["task_count"])
    tests = []
    for offset, region in enumerate(inference["regions"]):
        treatment = _task_margin_map(treatment_metrics, region, task_count)
        control = _task_margin_map(control_metrics, region, task_count)
        if set(treatment) != set(control):
            raise ValueError(f"{region} treatment/control task identities differ")
        shared = sorted(treatment)
        differences = [treatment[task_id] - control[task_id] for task_id in shared]
        lower, upper = bootstrap_mean_ci(
            differences,
            iterations=int(inference["bootstrap_iterations"]),
            confidence=float(inference["confidence"]),
            seed=int(inference["statistics_seed"]) + 1000 + offset,
        )
        p_value, method = sign_flip_p_value(
            differences,
            alternative=inference["alternative"],
            monte_carlo_samples=int(inference["monte_carlo_samples"]),
            seed=int(inference["statistics_seed"]) + 2000 + offset,
        )
        tests.append(
            {
                "region": region,
                "task_count": len(shared),
                "nonzero_task_count": sum(abs(value) > 1e-15 for value in differences),
                "mean_treatment_minus_control": mean(differences),
                "ci_lower": lower,
                "ci_upper": upper,
                "p_value": p_value,
                "test_method": method,
                "alternative": inference["alternative"],
                "tasks": [
                    {
                        "task_id": task_id,
                        "treatment_margin": treatment[task_id],
                        "control_margin": control[task_id],
                        "difference": treatment[task_id] - control[task_id],
                    }
                    for task_id in shared
                ],
            }
        )
    adjusted = holm_adjust([test["p_value"] for test in tests])
    alpha = float(inference["alpha"])
    for test, adjusted_p in zip(tests, adjusted):
        test["p_value_holm"] = adjusted_p
        test["rejected"] = bool(
            test["mean_treatment_minus_control"] > 0.0 and adjusted_p <= alpha
        )
    return {
        "alpha": alpha,
        "family": tests,
        "passed": all(test["rejected"] for test in tests),
        "criterion": (
            "positive treatment-minus-control task margin in joint/core/name "
            "under one Holm family"
        ),
    }


def aggregate_closed_loop_screen(
    aggregation: Mapping,
    experiment: Mapping,
    pilot: Mapping,
    control: Mapping,
    treatment: Mapping,
    *,
    aggregation_path: Path,
    experiment_path: Path,
) -> dict:
    validate_closed_loop_screen_contract(
        aggregation,
        experiment,
        aggregation_path=aggregation_path,
        experiment_path=experiment_path,
    )
    _validate_pilot(aggregation, pilot)
    paired = aggregation["paired_cells"]
    _validate_cell(aggregation, control, variant=paired["control"])
    _validate_cell(aggregation, treatment, variant=paired["treatment"])
    if control.get("run_commit") != treatment.get("run_commit"):
        raise ValueError("paired cells were not executed from one code commit")
    if control["training"].get("batch_policy") != treatment["training"].get(
        "batch_policy"
    ):
        raise ValueError("paired cells do not share one frozen training batch plan")
    for field in paired["shared_provenance_fields"]:
        if control["provenance"].get(field) != treatment["provenance"].get(field):
            raise ValueError(f"paired provenance differs for {field}")

    trajectory = aggregation["inference"]["trajectory"]
    control_metrics = control["development_gate_metrics"][trajectory]
    treatment_metrics = treatment["development_gate_metrics"][trajectory]
    control_rmse = _finite(control_metrics["normalized_residual_rmse"], "control RMSE")
    treatment_rmse = _finite(
        treatment_metrics["normalized_residual_rmse"], "treatment RMSE"
    )
    if control_rmse <= 0.0:
        raise ValueError("control RMSE must be positive")
    relative_reduction = (control_rmse - treatment_rmse) / control_rmse

    retrieval = {}
    for region in RETRIEVAL_REGIONS:
        retrieval[region] = {
            "control": _finite(
                control_metrics["regions"][region]["retrieval_top1"],
                f"{region} control retrieval",
            ),
            "treatment": _finite(
                treatment_metrics["regions"][region]["retrieval_top1"],
                f"{region} treatment retrieval",
            ),
        }
        retrieval[region]["difference"] = (
            retrieval[region]["treatment"] - retrieval[region]["control"]
        )
    control_mean_retrieval = mean(
        [retrieval[region]["control"] for region in RETRIEVAL_REGIONS]
    )
    treatment_mean_retrieval = mean(
        [retrieval[region]["treatment"] for region in RETRIEVAL_REGIONS]
    )
    family = _paired_margin_family(
        aggregation, treatment_metrics, control_metrics
    )

    gate_config = aggregation["gate"]
    tolerance = float(gate_config["numeric_tolerance"])
    rmse_passed = relative_reduction + tolerance >= float(
        gate_config["minimum_relative_rmse_reduction_vs_control"]
    )
    core_passed = (
        retrieval["core"]["treatment"] + tolerance
        >= retrieval["core"]["control"]
    )
    mean_passed = treatment_mean_retrieval + tolerance >= control_mean_retrieval
    passed = bool(rmse_passed and core_passed and mean_passed and family["passed"])
    return {
        "experiment_id": "LIP-H0-017",
        "aggregation_protocol_version": AGGREGATION_PROTOCOL_VERSION,
        "claim_status": aggregation["claim_status"],
        "complete": True,
        "provenance": {
            "aggregation_config_sha256": _lf_sha256_file(aggregation_path),
            "experiment_config_sha256": _lf_sha256_file(experiment_path),
            "pilot_run_commit": pilot["run_commit"],
            "paired_run_commit": treatment["run_commit"],
            "shared_cell_provenance": {
                field: treatment["provenance"][field]
                for field in paired["shared_provenance_fields"]
            },
        },
        "inputs": {
            "pilot": {
                "variant": pilot["variant"],
                "passed": True,
                "drive_file_id": aggregation["pilot_authorization"][
                    "drive_file_id"
                ],
            },
            "control": {
                "variant": control["variant"],
                "best_step": int(control["training"]["best_step"]),
            },
            "treatment": {
                "variant": treatment["variant"],
                "best_step": int(treatment["training"]["best_step"]),
            },
        },
        "paired_effects": {
            "incoming_trajectory_rmse": {
                "control": control_rmse,
                "treatment": treatment_rmse,
                "absolute_reduction": control_rmse - treatment_rmse,
                "relative_reduction": relative_reduction,
            },
            "retrieval_top1": {
                "regions": retrieval,
                "control_mean": control_mean_retrieval,
                "treatment_mean": treatment_mean_retrieval,
                "mean_difference": treatment_mean_retrieval
                - control_mean_retrieval,
            },
            "paired_margin_family": family,
        },
        "aggregate_gate": {
            "minimum_relative_rmse_reduction": float(
                gate_config["minimum_relative_rmse_reduction_vs_control"]
            ),
            "observed_relative_rmse_reduction": relative_reduction,
            "rmse_reduction_passed": rmse_passed,
            "core_retrieval_not_lower": core_passed,
            "mean_retrieval_not_lower": mean_passed,
            "holm_margin_family_passed": bool(family["passed"]),
            "passed": passed,
        },
        "decision": {
            "eval_038_design_authorized": passed,
            "eval_038_execution_authorized": False,
            "functional_transport_supported": False,
            "confirmation_used": False,
            "proto_015_status": "premature",
            "next_action": (
                gate_config["pass_action"] if passed else gate_config["fail_action"]
            ),
        },
    }
