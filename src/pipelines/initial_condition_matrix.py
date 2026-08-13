"""Run and summarize the frozen H0-010 two-variant development matrix."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.core.packet_bundle import sha256_file
from src.evaluation.packet_bridge import summarize_multi_replica_development_gate
from src.pipelines.initial_condition_bridge import (
    INITIAL_CONDITION_PROTOCOL_VERSION,
    _git_head,
    _validate_contract,
)
from src.pipelines.oracle_experiment import load_json_object, load_yaml
from src.pipelines.packet_trajectory import _atomic_json


MATRIX_VARIANTS = ("static_entry_snapshot", "unrolled_initial_condition")


def validate_pilot_authorization(
    registry_path: Path,
    summary_path: Path,
) -> dict:
    registry = load_json_object(registry_path)
    summary = load_json_object(summary_path)
    if registry.get("experiment_id") != "LIP-H0-010":
        raise ValueError("pilot registry is not LIP-H0-010")
    if registry.get("protocol_version") != INITIAL_CONDITION_PROTOCOL_VERSION:
        raise ValueError("pilot registry protocol differs from the matrix")
    if registry.get("claim_status") != "passed_numeric_feasibility_pilot_only":
        raise ValueError("pilot registry does not authorize the development matrix")
    expected_sha = registry.get("artifacts", {}).get("run_summary", {}).get("sha256")
    if expected_sha != sha256_file(summary_path):
        raise ValueError("pilot summary hash differs from the authorization registry")
    if summary.get("protocol_version") != INITIAL_CONDITION_PROTOCOL_VERSION:
        raise ValueError("pilot summary protocol differs from the matrix")
    if summary.get("stage") != "pilot" or summary.get("variant") != "unrolled_initial_condition":
        raise ValueError("pilot summary is not the frozen primary pilot cell")
    if int(summary.get("seed", -1)) != 4001:
        raise ValueError("pilot summary seed differs from the frozen pilot")
    if summary.get("run_commit") != registry.get("run_commit"):
        raise ValueError("pilot run commit differs from the authorization registry")
    if not summary.get("pilot_gate", {}).get("passed"):
        raise ValueError("pilot feasibility gate did not pass")
    return summary


def _mean_retrieval(metrics: Mapping) -> float:
    regions = metrics["regions"]
    return sum(
        float(regions[region]["retrieval_top1"])
        for region in ("joint", "core", "name")
    ) / 3.0


def summarize_initial_condition_matrix(
    cell_summaries: Sequence[Mapping],
    experiment: Mapping,
) -> dict:
    indexed = {
        (str(cell["variant"]), int(cell["seed"])): cell
        for cell in cell_summaries
    }
    seeds = [int(value) for value in experiment["training"]["seeds"]]
    expected = {(variant, seed) for variant in MATRIX_VARIANTS for seed in seeds}
    if set(indexed) != expected:
        raise ValueError("matrix summaries do not contain the frozen six cells")

    primary_reports = [
        indexed[("unrolled_initial_condition", seed)]["development_gate"]
        for seed in seeds
    ]
    multi_replica = summarize_multi_replica_development_gate(
        primary_reports,
        minimum_passing_replicas=int(
            experiment["development_gate"]["minimum_passing_primary_replicas"]
        ),
    )
    paired = []
    for seed in seeds:
        static = indexed[("static_entry_snapshot", seed)]
        primary = indexed[("unrolled_initial_condition", seed)]
        static_metrics = static["development_gate_metrics"]["induced_trajectory"]
        primary_metrics = primary["development_gate_metrics"]["induced_trajectory"]
        static_rmse = float(static_metrics["normalized_residual_rmse"])
        primary_rmse = float(primary_metrics["normalized_residual_rmse"])
        static_retrieval = _mean_retrieval(static_metrics)
        primary_retrieval = _mean_retrieval(primary_metrics)
        passed = primary_rmse < static_rmse and primary_retrieval >= static_retrieval
        paired.append(
            {
                "seed": seed,
                "static_rmse": static_rmse,
                "primary_rmse": primary_rmse,
                "rmse_improved": primary_rmse < static_rmse,
                "static_mean_retrieval": static_retrieval,
                "primary_mean_retrieval": primary_retrieval,
                "retrieval_not_lower": primary_retrieval >= static_retrieval,
                "passed": passed,
            }
        )
    required_pairs = int(
        experiment["development_gate"]["minimum_passing_primary_replicas"]
    )
    paired_passes = sum(bool(row["passed"]) for row in paired)
    paired_requirement = paired_passes >= required_pairs
    return {
        "primary_multi_replica_gate": multi_replica,
        "paired_primary_vs_static": {
            "minimum_passing_pairs": required_pairs,
            "passing_pairs": paired_passes,
            "passed": paired_requirement,
            "pairs": paired,
        },
        "passed": bool(multi_replica["passed"] and paired_requirement),
        "pass_action": experiment["development_gate"]["pass_action"],
        "fail_action": experiment["development_gate"]["fail_action"],
    }


def _cell_summary_is_complete(
    path: Path,
    *,
    variant: str,
    seed: int,
    run_commit: str,
) -> bool:
    if not path.is_file():
        return False
    payload = load_json_object(path)
    return bool(
        payload.get("complete")
        and payload.get("protocol_version") == INITIAL_CONDITION_PROTOCOL_VERSION
        and payload.get("stage") == "full_training_cell"
        and payload.get("variant") == variant
        and int(payload.get("seed", -1)) == int(seed)
        and payload.get("run_commit") == run_commit
    )


def run_initial_condition_matrix(
    *,
    experiment_path: Path,
    parent_path: Path,
    predecessor_registry_path: Path,
    pilot_registry_path: Path,
    pilot_summary_path: Path,
    bundle_dir: Path,
    output_root: Path,
    target_device: str,
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
    validate_pilot_authorization(pilot_registry_path, pilot_summary_path)
    run_commit = _git_head()
    state_path = output_root / "matrix_state.json"
    if output_root.exists() and any(output_root.iterdir()) and not state_path.is_file():
        raise FileExistsError("matrix output root is nonempty without resumable state")
    output_root.mkdir(parents=True, exist_ok=True)
    pilot_sha = sha256_file(pilot_summary_path)
    if state_path.is_file():
        state = load_json_object(state_path)
        if state.get("run_commit") != run_commit:
            raise ValueError("resumed matrix run commit differs from its frozen state")
        if state.get("pilot_summary_sha256") != pilot_sha:
            raise ValueError("resumed matrix pilot authorization differs")
    else:
        state = {
            "experiment_id": "LIP-H0-010",
            "protocol_version": INITIAL_CONDITION_PROTOCOL_VERSION,
            "status": "in_progress",
            "run_commit": run_commit,
            "pilot_summary_sha256": pilot_sha,
            "colab_compute_units_before": colab_compute_units_before,
            "completed_cells": [],
        }
        _atomic_json(state_path, state)

    seeds = [int(value) for value in experiment["training"]["seeds"]]
    cells = [(variant, seed) for seed in seeds for variant in MATRIX_VARIANTS]
    summaries = []
    for variant, seed in cells:
        cell_dir = output_root / variant / str(seed)
        summary_path = cell_dir / "run_summary.json"
        if not _cell_summary_is_complete(
            summary_path,
            variant=variant,
            seed=seed,
            run_commit=run_commit,
        ):
            if cell_dir.exists() and any(cell_dir.iterdir()):
                raise FileExistsError(f"incomplete matrix cell is nonempty: {cell_dir}")
            print(
                json.dumps(
                    {"event": "matrix_cell_start", "variant": variant, "seed": seed}
                ),
                flush=True,
            )
            command = [
                sys.executable,
                "-m",
                "src.scripts.run_initial_condition_bridge",
                "--experiment-config",
                str(experiment_path),
                "--parent-config",
                str(parent_path),
                "--predecessor-registry",
                str(predecessor_registry_path),
                "--bundle-dir",
                str(bundle_dir),
                "--output-dir",
                str(cell_dir),
                "--variant",
                variant,
                "--seed",
                str(seed),
                "--target-device",
                target_device,
            ]
            subprocess.run(command, check=True)
        if not _cell_summary_is_complete(
            summary_path,
            variant=variant,
            seed=seed,
            run_commit=run_commit,
        ):
            raise RuntimeError(f"matrix cell did not produce a valid summary: {cell_dir}")
        summary = load_json_object(summary_path)
        summaries.append(summary)
        completed = {
            "variant": variant,
            "seed": seed,
            "run_summary_sha256": sha256_file(summary_path),
        }
        state["completed_cells"] = [
            row
            for row in state["completed_cells"]
            if (row["variant"], int(row["seed"])) != (variant, seed)
        ] + [completed]
        _atomic_json(state_path, state)
        print(
            json.dumps(
                {"event": "matrix_cell_complete", "variant": variant, "seed": seed}
            ),
            flush=True,
        )

    gate = summarize_initial_condition_matrix(summaries, experiment)
    result = {
        "experiment_id": "LIP-H0-010",
        "protocol_version": INITIAL_CONDITION_PROTOCOL_VERSION,
        "claim_status": "development_matrix_only",
        "run_commit": run_commit,
        "pilot_authorization": {
            "registry": str(pilot_registry_path),
            "summary_sha256": pilot_sha,
        },
        "cell_order": [
            {"variant": variant, "seed": seed} for variant, seed in cells
        ],
        "cells": state["completed_cells"],
        "development_gate": gate,
        "confirmation_used": False,
        "telemetry": {
            "colab_compute_units_before": colab_compute_units_before,
            "colab_compute_units_after": None,
            "colab_compute_units_consumed": None,
        },
        "complete": True,
    }
    summary_path = output_root / "matrix_summary.json"
    _atomic_json(summary_path, result)
    state["status"] = "complete"
    state["matrix_summary_sha256"] = sha256_file(summary_path)
    _atomic_json(state_path, state)
    return result


def finalize_matrix_compute_units(summary_path: Path, after: float) -> dict:
    payload = load_json_object(summary_path)
    telemetry = payload.setdefault("telemetry", {})
    before = telemetry.get("colab_compute_units_before")
    if before is None:
        raise ValueError("matrix compute-unit finalization requires a before value")
    telemetry["colab_compute_units_after"] = float(after)
    telemetry["colab_compute_units_consumed"] = max(
        0.0, float(before) - float(after)
    )
    _atomic_json(summary_path, payload)
    return payload
