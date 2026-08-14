"""Validate and run the H0-014 core-only trajectory-margin screen."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from src.pipelines.initial_condition_bridge import run_initial_condition_training
from src.pipelines.oracle_experiment import load_json_object, load_yaml
from src.pipelines.receiver_aware_replay import _lf_sha256_file


CORE_ONLY_MARGIN_PROTOCOL_VERSION = "lip-core-only-margin-v1"


def _repo_path(experiment_path: Path, relative: str) -> Path:
    return experiment_path.resolve().parents[1] / relative


def _validate_training_freeze(experiment: Mapping, reference: Mapping) -> None:
    current = experiment["training"]
    frozen = reference["training"]
    for key in (
        "seeds",
        "learning_rate",
        "weight_decay",
        "gradient_clip",
        "fp16_autocast",
        "num_workers",
    ):
        if current[key] != frozen[key]:
            raise ValueError(f"training.{key} drifted from H0-013")
    for stage_name in ("pilot", "full_matrix"):
        stage = current[stage_name]
        frozen_stage = frozen[stage_name]
        for key in (
            "batch_size",
            "gradient_accumulation_steps",
            "max_updates",
            "validation_interval",
        ):
            if stage[key] != frozen_stage[key]:
                raise ValueError(f"training.{stage_name}.{key} drifted from H0-013")
    pilot = current["pilot"]
    if pilot["variant"] != "core_only_margin_unrolled" or int(pilot["seed"]) != 4007:
        raise ValueError("H0-014 pilot identity drifted")


def validate_core_only_margin_contract(
    experiment: Mapping,
    parent: Mapping,
    *,
    experiment_path: Path,
    parent_path: Path,
    predecessor_registry_path: Path,
) -> None:
    if experiment.get("experiment_id") != "LIP-H0-014":
        raise ValueError("unexpected core-only-margin experiment_id")
    if experiment.get("protocol_version") != CORE_ONLY_MARGIN_PROTOCOL_VERSION:
        raise ValueError("unexpected core-only-margin protocol_version")
    if experiment.get("claim_status") != "development_only_core_only_margin_screen":
        raise ValueError("H0-014 must remain a development-only screen")
    if parent.get("experiment_id") != "LIP-PROTO-014":
        raise ValueError("H0-014 parent must be LIP-PROTO-014")
    if experiment["parent"]["config_sha256"] != _lf_sha256_file(parent_path):
        raise ValueError("parent config hash differs from the frozen contract")

    predecessor = experiment["predecessor"]
    if predecessor["protocol"] != "LIP-EVAL-032":
        raise ValueError("H0-014 predecessor must be LIP-EVAL-032")
    if predecessor["registry_sha256"] != _lf_sha256_file(predecessor_registry_path):
        raise ValueError("LIP-EVAL-032 registry differs from the frozen contract")
    diagnostic = load_json_object(predecessor_registry_path)
    if diagnostic.get("experiment_id") != "LIP-EVAL-032":
        raise ValueError("predecessor registry is not LIP-EVAL-032")
    if diagnostic.get("routing", {}).get("selected") != "scale_limited":
        raise ValueError("LIP-EVAL-032 did not select the scale-limited route")
    decision = diagnostic.get("decision", {})
    if decision.get("H0_014_development_intervention_authorized") is not True:
        raise ValueError("LIP-EVAL-032 did not authorize H0-014")
    if decision.get("exact_family") != "explicit_core_only_or_adaptive_gradient_weight":
        raise ValueError("H0-014 left the authorized intervention family")

    reference_path = _repo_path(experiment_path, experiment["reference"]["config"])
    if experiment["reference"]["config_sha256"] != _lf_sha256_file(reference_path):
        raise ValueError("H0-013 reference config differs from the frozen contract")
    reference = load_yaml(reference_path)
    if reference.get("experiment_id") != "LIP-H0-013":
        raise ValueError("H0-014 reference must be H0-013")
    reference_registry = load_json_object(
        _repo_path(
            experiment_path,
            "experiments/registry/LIP-H0-013_core_margin_pressure.json",
        )
    )
    reference_screen = reference_registry["screen"]
    expected_reference = {
        "checkpoint_seed": 4007,
        "core_margin": float(reference_screen["regions"]["core"]["mean_diagonal_margin"]),
        "core_retrieval_top1": float(
            reference_screen["regions"]["core"]["retrieval_top1"]
        ),
        "mean_retrieval": float(reference_screen["mean_retrieval"]),
        "normalized_residual_rmse": float(reference_screen["normalized_residual_rmse"]),
    }
    for key, expected in expected_reference.items():
        observed = experiment["reference"][key]
        if float(observed) != float(expected):
            raise ValueError(f"reference.{key} drifted from H0-013 registry")

    for section in ("data", "receiver", "bridge"):
        if dict(experiment[section]) != dict(reference[section]):
            raise ValueError(f"{section} drifted from H0-013")
    systems = experiment["variants"]["systems"]
    if set(systems) != {"core_only_margin_unrolled"}:
        raise ValueError("H0-014 must contain only the core-only intervention")
    if experiment["variants"]["primary"] != "core_only_margin_unrolled":
        raise ValueError("core-only intervention must remain primary")
    variant = systems["core_only_margin_unrolled"]
    if (
        float(variant["lambda_entry_snapshot"]),
        float(variant["lambda_induced_trajectory"]),
    ) != (0.25, 1.0):
        raise ValueError("H0-014 variant weights drifted from H0-013")

    if dict(experiment["loss"]["entry_snapshot"]) != dict(
        reference["loss"]["entry_snapshot"]
    ):
        raise ValueError("entry snapshot loss drifted from H0-013")
    induced = dict(experiment["loss"]["induced_trajectory"])
    margin_region_weights = induced.pop("margin_region_weights", None)
    if induced != dict(reference["loss"]["induced_trajectory"]):
        raise ValueError("non-regional induced loss drifted from H0-013")
    if margin_region_weights != {"joint": 0.0, "core": 1.0, "name": 0.0}:
        raise ValueError("H0-014 induced margin must remain core-only")

    _validate_training_freeze(experiment, reference)
    gate = experiment["development_gate"]
    if (
        gate["reference_protocol"],
        int(gate["reference_seed"]),
        float(gate["reference_core_margin"]),
        float(gate["reference_core_retrieval_top1"]),
        float(gate["reference_mean_retrieval"]),
    ) != ("LIP-H0-013", 4007, 0.010905, 0.71875, 0.8125):
        raise ValueError("H0-014 paired development gate drifted")
    if experiment["confirmation"]["status"] != "prohibited_in_H0-014":
        raise ValueError("confirmation must remain prohibited")
    if experiment["compute"]["preferred_accelerator"] != "L4":
        raise ValueError("H0-014 is frozen for an L4")
    if experiment["compute"]["allow_silent_fallback"] is not False:
        raise ValueError("silent accelerator fallback is prohibited")
    if not experiment_path.is_file():
        raise ValueError("experiment contract path does not exist")


def run_core_only_margin_training(**kwargs) -> dict:
    return run_initial_condition_training(
        **kwargs,
        contract_validator=validate_core_only_margin_contract,
        result_experiment_id="LIP-H0-014",
        result_protocol_version=CORE_ONLY_MARGIN_PROTOCOL_VERSION,
    )
