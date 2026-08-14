"""Validate and run the H0-015 frozen hard-negative batch screen."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from torch.utils.data import DataLoader

from src.core.hard_negative_batching import (
    EpochShuffledBatchSampler,
    build_balanced_hard_negative_batches,
    hard_negative_mapping,
)
from src.core.packet_bundle import sha256_file
from src.pipelines.initial_condition_bridge import run_initial_condition_training
from src.pipelines.oracle_experiment import load_json_object, load_yaml
from src.pipelines.packet_bridge import packet_collate
from src.pipelines.receiver_aware_replay import _lf_sha256_file


HARD_NEGATIVE_COVERAGE_PROTOCOL_VERSION = "lip-hard-negative-batches-v1"


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
        for key in (
            "batch_size",
            "gradient_accumulation_steps",
            "max_updates",
            "validation_interval",
        ):
            if current[stage_name][key] != frozen[stage_name][key]:
                raise ValueError(f"training.{stage_name}.{key} drifted from H0-013")
    if current["pilot"]["variant"] != "hard_negative_batches_unrolled":
        raise ValueError("H0-015 pilot variant drifted")
    if int(current["pilot"]["seed"]) != 4007:
        raise ValueError("H0-015 pilot seed drifted")


def validate_hard_negative_coverage_contract(
    experiment: Mapping,
    parent: Mapping,
    *,
    experiment_path: Path,
    parent_path: Path,
    predecessor_registry_path: Path,
) -> None:
    if experiment.get("experiment_id") != "LIP-H0-015":
        raise ValueError("unexpected hard-negative-coverage experiment_id")
    if experiment.get("protocol_version") != HARD_NEGATIVE_COVERAGE_PROTOCOL_VERSION:
        raise ValueError("unexpected hard-negative-coverage protocol_version")
    if experiment.get("claim_status") != "development_only_hard_negative_batch_screen":
        raise ValueError("H0-015 must remain a development-only screen")
    if parent.get("experiment_id") != "LIP-PROTO-014":
        raise ValueError("H0-015 parent must be LIP-PROTO-014")
    if experiment["parent"]["config_sha256"] != _lf_sha256_file(parent_path):
        raise ValueError("parent config hash differs from the frozen contract")

    if experiment["predecessor"]["registry_sha256"] != _lf_sha256_file(
        predecessor_registry_path
    ):
        raise ValueError("LIP-H0-014 registry differs from the frozen contract")
    predecessor = load_json_object(predecessor_registry_path)
    if predecessor.get("experiment_id") != "LIP-H0-014":
        raise ValueError("H0-015 predecessor must be LIP-H0-014")
    decision = predecessor.get("decision", {})
    if decision.get("H0_015_development_intervention_authorized") is not True:
        raise ValueError("LIP-H0-014 did not authorize H0-015")
    if decision.get("H0_015_exact_family") != "global_hard_negative_coverage":
        raise ValueError("H0-015 left the authorized intervention family")
    if predecessor.get("paired_to_H0_013_seed_4007", {}).get(
        "directional_success"
    ) is not False:
        raise ValueError("H0-015 requires the negative H0-014 directional gate")

    reference_path = _repo_path(experiment_path, experiment["reference"]["config"])
    if experiment["reference"]["config_sha256"] != _lf_sha256_file(reference_path):
        raise ValueError("H0-013 reference config differs from the frozen contract")
    reference_registry_path = _repo_path(
        experiment_path, experiment["reference"]["registry"]
    )
    if experiment["reference"]["registry_sha256"] != _lf_sha256_file(
        reference_registry_path
    ):
        raise ValueError("H0-013 reference registry differs from the frozen contract")
    reference = load_yaml(reference_path)
    reference_registry = load_json_object(reference_registry_path)
    if reference.get("experiment_id") != "LIP-H0-013":
        raise ValueError("H0-015 training reference must be H0-013")
    reference_screen = reference_registry["screen"]
    expected_reference = {
        "checkpoint_seed": 4007,
        "core_margin": float(reference_screen["regions"]["core"]["mean_diagonal_margin"]),
        "core_retrieval_top1": float(
            reference_screen["regions"]["core"]["retrieval_top1"]
        ),
        "mean_retrieval": float(reference_screen["mean_retrieval"]),
        "normalized_residual_rmse": float(
            reference_screen["normalized_residual_rmse"]
        ),
    }
    for key, expected in expected_reference.items():
        if float(experiment["reference"][key]) != float(expected):
            raise ValueError(f"reference.{key} drifted from H0-013 registry")

    diagnostic_path = _repo_path(
        experiment_path, experiment["diagnostic_source"]["registry"]
    )
    if experiment["diagnostic_source"]["registry_sha256"] != _lf_sha256_file(
        diagnostic_path
    ):
        raise ValueError("LIP-EVAL-032 registry differs from the frozen contract")
    diagnostic = load_json_object(diagnostic_path)
    candidate = experiment["diagnostic_source"]["candidate_bank"]
    observed_candidate_hash = diagnostic.get("artifacts", {}).get("full", {}).get(
        "candidate_banks", {}
    ).get("sha256")
    if observed_candidate_hash != candidate["sha256"]:
        raise ValueError("candidate-bank artifact differs from LIP-EVAL-032")
    if candidate["checkpoint_label"] != "H0_013":
        raise ValueError("H0-015 must mine the frozen H0-013 candidate bank")
    if float(candidate["diagnostic_partition_coverage"]) != 0.04296875:
        raise ValueError("diagnostic partition coverage drifted")

    for section in ("data", "receiver", "bridge", "loss"):
        if dict(experiment[section]) != dict(reference[section]):
            raise ValueError(f"{section} drifted from H0-013")
    systems = experiment["variants"]["systems"]
    if set(systems) != {"hard_negative_batches_unrolled"}:
        raise ValueError("H0-015 must contain only its frozen intervention")
    if experiment["variants"]["primary"] != "hard_negative_batches_unrolled":
        raise ValueError("hard-negative batch intervention must remain primary")
    variant = systems["hard_negative_batches_unrolled"]
    reference_variant = reference["variants"]["systems"][
        "core_margin_pressure_unrolled"
    ]
    for key in (
        "training_loss_scope",
        "lambda_entry_snapshot",
        "lambda_induced_trajectory",
    ):
        if variant[key] != reference_variant[key]:
            raise ValueError(f"variant.{key} drifted from H0-013")
    _validate_training_freeze(experiment, reference)

    policy = experiment["training"]["batch_policy"]
    expected_policy = {
        "kind": "frozen_global_hardest_balanced_partition",
        "candidate_label": "H0_013",
        "partition_seed": 4007,
        "search_restarts": 8,
        "maximum_improving_swaps_per_restart": 128,
        "expected_global_hardest_covered_anchors": 224,
        "expected_global_hardest_coverage": 0.875,
        "random_partition_expected_coverage": 15 / 255,
        "one_exposure_per_task_per_epoch": True,
        "freeze_partition_across_epochs": True,
        "shuffle_batch_order_each_epoch": True,
        "shuffle_row_order_each_epoch": True,
    }
    if dict(policy) != expected_policy:
        raise ValueError("H0-015 frozen batch policy drifted")

    gate = experiment["development_gate"]
    if (
        gate["reference_protocol"],
        int(gate["reference_seed"]),
        float(gate["reference_core_margin"]),
        float(gate["reference_core_retrieval_top1"]),
        float(gate["reference_mean_retrieval"]),
    ) != ("LIP-H0-013", 4007, 0.010905, 0.71875, 0.8125):
        raise ValueError("H0-015 paired development gate drifted")
    if experiment["confirmation"]["status"] != "prohibited_in_H0-015":
        raise ValueError("confirmation must remain prohibited")
    if experiment["compute"]["preferred_accelerator"] != "L4":
        raise ValueError("H0-015 is frozen for an L4")
    if experiment["compute"]["allow_silent_fallback"] is not False:
        raise ValueError("silent accelerator fallback is prohibited")
    if not experiment_path.is_file():
        raise ValueError("experiment contract path does not exist")


def build_hard_negative_batch_plan(
    experiment: Mapping,
    candidate_bank_path: Path,
    *,
    task_ids: Sequence[str] | None = None,
) -> tuple[list[list[int]], dict]:
    candidate = experiment["diagnostic_source"]["candidate_bank"]
    if not candidate_bank_path.is_file():
        raise FileNotFoundError(candidate_bank_path)
    observed_hash = sha256_file(candidate_bank_path)
    if observed_hash != candidate["sha256"]:
        raise ValueError("candidate-bank file hash differs from the frozen contract")
    payload = load_json_object(candidate_bank_path)
    mapping = hard_negative_mapping(payload, label=candidate["checkpoint_label"])
    ids = list(mapping) if task_ids is None else [str(task_id) for task_id in task_ids]
    policy = experiment["training"]["batch_policy"]
    batches, metadata = build_balanced_hard_negative_batches(
        ids,
        mapping,
        batch_size=int(experiment["training"]["full_matrix"]["batch_size"]),
        seed=int(policy["partition_seed"]),
        restarts=int(policy["search_restarts"]),
        max_swaps=int(policy["maximum_improving_swaps_per_restart"]),
    )
    if metadata["global_hardest_covered_anchors"] != int(
        policy["expected_global_hardest_covered_anchors"]
    ):
        raise ValueError("observed hard-negative covered-anchor count drifted")
    if metadata["global_hardest_coverage"] != float(
        policy["expected_global_hardest_coverage"]
    ):
        raise ValueError("observed hard-negative coverage drifted")
    metadata.update(
        {
            "policy_kind": policy["kind"],
            "candidate_label": candidate["checkpoint_label"],
            "candidate_bank_sha256": observed_hash,
            "candidate_bank_drive_file_id": candidate["drive_file_id"],
            "partition_frozen_across_epochs": True,
            "historical_cumulative_coverage_measured": False,
        }
    )
    return batches, metadata


def _train_loader_builder(experiment: Mapping, candidate_bank_path: Path):
    def build(dataset, *, batch_size: int, seed: int, num_workers: int):
        frozen_batch_size = int(experiment["training"]["full_matrix"]["batch_size"])
        if batch_size != frozen_batch_size:
            raise ValueError("runtime batch size differs from H0-015 batch plan")
        task_ids = [str(record["task_id"]) for record in dataset.records]
        batches, metadata = build_hard_negative_batch_plan(
            experiment, candidate_bank_path, task_ids=task_ids
        )
        sampler = EpochShuffledBatchSampler(batches, seed=int(seed))
        return (
            DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=int(num_workers),
                pin_memory=False,
                collate_fn=packet_collate,
            ),
            metadata,
        )

    return build


def run_hard_negative_coverage_training(
    *, candidate_bank_path: Path, **kwargs
) -> dict:
    experiment = load_yaml(kwargs["experiment_path"])
    return run_initial_condition_training(
        **kwargs,
        contract_validator=validate_hard_negative_coverage_contract,
        result_experiment_id="LIP-H0-015",
        result_protocol_version=HARD_NEGATIVE_COVERAGE_PROTOCOL_VERSION,
        train_loader_builder=_train_loader_builder(experiment, candidate_bank_path),
    )
