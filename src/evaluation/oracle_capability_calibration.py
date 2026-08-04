"""Frozen design helpers for capability-calibrated layer-depth replication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.evaluation.oracle_layer_depth import (
    ORACLE_LAYER_DEPTH_CONDITIONS,
    ORACLE_LAYER_DEPTH_LAYER_COUNT,
    ORACLE_LAYER_DEPTH_PACKET_SIZE,
    ORACLE_LAYER_DEPTH_SCOPE_ORDER,
    build_condition_plan as build_layer_depth_condition_plan,
    plan_as_dicts,
    validate_layer_depth_contract,
)
from src.pipelines.oracle_experiment import load_json_object, sha256_path


ORACLE_CAPABILITY_EXPERIMENT_ID = "LIP-PROTO-010"
ORACLE_CAPABILITY_PROTOCOL_VERSION = "lip-oracle-capability-calibrated-depth-v1"
ORACLE_CAPABILITY_SCREENING_SCOPE = "capability_screening"
ORACLE_CAPABILITY_SCREENING_CONDITION = "text_only_no_lip"
ORACLE_CAPABILITY_CANDIDATE_COUNT = 192
ORACLE_CAPABILITY_SELECTED_COUNT = 32
ORACLE_CAPABILITY_SCREENING_SEEDS = (17, 29)
ORACLE_CAPABILITY_CONFIRMATION_SEEDS = (401, 509, 631)
ORACLE_CAPABILITY_ELIGIBILITY_RULE = "any_functional_pass_across_screening_seeds"

ORACLE_CAPABILITY_CONDITIONS = ORACLE_LAYER_DEPTH_CONDITIONS
ORACLE_CAPABILITY_LAYER_COUNT = ORACLE_LAYER_DEPTH_LAYER_COUNT
ORACLE_CAPABILITY_PACKET_SIZE = ORACLE_LAYER_DEPTH_PACKET_SIZE
ORACLE_CAPABILITY_SCOPE_ORDER = ORACLE_LAYER_DEPTH_SCOPE_ORDER


def candidate_binding_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Project the 010 config onto the standard immutable-task binding API."""

    data = config.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("data must be a mapping")
    return {
        **config,
        "data": {
            "tasks_jsonl": data.get("candidate_tasks_jsonl"),
            "task_manifest": data.get("candidate_task_manifest"),
            "task_count": data.get("candidate_task_count"),
        },
    }


def validate_selected_task_manifest(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    """Require the confirmation registry to derive from the hardened screen."""

    data = config.get("data", {})
    output = config.get("output", {})
    candidate_manifest_path = Path(str(data.get("candidate_task_manifest", "")))
    summary_path = Path(str(output.get("screening_evaluation_dir", ""))) / (
        "summary.json"
    )
    scored_path = Path(str(output.get("screening_evaluation_dir", ""))) / (
        "scored_generations.jsonl"
    )
    report_path = Path(str(output.get("selection_report_json", "")))
    for path in (candidate_manifest_path, summary_path, scored_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = load_json_object(report_path)
    sampled_ids = [str(task_id) for task_id in manifest.get("sampled_ids", [])]
    checks = {
        "selection_kind": manifest.get("selection_kind")
        == "capability_calibrated_confirmation",
        "selected_count": len(sampled_ids) == ORACLE_CAPABILITY_SELECTED_COUNT,
        "eligibility_rule": manifest.get("eligibility_rule")
        == ORACLE_CAPABILITY_ELIGIBILITY_RULE,
        "screening_seeds": manifest.get("screening_seeds")
        == list(ORACLE_CAPABILITY_SCREENING_SEEDS),
        "eligible_prefix": manifest.get("selected_ids_are_eligible_prefix") is True,
        "disjoint": manifest.get("sampled_ids_disjoint_from_exclusions") is True,
        "candidate_path": manifest.get("candidate_manifest")
        == str(candidate_manifest_path),
        "candidate_hash": manifest.get("candidate_manifest_sha256")
        == sha256_path(candidate_manifest_path),
        "summary_path": manifest.get("screening_summary") == str(summary_path),
        "summary_hash": manifest.get("screening_summary_sha256")
        == sha256_path(summary_path),
        "scored_path": manifest.get("screening_scored_jsonl") == str(scored_path),
        "scored_hash": manifest.get("screening_scored_jsonl_sha256")
        == sha256_path(scored_path),
        "report_passed": report.get("passed") is True,
        "report_selected_ids": report.get("selected_task_ids") == sampled_ids,
        "report_manifest_path": report.get("selected_task_manifest")
        == str(manifest_path),
        "report_manifest_hash": report.get("selected_task_manifest_sha256")
        == sha256_path(manifest_path),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "selected task manifest failed calibration provenance: "
            + ", ".join(failed)
        )


def validate_capability_memory_contract(memory: Mapping) -> tuple[dict, ...]:
    """Reuse the cumulative 8/16/24/32 prefix ladder without changing it."""

    return validate_layer_depth_contract(
        memory,
        experiment_id=ORACLE_CAPABILITY_EXPERIMENT_ID,
    )


def build_condition_plan(
    task_ids: Iterable[str],
    conditions: Iterable[str],
    *,
    shuffle_seed: int,
):
    return build_layer_depth_condition_plan(
        task_ids,
        conditions,
        shuffle_seed=shuffle_seed,
    )


def design_fingerprint(config: Mapping) -> str:
    """Bind both calibration and confirmation generation choices."""

    payload = {
        "protocol_version": ORACLE_CAPABILITY_PROTOCOL_VERSION,
        "experiment_id": config.get("experiment_id"),
        "predecessor_experiment": config.get("predecessor_experiment"),
        "models": config.get("models", {}),
        "prompt_protocol": config.get("prompt_protocol", {}),
        "runtime": config.get("runtime", {}),
        "data": config.get("data", {}),
        "screening": config.get("screening", {}),
        "neutral_target_prompt": config.get("neutral_target_prompt"),
        "carrier": config.get("carrier", {}),
        "memory": config.get("memory", {}),
        "diagnostics": config.get("diagnostics", {}),
        "conditions": config.get("conditions", []),
        "controls": config.get("controls", {}),
        "generation": config.get("generation", {}),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def primary_fixed_sequence() -> tuple[tuple[str, str], ...]:
    """Test the prospective 24 -> 16 -> 8 early-prefix hypothesis family."""

    return tuple(
        (
            f"oracle_{scope}_k{ORACLE_CAPABILITY_PACKET_SIZE}",
            f"shuffled_oracle_{scope}_k{ORACLE_CAPABILITY_PACKET_SIZE}",
        )
        for scope in (
            "early_three_quarters_input",
            "early_half_input",
            "early_quarter_input",
        )
    )


def eligible_task_ids(
    records: Sequence[Mapping],
    candidate_task_ids: Sequence[str],
    *,
    screening_seeds: Sequence[int] = ORACLE_CAPABILITY_SCREENING_SEEDS,
) -> list[str]:
    """Return eligible IDs in the immutable candidate-manifest order."""

    task_ids = [str(task_id) for task_id in candidate_task_ids]
    seeds = [int(seed) for seed in screening_seeds]
    if len(task_ids) != ORACLE_CAPABILITY_CANDIDATE_COUNT:
        raise ValueError(
            f"capability screening requires {ORACLE_CAPABILITY_CANDIDATE_COUNT} candidates"
        )
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("candidate task IDs must be unique")
    if tuple(seeds) != ORACLE_CAPABILITY_SCREENING_SEEDS:
        raise ValueError("screening seeds do not match the frozen calibration design")

    by_key: dict[tuple[str, int], Mapping] = {}
    for row in records:
        if row.get("condition") != ORACLE_CAPABILITY_SCREENING_CONDITION:
            raise ValueError("screening contains a non-text condition")
        key = (str(row.get("task_id", "")), int(row.get("generation_seed", -1)))
        if key in by_key:
            raise ValueError(f"duplicate screening record: {key}")
        if not isinstance(row.get("functional_pass"), bool):
            raise ValueError("every screening record needs a boolean functional_pass")
        by_key[key] = row

    expected = {(task_id, seed) for task_id in task_ids for seed in seeds}
    if set(by_key) != expected:
        missing = len(expected.difference(by_key))
        unexpected = len(set(by_key).difference(expected))
        raise ValueError(
            "screening grid mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return [
        task_id
        for task_id in task_ids
        if any(bool(by_key[(task_id, seed)]["functional_pass"]) for seed in seeds)
    ]


def semantic_gate(
    condition_means: Mapping[str, float],
    primary_inference: Mapping,
) -> dict:
    """Apply the prospective task-specific early-prefix decision rule."""

    missing = sorted(set(ORACLE_CAPABILITY_CONDITIONS).difference(condition_means))
    if missing:
        raise ValueError(
            f"capability-calibrated gate is missing condition(s): {', '.join(missing)}"
        )
    means = {
        condition: float(condition_means[condition])
        for condition in ORACLE_CAPABILITY_CONDITIONS
    }
    hypotheses = primary_inference.get("hypotheses")
    if not isinstance(hypotheses, list):
        raise ValueError("capability gate requires fixed-sequence hypotheses")
    rejected = {
        str(item.get("treatment")): bool(item.get("rejected"))
        for item in hypotheses
        if isinstance(item, Mapping)
    }
    expected_treatments = {treatment for treatment, _ in primary_fixed_sequence()}
    if set(rejected) != expected_treatments:
        raise ValueError("primary inference does not match the frozen 24 -> 16 -> 8 order")

    scope_checks = {}
    supported = []
    for scope in (
        "early_quarter_input",
        "early_half_input",
        "early_three_quarters_input",
    ):
        matched = f"oracle_{scope}_k{ORACLE_CAPABILITY_PACKET_SIZE}"
        shuffled = f"shuffled_oracle_{scope}_k{ORACLE_CAPABILITY_PACKET_SIZE}"
        checks = {
            "beats_neutral": means[matched] > means["neutral_no_lip"],
            "beats_task_mismatched": means[matched] > means[shuffled],
            "fixed_sequence_rejected": rejected[matched],
        }
        passed = all(checks.values())
        scope_checks[scope] = {"checks": checks, "passed": passed}
        if passed:
            supported.append(scope)

    all_layer_matched = "oracle_all_layer_input_k32"
    all_layer_shuffled = "shuffled_oracle_all_layer_input_k32"
    all_layer_anchor = {
        "beats_neutral": means[all_layer_matched] > means["neutral_no_lip"],
        "beats_task_mismatched": means[all_layer_matched]
        > means[all_layer_shuffled],
    }
    checks = {
        "text_control_nonzero": means["text_only_no_lip"] > 0.0,
        "early_three_quarters_confirmed": scope_checks[
            "early_three_quarters_input"
        ]["passed"],
    }
    return {
        "metric": "functional_pass",
        "condition_means": means,
        "scope_checks": scope_checks,
        "supported_scopes": supported,
        "smallest_confirmed_scope": supported[0] if supported else None,
        "all_layer_descriptive_anchor": {
            "checks": all_layer_anchor,
            "passed": all(all_layer_anchor.values()),
            "confirmatory_family_member": False,
        },
        "primary_inference": dict(primary_inference),
        "checks": checks,
        "passed": all(checks.values()),
    }


__all__ = [
    "ORACLE_CAPABILITY_CANDIDATE_COUNT",
    "ORACLE_CAPABILITY_CONDITIONS",
    "ORACLE_CAPABILITY_CONFIRMATION_SEEDS",
    "ORACLE_CAPABILITY_ELIGIBILITY_RULE",
    "ORACLE_CAPABILITY_EXPERIMENT_ID",
    "ORACLE_CAPABILITY_LAYER_COUNT",
    "ORACLE_CAPABILITY_PACKET_SIZE",
    "ORACLE_CAPABILITY_PROTOCOL_VERSION",
    "ORACLE_CAPABILITY_SCOPE_ORDER",
    "ORACLE_CAPABILITY_SCREENING_CONDITION",
    "ORACLE_CAPABILITY_SCREENING_SCOPE",
    "ORACLE_CAPABILITY_SCREENING_SEEDS",
    "ORACLE_CAPABILITY_SELECTED_COUNT",
    "build_condition_plan",
    "candidate_binding_config",
    "design_fingerprint",
    "eligible_task_ids",
    "plan_as_dicts",
    "primary_fixed_sequence",
    "semantic_gate",
    "validate_selected_task_manifest",
    "validate_capability_memory_contract",
]
