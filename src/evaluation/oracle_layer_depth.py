"""Frozen design helpers for oracle layer-depth localization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

from src.evaluation.source_only import derangement_indices


ORACLE_LAYER_DEPTH_PROTOCOL_VERSION = "lip-oracle-layer-depth-v1"
ORACLE_LAYER_DEPTH_PACKET_SIZE = 32
ORACLE_LAYER_DEPTH_LAYER_COUNT = 32
ORACLE_LAYER_DEPTH_SCOPE_ORDER = (
    "early_quarter_input",
    "early_half_input",
    "early_three_quarters_input",
    "all_layer_input",
)
ORACLE_LAYER_DEPTH_SCOPE_CONTRACT = (
    {
        "name": "early_quarter_input",
        "boundary": "block_input",
        "layers": list(range(-32, -24)),
    },
    {
        "name": "early_half_input",
        "boundary": "block_input",
        "layers": list(range(-32, -16)),
    },
    {
        "name": "early_three_quarters_input",
        "boundary": "block_input",
        "layers": list(range(-32, -8)),
    },
    {
        "name": "all_layer_input",
        "boundary": "block_input",
        "layers": list(range(-32, 0)),
    },
)


def expected_layer_depth_conditions(
    scope_names: Sequence[str] = ORACLE_LAYER_DEPTH_SCOPE_ORDER,
) -> tuple[str, ...]:
    names = tuple(str(name) for name in scope_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("layer-depth scope names must be a non-empty unique sequence")
    conditions = ["neutral_no_lip", "text_only_no_lip"]
    for name in names:
        conditions.extend(
            (
                f"oracle_{name}_k{ORACLE_LAYER_DEPTH_PACKET_SIZE}",
                f"shuffled_oracle_{name}_k{ORACLE_LAYER_DEPTH_PACKET_SIZE}",
            )
        )
    return tuple(conditions)


ORACLE_LAYER_DEPTH_CONDITIONS = expected_layer_depth_conditions()


def validate_layer_depth_contract(memory: Mapping) -> tuple[dict, ...]:
    if not isinstance(memory, Mapping):
        raise ValueError("memory must be a mapping")
    if int(memory.get("packet_size", 0)) != ORACLE_LAYER_DEPTH_PACKET_SIZE:
        raise ValueError("LIP-PROTO-009 freezes memory.packet_size=32")
    if int(memory.get("decoder_layer_count", 0)) != ORACLE_LAYER_DEPTH_LAYER_COUNT:
        raise ValueError("LIP-PROTO-009 freezes a 32-layer target decoder")
    if int(memory.get("self_check_tasks", 0)) != 1:
        raise ValueError("LIP-PROTO-009 freezes memory.self_check_tasks=1")
    if float(memory.get("maximum_self_logit_delta", -1.0)) != 0.0001:
        raise ValueError(
            "LIP-PROTO-009 freezes memory.maximum_self_logit_delta=0.0001"
        )
    scopes = memory.get("scopes")
    if not isinstance(scopes, list):
        raise ValueError("memory.scopes must be a list")
    normalized = tuple(
        {
            "name": str(scope.get("name", "")),
            "boundary": str(scope.get("boundary", "")),
            "layers": [int(layer) for layer in scope.get("layers", [])],
        }
        for scope in scopes
        if isinstance(scope, Mapping)
    )
    if (
        len(normalized) != len(scopes)
        or normalized != ORACLE_LAYER_DEPTH_SCOPE_CONTRACT
    ):
        raise ValueError("memory.scopes must match the frozen layer-depth contract")
    return normalized


def design_fingerprint(config: Mapping) -> str:
    payload = {
        "protocol_version": ORACLE_LAYER_DEPTH_PROTOCOL_VERSION,
        "experiment_id": config.get("experiment_id"),
        "predecessor_experiment": config.get("predecessor_experiment"),
        "models": config.get("models", {}),
        "prompt_protocol": config.get("prompt_protocol", {}),
        "data": config.get("data", {}),
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


@dataclass(frozen=True)
class OracleLayerDepthCondition:
    task_id: str
    task_index: int
    condition: str
    target_prompt_kind: str
    scope_name: str | None
    oracle_index: int | None


def build_condition_plan(
    task_ids: Iterable[str],
    conditions: Iterable[str],
    *,
    shuffle_seed: int,
) -> list[OracleLayerDepthCondition]:
    ids = [str(task_id) for task_id in task_ids]
    selected = list(conditions)
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("task_ids must contain at least two unique tasks")
    if selected != list(ORACLE_LAYER_DEPTH_CONDITIONS):
        raise ValueError("conditions must match the frozen layer-depth design")
    shuffled = derangement_indices(len(ids), int(shuffle_seed))
    matched = {
        f"oracle_{scope}_k{ORACLE_LAYER_DEPTH_PACKET_SIZE}": scope
        for scope in ORACLE_LAYER_DEPTH_SCOPE_ORDER
    }
    mismatched = {
        f"shuffled_oracle_{scope}_k{ORACLE_LAYER_DEPTH_PACKET_SIZE}": scope
        for scope in ORACLE_LAYER_DEPTH_SCOPE_ORDER
    }
    plan = []
    for task_index, task_id in enumerate(ids):
        for condition in selected:
            scope_name = None
            oracle_index = None
            if condition in matched:
                scope_name = matched[condition]
                oracle_index = task_index
            elif condition in mismatched:
                scope_name = mismatched[condition]
                oracle_index = shuffled[task_index]
            plan.append(
                OracleLayerDepthCondition(
                    task_id=task_id,
                    task_index=task_index,
                    condition=condition,
                    target_prompt_kind=(
                        "task" if condition == "text_only_no_lip" else "neutral"
                    ),
                    scope_name=scope_name,
                    oracle_index=oracle_index,
                )
            )
    return plan


def plan_as_dicts(plan: Iterable[OracleLayerDepthCondition]) -> list[dict]:
    return [asdict(item) for item in plan]


def primary_fixed_sequence() -> tuple[tuple[str, str], ...]:
    """Return the preregistered descending-depth primary hypothesis order."""

    return tuple(
        (
            f"oracle_{scope}_k{ORACLE_LAYER_DEPTH_PACKET_SIZE}",
            f"shuffled_oracle_{scope}_k{ORACLE_LAYER_DEPTH_PACKET_SIZE}",
        )
        for scope in reversed(ORACLE_LAYER_DEPTH_SCOPE_ORDER)
    )


def semantic_gate(
    condition_means: Mapping[str, float],
    primary_inference: Mapping,
) -> dict:
    missing = sorted(set(ORACLE_LAYER_DEPTH_CONDITIONS).difference(condition_means))
    if missing:
        raise ValueError(
            f"layer-depth gate is missing condition(s): {', '.join(missing)}"
        )
    means = {
        condition: float(condition_means[condition])
        for condition in ORACLE_LAYER_DEPTH_CONDITIONS
    }
    hypotheses = primary_inference.get("hypotheses")
    if not isinstance(hypotheses, list):
        raise ValueError("layer-depth gate requires fixed-sequence hypotheses")
    rejected = {
        str(item.get("treatment")): bool(item.get("rejected"))
        for item in hypotheses
        if isinstance(item, Mapping)
    }
    expected_treatments = {treatment for treatment, _ in primary_fixed_sequence()}
    if set(rejected) != expected_treatments:
        raise ValueError("primary inference does not match the frozen depth sequence")

    scope_checks = {}
    supported = []
    for scope in ORACLE_LAYER_DEPTH_SCOPE_ORDER:
        matched_name = f"oracle_{scope}_k{ORACLE_LAYER_DEPTH_PACKET_SIZE}"
        shuffled_name = (
            f"shuffled_oracle_{scope}_k{ORACLE_LAYER_DEPTH_PACKET_SIZE}"
        )
        checks = {
            "beats_neutral": means[matched_name] > means["neutral_no_lip"],
            "beats_task_mismatched": means[matched_name] > means[shuffled_name],
            "fixed_sequence_rejected": rejected[matched_name],
        }
        passed = all(checks.values())
        scope_checks[scope] = {"checks": checks, "passed": passed}
        if passed:
            supported.append(scope)

    replication_passed = scope_checks["all_layer_input"]["passed"]
    checks = {
        "text_control_nonzero": means["text_only_no_lip"] > 0.0,
        "all_layer_replication_passed": replication_passed,
    }
    return {
        "metric": "functional_pass",
        "condition_means": means,
        "scope_checks": scope_checks,
        "supported_scopes": supported,
        "minimum_supported_scope": supported[0] if supported else None,
        "primary_inference": dict(primary_inference),
        "checks": checks,
        "passed": all(checks.values()),
    }
