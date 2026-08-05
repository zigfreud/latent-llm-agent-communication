"""Frozen helpers for the LIP-PROTO-013 terminal-source factorial."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.evaluation.source_only import derangement_indices
from src.pipelines.oracle_experiment import load_json_object, sha256_path


ORACLE_TERMINAL_EXPERIMENT_ID = "LIP-PROTO-013"
ORACLE_TERMINAL_PREDECESSOR = "LIP-PROTO-012"
ORACLE_TERMINAL_PROTOCOL_VERSION = "lip-oracle-terminal-source-factorial-v1"
ORACLE_TERMINAL_SCREENING_SCOPE = "terminal_layout_capability_screening"
ORACLE_TERMINAL_SCREENING_CONDITION = "text_only_no_lip"
ORACLE_TERMINAL_SCREENING_SEEDS = (1423, 1559)
ORACLE_TERMINAL_CONFIRMATION_SEEDS = (1667, 1789, 1901)
ORACLE_TERMINAL_CANDIDATE_COUNT = 179
ORACLE_TERMINAL_SELECTED_PER_STRATUM = 16
ORACLE_TERMINAL_SELECTED_COUNT = 32
ORACLE_TERMINAL_ELIGIBILITY_RULE = "any_functional_pass_across_screening_seeds"
ORACLE_TERMINAL_LAYER_COUNT = 32
ORACLE_TERMINAL_CAPTURE_SIZE = 32
ORACLE_TERMINAL_SCOPE_NAME = "early_quarter_input"
ORACLE_TERMINAL_CAPTURE_LAYERS = tuple(range(-32, 0))
ORACLE_TERMINAL_REPLAY_LAYERS = tuple(range(-32, -24))
ORACLE_TERMINAL_FULL_OFFSETS = tuple(range(-32, 0))
ORACLE_TERMINAL_TAIL_OFFSETS = tuple(range(-24, 0))
ORACLE_TERMINAL_BOUNDARY_OFFSETS = tuple(range(-6, 0))
ORACLE_TERMINAL_NAME_OFFSETS = {
    2: (-8, -7),
    3: (-9, -8, -7),
}
ORACLE_TERMINAL_CORE_OFFSETS = {
    2: tuple(range(-24, -8)),
    3: tuple(range(-24, -9)),
}
ORACLE_TERMINAL_COMPONENT_ORDER = ("core", "name", "boundary")
ORACLE_TERMINAL_ASSIGNMENTS = (
    "mmm",
    "smm",
    "msm",
    "mms",
    "ssm",
    "sms",
    "mss",
    "sss",
)
ORACLE_TERMINAL_PATTERN_CONTRACT = (
    {"name": "full_k32", "packet_offsets": list(ORACLE_TERMINAL_FULL_OFFSETS)},
    {"name": "terminal_k24", "packet_offsets": list(ORACLE_TERMINAL_TAIL_OFFSETS)},
)


def _factorial_condition(assignment: str) -> str:
    return (
        f"oracle_{ORACLE_TERMINAL_SCOPE_NAME}_terminal_k24_{assignment}"
    )


ORACLE_TERMINAL_FULL_MATCHED = (
    f"oracle_{ORACLE_TERMINAL_SCOPE_NAME}_full_k32"
)
ORACLE_TERMINAL_FULL_SHUFFLED = (
    f"shuffled_oracle_{ORACLE_TERMINAL_SCOPE_NAME}_full_k32"
)
ORACLE_TERMINAL_CONDITIONS = (
    "neutral_no_lip",
    "text_only_no_lip",
    ORACLE_TERMINAL_FULL_MATCHED,
    ORACLE_TERMINAL_FULL_SHUFFLED,
    *(_factorial_condition(value) for value in ORACLE_TERMINAL_ASSIGNMENTS),
)


def terminal_components(name_token_count: int) -> dict[str, tuple[int, ...]]:
    """Return the exhaustive 24-position component partition for one stratum."""

    count = int(name_token_count)
    if count not in ORACLE_TERMINAL_NAME_OFFSETS:
        raise ValueError("name_token_count must be 2 or 3")
    components = {
        "core": ORACLE_TERMINAL_CORE_OFFSETS[count],
        "name": ORACLE_TERMINAL_NAME_OFFSETS[count],
        "boundary": ORACLE_TERMINAL_BOUNDARY_OFFSETS,
    }
    flattened = tuple(
        offset
        for component in ORACLE_TERMINAL_COMPONENT_ORDER
        for offset in components[component]
    )
    if flattened != ORACLE_TERMINAL_TAIL_OFFSETS:
        raise RuntimeError("terminal components do not partition the K=24 tail")
    return components


def validate_terminal_layout(layout: Mapping[str, Any]) -> int:
    """Validate task-level tokenizer metadata and return its stratum."""

    if not isinstance(layout, Mapping):
        raise ValueError("terminal_layout must be a mapping")
    count = int(layout.get("name_token_count", 0))
    components = terminal_components(count)
    expected = {
        "name_token_count": count,
        "core_offsets": list(components["core"]),
        "name_offsets": list(components["name"]),
        "boundary_offsets": list(components["boundary"]),
        "tail_offsets": list(ORACLE_TERMINAL_TAIL_OFFSETS),
    }
    for field, value in expected.items():
        if layout.get(field) != value:
            raise ValueError(f"terminal_layout.{field} violates the frozen stratum")
    selection_hash = str(layout.get("selection_hash", ""))
    if len(selection_hash) != 64 or any(
        character not in "0123456789abcdef" for character in selection_hash
    ):
        raise ValueError("terminal_layout.selection_hash must be SHA-256")
    return count


def terminal_patterns(memory: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    raw = memory.get("position_patterns")
    if not isinstance(raw, list):
        raise ValueError("memory.position_patterns must be a list")
    normalized = tuple(
        {
            "name": str(item.get("name", "")),
            "packet_offsets": [int(offset) for offset in item.get("packet_offsets", [])],
        }
        for item in raw
        if isinstance(item, Mapping)
    )
    if len(normalized) != len(raw) or normalized != ORACLE_TERMINAL_PATTERN_CONTRACT:
        raise ValueError("position patterns do not match the terminal factorial")
    return {
        item["name"]: tuple(item["packet_offsets"])
        for item in normalized
    }


def validate_terminal_design(design: Mapping[str, Any]) -> None:
    expected = {
        "method": "constant_capacity_source_identity_factorial",
        "factors": list(ORACLE_TERMINAL_COMPONENT_ORDER),
        "levels": {"m": "matched_task", "s": "same_stratum_donor"},
        "assignments": list(ORACLE_TERMINAL_ASSIGNMENTS),
        "packet_size": 24,
        "same_donor_across_s_components": True,
        "strata": [
            {
                "name_token_count": count,
                "selected_task_count": ORACLE_TERMINAL_SELECTED_PER_STRATUM,
                "components": {
                    name: list(offsets)
                    for name, offsets in terminal_components(count).items()
                },
            }
            for count in (2, 3)
        ],
    }
    if dict(design) != expected:
        raise ValueError("terminal_factorial must match the frozen 2x2x2 design")


def validate_terminal_memory_contract(memory: Mapping[str, Any]) -> tuple[dict, ...]:
    expected_scalars = {
        "packet_size": ORACLE_TERMINAL_CAPTURE_SIZE,
        "decoder_layer_count": ORACLE_TERMINAL_LAYER_COUNT,
        "self_check_tasks": 1,
        "maximum_self_logit_delta": 0.0001,
    }
    for field, expected in expected_scalars.items():
        if memory.get(field) != expected:
            raise ValueError(f"LIP-PROTO-013 freezes memory.{field}={expected}")
    if [int(value) for value in memory.get("state_capture_layers", [])] != list(
        ORACLE_TERMINAL_CAPTURE_LAYERS
    ):
        raise ValueError("state_capture_layers must cover all 32 decoder blocks")
    scope = {
        "name": ORACLE_TERMINAL_SCOPE_NAME,
        "boundary": "block_input",
        "layers": list(ORACLE_TERMINAL_REPLAY_LAYERS),
    }
    if memory.get("replay_scope") != scope:
        raise ValueError("replay_scope must freeze the confirmed first 8 blocks")
    terminal_patterns(memory)
    return (scope,)


def candidate_binding_config(config: Mapping[str, Any]) -> dict[str, Any]:
    data = config.get("data", {})
    return {
        **config,
        "data": {
            "tasks_jsonl": data.get("candidate_tasks_jsonl"),
            "task_manifest": data.get("candidate_task_manifest"),
            "task_count": data.get("candidate_task_count"),
        },
    }


def _ids_sha256(task_ids: Sequence[str]) -> str:
    payload = json.dumps(list(task_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def eligible_task_ids(
    records: Sequence[Mapping[str, Any]],
    candidate_tasks: Sequence[Mapping[str, Any]],
) -> dict[int, list[str]]:
    """Return capable candidates in frozen within-stratum hash order."""

    if len(candidate_tasks) != ORACLE_TERMINAL_CANDIDATE_COUNT:
        raise ValueError("terminal screening requires exactly 179 candidates")
    ordered = sorted(
        candidate_tasks,
        key=lambda task: (
            validate_terminal_layout(task.get("terminal_layout", {})),
            str(task["terminal_layout"]["selection_hash"]),
            str(task["task_id"]),
        ),
    )
    task_ids = [str(task["task_id"]) for task in ordered]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("candidate task IDs must be unique")
    expected = {
        (task_id, seed)
        for task_id in task_ids
        for seed in ORACLE_TERMINAL_SCREENING_SEEDS
    }
    observed: dict[tuple[str, int], bool] = {}
    for row in records:
        if row.get("condition") != ORACLE_TERMINAL_SCREENING_CONDITION:
            raise ValueError("terminal screening contains a non-text condition")
        key = (str(row.get("task_id", "")), int(row.get("generation_seed", -1)))
        if key in observed:
            raise ValueError(f"duplicate screening record: {key}")
        if not isinstance(row.get("functional_pass"), bool):
            raise ValueError("every screening record needs boolean functional_pass")
        observed[key] = bool(row["functional_pass"])
    if set(observed) != expected:
        raise ValueError("terminal screening grid does not match the candidate registry")
    by_stratum = {2: [], 3: []}
    task_by_id = {str(task["task_id"]): task for task in ordered}
    for task_id in task_ids:
        if any(observed[(task_id, seed)] for seed in ORACLE_TERMINAL_SCREENING_SEEDS):
            count = validate_terminal_layout(task_by_id[task_id]["terminal_layout"])
            by_stratum[count].append(task_id)
    return by_stratum


def validate_selected_task_manifest(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    data = config.get("data", {})
    output = config.get("output", {})
    candidate_path = Path(str(data.get("candidate_task_manifest", "")))
    summary_path = Path(str(output.get("screening_evaluation_dir", ""))) / "summary.json"
    scored_path = (
        Path(str(output.get("screening_evaluation_dir", "")))
        / "scored_generations.jsonl"
    )
    report_path = Path(str(output.get("selection_report_json", "")))
    for path in (candidate_path, summary_path, scored_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = load_json_object(report_path)
    sampled = [str(task_id) for task_id in manifest.get("sampled_ids", [])]
    strata = manifest.get("selected_task_ids_by_name_token_count", {})
    expected_sampled = [
        *[str(task_id) for task_id in strata.get("2", [])],
        *[str(task_id) for task_id in strata.get("3", [])],
    ] if isinstance(strata, Mapping) else []
    checks = {
        "selection_kind": manifest.get("selection_kind")
        == "terminal_layout_stratified_capability_confirmation",
        "selected_count": len(sampled) == ORACLE_TERMINAL_SELECTED_COUNT,
        "balanced": isinstance(strata, Mapping)
        and all(
            len(strata.get(str(count), [])) == ORACLE_TERMINAL_SELECTED_PER_STRATUM
            for count in (2, 3)
        ),
        "stratum_order": sampled == expected_sampled,
        "eligibility_rule": manifest.get("eligibility_rule")
        == ORACLE_TERMINAL_ELIGIBILITY_RULE,
        "screening_seeds": manifest.get("screening_seeds")
        == list(ORACLE_TERMINAL_SCREENING_SEEDS),
        "candidate_path": manifest.get("candidate_manifest") == str(candidate_path),
        "candidate_hash": manifest.get("candidate_manifest_sha256")
        == sha256_path(candidate_path),
        "summary_path": manifest.get("screening_summary") == str(summary_path),
        "summary_hash": manifest.get("screening_summary_sha256")
        == sha256_path(summary_path),
        "scored_path": manifest.get("screening_scored_jsonl") == str(scored_path),
        "scored_hash": manifest.get("screening_scored_jsonl_sha256")
        == sha256_path(scored_path),
        "report_passed": report.get("passed") is True,
        "report_ids": report.get("selected_task_ids") == sampled,
        "report_manifest": report.get("selected_task_manifest")
        == str(manifest_path),
        "report_manifest_hash": report.get("selected_task_manifest_sha256")
        == sha256_path(manifest_path),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "selected task manifest failed 013 provenance: " + ", ".join(failed)
        )


@dataclass(frozen=True)
class OracleTerminalCondition:
    task_id: str
    task_index: int
    condition: str
    target_prompt_kind: str
    scope_name: str | None
    oracle_index: int | None
    position_pattern: str | None
    packet_offsets: tuple[int, ...] | None
    component_oracle_indices: tuple[int, int, int] | None
    component_sources: tuple[str, str, str] | None
    component_offsets: tuple[tuple[int, ...], ...] | None


def _stratified_donors(
    tasks: Sequence[Mapping[str, Any]], shuffle_seed: int
) -> dict[int, int]:
    strata: dict[int, list[int]] = {2: [], 3: []}
    for index, task in enumerate(tasks):
        count = validate_terminal_layout(task.get("terminal_layout", {}))
        strata[count].append(index)
    if any(
        len(indices) != ORACLE_TERMINAL_SELECTED_PER_STRATUM
        for indices in strata.values()
    ):
        raise ValueError("confirmation tasks must contain balanced 16-task strata")
    donors = {}
    for count, indices in strata.items():
        permutation = derangement_indices(len(indices), int(shuffle_seed) + count)
        donors.update(
            {
                target_index: indices[permutation[local_index]]
                for local_index, target_index in enumerate(indices)
            }
        )
    return donors


def build_condition_plan(
    tasks: Iterable[Mapping[str, Any]],
    conditions: Iterable[str],
    *,
    shuffle_seed: int,
) -> list[OracleTerminalCondition]:
    task_list = list(tasks)
    ids = [str(task["task_id"]) for task in task_list]
    if len(ids) != ORACLE_TERMINAL_SELECTED_COUNT or len(set(ids)) != len(ids):
        raise ValueError("terminal plan requires 32 unique selected tasks")
    selected_conditions = list(conditions)
    if selected_conditions != list(ORACLE_TERMINAL_CONDITIONS):
        raise ValueError("conditions must match the terminal source factorial")
    donors = _stratified_donors(task_list, int(shuffle_seed))
    plan = []
    for task_index, task in enumerate(task_list):
        task_id = ids[task_index]
        donor_index = donors[task_index]
        count = validate_terminal_layout(task.get("terminal_layout", {}))
        components = terminal_components(count)
        component_offsets = tuple(
            components[name] for name in ORACLE_TERMINAL_COMPONENT_ORDER
        )
        for condition in selected_conditions:
            scope_name = None
            oracle_index = None
            pattern_name = None
            packet_offsets = None
            component_indices = None
            component_sources = None
            if condition == ORACLE_TERMINAL_FULL_MATCHED:
                scope_name = ORACLE_TERMINAL_SCOPE_NAME
                oracle_index = task_index
                pattern_name = "full_k32"
                packet_offsets = ORACLE_TERMINAL_FULL_OFFSETS
            elif condition == ORACLE_TERMINAL_FULL_SHUFFLED:
                scope_name = ORACLE_TERMINAL_SCOPE_NAME
                oracle_index = donor_index
                pattern_name = "full_k32"
                packet_offsets = ORACLE_TERMINAL_FULL_OFFSETS
            elif condition.startswith(
                f"oracle_{ORACLE_TERMINAL_SCOPE_NAME}_terminal_k24_"
            ):
                assignment = condition.rsplit("_", 1)[-1]
                if assignment not in ORACLE_TERMINAL_ASSIGNMENTS:
                    raise ValueError(f"unknown factorial assignment: {assignment}")
                scope_name = ORACLE_TERMINAL_SCOPE_NAME
                pattern_name = "terminal_k24"
                packet_offsets = ORACLE_TERMINAL_TAIL_OFFSETS
                component_sources = tuple(assignment)
                component_indices = tuple(
                    task_index if source == "m" else donor_index
                    for source in component_sources
                )
            plan.append(
                OracleTerminalCondition(
                    task_id=task_id,
                    task_index=task_index,
                    condition=condition,
                    target_prompt_kind=(
                        "task" if condition == "text_only_no_lip" else "neutral"
                    ),
                    scope_name=scope_name,
                    oracle_index=oracle_index,
                    position_pattern=pattern_name,
                    packet_offsets=packet_offsets,
                    component_oracle_indices=component_indices,
                    component_sources=component_sources,
                    component_offsets=(
                        component_offsets if component_indices is not None else None
                    ),
                )
            )
    return plan


def plan_as_dicts(plan: Iterable[OracleTerminalCondition]) -> list[dict[str, Any]]:
    return [asdict(item) for item in plan]


def design_fingerprint(config: Mapping[str, Any]) -> str:
    payload = {
        "protocol_version": ORACLE_TERMINAL_PROTOCOL_VERSION,
        "experiment_id": config.get("experiment_id"),
        "predecessor_experiment": config.get("predecessor_experiment"),
        "population_source": config.get("population_source", {}),
        "models": config.get("models", {}),
        "prompt_protocol": config.get("prompt_protocol", {}),
        "runtime": config.get("runtime", {}),
        "data": config.get("data", {}),
        "screening": config.get("screening", {}),
        "terminal_factorial": config.get("terminal_factorial", {}),
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


def primary_gates() -> tuple[tuple[str, str], ...]:
    return (
        (ORACLE_TERMINAL_FULL_MATCHED, ORACLE_TERMINAL_FULL_SHUFFLED),
        (_factorial_condition("mmm"), _factorial_condition("sss")),
    )


def primary_family() -> tuple[tuple[str, str], ...]:
    mmm = _factorial_condition("mmm")
    sss = _factorial_condition("sss")
    return (
        (mmm, _factorial_condition("smm")),
        (mmm, _factorial_condition("msm")),
        (mmm, _factorial_condition("mms")),
        (_factorial_condition("mss"), sss),
        (_factorial_condition("sms"), sss),
        (_factorial_condition("ssm"), sss),
        (_factorial_condition("smm"), sss),
    )


ORACLE_TERMINAL_HYPOTHESIS_LABELS = (
    "core_contribution",
    "name_contribution",
    "boundary_contribution",
    "core_only_sufficiency",
    "name_only_sufficiency",
    "boundary_only_sufficiency",
    "tail_only_sufficiency",
)


def semantic_gate(
    condition_means: Mapping[str, float],
    primary_inference: Mapping[str, Any],
) -> dict[str, Any]:
    missing = sorted(set(ORACLE_TERMINAL_CONDITIONS).difference(condition_means))
    if missing:
        raise ValueError("terminal gate is missing condition(s): " + ", ".join(missing))
    means = {
        condition: float(condition_means[condition])
        for condition in ORACLE_TERMINAL_CONDITIONS
    }
    gates = primary_inference.get("gates")
    family = primary_inference.get("family")
    if not isinstance(gates, list) or not isinstance(family, list):
        raise ValueError("terminal gate requires two-gate Holm inference")
    observed_gates = [
        (str(item.get("treatment")), str(item.get("control")))
        for item in gates
        if isinstance(item, Mapping)
    ]
    observed_family = [
        (str(item.get("treatment")), str(item.get("control")))
        for item in family
        if isinstance(item, Mapping)
    ]
    if observed_gates != list(primary_gates()) or observed_family != list(
        primary_family()
    ):
        raise ValueError("primary inference does not match the terminal factorial")
    gate_checks = {
        "full_k32_replication": bool(gates[0].get("rejected"))
        and means[gates[0]["treatment"]] > means[gates[0]["control"]],
        "tail_k24_replication": bool(gates[1].get("rejected"))
        and means[gates[1]["treatment"]] > means[gates[1]["control"]],
    }
    component_results = {}
    supported = []
    for label, pair, inference in zip(
        ORACLE_TERMINAL_HYPOTHESIS_LABELS,
        primary_family(),
        family,
    ):
        treatment, control = pair
        checks = {
            "positive_direction": means[treatment] > means[control],
            "tested": bool(inference.get("tested")),
            "holm_rejected": bool(inference.get("rejected")),
        }
        passed = all(checks.values())
        component_results[label] = {
            "treatment": treatment,
            "control": control,
            "checks": checks,
            "passed": passed,
        }
        if passed:
            supported.append(label)
    checks = {
        "text_control_nonzero": means["text_only_no_lip"] > 0.0,
        **gate_checks,
        "at_least_one_component_claim": bool(supported),
    }
    return {
        "metric": "functional_pass",
        "condition_means": means,
        "replication_gates": gate_checks,
        "component_results": component_results,
        "supported_component_claims": supported,
        "terminal_source_attribution_supported": bool(supported),
        "primary_inference": dict(primary_inference),
        "checks": checks,
        "passed": all(checks.values()),
    }


__all__ = [
    "ORACLE_TERMINAL_ASSIGNMENTS",
    "ORACLE_TERMINAL_BOUNDARY_OFFSETS",
    "ORACLE_TERMINAL_CANDIDATE_COUNT",
    "ORACLE_TERMINAL_CAPTURE_LAYERS",
    "ORACLE_TERMINAL_CAPTURE_SIZE",
    "ORACLE_TERMINAL_COMPONENT_ORDER",
    "ORACLE_TERMINAL_CONDITIONS",
    "ORACLE_TERMINAL_CONFIRMATION_SEEDS",
    "ORACLE_TERMINAL_ELIGIBILITY_RULE",
    "ORACLE_TERMINAL_EXPERIMENT_ID",
    "ORACLE_TERMINAL_FULL_MATCHED",
    "ORACLE_TERMINAL_FULL_SHUFFLED",
    "ORACLE_TERMINAL_HYPOTHESIS_LABELS",
    "ORACLE_TERMINAL_LAYER_COUNT",
    "ORACLE_TERMINAL_PATTERN_CONTRACT",
    "ORACLE_TERMINAL_PROTOCOL_VERSION",
    "ORACLE_TERMINAL_SCOPE_NAME",
    "ORACLE_TERMINAL_SCREENING_CONDITION",
    "ORACLE_TERMINAL_SCREENING_SCOPE",
    "ORACLE_TERMINAL_SCREENING_SEEDS",
    "ORACLE_TERMINAL_SELECTED_COUNT",
    "ORACLE_TERMINAL_SELECTED_PER_STRATUM",
    "ORACLE_TERMINAL_TAIL_OFFSETS",
    "build_condition_plan",
    "candidate_binding_config",
    "design_fingerprint",
    "eligible_task_ids",
    "plan_as_dicts",
    "primary_family",
    "primary_gates",
    "semantic_gate",
    "terminal_components",
    "terminal_patterns",
    "validate_selected_task_manifest",
    "validate_terminal_design",
    "validate_terminal_layout",
    "validate_terminal_memory_contract",
]
