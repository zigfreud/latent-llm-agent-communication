"""Frozen helpers for the LIP-PROTO-012 block-deletion oracle packet."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.evaluation.oracle_capability_calibration import eligible_task_ids
from src.evaluation.source_only import derangement_indices
from src.pipelines.oracle_experiment import load_json_object, sha256_path


ORACLE_DELETION_EXPERIMENT_ID = "LIP-PROTO-012"
ORACLE_DELETION_PREDECESSOR = "LIP-PROTO-011"
ORACLE_DELETION_CALIBRATION_EXPERIMENT = "LIP-PROTO-010"
ORACLE_DELETION_PROTOCOL_VERSION = "lip-oracle-block-deletion-v1"
ORACLE_DELETION_LAYER_COUNT = 32
ORACLE_DELETION_CAPTURE_SIZE = 32
ORACLE_DELETION_SELECTED_COUNT = 17
ORACLE_DELETION_ELIGIBLE_START = 64
ORACLE_DELETION_ELIGIBLE_STOP = 81
ORACLE_DELETION_CONFIRMATION_SEEDS = (1103, 1217, 1301)
ORACLE_DELETION_SCOPE_NAME = "early_quarter_input"
ORACLE_DELETION_CAPTURE_LAYERS = tuple(range(-32, 0))
ORACLE_DELETION_REPLAY_LAYERS = tuple(range(-32, -24))
ORACLE_DELETION_OCTETS = (
    tuple(range(-32, -24)),
    tuple(range(-24, -16)),
    tuple(range(-16, -8)),
    tuple(range(-8, 0)),
)
ORACLE_DELETION_PATTERN_CONTRACT = (
    {"name": "full_k32", "packet_offsets": list(range(-32, 0))},
    *(
        {
            "name": f"drop_octet_{index}_k24",
            "packet_offsets": [
                offset
                for offset in range(-32, 0)
                if offset not in deleted_octet
            ],
        }
        for index, deleted_octet in enumerate(ORACLE_DELETION_OCTETS, start=1)
    ),
)
ORACLE_DELETION_PATTERN_ORDER = tuple(
    pattern["name"] for pattern in ORACLE_DELETION_PATTERN_CONTRACT
)
ORACLE_DELETION_K24_PATTERN_ORDER = ORACLE_DELETION_PATTERN_ORDER[1:]


def expected_conditions() -> tuple[str, ...]:
    conditions = ["neutral_no_lip", "text_only_no_lip"]
    for pattern_name in ORACLE_DELETION_PATTERN_ORDER:
        conditions.extend(
            (
                f"oracle_{ORACLE_DELETION_SCOPE_NAME}_{pattern_name}",
                f"shuffled_oracle_{ORACLE_DELETION_SCOPE_NAME}_{pattern_name}",
            )
        )
    return tuple(conditions)


ORACLE_DELETION_CONDITIONS = expected_conditions()


def deletion_patterns(memory: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    raw_patterns = memory.get("position_patterns")
    if not isinstance(raw_patterns, list):
        raise ValueError("memory.position_patterns must be a list")
    normalized = tuple(
        {
            "name": str(pattern.get("name", "")),
            "packet_offsets": [
                int(offset) for offset in pattern.get("packet_offsets", [])
            ],
        }
        for pattern in raw_patterns
        if isinstance(pattern, Mapping)
    )
    if (
        len(normalized) != len(raw_patterns)
        or normalized != ORACLE_DELETION_PATTERN_CONTRACT
    ):
        raise ValueError("position patterns do not match the frozen deletion contract")
    return {
        pattern["name"]: tuple(pattern["packet_offsets"])
        for pattern in normalized
    }


def validate_deletion_design(design: Mapping[str, Any]) -> None:
    expected = {
        "method": "leave_one_contiguous_octet_out",
        "partition": [list(octet) for octet in ORACLE_DELETION_OCTETS],
        "kept_packet_size": 24,
        "interpretation_unit": "deleted_octet",
    }
    if dict(design) != expected:
        raise ValueError("deletion_design must match the exhaustive octet partition")


def validate_deletion_memory_contract(memory: Mapping) -> tuple[dict, ...]:
    if not isinstance(memory, Mapping):
        raise ValueError("memory must be a mapping")
    expected_scalars = {
        "packet_size": ORACLE_DELETION_CAPTURE_SIZE,
        "decoder_layer_count": ORACLE_DELETION_LAYER_COUNT,
        "self_check_tasks": 1,
        "maximum_self_logit_delta": 0.0001,
    }
    for field, expected in expected_scalars.items():
        if memory.get(field) != expected:
            raise ValueError(f"LIP-PROTO-012 freezes memory.{field}={expected}")
    if [int(layer) for layer in memory.get("state_capture_layers", [])] != list(
        ORACLE_DELETION_CAPTURE_LAYERS
    ):
        raise ValueError("state_capture_layers must cover all 32 decoder blocks")
    expected_scope = {
        "name": ORACLE_DELETION_SCOPE_NAME,
        "boundary": "block_input",
        "layers": list(ORACLE_DELETION_REPLAY_LAYERS),
    }
    if memory.get("replay_scope") != expected_scope:
        raise ValueError("replay_scope must freeze the confirmed first 8 blocks")
    deletion_patterns(memory)
    return (dict(expected_scope),)


def select_eligible_holdout(eligible_task_ids: Sequence[str]) -> list[str]:
    ids = [str(task_id) for task_id in eligible_task_ids]
    if len(ids) != ORACLE_DELETION_ELIGIBLE_STOP or len(set(ids)) != len(ids):
        raise ValueError("eligible registry must contain exactly 81 unique tasks")
    return ids[ORACLE_DELETION_ELIGIBLE_START:ORACLE_DELETION_ELIGIBLE_STOP]


def _source_checks(
    source: Mapping[str, Any],
    manifest: Mapping[str, Any],
    fields: Sequence[tuple[str, str]],
) -> dict[str, bool]:
    checks = {}
    for source_field, manifest_field in fields:
        path = Path(str(source.get(source_field, "")))
        expected_hash = source.get(f"{source_field}_sha256")
        checks[f"{source_field}_exists"] = path.is_file()
        checks[f"{source_field}_path"] = manifest.get(manifest_field) == str(path)
        checks[f"{source_field}_hash"] = bool(
            path.is_file()
            and sha256_path(path) == expected_hash
            and manifest.get(f"{manifest_field}_sha256") == expected_hash
        )
    return checks


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row must be an object: {path}")
                rows.append(row)
    return rows


def _ids_sha256(task_ids: Sequence[str]) -> str:
    canonical = json.dumps(list(task_ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_selected_task_manifest(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    """Require the final 17-task slice and both sealed predecessor bindings."""

    calibration = config.get("calibration_source", {})
    predecessor = config.get("predecessor_source", {})
    output = config.get("output", {})
    report_path = Path(str(output.get("selection_report_json", "")))
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = load_json_object(report_path)
    sampled_ids = [str(task_id) for task_id in manifest.get("sampled_ids", [])]
    prior_ids = [
        str(task_id) for task_id in report.get("prior_selected_task_ids", [])
    ]
    calibration_fields = (
        ("candidate_task_manifest", "candidate_manifest"),
        ("screening_scored_jsonl", "screening_scored_jsonl"),
        ("screening_summary", "screening_summary"),
        ("selection_report", "calibration_selection_report"),
    )
    predecessor_fields = (
        ("selected_tasks_jsonl", "predecessor_selected_tasks_jsonl"),
        ("selected_task_manifest", "predecessor_selected_task_manifest"),
        ("selection_report", "predecessor_selection_report"),
        ("functional_summary", "predecessor_functional_summary"),
    )
    candidate_manifest = load_json_object(
        Path(str(calibration.get("candidate_task_manifest", "")))
    )
    candidate_ids = [
        str(task_id) for task_id in candidate_manifest.get("sampled_ids", [])
    ]
    eligible_ids = eligible_task_ids(
        _read_jsonl(Path(str(calibration.get("screening_scored_jsonl", "")))),
        candidate_ids,
    )
    calibration_report = load_json_object(
        Path(str(calibration.get("selection_report", "")))
    )
    predecessor_manifest = load_json_object(
        Path(str(predecessor.get("selected_task_manifest", "")))
    )
    predecessor_report = load_json_object(
        Path(str(predecessor.get("selection_report", "")))
    )
    expected_prior_ids = [
        str(task_id)
        for task_id in calibration_report.get("selected_task_ids", [])
    ] + [
        str(task_id)
        for task_id in predecessor_manifest.get("sampled_ids", [])
    ]
    checks = {
        "selection_kind": manifest.get("selection_kind")
        == "capability_calibrated_final_latent_unseen_holdout",
        "selected_count": len(sampled_ids) == ORACLE_DELETION_SELECTED_COUNT,
        "rank_start": manifest.get("eligible_rank_start_zero_based")
        == ORACLE_DELETION_ELIGIBLE_START,
        "rank_stop": manifest.get("eligible_rank_stop_exclusive")
        == ORACLE_DELETION_ELIGIBLE_STOP,
        "registered_slice": manifest.get(
            "selected_ids_are_registered_holdout_slice"
        )
        is True
        and sampled_ids
        == eligible_ids[
            ORACLE_DELETION_ELIGIBLE_START:ORACLE_DELETION_ELIGIBLE_STOP
        ],
        "prior_disjoint": manifest.get("sampled_ids_disjoint_from_prior_latent")
        is True
        and not bool(set(sampled_ids).intersection(prior_ids)),
        "prior_partition": prior_ids == expected_prior_ids == eligible_ids[:64],
        "prior_hash": manifest.get("prior_selected_task_ids_sha256")
        == _ids_sha256(prior_ids),
        "calibration_report_passed": calibration_report.get("passed") is True,
        "predecessor_report_passed": predecessor_report.get("passed") is True,
        "predecessor_report_ids": predecessor_report.get("selected_task_ids")
        == predecessor_manifest.get("sampled_ids"),
        "exclusion_disjoint": manifest.get("sampled_ids_disjoint_from_exclusions")
        is True,
        "calibration_manifest": manifest.get(
            "calibration_artifact_manifest_sha256"
        )
        == calibration.get("artifact_manifest_sha256"),
        "predecessor_manifest": manifest.get(
            "predecessor_artifact_manifest_sha256"
        )
        == predecessor.get("artifact_manifest_sha256"),
        "eligible_count": manifest.get("eligible_task_count")
        == calibration.get("eligible_task_count")
        == len(eligible_ids),
        "eligible_hash": manifest.get("eligible_ids_sha256")
        == calibration.get("eligible_ids_sha256")
        == _ids_sha256(eligible_ids),
        "deletion_design": manifest.get("deletion_design")
        == config.get("deletion_design"),
        "report_experiment": report.get("experiment_id")
        == ORACLE_DELETION_EXPERIMENT_ID,
        "report_passed": report.get("passed") is True,
        "report_selected_ids": report.get("selected_task_ids") == sampled_ids,
        "report_manifest_path": report.get("selected_task_manifest")
        == str(manifest_path),
        "report_manifest_hash": report.get("selected_task_manifest_sha256")
        == sha256_path(manifest_path),
        **_source_checks(calibration, manifest, calibration_fields),
        **{
            f"predecessor_{name}": passed
            for name, passed in _source_checks(
                predecessor, manifest, predecessor_fields
            ).items()
        },
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "selected task manifest failed 012 provenance: " + ", ".join(failed)
        )


@dataclass(frozen=True)
class OracleDeletionCondition:
    task_id: str
    task_index: int
    condition: str
    target_prompt_kind: str
    scope_name: str | None
    oracle_index: int | None
    position_pattern: str | None
    packet_offsets: tuple[int, ...] | None


def build_condition_plan(
    task_ids: Iterable[str],
    conditions: Iterable[str],
    *,
    shuffle_seed: int,
) -> list[OracleDeletionCondition]:
    ids = [str(task_id) for task_id in task_ids]
    selected = list(conditions)
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("task_ids must contain at least two unique tasks")
    if selected != list(ORACLE_DELETION_CONDITIONS):
        raise ValueError("conditions must match the frozen block-deletion design")
    shuffled = derangement_indices(len(ids), int(shuffle_seed))
    patterns = {
        pattern["name"]: tuple(pattern["packet_offsets"])
        for pattern in ORACLE_DELETION_PATTERN_CONTRACT
    }
    matched = {
        f"oracle_{ORACLE_DELETION_SCOPE_NAME}_{name}": name for name in patterns
    }
    mismatched = {
        f"shuffled_oracle_{ORACLE_DELETION_SCOPE_NAME}_{name}": name
        for name in patterns
    }
    plan = []
    for task_index, task_id in enumerate(ids):
        for condition in selected:
            pattern_name = matched.get(condition) or mismatched.get(condition)
            oracle_index = None
            if condition in matched:
                oracle_index = task_index
            elif condition in mismatched:
                oracle_index = shuffled[task_index]
            plan.append(
                OracleDeletionCondition(
                    task_id=task_id,
                    task_index=task_index,
                    condition=condition,
                    target_prompt_kind=(
                        "task" if condition == "text_only_no_lip" else "neutral"
                    ),
                    scope_name=(
                        ORACLE_DELETION_SCOPE_NAME
                        if pattern_name is not None
                        else None
                    ),
                    oracle_index=oracle_index,
                    position_pattern=pattern_name,
                    packet_offsets=(
                        patterns[pattern_name] if pattern_name is not None else None
                    ),
                )
            )
    return plan


def plan_as_dicts(plan: Iterable[OracleDeletionCondition]) -> list[dict]:
    return [asdict(item) for item in plan]


def design_fingerprint(config: Mapping[str, Any]) -> str:
    payload = {
        "protocol_version": ORACLE_DELETION_PROTOCOL_VERSION,
        "experiment_id": config.get("experiment_id"),
        "predecessor_experiment": config.get("predecessor_experiment"),
        "calibration_source": config.get("calibration_source", {}),
        "predecessor_source": config.get("predecessor_source", {}),
        "models": config.get("models", {}),
        "prompt_protocol": config.get("prompt_protocol", {}),
        "runtime": config.get("runtime", {}),
        "data": config.get("data", {}),
        "deletion_design": config.get("deletion_design", {}),
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


def primary_anchor() -> tuple[str, str]:
    prefix = f"oracle_{ORACLE_DELETION_SCOPE_NAME}_"
    shuffled = f"shuffled_oracle_{ORACLE_DELETION_SCOPE_NAME}_"
    return f"{prefix}full_k32", f"{shuffled}full_k32"


def primary_family() -> tuple[tuple[str, str], ...]:
    prefix = f"oracle_{ORACLE_DELETION_SCOPE_NAME}_"
    shuffled = f"shuffled_oracle_{ORACLE_DELETION_SCOPE_NAME}_"
    return tuple(
        (f"{prefix}{pattern}", f"{shuffled}{pattern}")
        for pattern in ORACLE_DELETION_K24_PATTERN_ORDER
    )


def semantic_gate(
    condition_means: Mapping[str, float],
    primary_inference: Mapping,
) -> dict[str, Any]:
    missing = sorted(set(ORACLE_DELETION_CONDITIONS).difference(condition_means))
    if missing:
        raise ValueError(
            "block-deletion gate is missing condition(s): " + ", ".join(missing)
        )
    means = {
        condition: float(condition_means[condition])
        for condition in ORACLE_DELETION_CONDITIONS
    }
    anchor = primary_inference.get("anchor")
    family = primary_inference.get("family")
    if not isinstance(anchor, Mapping) or not isinstance(family, list):
        raise ValueError("block-deletion gate requires anchor-gated Holm inference")
    if (
        (str(anchor.get("treatment")), str(anchor.get("control")))
        != primary_anchor()
        or {
            (str(item.get("treatment")), str(item.get("control")))
            for item in family
            if isinstance(item, Mapping)
        }
        != set(primary_family())
    ):
        raise ValueError("primary inference does not match the deletion contract")

    prefix = f"oracle_{ORACLE_DELETION_SCOPE_NAME}_"
    shuffled = f"shuffled_oracle_{ORACLE_DELETION_SCOPE_NAME}_"
    full_matched, full_shuffled = primary_anchor()
    full_checks = {
        "beats_neutral": means[full_matched] > means["neutral_no_lip"],
        "beats_task_mismatched": means[full_matched] > means[full_shuffled],
        "anchor_rejected": bool(anchor.get("rejected")),
    }
    pattern_checks = {
        "full_k32": {"checks": full_checks, "passed": all(full_checks.values())}
    }
    family_by_treatment = {
        str(item["treatment"]): item for item in family if isinstance(item, Mapping)
    }
    dispensable = []
    for index, pattern_name in enumerate(ORACLE_DELETION_K24_PATTERN_ORDER, start=1):
        matched = f"{prefix}{pattern_name}"
        mismatched = f"{shuffled}{pattern_name}"
        inference = family_by_treatment[matched]
        checks = {
            "beats_neutral": means[matched] > means["neutral_no_lip"],
            "beats_task_mismatched": means[matched] > means[mismatched],
            "family_tested": bool(inference.get("tested")),
            "holm_rejected": bool(inference.get("rejected")),
        }
        passed = all(checks.values())
        pattern_checks[pattern_name] = {"checks": checks, "passed": passed}
        if passed:
            dispensable.append(f"octet_{index}")
    any_k24 = bool(dispensable)
    all_k24 = len(dispensable) == len(ORACLE_DELETION_K24_PATTERN_ORDER)
    checks = {
        "text_control_nonzero": means["text_only_no_lip"] > 0.0,
        "full_k32_replication_confirmed": pattern_checks["full_k32"]["passed"],
        "at_least_one_k24_deletion_confirmed": any_k24,
    }
    return {
        "metric": "functional_pass",
        "condition_means": means,
        "pattern_checks": pattern_checks,
        "dispensable_octets": dispensable,
        "all_octet_deletions_supported": all_k24,
        "block_deletion_transport_supported": any_k24,
        "smallest_confirmed_packet_size": (
            24 if any_k24 else 32 if pattern_checks["full_k32"]["passed"] else None
        ),
        "primary_inference": dict(primary_inference),
        "checks": checks,
        "passed": all(checks.values()),
    }


__all__ = [
    "ORACLE_DELETION_CAPTURE_LAYERS",
    "ORACLE_DELETION_CAPTURE_SIZE",
    "ORACLE_DELETION_CONDITIONS",
    "ORACLE_DELETION_CONFIRMATION_SEEDS",
    "ORACLE_DELETION_ELIGIBLE_START",
    "ORACLE_DELETION_ELIGIBLE_STOP",
    "ORACLE_DELETION_EXPERIMENT_ID",
    "ORACLE_DELETION_K24_PATTERN_ORDER",
    "ORACLE_DELETION_LAYER_COUNT",
    "ORACLE_DELETION_OCTETS",
    "ORACLE_DELETION_PATTERN_CONTRACT",
    "ORACLE_DELETION_PATTERN_ORDER",
    "ORACLE_DELETION_PREDECESSOR",
    "ORACLE_DELETION_PROTOCOL_VERSION",
    "ORACLE_DELETION_REPLAY_LAYERS",
    "ORACLE_DELETION_SCOPE_NAME",
    "ORACLE_DELETION_SELECTED_COUNT",
    "build_condition_plan",
    "deletion_patterns",
    "design_fingerprint",
    "plan_as_dicts",
    "primary_anchor",
    "primary_family",
    "select_eligible_holdout",
    "semantic_gate",
    "validate_deletion_design",
    "validate_deletion_memory_contract",
    "validate_selected_task_manifest",
]
