"""Frozen helpers for capability-calibrated position-sparse oracle packets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.evaluation.source_only import derangement_indices
from src.pipelines.oracle_experiment import load_json_object, sha256_path


ORACLE_POSITION_EXPERIMENT_ID = "LIP-PROTO-011"
ORACLE_POSITION_PREDECESSOR = "LIP-PROTO-010"
ORACLE_POSITION_PROTOCOL_VERSION = "lip-oracle-position-sparse-packet-v1"
ORACLE_POSITION_LAYER_COUNT = 32
ORACLE_POSITION_CAPTURE_SIZE = 32
ORACLE_POSITION_SELECTED_COUNT = 32
ORACLE_POSITION_ELIGIBLE_START = 32
ORACLE_POSITION_ELIGIBLE_STOP = 64
ORACLE_POSITION_CONFIRMATION_SEEDS = (743, 887, 991)
ORACLE_POSITION_SCOPE_NAME = "early_quarter_input"
ORACLE_POSITION_CAPTURE_LAYERS = tuple(range(-32, 0))
ORACLE_POSITION_REPLAY_LAYERS = tuple(range(-32, -24))

ORACLE_POSITION_PATTERN_CONTRACT = (
    {
        "name": "full_k32",
        "packet_offsets": list(range(-32, 0)),
    },
    {
        "name": "diagnostic_top_k8",
        "packet_offsets": [-32, -30, -23, -22, -21, -20, -19, -18],
    },
    {
        "name": "peak_window_k8",
        "packet_offsets": list(range(-23, -15)),
    },
    {
        "name": "suffix_k8",
        "packet_offsets": list(range(-8, 0)),
    },
)
ORACLE_POSITION_PATTERN_ORDER = tuple(
    pattern["name"] for pattern in ORACLE_POSITION_PATTERN_CONTRACT
)


def expected_conditions() -> tuple[str, ...]:
    conditions = ["neutral_no_lip", "text_only_no_lip"]
    for pattern_name in ORACLE_POSITION_PATTERN_ORDER:
        conditions.extend(
            (
                f"oracle_{ORACLE_POSITION_SCOPE_NAME}_{pattern_name}",
                f"shuffled_oracle_{ORACLE_POSITION_SCOPE_NAME}_{pattern_name}",
            )
        )
    return tuple(conditions)


ORACLE_POSITION_CONDITIONS = expected_conditions()


def position_patterns(memory: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
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
        or normalized != ORACLE_POSITION_PATTERN_CONTRACT
    ):
        raise ValueError("position patterns do not match the frozen contract")
    return {
        pattern["name"]: tuple(pattern["packet_offsets"])
        for pattern in normalized
    }


def validate_position_memory_contract(memory: Mapping) -> tuple[dict, ...]:
    """Validate capture span, replay depth, and sparse-position patterns."""

    if not isinstance(memory, Mapping):
        raise ValueError("memory must be a mapping")
    expected_scalars = {
        "packet_size": ORACLE_POSITION_CAPTURE_SIZE,
        "decoder_layer_count": ORACLE_POSITION_LAYER_COUNT,
        "self_check_tasks": 1,
        "maximum_self_logit_delta": 0.0001,
    }
    for field, expected in expected_scalars.items():
        if memory.get(field) != expected:
            raise ValueError(f"LIP-PROTO-011 freezes memory.{field}={expected}")
    capture_layers = [int(layer) for layer in memory.get("state_capture_layers", [])]
    if capture_layers != list(ORACLE_POSITION_CAPTURE_LAYERS):
        raise ValueError("state_capture_layers must cover all 32 decoder blocks")
    replay_scope = memory.get("replay_scope")
    expected_scope = {
        "name": ORACLE_POSITION_SCOPE_NAME,
        "boundary": "block_input",
        "layers": list(ORACLE_POSITION_REPLAY_LAYERS),
    }
    if replay_scope != expected_scope:
        raise ValueError("replay_scope must freeze the confirmed first 8 blocks")
    position_patterns(memory)
    return (dict(expected_scope),)


def validate_position_selection_contract(selection: Mapping[str, Any]) -> None:
    expected = {
        "source_experiment": ORACLE_POSITION_PREDECESSOR,
        "state_type": "residual_input",
        "metric": "task_signal_fraction",
        "layer_reduction": "arithmetic_mean",
        "rank_tie_break": "packet_offset_ascending",
        "diagnostic_top_k": 8,
        "contiguous_window_size": 8,
        "expected_diagnostic_top_offsets": [
            -32,
            -30,
            -23,
            -22,
            -21,
            -20,
            -19,
            -18,
        ],
        "expected_peak_window_offsets": list(range(-23, -15)),
    }
    if dict(selection) != expected:
        raise ValueError("position_selection must match the frozen 010-derived rule")


def derive_position_patterns(
    diagnostics: Mapping[str, Any],
) -> dict[str, list[int]]:
    """Derive the registered top-eight and best contiguous residual windows."""

    offsets = [int(offset) for offset in diagnostics.get("packet_offsets", [])]
    layers = [int(layer) for layer in diagnostics.get("layer_indices", [])]
    if offsets != list(range(-32, 0)) or layers != list(range(-32, 0)):
        raise ValueError("source diagnostics must cover the frozen 32x32 grid")
    cells = diagnostics.get("cells")
    if not isinstance(cells, list):
        raise ValueError("source diagnostics must contain cells")
    residual = [
        cell
        for cell in cells
        if isinstance(cell, Mapping) and cell.get("state_type") == "residual_input"
    ]
    by_key = {
        (int(cell["layer_index"]), int(cell["packet_offset"])): float(
            cell["task_signal_fraction"]
        )
        for cell in residual
    }
    expected_keys = {(layer, offset) for layer in layers for offset in offsets}
    if set(by_key) != expected_keys:
        raise ValueError("source residual diagnostics do not form a complete grid")
    position_means = {
        offset: sum(by_key[(layer, offset)] for layer in layers) / len(layers)
        for offset in offsets
    }
    ranked = sorted(offsets, key=lambda offset: (-position_means[offset], offset))
    top_offsets = sorted(ranked[:8])
    windows = [offsets[start : start + 8] for start in range(len(offsets) - 7)]
    peak_window = min(
        windows,
        key=lambda window: (
            -sum(position_means[offset] for offset in window) / len(window),
            window[0],
        ),
    )
    return {
        "diagnostic_top_k8": top_offsets,
        "peak_window_k8": peak_window,
    }


def select_eligible_holdout(eligible_task_ids: Sequence[str]) -> list[str]:
    ids = [str(task_id) for task_id in eligible_task_ids]
    if len(ids) < ORACLE_POSITION_ELIGIBLE_STOP or len(set(ids)) != len(ids):
        raise ValueError("eligible registry must contain at least 64 unique tasks")
    return ids[ORACLE_POSITION_ELIGIBLE_START:ORACLE_POSITION_ELIGIBLE_STOP]


def validate_selected_task_manifest(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    """Require a hash-bound, latent-unseen slice of the sealed 010 screen."""

    source = config.get("calibration_source", {})
    output = config.get("output", {})
    report_path = Path(str(output.get("selection_report_json", "")))
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = load_json_object(report_path)
    sampled_ids = [str(task_id) for task_id in manifest.get("sampled_ids", [])]
    predecessor_ids = [
        str(task_id)
        for task_id in report.get("predecessor_selected_task_ids", [])
    ]
    source_fields = (
        ("candidate_task_manifest", "candidate_manifest"),
        ("screening_scored_jsonl", "screening_scored_jsonl"),
        ("screening_summary", "screening_summary"),
        ("selection_report", "predecessor_selection_report"),
        ("state_diagnostics", "source_state_diagnostics"),
    )
    source_checks = {}
    for source_field, manifest_field in source_fields:
        path = Path(str(source.get(source_field, "")))
        source_checks[f"{source_field}_exists"] = path.is_file()
        source_checks[f"{source_field}_path"] = manifest.get(manifest_field) == str(
            path
        )
        source_checks[f"{source_field}_hash"] = bool(
            path.is_file()
            and sha256_path(path) == source.get(f"{source_field}_sha256")
            and manifest.get(f"{manifest_field}_sha256")
            == source.get(f"{source_field}_sha256")
        )
    checks = {
        "selection_kind": manifest.get("selection_kind")
        == "capability_calibrated_latent_unseen_holdout",
        "selected_count": len(sampled_ids) == ORACLE_POSITION_SELECTED_COUNT,
        "rank_start": manifest.get("eligible_rank_start_zero_based")
        == ORACLE_POSITION_ELIGIBLE_START,
        "rank_stop": manifest.get("eligible_rank_stop_exclusive")
        == ORACLE_POSITION_ELIGIBLE_STOP,
        "registered_slice": manifest.get(
            "selected_ids_are_registered_holdout_slice"
        )
        is True,
        "predecessor_disjoint": manifest.get(
            "sampled_ids_disjoint_from_predecessor_selection"
        )
        is True
        and not bool(set(sampled_ids).intersection(predecessor_ids)),
        "exclusion_disjoint": manifest.get(
            "sampled_ids_disjoint_from_exclusions"
        )
        is True,
        "artifact_manifest": manifest.get(
            "calibration_artifact_manifest_sha256"
        )
        == source.get("artifact_manifest_sha256"),
        "eligible_count": manifest.get("eligible_task_count")
        == source.get("eligible_task_count"),
        "eligible_hash": manifest.get("eligible_ids_sha256")
        == source.get("eligible_ids_sha256"),
        "position_selection": manifest.get("position_selection")
        == config.get("position_selection"),
        "derived_patterns": manifest.get("derived_position_patterns")
        == {
            "diagnostic_top_k8": config.get("position_selection", {}).get(
                "expected_diagnostic_top_offsets"
            ),
            "peak_window_k8": config.get("position_selection", {}).get(
                "expected_peak_window_offsets"
            ),
        },
        "report_experiment": report.get("experiment_id")
        == ORACLE_POSITION_EXPERIMENT_ID,
        "report_passed": report.get("passed") is True,
        "report_selected_ids": report.get("selected_task_ids") == sampled_ids,
        "report_manifest_path": report.get("selected_task_manifest")
        == str(manifest_path),
        "report_manifest_hash": report.get("selected_task_manifest_sha256")
        == sha256_path(manifest_path),
        **source_checks,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "selected task manifest failed 011 provenance: "
            + ", ".join(failed)
        )


@dataclass(frozen=True)
class OraclePositionCondition:
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
) -> list[OraclePositionCondition]:
    ids = [str(task_id) for task_id in task_ids]
    selected = list(conditions)
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("task_ids must contain at least two unique tasks")
    if selected != list(ORACLE_POSITION_CONDITIONS):
        raise ValueError("conditions must match the frozen position-sparse design")
    shuffled = derangement_indices(len(ids), int(shuffle_seed))
    patterns = {
        pattern["name"]: tuple(pattern["packet_offsets"])
        for pattern in ORACLE_POSITION_PATTERN_CONTRACT
    }
    matched = {
        f"oracle_{ORACLE_POSITION_SCOPE_NAME}_{name}": name for name in patterns
    }
    mismatched = {
        f"shuffled_oracle_{ORACLE_POSITION_SCOPE_NAME}_{name}": name
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
                OraclePositionCondition(
                    task_id=task_id,
                    task_index=task_index,
                    condition=condition,
                    target_prompt_kind=(
                        "task" if condition == "text_only_no_lip" else "neutral"
                    ),
                    scope_name=(
                        ORACLE_POSITION_SCOPE_NAME
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


def plan_as_dicts(plan: Iterable[OraclePositionCondition]) -> list[dict]:
    return [asdict(item) for item in plan]


def design_fingerprint(config: Mapping[str, Any]) -> str:
    payload = {
        "protocol_version": ORACLE_POSITION_PROTOCOL_VERSION,
        "experiment_id": config.get("experiment_id"),
        "predecessor_experiment": config.get("predecessor_experiment"),
        "calibration_source": config.get("calibration_source", {}),
        "models": config.get("models", {}),
        "prompt_protocol": config.get("prompt_protocol", {}),
        "runtime": config.get("runtime", {}),
        "data": config.get("data", {}),
        "position_selection": config.get("position_selection", {}),
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
    return tuple(
        (
            f"oracle_{ORACLE_POSITION_SCOPE_NAME}_{pattern_name}",
            f"shuffled_oracle_{ORACLE_POSITION_SCOPE_NAME}_{pattern_name}",
        )
        for pattern_name in ORACLE_POSITION_PATTERN_ORDER
    )


def semantic_gate(
    condition_means: Mapping[str, float],
    primary_inference: Mapping,
) -> dict[str, Any]:
    missing = sorted(set(ORACLE_POSITION_CONDITIONS).difference(condition_means))
    if missing:
        raise ValueError(
            "position-sparse gate is missing condition(s): " + ", ".join(missing)
        )
    means = {
        condition: float(condition_means[condition])
        for condition in ORACLE_POSITION_CONDITIONS
    }
    hypotheses = primary_inference.get("hypotheses")
    if not isinstance(hypotheses, list):
        raise ValueError("position-sparse gate requires fixed-sequence hypotheses")
    rejected = {
        str(item.get("treatment")): bool(item.get("rejected"))
        for item in hypotheses
        if isinstance(item, Mapping)
    }
    expected = {treatment for treatment, _ in primary_fixed_sequence()}
    if set(rejected) != expected:
        raise ValueError("primary inference does not match the frozen pattern order")

    pattern_checks = {}
    confirmed = []
    for pattern_name in ORACLE_POSITION_PATTERN_ORDER:
        matched = f"oracle_{ORACLE_POSITION_SCOPE_NAME}_{pattern_name}"
        shuffled = f"shuffled_oracle_{ORACLE_POSITION_SCOPE_NAME}_{pattern_name}"
        checks = {
            "beats_neutral": means[matched] > means["neutral_no_lip"],
            "beats_task_mismatched": means[matched] > means[shuffled],
            "fixed_sequence_rejected": rejected[matched],
        }
        passed = all(checks.values())
        pattern_checks[pattern_name] = {"checks": checks, "passed": passed}
        if passed:
            confirmed.append(pattern_name)
    sparse_confirmed = pattern_checks["diagnostic_top_k8"]["passed"]
    checks = {
        "text_control_nonzero": means["text_only_no_lip"] > 0.0,
        "full_k32_replication_confirmed": pattern_checks["full_k32"]["passed"],
        "diagnostic_top_k8_confirmed": sparse_confirmed,
    }
    return {
        "metric": "functional_pass",
        "condition_means": means,
        "pattern_checks": pattern_checks,
        "confirmed_patterns": confirmed,
        "position_sparse_transport_supported": sparse_confirmed,
        "smallest_confirmed_packet_size": (
            8 if any(name.endswith("k8") for name in confirmed) else 32
            if "full_k32" in confirmed
            else None
        ),
        "primary_inference": dict(primary_inference),
        "checks": checks,
        "passed": all(checks.values()),
    }


__all__ = [
    "ORACLE_POSITION_CAPTURE_LAYERS",
    "ORACLE_POSITION_CAPTURE_SIZE",
    "ORACLE_POSITION_CONDITIONS",
    "ORACLE_POSITION_CONFIRMATION_SEEDS",
    "ORACLE_POSITION_ELIGIBLE_START",
    "ORACLE_POSITION_ELIGIBLE_STOP",
    "ORACLE_POSITION_EXPERIMENT_ID",
    "ORACLE_POSITION_LAYER_COUNT",
    "ORACLE_POSITION_PATTERN_CONTRACT",
    "ORACLE_POSITION_PATTERN_ORDER",
    "ORACLE_POSITION_PREDECESSOR",
    "ORACLE_POSITION_PROTOCOL_VERSION",
    "ORACLE_POSITION_REPLAY_LAYERS",
    "ORACLE_POSITION_SCOPE_NAME",
    "ORACLE_POSITION_SELECTED_COUNT",
    "build_condition_plan",
    "derive_position_patterns",
    "design_fingerprint",
    "plan_as_dicts",
    "position_patterns",
    "primary_fixed_sequence",
    "select_eligible_holdout",
    "semantic_gate",
    "validate_position_memory_contract",
    "validate_position_selection_contract",
    "validate_selected_task_manifest",
]
