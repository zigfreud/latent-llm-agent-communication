"""Pure design helpers for functional target-oracle packet generation."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

from src.evaluation.source_only import derangement_indices


ORACLE_FUNCTIONAL_PROTOCOL_VERSION = "lip-oracle-packet-functional-v1"
ORACLE_CAPACITY_PROTOCOL_VERSION = "lip-oracle-packet-functional-v2"
ORACLE_FUNCTIONAL_CONDITIONS = (
    "neutral_no_lip",
    "text_only_no_lip",
    "oracle_packet_k1",
    "oracle_packet_k8",
    "shuffled_oracle_packet_k8",
)
ORACLE_CAPACITY_PACKET_SIZES = (8, 16, 32)


def protocol_version_for_config(config: Mapping) -> str:
    """Resolve the immutable record schema from the registered experiment."""

    experiment_id = config.get("experiment_id")
    if experiment_id == "LIP-PROTO-005":
        return ORACLE_FUNCTIONAL_PROTOCOL_VERSION
    if experiment_id == "LIP-PROTO-007":
        return ORACLE_CAPACITY_PROTOCOL_VERSION
    raise ValueError(f"unsupported functional oracle experiment: {experiment_id}")


def packet_contract(config: Mapping) -> tuple[tuple[int, ...], int | None]:
    """Return tested packet sizes and the optional replication control size."""

    packet = config.get("packet", {})
    if not isinstance(packet, Mapping):
        raise ValueError("packet must be a mapping")
    if protocol_version_for_config(config) == ORACLE_FUNCTIONAL_PROTOCOL_VERSION:
        return (
            (int(packet.get("selected_size", 0)),),
            int(packet.get("replication_size", 0)),
        )
    raw_sizes = packet.get("sizes", [])
    if not isinstance(raw_sizes, list):
        raise ValueError("packet.sizes must be a list")
    return tuple(int(size) for size in raw_sizes), None


def expected_functional_conditions(
    packet_sizes: Sequence[int],
    *,
    replication_size: int | None,
) -> tuple[str, ...]:
    """Construct the ordered factorial condition contract."""

    sizes = tuple(int(size) for size in packet_sizes)
    if not sizes or any(size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("packet sizes must be a non-empty unique positive sequence")
    if tuple(sorted(sizes)) != sizes:
        raise ValueError("packet sizes must be strictly increasing")
    if replication_size is not None:
        replication_size = int(replication_size)
        if replication_size <= 0 or replication_size in sizes:
            raise ValueError("replication size must be positive and outside packet sizes")

    conditions = ["neutral_no_lip", "text_only_no_lip"]
    if replication_size is not None:
        conditions.append(f"oracle_packet_k{replication_size}")
    for size in sizes:
        conditions.extend(
            (f"oracle_packet_k{size}", f"shuffled_oracle_packet_k{size}")
        )
    return tuple(conditions)


def design_fingerprint(config: dict) -> str:
    protocol_version = protocol_version_for_config(config)
    payload = {
        "protocol_version": protocol_version,
        "experiment_id": config.get("experiment_id"),
        "capacity_selection_experiment": config.get(
            "capacity_selection_experiment"
        ),
        "models": config.get("models", {}),
        "prompt_protocol": config.get("prompt_protocol", {}),
        "data": config.get("data", {}),
        "neutral_target_prompt": config.get("neutral_target_prompt"),
        "carrier": config.get("carrier", {}),
        "packet": config.get("packet", {}),
        "conditions": config.get("conditions", []),
        "controls": config.get("controls", {}),
        "generation": config.get("generation", {}),
    }
    if protocol_version == ORACLE_CAPACITY_PROTOCOL_VERSION:
        payload.update(
            {
                "source_protocol_experiment": config.get(
                    "source_protocol_experiment"
                ),
                "functional_anchor_experiment": config.get(
                    "functional_anchor_experiment"
                ),
                "position_audit_experiment": config.get(
                    "position_audit_experiment"
                ),
            }
        )
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def stable_seed(*values: int) -> int:
    payload = ":".join(str(int(value)) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


@dataclass(frozen=True)
class OracleCondition:
    task_id: str
    task_index: int
    condition: str
    target_prompt_kind: str
    packet_size: int | None
    packet_index: int | None


def build_condition_plan(
    task_ids: Iterable[str],
    conditions: Iterable[str],
    *,
    shuffle_seed: int,
    packet_sizes: Sequence[int] = (8,),
    replication_size: int | None = 1,
) -> list[OracleCondition]:
    ids = [str(task_id) for task_id in task_ids]
    selected = list(conditions)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("task_ids must be a non-empty unique sequence")
    expected = expected_functional_conditions(
        packet_sizes,
        replication_size=replication_size,
    )
    if selected != list(expected):
        raise ValueError("conditions must match the frozen oracle functional design")

    matched_conditions = {
        f"oracle_packet_k{size}": int(size) for size in packet_sizes
    }
    if replication_size is not None:
        matched_conditions[f"oracle_packet_k{replication_size}"] = int(
            replication_size
        )
    shuffled_conditions = {
        f"shuffled_oracle_packet_k{size}": int(size) for size in packet_sizes
    }
    shuffled = derangement_indices(len(ids), shuffle_seed)
    plan = []
    for task_index, task_id in enumerate(ids):
        for condition in selected:
            target_prompt_kind = (
                "task" if condition == "text_only_no_lip" else "neutral"
            )
            packet_size = None
            packet_index = None
            if condition in matched_conditions:
                packet_size = matched_conditions[condition]
                packet_index = task_index
            elif condition in shuffled_conditions:
                packet_size = shuffled_conditions[condition]
                packet_index = shuffled[task_index]
            plan.append(
                OracleCondition(
                    task_id=task_id,
                    task_index=task_index,
                    condition=condition,
                    target_prompt_kind=target_prompt_kind,
                    packet_size=packet_size,
                    packet_index=packet_index,
                )
            )
    return plan


def plan_as_dicts(plan: Iterable[OracleCondition]) -> list[dict]:
    return [asdict(item) for item in plan]


def declares_entry_point(code: str, entry_point: str | None) -> bool:
    """Return whether syntactically valid candidate code declares the required name."""

    if not isinstance(entry_point, str) or not entry_point.strip():
        raise ValueError("functional task must define a non-empty entry_point")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_point
        for node in ast.walk(tree)
    )


def semantic_gate(
    condition_means: Mapping[str, float],
    *,
    packet_sizes: Sequence[int] = (8,),
    replication_size: int | None = 1,
) -> dict:
    """Apply the frozen functional decision rule to task-clustered pass rates."""

    expected = expected_functional_conditions(
        packet_sizes,
        replication_size=replication_size,
    )
    missing = sorted(set(expected).difference(condition_means))
    if missing:
        raise ValueError(f"functional gate is missing condition(s): {', '.join(missing)}")
    means = {condition: float(condition_means[condition]) for condition in expected}

    if replication_size is not None and tuple(packet_sizes) == (8,):
        checks = {
            "text_control_nonzero": means["text_only_no_lip"] > 0.0,
            "k8_beats_neutral": means["oracle_packet_k8"]
            > means["neutral_no_lip"],
            "k8_beats_shuffled_k8": means["oracle_packet_k8"]
            > means["shuffled_oracle_packet_k8"],
            "k8_beats_k1": means["oracle_packet_k8"]
            > means[f"oracle_packet_k{replication_size}"],
        }
        return {
            "metric": "functional_pass",
            "condition_means": means,
            "checks": checks,
            "passed": all(checks.values()),
        }

    capacity_checks = {}
    for size in packet_sizes:
        matched = means[f"oracle_packet_k{size}"]
        neutral = means["neutral_no_lip"]
        shuffled = means[f"shuffled_oracle_packet_k{size}"]
        checks = {
            "beats_neutral": matched > neutral,
            "beats_task_mismatched": matched > shuffled,
        }
        capacity_checks[str(size)] = {"checks": checks, "passed": all(checks.values())}
    supported = [
        int(size) for size, result in capacity_checks.items() if result["passed"]
    ]
    checks = {
        "text_control_nonzero": means["text_only_no_lip"] > 0.0,
        "any_capacity_task_specific": bool(supported),
    }
    return {
        "metric": "functional_pass",
        "condition_means": means,
        "capacity_checks": capacity_checks,
        "supported_capacities": supported,
        "smallest_supported_capacity": min(supported) if supported else None,
        "checks": checks,
        "passed": all(checks.values()),
    }
