"""Pure design helpers for functional target-oracle packet generation."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable

from src.evaluation.source_only import derangement_indices


ORACLE_FUNCTIONAL_PROTOCOL_VERSION = "lip-oracle-packet-functional-v1"
ORACLE_FUNCTIONAL_CONDITIONS = (
    "neutral_no_lip",
    "text_only_no_lip",
    "oracle_packet_k1",
    "oracle_packet_k8",
    "shuffled_oracle_packet_k8",
)


def design_fingerprint(config: dict) -> str:
    payload = {
        "protocol_version": ORACLE_FUNCTIONAL_PROTOCOL_VERSION,
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
) -> list[OracleCondition]:
    ids = [str(task_id) for task_id in task_ids]
    selected = list(conditions)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("task_ids must be a non-empty unique sequence")
    if selected != list(ORACLE_FUNCTIONAL_CONDITIONS):
        raise ValueError("conditions must match the frozen oracle functional design")
    shuffled = derangement_indices(len(ids), shuffle_seed)
    plan = []
    for task_index, task_id in enumerate(ids):
        for condition in selected:
            target_prompt_kind = (
                "task" if condition == "text_only_no_lip" else "neutral"
            )
            packet_size = None
            packet_index = None
            if condition == "oracle_packet_k1":
                packet_size, packet_index = 1, task_index
            elif condition == "oracle_packet_k8":
                packet_size, packet_index = 8, task_index
            elif condition == "shuffled_oracle_packet_k8":
                packet_size, packet_index = 8, shuffled[task_index]
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
