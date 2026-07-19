"""Pure protocol helpers for source-only experimental conditions."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Iterable, Optional


SOURCE_ONLY_PROTOCOL_VERSION = "lip-source-only-v1"
CONDITIONS = (
    "neutral_no_lip",
    "text_only_no_lip",
    "source_latent",
    "shuffled_source_latent",
    "random_norm_matched",
    "oracle_target_latent",
)


def design_fingerprint(config: dict) -> str:
    """Hash all configuration fields that can change generated records."""

    payload = {
        "protocol_version": SOURCE_ONLY_PROTOCOL_VERSION,
        "conditions": config.get("conditions", []),
        "neutral_target_prompt": str(config.get("neutral_target_prompt", "")).strip(),
        "models": config.get("models", {}),
        "prompt_protocol": config.get("prompt_protocol", {}),
        "extraction": config.get("extraction", {}),
        "lip": config.get("lip", {}),
        "adapter": config.get("adapter", {}),
        "data": config.get("data", {}),
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
class ConditionPlan:
    task_id: str
    task_index: int
    condition: str
    target_prompt_kind: str
    vector_kind: Optional[str]
    vector_index: Optional[int]


def validate_conditions(conditions: Iterable[str]) -> list[str]:
    normalized = list(conditions)
    if not normalized:
        raise ValueError("conditions must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("conditions must not contain duplicates")
    unknown = sorted(set(normalized).difference(CONDITIONS))
    if unknown:
        raise ValueError(f"unknown source-only condition(s): {', '.join(unknown)}")
    if "source_latent" not in normalized:
        raise ValueError("conditions must include source_latent")
    if "neutral_no_lip" not in normalized:
        raise ValueError("conditions must include neutral_no_lip")
    return normalized


def derangement_indices(size: int, seed: int) -> list[int]:
    """Generate a deterministic Sattolo derangement for a shuffled-vector control."""

    if size < 2:
        raise ValueError("shuffled_source_latent requires at least two tasks")
    indices = list(range(size))
    rng = random.Random(seed)
    for index in range(size - 1, 0, -1):
        swap_index = rng.randrange(index)
        indices[index], indices[swap_index] = indices[swap_index], indices[index]
    if any(index == value for index, value in enumerate(indices)):
        raise RuntimeError("internal error: shuffled control was not a derangement")
    return indices


def target_prompt_for_condition(
    condition: str,
    task_prompt: str,
    neutral_prompt: str,
) -> str:
    """Return the only target-visible text for a condition."""

    if condition not in CONDITIONS:
        raise ValueError(f"unknown source-only condition: {condition}")
    if condition == "text_only_no_lip":
        return task_prompt
    return neutral_prompt


def build_condition_plan(
    task_ids: Iterable[str],
    conditions: Iterable[str],
    seed: int,
) -> list[ConditionPlan]:
    """Create an auditable task/condition/vector mapping."""

    ids = [str(task_id) for task_id in task_ids]
    if not ids or any(not task_id for task_id in ids):
        raise ValueError("task_ids must contain non-empty identifiers")
    if len(set(ids)) != len(ids):
        raise ValueError("task_ids must be unique")
    selected = validate_conditions(conditions)
    shuffled = (
        derangement_indices(len(ids), seed)
        if "shuffled_source_latent" in selected
        else None
    )

    plan = []
    for task_index, task_id in enumerate(ids):
        for condition in selected:
            target_prompt_kind = (
                "task" if condition == "text_only_no_lip" else "neutral"
            )
            vector_kind = None
            vector_index = None
            if condition == "source_latent":
                vector_kind, vector_index = "translated_source", task_index
            elif condition == "shuffled_source_latent":
                vector_kind, vector_index = "translated_source", shuffled[task_index]
            elif condition == "random_norm_matched":
                vector_kind, vector_index = "random_norm_matched", task_index
            elif condition == "oracle_target_latent":
                vector_kind, vector_index = "target_hidden", task_index

            plan.append(
                ConditionPlan(
                    task_id=task_id,
                    task_index=task_index,
                    condition=condition,
                    target_prompt_kind=target_prompt_kind,
                    vector_kind=vector_kind,
                    vector_index=vector_index,
                )
            )
    return plan


def plan_as_dicts(plan: Iterable[ConditionPlan]) -> list[dict]:
    return [asdict(item) for item in plan]
