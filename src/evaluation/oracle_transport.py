"""Pure metrics and decision rules for the target-oracle transport audit."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


def normalize_layer_indices(layers: Iterable[int], layer_count: int) -> list[int]:
    """Validate negative transformer-layer indices without changing their order."""

    if layer_count <= 0:
        raise ValueError("layer_count must be positive")
    normalized = [int(layer) for layer in layers]
    if not normalized:
        raise ValueError("at least one audit layer is required")
    if len(set(normalized)) != len(normalized):
        raise ValueError("audit layers must be unique")
    if any(layer >= 0 or layer < -layer_count for layer in normalized):
        raise ValueError(
            f"audit layers must be negative indices in [-{layer_count}, -1]"
        )
    return normalized


def continuation_token_metrics(
    logits: torch.Tensor,
    reference_ids: torch.Tensor,
    prompt_length: int,
) -> dict[str, float | int]:
    """Score reference tokens after a prompt under causal next-token alignment."""

    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("logits must have shape (1, sequence, vocabulary)")
    if reference_ids.ndim != 1 or reference_ids.numel() == 0:
        raise ValueError("reference_ids must be a non-empty rank-1 tensor")
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")

    reference_length = int(reference_ids.numel())
    start = prompt_length - 1
    stop = start + reference_length
    if stop > logits.shape[1]:
        raise ValueError("logits do not cover every reference token")

    aligned = logits[0, start:stop, :].float()
    targets = reference_ids.to(device=aligned.device, dtype=torch.long)
    nll = F.cross_entropy(aligned, targets, reduction="mean")
    accuracy = aligned.argmax(dim=-1).eq(targets).float().mean()
    return {
        "token_count": reference_length,
        "nll": float(nll.detach().cpu().item()),
        "top1_accuracy": float(accuracy.detach().cpu().item()),
    }


def recovery_fraction(
    task_nll: float,
    neutral_nll: float,
    injected_nll: float,
    *,
    minimum_task_advantage: float,
) -> tuple[float | None, float, bool]:
    """Return recovered task-prompt advantage, its denominator, and informativeness."""

    advantage = float(neutral_nll) - float(task_nll)
    informative = advantage >= float(minimum_task_advantage)
    if not informative:
        return None, advantage, False
    recovery = (float(neutral_nll) - float(injected_nll)) / advantage
    return recovery, advantage, True


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def summarize_oracle_transport(
    records: Sequence[Mapping[str, Any]],
    *,
    task_ids: Sequence[str],
    layers: Sequence[int],
    selection_task_count: int,
    minimum_informative_tasks_per_split: int,
    minimum_confirmation_recovery: float,
    maximum_self_nll_delta: float,
    run_scope: str,
) -> dict[str, Any]:
    """Select a layer on one task split and apply the gate on a disjoint split."""

    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be a non-empty unique sequence")
    if not 0 < selection_task_count < len(task_ids):
        raise ValueError("selection_task_count must leave a non-empty confirmation split")
    if minimum_informative_tasks_per_split <= 0:
        raise ValueError("minimum_informative_tasks_per_split must be positive")

    expected = {(task_id, int(layer)) for task_id in task_ids for layer in layers}
    observed = [(str(row["task_id"]), int(row["layer_idx"])) for row in records]
    duplicates = [key for key, count in Counter(observed).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate task/layer record(s): {duplicates[:3]}")
    missing = expected.difference(observed)
    unexpected = set(observed).difference(expected)
    if missing or unexpected:
        raise ValueError(
            f"audit record grid mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )

    selection_ids = set(task_ids[:selection_task_count])
    confirmation_ids = set(task_ids[selection_task_count:])
    by_layer: dict[str, Any] = {}
    for layer in layers:
        layer_rows = [row for row in records if int(row["layer_idx"]) == int(layer)]
        split_metrics: dict[str, Any] = {}
        for split_name, split_ids in (
            ("selection", selection_ids),
            ("confirmation", confirmation_ids),
        ):
            split_rows = [row for row in layer_rows if str(row["task_id"]) in split_ids]
            informative = [row for row in split_rows if bool(row["informative"])]
            recoveries = [float(row["recovery_fraction"]) for row in informative]
            split_metrics[split_name] = {
                "task_count": len(split_rows),
                "informative_task_count": len(informative),
                "mean_task_advantage_nll": _mean(
                    [float(row["task_advantage_nll"]) for row in split_rows]
                ),
                "mean_recovery_fraction": _mean(recoveries),
            }
        by_layer[str(layer)] = split_metrics

    selectable = [
        int(layer)
        for layer in layers
        if by_layer[str(layer)]["selection"]["informative_task_count"]
        >= minimum_informative_tasks_per_split
        and by_layer[str(layer)]["selection"]["mean_recovery_fraction"] is not None
    ]
    selected_layer = max(
        selectable,
        key=lambda layer: (
            by_layer[str(layer)]["selection"]["mean_recovery_fraction"],
            -layers.index(layer),
        ),
        default=None,
    )

    self_deltas = [
        abs(float(row["self_nll_delta"]))
        for row in records
        if row.get("self_nll_delta") is not None
    ]
    self_check_max = max(self_deltas, default=None)
    self_check_pass = (
        self_check_max is not None and self_check_max <= maximum_self_nll_delta
    )

    confirmation = (
        by_layer[str(selected_layer)]["confirmation"]
        if selected_layer is not None
        else None
    )
    confirmation_has_support = bool(
        confirmation
        and confirmation["informative_task_count"]
        >= minimum_informative_tasks_per_split
    )
    confirmation_recovery = (
        confirmation["mean_recovery_fraction"] if confirmation_has_support else None
    )
    recovery_pass = bool(
        confirmation_recovery is not None
        and confirmation_recovery >= minimum_confirmation_recovery
    )
    claim_eligible = run_scope == "full"

    return {
        "run_scope": run_scope,
        "claim_eligible": claim_eligible,
        "task_count": len(task_ids),
        "selection_task_ids": list(task_ids[:selection_task_count]),
        "confirmation_task_ids": list(task_ids[selection_task_count:]),
        "layers": [int(layer) for layer in layers],
        "by_layer": by_layer,
        "selected_layer": selected_layer,
        "self_reconstruction": {
            "observations": len(self_deltas),
            "maximum_absolute_nll_delta": self_check_max,
            "threshold": maximum_self_nll_delta,
            "passed": self_check_pass,
        },
        "confirmation": confirmation,
        "gate": {
            "minimum_informative_tasks_per_split": minimum_informative_tasks_per_split,
            "minimum_confirmation_recovery": minimum_confirmation_recovery,
            "selection_supported": selected_layer is not None,
            "confirmation_supported": confirmation_has_support,
            "recovery_passed": recovery_pass,
            "passed": bool(
                claim_eligible
                and self_check_pass
                and selected_layer is not None
                and confirmation_has_support
                and recovery_pass
            ),
        },
    }


def summarize_packet_capacity(
    records: Sequence[Mapping[str, Any]],
    *,
    task_ids: Sequence[str],
    packet_sizes: Sequence[int],
    selection_task_count: int,
    minimum_informative_tasks_per_split: int,
    minimum_recovery: float,
    maximum_self_nll_delta: float,
    run_scope: str,
) -> dict[str, Any]:
    """Select the smallest sufficient packet and confirm it on disjoint tasks."""

    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be a non-empty unique sequence")
    sizes = [int(size) for size in packet_sizes]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("packet_sizes must be positive")
    if sizes != sorted(set(sizes)):
        raise ValueError("packet_sizes must be strictly increasing and unique")
    if not 0 < selection_task_count < len(task_ids):
        raise ValueError("selection_task_count must leave a non-empty confirmation split")
    if minimum_informative_tasks_per_split <= 0:
        raise ValueError("minimum_informative_tasks_per_split must be positive")

    expected = {(task_id, size) for task_id in task_ids for size in sizes}
    observed = [(str(row["task_id"]), int(row["packet_size"])) for row in records]
    duplicates = [key for key, count in Counter(observed).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate task/packet record(s): {duplicates[:3]}")
    missing = expected.difference(observed)
    unexpected = set(observed).difference(expected)
    if missing or unexpected:
        raise ValueError(
            f"packet record grid mismatch: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    selection_ids = set(task_ids[:selection_task_count])
    confirmation_ids = set(task_ids[selection_task_count:])
    by_packet_size: dict[str, Any] = {}
    for size in sizes:
        size_rows = [row for row in records if int(row["packet_size"]) == size]
        split_metrics: dict[str, Any] = {}
        for split_name, split_ids in (
            ("selection", selection_ids),
            ("confirmation", confirmation_ids),
        ):
            split_rows = [row for row in size_rows if str(row["task_id"]) in split_ids]
            informative = [row for row in split_rows if bool(row["informative"])]
            split_metrics[split_name] = {
                "task_count": len(split_rows),
                "informative_task_count": len(informative),
                "mean_task_advantage_nll": _mean(
                    [float(row["task_advantage_nll"]) for row in split_rows]
                ),
                "mean_recovery_fraction": _mean(
                    [float(row["recovery_fraction"]) for row in informative]
                ),
            }
        by_packet_size[str(size)] = split_metrics

    selected_packet_size = next(
        (
            size
            for size in sizes
            if by_packet_size[str(size)]["selection"]["informative_task_count"]
            >= minimum_informative_tasks_per_split
            and by_packet_size[str(size)]["selection"]["mean_recovery_fraction"]
            is not None
            and by_packet_size[str(size)]["selection"]["mean_recovery_fraction"]
            >= minimum_recovery
        ),
        None,
    )

    self_deltas = [
        abs(float(row["self_nll_delta"]))
        for row in records
        if row.get("self_nll_delta") is not None
    ]
    self_check_max = max(self_deltas, default=None)
    self_check_pass = (
        self_check_max is not None and self_check_max <= maximum_self_nll_delta
    )
    confirmation = (
        by_packet_size[str(selected_packet_size)]["confirmation"]
        if selected_packet_size is not None
        else None
    )
    confirmation_has_support = bool(
        confirmation
        and confirmation["informative_task_count"]
        >= minimum_informative_tasks_per_split
    )
    confirmation_recovery = (
        confirmation["mean_recovery_fraction"] if confirmation_has_support else None
    )
    confirmation_pass = bool(
        confirmation_recovery is not None and confirmation_recovery >= minimum_recovery
    )
    claim_eligible = run_scope == "full"

    return {
        "run_scope": run_scope,
        "claim_eligible": claim_eligible,
        "task_count": len(task_ids),
        "selection_task_ids": list(task_ids[:selection_task_count]),
        "confirmation_task_ids": list(task_ids[selection_task_count:]),
        "packet_sizes": sizes,
        "by_packet_size": by_packet_size,
        "selected_packet_size": selected_packet_size,
        "self_reconstruction": {
            "observations": len(self_deltas),
            "maximum_absolute_nll_delta": self_check_max,
            "threshold": maximum_self_nll_delta,
            "passed": self_check_pass,
        },
        "confirmation": confirmation,
        "gate": {
            "minimum_informative_tasks_per_split": minimum_informative_tasks_per_split,
            "minimum_recovery": minimum_recovery,
            "selection_supported": selected_packet_size is not None,
            "confirmation_supported": confirmation_has_support,
            "recovery_passed": confirmation_pass,
            "passed": bool(
                claim_eligible
                and self_check_pass
                and selected_packet_size is not None
                and confirmation_has_support
                and confirmation_pass
            ),
        },
    }
