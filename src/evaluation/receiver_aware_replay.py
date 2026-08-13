"""Frozen decision rule for the H0-007 receiver-aware replay gate."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence


RECEIVER_AWARE_REPLAY_PROTOCOL_VERSION = "lip-receiver-aware-replay-operator-v1"


def _metrics(row: Mapping) -> dict[str, float]:
    alignment = row["oracle_replay_alignment"]["state_summaries"]
    return {
        "transition_jump_nrmse": float(
            row["intervention_jump"]["transition_summary"][
                "mean_relative_jump_rms"
            ]
        ),
        "attention_output_nrmse": float(
            alignment["attention_output"]["mean_normalized_rmse"]
        ),
        "residual_output_nrmse": float(
            alignment["residual_output"]["mean_normalized_rmse"]
        ),
    }


def summarize_receiver_aware_gate(
    rows: Sequence[Mapping],
    *,
    variants: Sequence[str],
    seeds: Sequence[int],
    primary_variant: str,
    minimum_taskwise_improvements: int,
    minimum_passing_replicas: int,
) -> dict:
    """Apply the preregistered paired add-versus-replace decision rule."""

    if primary_variant not in variants:
        raise ValueError("primary_variant must be one of the configured variants")
    if minimum_taskwise_improvements <= 0 or minimum_passing_replicas <= 0:
        raise ValueError("gate thresholds must be positive")
    learned = [row for row in rows if row.get("variant") in variants]
    by_key = {
        (str(row["task_id"]), str(row["variant"]), int(row["training_seed"]), str(row["operator"])): row
        for row in learned
    }
    task_ids = sorted({str(row["task_id"]) for row in learned})
    if minimum_taskwise_improvements > len(task_ids):
        raise ValueError("taskwise threshold exceeds the observed task count")

    metrics = (
        "transition_jump_nrmse",
        "attention_output_nrmse",
        "residual_output_nrmse",
    )
    replica_results = []
    for variant in variants:
        for seed in seeds:
            pairs = []
            for task_id in task_ids:
                replace_key = (task_id, str(variant), int(seed), "absolute_replace")
                add_key = (task_id, str(variant), int(seed), "live_task_delta_add")
                if replace_key not in by_key or add_key not in by_key:
                    raise ValueError(
                        f"missing paired operator row for {variant} seed {seed} task {task_id}"
                    )
                pairs.append((by_key[replace_key], by_key[add_key]))
            metric_results = {}
            for metric in metrics:
                replace_values = [_metrics(replace)[metric] for replace, _ in pairs]
                add_values = [_metrics(add)[metric] for _, add in pairs]
                improvement_count = sum(
                    add < replace
                    for replace, add in zip(replace_values, add_values, strict=True)
                )
                replace_mean = float(statistics.fmean(replace_values))
                add_mean = float(statistics.fmean(add_values))
                metric_results[metric] = {
                    "absolute_replace_mean": replace_mean,
                    "live_task_delta_add_mean": add_mean,
                    "mean_difference_add_minus_replace": add_mean - replace_mean,
                    "taskwise_improvement_count": int(improvement_count),
                    "task_count": len(pairs),
                    "passed": bool(
                        add_mean < replace_mean
                        and improvement_count >= minimum_taskwise_improvements
                    ),
                }
            replica_results.append(
                {
                    "variant": str(variant),
                    "training_seed": int(seed),
                    "metrics": metric_results,
                    "passed": all(result["passed"] for result in metric_results.values()),
                }
            )

    primary_replicas = [
        row for row in replica_results if row["variant"] == primary_variant
    ]
    passing_primary_replicas = sum(row["passed"] for row in primary_replicas)
    return {
        "protocol_version": RECEIVER_AWARE_REPLAY_PROTOCOL_VERSION,
        "comparison": "live_task_delta_add_vs_absolute_replace",
        "direction": "lower_is_better_for_all_metrics",
        "primary_variant": primary_variant,
        "minimum_taskwise_improvements": int(minimum_taskwise_improvements),
        "minimum_passing_replicas": int(minimum_passing_replicas),
        "replicas": replica_results,
        "passing_primary_replicas": int(passing_primary_replicas),
        "advance_to_functional_identity_test": bool(
            passing_primary_replicas >= minimum_passing_replicas
        ),
    }
