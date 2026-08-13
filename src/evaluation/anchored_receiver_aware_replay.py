"""Frozen decision rule for the H0-008 anchored replay gate."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence


ANCHORED_REPLAY_PROTOCOL_VERSION = "lip-anchored-receiver-aware-replay-v1"
ABSOLUTE_OPERATOR = "absolute_replace"
ANCHORED_OPERATOR = "anchored_layer0_replace_then_delta_add"


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


def summarize_anchored_gate(
    rows: Sequence[Mapping],
    *,
    variants: Sequence[str],
    seeds: Sequence[int],
    primary_variant: str,
    minimum_taskwise_improvements: int,
    minimum_passing_replicas: int,
    oracle_unanchored_reference: Mapping[str, float],
    oracle_maximum_fraction: Mapping[str, float],
) -> dict:
    """Apply the preregistered oracle-origin and paired learned gates."""

    metrics = (
        "transition_jump_nrmse",
        "attention_output_nrmse",
        "residual_output_nrmse",
    )
    if primary_variant not in variants:
        raise ValueError("primary_variant must be one of the configured variants")
    if set(oracle_unanchored_reference) != set(metrics):
        raise ValueError("oracle reference must cover every frozen metric")
    if set(oracle_maximum_fraction) != set(metrics):
        raise ValueError("oracle fractions must cover every frozen metric")

    oracle_rows = [
        row
        for row in rows
        if row.get("variant") == "oracle"
        and row.get("operator") == ANCHORED_OPERATOR
    ]
    if not oracle_rows:
        raise ValueError("missing anchored oracle diagnostic rows")
    oracle_metrics = {}
    for metric in metrics:
        observed = float(statistics.fmean(_metrics(row)[metric] for row in oracle_rows))
        reference = float(oracle_unanchored_reference[metric])
        maximum_fraction = float(oracle_maximum_fraction[metric])
        if reference <= 0.0 or not 0.0 < maximum_fraction <= 1.0:
            raise ValueError("oracle references and maximum fractions must be positive")
        ratio = observed / reference
        oracle_metrics[metric] = {
            "unanchored_oracle_mean": reference,
            "anchored_oracle_mean": observed,
            "anchored_to_unanchored_ratio": ratio,
            "maximum_fraction": maximum_fraction,
            "passed": bool(ratio <= maximum_fraction),
        }
    oracle_entry_origin_passed = all(
        result["passed"] for result in oracle_metrics.values()
    )

    learned = [row for row in rows if row.get("variant") in variants]
    by_key = {
        (
            str(row["task_id"]),
            str(row["variant"]),
            int(row["training_seed"]),
            str(row["operator"]),
        ): row
        for row in learned
    }
    task_ids = sorted({str(row["task_id"]) for row in learned})
    if minimum_taskwise_improvements > len(task_ids):
        raise ValueError("taskwise threshold exceeds the observed task count")

    replica_results = []
    for variant in variants:
        for seed in seeds:
            pairs = []
            for task_id in task_ids:
                absolute_key = (task_id, str(variant), int(seed), ABSOLUTE_OPERATOR)
                anchored_key = (task_id, str(variant), int(seed), ANCHORED_OPERATOR)
                if absolute_key not in by_key or anchored_key not in by_key:
                    raise ValueError(
                        f"missing paired operator row for {variant} seed {seed} task {task_id}"
                    )
                pairs.append((by_key[absolute_key], by_key[anchored_key]))
            metric_results = {}
            for metric in metrics:
                absolute_values = [_metrics(absolute)[metric] for absolute, _ in pairs]
                anchored_values = [_metrics(anchored)[metric] for _, anchored in pairs]
                improvement_count = sum(
                    anchored < absolute
                    for absolute, anchored in zip(
                        absolute_values, anchored_values, strict=True
                    )
                )
                absolute_mean = float(statistics.fmean(absolute_values))
                anchored_mean = float(statistics.fmean(anchored_values))
                metric_results[metric] = {
                    "absolute_replace_mean": absolute_mean,
                    "anchored_hybrid_mean": anchored_mean,
                    "mean_difference_anchored_minus_absolute": (
                        anchored_mean - absolute_mean
                    ),
                    "taskwise_improvement_count": int(improvement_count),
                    "task_count": len(pairs),
                    "passed": bool(
                        anchored_mean < absolute_mean
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
    learned_gate_passed = passing_primary_replicas >= minimum_passing_replicas
    return {
        "protocol_version": ANCHORED_REPLAY_PROTOCOL_VERSION,
        "comparison": f"{ANCHORED_OPERATOR}_vs_{ABSOLUTE_OPERATOR}",
        "direction": "lower_is_better_for_all_metrics",
        "oracle_entry_origin": {
            "metrics": oracle_metrics,
            "passed": bool(oracle_entry_origin_passed),
        },
        "learned_operator": {
            "primary_variant": primary_variant,
            "minimum_taskwise_improvements": int(minimum_taskwise_improvements),
            "minimum_passing_replicas": int(minimum_passing_replicas),
            "replicas": replica_results,
            "passing_primary_replicas": int(passing_primary_replicas),
            "passed": bool(learned_gate_passed),
        },
        "advance_to_functional_identity_test": bool(
            oracle_entry_origin_passed and learned_gate_passed
        ),
    }
