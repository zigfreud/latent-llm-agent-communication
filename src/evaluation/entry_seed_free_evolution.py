"""Frozen decision rule for the H0-009 entry-seed free-evolution gate."""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence


ENTRY_SEED_PROTOCOL_VERSION = "lip-entry-seed-free-evolution-v1"
ABSOLUTE_OPERATOR = "absolute_replace"
FREE_EVOLUTION_OPERATOR = "entry_seed_then_free_evolution"


def _metrics(row: Mapping) -> dict[str, float]:
    states = row["oracle_replay_alignment"]["state_summaries"]
    return {
        "attention_output_nrmse": float(
            states["attention_output"]["mean_normalized_rmse"]
        ),
        "residual_output_nrmse": float(
            states["residual_output"]["mean_normalized_rmse"]
        ),
    }


def summarize_entry_seed_gate(
    rows: Sequence[Mapping],
    *,
    variants: Sequence[str],
    seeds: Sequence[int],
    primary_variant: str,
    minimum_taskwise_improvements: int,
    minimum_passing_replicas: int,
    oracle_anchored_delta_reference: Mapping[str, float],
    oracle_maximum_fraction: float,
) -> dict:
    """Require free evolution to beat repeated deltas and learned snapshots."""

    metrics = ("attention_output_nrmse", "residual_output_nrmse")
    if primary_variant not in variants:
        raise ValueError("primary_variant must be one of the configured variants")
    if set(oracle_anchored_delta_reference) != set(metrics):
        raise ValueError("oracle reference must cover both downstream metrics")
    if not 0.0 < oracle_maximum_fraction <= 1.0:
        raise ValueError("oracle maximum fraction must be in (0, 1]")

    oracle_rows = [
        row
        for row in rows
        if row.get("variant") == "oracle"
        and row.get("operator") == FREE_EVOLUTION_OPERATOR
    ]
    if not oracle_rows:
        raise ValueError("missing free-evolution oracle rows")
    oracle_metrics = {}
    for metric in metrics:
        observed = float(statistics.fmean(_metrics(row)[metric] for row in oracle_rows))
        reference = float(oracle_anchored_delta_reference[metric])
        if reference <= 0.0:
            raise ValueError("oracle reference means must be positive")
        ratio = observed / reference
        oracle_metrics[metric] = {
            "anchored_repeated_delta_mean": reference,
            "entry_seed_free_evolution_mean": observed,
            "free_to_repeated_delta_ratio": ratio,
            "maximum_fraction": float(oracle_maximum_fraction),
            "passed": bool(ratio <= oracle_maximum_fraction),
        }
    oracle_free_evolution_passed = all(
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

    replicas = []
    for variant in variants:
        for seed in seeds:
            pairs = []
            for task_id in task_ids:
                absolute_key = (task_id, str(variant), int(seed), ABSOLUTE_OPERATOR)
                free_key = (
                    task_id,
                    str(variant),
                    int(seed),
                    FREE_EVOLUTION_OPERATOR,
                )
                if absolute_key not in by_key or free_key not in by_key:
                    raise ValueError(
                        f"missing paired operator row for {variant} seed {seed} task {task_id}"
                    )
                pairs.append((by_key[absolute_key], by_key[free_key]))
            metric_results = {}
            for metric in metrics:
                absolute_values = [_metrics(absolute)[metric] for absolute, _ in pairs]
                free_values = [_metrics(free)[metric] for _, free in pairs]
                improvements = sum(
                    free < absolute
                    for absolute, free in zip(absolute_values, free_values, strict=True)
                )
                absolute_mean = float(statistics.fmean(absolute_values))
                free_mean = float(statistics.fmean(free_values))
                metric_results[metric] = {
                    "absolute_replace_mean": absolute_mean,
                    "entry_seed_free_evolution_mean": free_mean,
                    "mean_difference_free_minus_absolute": free_mean - absolute_mean,
                    "taskwise_improvement_count": int(improvements),
                    "task_count": len(pairs),
                    "passed": bool(
                        free_mean < absolute_mean
                        and improvements >= minimum_taskwise_improvements
                    ),
                }
            replicas.append(
                {
                    "variant": str(variant),
                    "training_seed": int(seed),
                    "metrics": metric_results,
                    "passed": all(value["passed"] for value in metric_results.values()),
                }
            )

    primary = [row for row in replicas if row["variant"] == primary_variant]
    passing_primary = sum(row["passed"] for row in primary)
    learned_passed = passing_primary >= minimum_passing_replicas
    return {
        "protocol_version": ENTRY_SEED_PROTOCOL_VERSION,
        "comparison": f"{FREE_EVOLUTION_OPERATOR}_vs_{ABSOLUTE_OPERATOR}",
        "jump_metric_role": "diagnostic_only_because_free_evolution_has_no_post_entry_intervention",
        "oracle_free_evolution": {
            "metrics": oracle_metrics,
            "passed": bool(oracle_free_evolution_passed),
        },
        "learned_operator": {
            "primary_variant": primary_variant,
            "minimum_taskwise_improvements": int(minimum_taskwise_improvements),
            "minimum_passing_replicas": int(minimum_passing_replicas),
            "replicas": replicas,
            "passing_primary_replicas": int(passing_primary),
            "passed": bool(learned_passed),
        },
        "advance_to_functional_identity_test": bool(
            oracle_free_evolution_passed and learned_passed
        ),
    }
