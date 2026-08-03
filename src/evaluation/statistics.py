"""Task-clustered uncertainty estimates for repeated code generations."""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute a mean of zero values")
    return math.fsum(values) / len(values)


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of zero values")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _numeric(value) -> float:
    if isinstance(value, bool):
        return float(value)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("metric values must be finite")
    return number


def task_means(
    records: Iterable[Mapping],
    condition: str,
    metric: str,
) -> dict[str, float]:
    grouped = defaultdict(list)
    for record in records:
        if record.get("condition") != condition or record.get(metric) is None:
            continue
        task_id = str(record.get("task_id", ""))
        if not task_id:
            raise ValueError("every scored record must include task_id")
        grouped[task_id].append(_numeric(record[metric]))
    return {task_id: mean(values) for task_id, values in grouped.items()}


def _training_seed_values(
    records: Sequence[Mapping],
    condition: str,
    metric: str,
) -> dict[str, dict[str, float]]:
    seeds = sorted(
        {
            str(record["training_seed"])
            for record in records
            if record.get("condition") == condition
            and record.get(metric) is not None
            and record.get("training_seed") is not None
        }
    )
    return {
        seed: task_means(
            [record for record in records if str(record.get("training_seed")) == seed],
            condition,
            metric,
        )
        for seed in seeds
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one task")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    rng = random.Random(seed)
    count = len(values)
    estimates = [
        mean([values[rng.randrange(count)] for _ in range(count)])
        for _ in range(iterations)
    ]
    alpha = (1.0 - confidence) / 2.0
    return percentile(estimates, alpha), percentile(estimates, 1.0 - alpha)


def sign_flip_p_value(
    differences: Sequence[float],
    *,
    monte_carlo_samples: int = 100_000,
    seed: int = 2718,
) -> tuple[float, str]:
    """Two-sided paired randomization test over task-level differences."""

    if not differences:
        raise ValueError("paired test requires at least one task")
    observed = abs(mean(differences))
    tolerance = 1e-15
    count = len(differences)

    if count <= 20:
        total = 1 << count
        extreme = 0
        for signs in itertools.product((-1.0, 1.0), repeat=count):
            statistic = abs(mean([d * sign for d, sign in zip(differences, signs)]))
            extreme += int(statistic + tolerance >= observed)
        return extreme / total, "exact"

    if monte_carlo_samples <= 0:
        raise ValueError("monte_carlo_samples must be positive")
    rng = random.Random(seed)
    extreme = 0
    for _ in range(monte_carlo_samples):
        statistic = abs(
            mean([value if rng.getrandbits(1) else -value for value in differences])
        )
        extreme += int(statistic + tolerance >= observed)
    return (extreme + 1) / (monte_carlo_samples + 1), "monte_carlo"


def summarize_metric(
    records: Sequence[Mapping],
    metric: str,
    conditions: Sequence[str],
    comparisons: Sequence[Sequence[str]],
    *,
    bootstrap_iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> dict:
    """Summarize conditions and paired differences, clustering by task."""

    condition_summary = {}
    cached = {}
    for offset, condition in enumerate(conditions):
        per_task = task_means(records, condition, metric)
        if not per_task:
            continue
        values = list(per_task.values())
        lower, upper = bootstrap_mean_ci(
            values,
            iterations=bootstrap_iterations,
            confidence=confidence,
            seed=seed + offset,
        )
        cached[condition] = per_task
        per_seed = _training_seed_values(records, condition, metric)
        condition_summary[condition] = {
            "task_count": len(values),
            "observation_count": sum(
                1
                for record in records
                if record.get("condition") == condition and record.get(metric) is not None
            ),
            "mean": mean(values),
            "ci_lower": lower,
            "ci_upper": upper,
            "by_training_seed": {
                training_seed: {
                    "task_count": len(seed_tasks),
                    "mean": mean(list(seed_tasks.values())),
                }
                for training_seed, seed_tasks in per_seed.items()
                if seed_tasks
            },
        }

    comparison_summary = []
    for offset, comparison in enumerate(comparisons):
        if len(comparison) != 2:
            raise ValueError("each comparison must contain [treatment, control]")
        treatment, control = comparison
        treatment_tasks = cached.get(treatment) or task_means(records, treatment, metric)
        control_tasks = cached.get(control) or task_means(records, control, metric)
        shared = sorted(set(treatment_tasks).intersection(control_tasks))
        if not shared:
            raise ValueError(f"comparison {treatment} vs {control} has no shared tasks")
        differences = [
            treatment_tasks[task_id] - control_tasks[task_id]
            for task_id in shared
        ]
        lower, upper = bootstrap_mean_ci(
            differences,
            iterations=bootstrap_iterations,
            confidence=confidence,
            seed=seed + 1000 + offset,
        )
        p_value, method = sign_flip_p_value(differences, seed=seed + 2000 + offset)
        treatment_seeds = _training_seed_values(records, treatment, metric)
        control_seeds = _training_seed_values(records, control, metric)
        per_seed_differences = {}
        for training_seed in sorted(set(treatment_seeds).intersection(control_seeds)):
            shared_seed_tasks = sorted(
                set(treatment_seeds[training_seed]).intersection(
                    control_seeds[training_seed]
                )
            )
            if shared_seed_tasks:
                seed_differences = [
                    treatment_seeds[training_seed][task_id]
                    - control_seeds[training_seed][task_id]
                    for task_id in shared_seed_tasks
                ]
                per_seed_differences[training_seed] = {
                    "task_count": len(shared_seed_tasks),
                    "mean_difference": mean(seed_differences),
                }
        comparison_summary.append(
            {
                "treatment": treatment,
                "control": control,
                "task_count": len(shared),
                "mean_difference": mean(differences),
                "ci_lower": lower,
                "ci_upper": upper,
                "p_value_two_sided": p_value,
                "p_value_method": method,
                "by_training_seed": per_seed_differences,
            }
        )

    if comparison_summary:
        ordered = sorted(
            range(len(comparison_summary)),
            key=lambda index: comparison_summary[index]["p_value_two_sided"],
        )
        running_max = 0.0
        comparison_count = len(comparison_summary)
        for rank, index in enumerate(ordered):
            raw = comparison_summary[index]["p_value_two_sided"]
            adjusted = min(1.0, (comparison_count - rank) * raw)
            running_max = max(running_max, adjusted)
            comparison_summary[index]["p_value_holm"] = running_max

    return {
        "metric": metric,
        "cluster_unit": "task_id",
        "replicates_within_task": "averaged before task-level interval and test",
        "confidence": confidence,
        "bootstrap_iterations": bootstrap_iterations,
        "multiplicity_adjustment": "Holm across configured comparisons",
        "conditions": condition_summary,
        "comparisons": comparison_summary,
    }
