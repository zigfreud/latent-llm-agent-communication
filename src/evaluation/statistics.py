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


def _seed_values(
    records: Sequence[Mapping],
    condition: str,
    metric: str,
    seed_field: str,
) -> dict[str, dict[str, float]]:
    seeds = sorted(
        {
            str(record[seed_field])
            for record in records
            if record.get("condition") == condition
            and record.get(metric) is not None
            and record.get(seed_field) is not None
        }
    )
    return {
        seed: task_means(
            [record for record in records if str(record.get(seed_field)) == seed],
            condition,
            metric,
        )
        for seed in seeds
    }


def _training_seed_values(
    records: Sequence[Mapping],
    condition: str,
    metric: str,
) -> dict[str, dict[str, float]]:
    return _seed_values(records, condition, metric, "training_seed")


def _generation_seed_values(
    records: Sequence[Mapping],
    condition: str,
    metric: str,
) -> dict[str, dict[str, float]]:
    return _seed_values(records, condition, metric, "generation_seed")


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
    alternative: str = "two-sided",
    monte_carlo_samples: int = 100_000,
    seed: int = 2718,
) -> tuple[float, str]:
    """Paired randomization test over task-level differences."""

    if not differences:
        raise ValueError("paired test requires at least one task")
    if alternative not in {"two-sided", "greater"}:
        raise ValueError("alternative must be 'two-sided' or 'greater'")
    tolerance = 1e-15
    nonzero = [float(value) for value in differences if abs(float(value)) > tolerance]
    if not nonzero:
        return 1.0, "exact"
    observed_mean = mean(nonzero)
    observed = abs(observed_mean) if alternative == "two-sided" else observed_mean
    count = len(nonzero)

    def is_extreme(statistic: float) -> bool:
        value = abs(statistic) if alternative == "two-sided" else statistic
        return value + tolerance >= observed

    if count <= 20:
        total = 1 << count
        extreme = 0
        for signs in itertools.product((-1.0, 1.0), repeat=count):
            statistic = mean([d * sign for d, sign in zip(nonzero, signs)])
            extreme += int(is_extreme(statistic))
        return extreme / total, "exact"

    if monte_carlo_samples <= 0:
        raise ValueError("monte_carlo_samples must be positive")
    rng = random.Random(seed)
    extreme = 0
    for _ in range(monte_carlo_samples):
        statistic = mean(
            [value if rng.getrandbits(1) else -value for value in nonzero]
        )
        extreme += int(is_extreme(statistic))
    return (extreme + 1) / (monte_carlo_samples + 1), "monte_carlo"


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in the input order."""

    values = [float(value) for value in p_values]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("p-values must be finite and between zero and one")
    ordered = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running_max = 0.0
    count = len(values)
    for rank, index in enumerate(ordered):
        running_max = max(running_max, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running_max
    return adjusted


def summarize_fixed_sequence(
    records: Sequence[Mapping],
    metric: str,
    hypotheses: Sequence[Sequence[str]],
    *,
    alpha: float = 0.05,
    alternative: str = "greater",
    bootstrap_iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> dict:
    """Test an ordered family, stopping confirmatory rejection at first failure."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if not hypotheses:
        raise ValueError("fixed sequence requires at least one hypothesis")
    cached: dict[str, dict[str, float]] = {}
    results = []
    sequence_active = True
    for offset, hypothesis in enumerate(hypotheses):
        if len(hypothesis) != 2:
            raise ValueError("each hypothesis must contain [treatment, control]")
        treatment, control = (str(value) for value in hypothesis)
        treatment_tasks = cached.setdefault(
            treatment, task_means(records, treatment, metric)
        )
        control_tasks = cached.setdefault(control, task_means(records, control, metric))
        shared = sorted(set(treatment_tasks).intersection(control_tasks))
        if not shared:
            raise ValueError(f"hypothesis {treatment} vs {control} has no shared tasks")
        differences = [
            treatment_tasks[task_id] - control_tasks[task_id] for task_id in shared
        ]
        lower, upper = bootstrap_mean_ci(
            differences,
            iterations=bootstrap_iterations,
            confidence=confidence,
            seed=seed + 3000 + offset,
        )
        p_value, method = sign_flip_p_value(
            differences,
            alternative=alternative,
            seed=seed + 4000 + offset,
        )
        tested = sequence_active
        rejected = bool(tested and p_value <= alpha)
        results.append(
            {
                "sequence_index": offset,
                "treatment": treatment,
                "control": control,
                "task_count": len(shared),
                "nonzero_task_count": sum(
                    abs(difference) > 1e-15 for difference in differences
                ),
                "mean_difference": mean(differences),
                "ci_lower": lower,
                "ci_upper": upper,
                "p_value": p_value,
                "p_value_method": method,
                "alternative": alternative,
                "tested": tested,
                "rejected": rejected,
            }
        )
        sequence_active = bool(sequence_active and rejected)
    return {
        "metric": metric,
        "method": "fixed_sequence_gatekeeping",
        "familywise_alpha": alpha,
        "alternative": alternative,
        "cluster_unit": "task_id",
        "replicates_within_task": "averaged before task-level test",
        "stopping_rule": "stop confirmatory testing after the first non-rejection",
        "hypotheses": results,
    }


def summarize_gatekept_holm(
    records: Sequence[Mapping],
    metric: str,
    anchor: Sequence[str],
    family: Sequence[Sequence[str]],
    *,
    alpha: float = 0.05,
    alternative: str = "greater",
    bootstrap_iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> dict:
    """Gate a Holm-adjusted hypothesis family behind one anchor contrast."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    hypotheses = [anchor, *family]
    if len(anchor) != 2 or not family or any(len(pair) != 2 for pair in family):
        raise ValueError(
            "anchor and family hypotheses must contain [treatment, control]"
        )

    cached: dict[str, dict[str, float]] = {}
    results = []
    for offset, hypothesis in enumerate(hypotheses):
        treatment, control = (str(value) for value in hypothesis)
        treatment_tasks = cached.setdefault(
            treatment, task_means(records, treatment, metric)
        )
        control_tasks = cached.setdefault(control, task_means(records, control, metric))
        shared = sorted(set(treatment_tasks).intersection(control_tasks))
        if not shared:
            raise ValueError(f"hypothesis {treatment} vs {control} has no shared tasks")
        differences = [
            treatment_tasks[task_id] - control_tasks[task_id] for task_id in shared
        ]
        lower, upper = bootstrap_mean_ci(
            differences,
            iterations=bootstrap_iterations,
            confidence=confidence,
            seed=seed + 3000 + offset,
        )
        p_value, method = sign_flip_p_value(
            differences,
            alternative=alternative,
            seed=seed + 4000 + offset,
        )
        results.append(
            {
                "treatment": treatment,
                "control": control,
                "task_count": len(shared),
                "nonzero_task_count": sum(
                    abs(difference) > 1e-15 for difference in differences
                ),
                "mean_difference": mean(differences),
                "ci_lower": lower,
                "ci_upper": upper,
                "p_value": p_value,
                "p_value_method": method,
                "alternative": alternative,
            }
        )

    anchor_result = results[0]
    anchor_result["tested"] = True
    anchor_result["rejected"] = bool(anchor_result["p_value"] <= alpha)
    adjusted = holm_adjust([item["p_value"] for item in results[1:]])
    family_results = []
    for item, adjusted_p in zip(results[1:], adjusted):
        item["p_value_holm"] = adjusted_p
        item["tested"] = anchor_result["rejected"]
        item["rejected"] = bool(item["tested"] and adjusted_p <= alpha)
        family_results.append(item)

    return {
        "metric": metric,
        "method": "anchor_gate_then_holm",
        "familywise_alpha": alpha,
        "alternative": alternative,
        "cluster_unit": "task_id",
        "replicates_within_task": "averaged before task-level test",
        "stopping_rule": (
            "test the anchor first; open the Holm-adjusted family only after "
            "anchor rejection"
        ),
        "anchor": anchor_result,
        "family": family_results,
    }


def summarize_two_gate_holm(
    records: Sequence[Mapping],
    metric: str,
    gates: Sequence[Sequence[str]],
    family: Sequence[Sequence[str]],
    *,
    alpha: float = 0.05,
    alternative: str = "greater",
    bootstrap_iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 1729,
) -> dict:
    """Open one Holm family only after two ordered replication gates pass."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if len(gates) != 2 or any(len(pair) != 2 for pair in gates):
        raise ValueError("two ordered gates must contain [treatment, control]")
    if not family or any(len(pair) != 2 for pair in family):
        raise ValueError("family hypotheses must contain [treatment, control]")

    cached: dict[str, dict[str, float]] = {}
    results = []
    for offset, hypothesis in enumerate([*gates, *family]):
        treatment, control = (str(value) for value in hypothesis)
        treatment_tasks = cached.setdefault(
            treatment, task_means(records, treatment, metric)
        )
        control_tasks = cached.setdefault(
            control, task_means(records, control, metric)
        )
        shared = sorted(set(treatment_tasks).intersection(control_tasks))
        if not shared:
            raise ValueError(f"hypothesis {treatment} vs {control} has no shared tasks")
        differences = [
            treatment_tasks[task_id] - control_tasks[task_id]
            for task_id in shared
        ]
        lower, upper = bootstrap_mean_ci(
            differences,
            iterations=bootstrap_iterations,
            confidence=confidence,
            seed=seed + 3000 + offset,
        )
        p_value, method = sign_flip_p_value(
            differences,
            alternative=alternative,
            seed=seed + 4000 + offset,
        )
        results.append(
            {
                "treatment": treatment,
                "control": control,
                "task_count": len(shared),
                "nonzero_task_count": sum(
                    abs(difference) > 1e-15 for difference in differences
                ),
                "mean_difference": mean(differences),
                "ci_lower": lower,
                "ci_upper": upper,
                "p_value": p_value,
                "p_value_method": method,
                "alternative": alternative,
            }
        )

    gate_results = results[:2]
    sequence_active = True
    for index, item in enumerate(gate_results):
        item["gate_index"] = index
        item["tested"] = sequence_active
        item["rejected"] = bool(sequence_active and item["p_value"] <= alpha)
        sequence_active = bool(sequence_active and item["rejected"])

    raw_family = results[2:]
    adjusted = holm_adjust([item["p_value"] for item in raw_family])
    family_results = []
    for item, adjusted_p in zip(raw_family, adjusted):
        item["p_value_holm"] = adjusted_p
        item["tested"] = sequence_active
        item["rejected"] = bool(sequence_active and adjusted_p <= alpha)
        family_results.append(item)

    return {
        "metric": metric,
        "method": "two_gate_then_holm",
        "familywise_alpha": alpha,
        "alternative": alternative,
        "cluster_unit": "task_id",
        "replicates_within_task": "averaged before task-level test",
        "stopping_rule": (
            "test both replication gates in order; open one Holm-adjusted "
            "component family only after both reject"
        ),
        "gates": gate_results,
        "family": family_results,
    }


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
        per_generation_seed = _generation_seed_values(records, condition, metric)
        condition_summary[condition] = {
            "task_count": len(values),
            "observation_count": sum(
                1
                for record in records
                if record.get("condition") == condition
                and record.get(metric) is not None
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
            "by_generation_seed": {
                generation_seed: {
                    "task_count": len(seed_tasks),
                    "mean": mean(list(seed_tasks.values())),
                }
                for generation_seed, seed_tasks in per_generation_seed.items()
                if seed_tasks
            },
        }

    comparison_summary = []
    for offset, comparison in enumerate(comparisons):
        if len(comparison) != 2:
            raise ValueError("each comparison must contain [treatment, control]")
        treatment, control = comparison
        treatment_tasks = cached.get(treatment) or task_means(
            records, treatment, metric
        )
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
        treatment_generation_seeds = _generation_seed_values(
            records, treatment, metric
        )
        control_generation_seeds = _generation_seed_values(records, control, metric)
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
        per_generation_seed_differences = {}
        for generation_seed in sorted(
            set(treatment_generation_seeds).intersection(control_generation_seeds)
        ):
            shared_seed_tasks = sorted(
                set(treatment_generation_seeds[generation_seed]).intersection(
                    control_generation_seeds[generation_seed]
                )
            )
            if shared_seed_tasks:
                seed_differences = [
                    treatment_generation_seeds[generation_seed][task_id]
                    - control_generation_seeds[generation_seed][task_id]
                    for task_id in shared_seed_tasks
                ]
                per_generation_seed_differences[generation_seed] = {
                    "task_count": len(shared_seed_tasks),
                    "mean_difference": mean(seed_differences),
                }
        comparison_summary.append(
            {
                "treatment": treatment,
                "control": control,
                "task_count": len(shared),
                "nonzero_task_count": sum(
                    abs(difference) > 1e-15 for difference in differences
                ),
                "mean_difference": mean(differences),
                "ci_lower": lower,
                "ci_upper": upper,
                "p_value_two_sided": p_value,
                "p_value_method": method,
                "by_training_seed": per_seed_differences,
                "by_generation_seed": per_generation_seed_differences,
            }
        )

    if comparison_summary:
        adjusted = holm_adjust(
            [item["p_value_two_sided"] for item in comparison_summary]
        )
        for item, adjusted_p in zip(comparison_summary, adjusted):
            item["p_value_holm"] = adjusted_p

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
