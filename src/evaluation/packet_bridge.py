"""Latent and functional summaries for learned LIP packet bridges."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch

from src.core.packet_loss import packet_similarity
from src.evaluation.statistics import holm_adjust, sign_flip_p_value


RETRIEVAL_REGIONS = ("joint", "core", "name")


def _task_region_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> list[float]:
    squared = (prediction - target).square()
    if mask is None:
        task_mse = squared.mean(dim=(1, 2, 3))
    else:
        expanded = mask[:, None, :, None].expand_as(squared)
        task_mse = (squared * expanded).sum(dim=(1, 2, 3)) / expanded.sum(
            dim=(1, 2, 3)
        )
    return torch.sqrt(task_mse).tolist()


def summarize_packet_latent_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    component_masks: Mapping[str, torch.Tensor],
    *,
    task_ids: Sequence[str] | None = None,
) -> dict:
    """Compute task-level retrieval, margin, and residual reconstruction metrics."""

    if prediction.ndim != 4 or tuple(prediction.shape) != tuple(target.shape):
        raise ValueError("prediction and target must share [batch, layers, positions, width]")
    batch_size = prediction.shape[0]
    if batch_size < 2:
        raise ValueError("packet retrieval metrics require at least two tasks")
    if task_ids is None:
        task_ids = [str(index) for index in range(batch_size)]
    if len(task_ids) != batch_size or len(set(task_ids)) != batch_size:
        raise ValueError("task_ids must uniquely identify every packet")

    report = {
        "task_count": batch_size,
        "normalized_residual_rmse": float(
            torch.sqrt((prediction - target).square().mean()).item()
        ),
        "regions": {},
    }
    for region in RETRIEVAL_REGIONS:
        mask = None if region == "joint" else component_masks[region]
        similarity = packet_similarity(prediction, target, mask)
        labels = torch.arange(batch_size, device=similarity.device)
        diagonal = torch.diagonal(similarity)
        diagonal_mask = torch.eye(batch_size, dtype=torch.bool, device=similarity.device)
        hardest_negative = similarity.masked_fill(diagonal_mask, float("-inf")).max(
            dim=1
        ).values
        margins = diagonal - hardest_negative
        retrieved = similarity.argmax(dim=1) == labels
        task_rmse = _task_region_rmse(prediction, target, mask)
        rows = [
            {
                "task_id": str(task_id),
                "retrieved_top1": bool(retrieved[index].item()),
                "diagonal_cosine": float(diagonal[index].item()),
                "hard_negative_cosine": float(hardest_negative[index].item()),
                "diagonal_margin": float(margins[index].item()),
                "normalized_residual_rmse": float(task_rmse[index]),
            }
            for index, task_id in enumerate(task_ids)
        ]
        report["regions"][region] = {
            "retrieval_top1": float(retrieved.float().mean().item()),
            "diagonal_cosine_mean": float(diagonal.mean().item()),
            "hard_negative_cosine_mean": float(hardest_negative.mean().item()),
            "diagonal_margin_mean": float(margins.mean().item()),
            "normalized_residual_rmse": float(sum(task_rmse) / len(task_rmse)),
            "tasks": rows,
        }
    return report


def checkpoint_selection_key(metrics: Mapping, *, step: int) -> tuple[float, ...]:
    """Lexicographic checkpoint key protecting the weakest causal region."""

    regions = metrics.get("regions", {})
    if any(region not in regions for region in RETRIEVAL_REGIONS):
        raise ValueError("checkpoint metrics must contain joint, core, and name regions")
    retrievals = [float(regions[region]["retrieval_top1"]) for region in RETRIEVAL_REGIONS]
    margins = [float(regions[region]["diagonal_margin_mean"]) for region in RETRIEVAL_REGIONS]
    rmse = float(metrics["normalized_residual_rmse"])
    if not all(math.isfinite(value) for value in [*retrievals, *margins, rmse]):
        raise ValueError("checkpoint metrics must be finite")
    return (
        min(retrievals),
        sum(retrievals) / len(retrievals),
        min(margins),
        sum(margins) / len(margins),
        -rmse,
        -int(step),
    )


def summarize_replica_development_gate(
    metrics: Mapping,
    *,
    alpha: float = 0.05,
    statistics_seed: int = 4481,
) -> dict:
    """Apply one Holm family to untouched development-gate packet margins."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    tests = []
    for offset, region in enumerate(RETRIEVAL_REGIONS):
        rows = metrics["regions"][region]["tasks"]
        margins = [float(row["diagonal_margin"]) for row in rows]
        p_value, method = sign_flip_p_value(
            margins,
            alternative="greater",
            seed=statistics_seed + offset,
        )
        tests.append(
            {
                "region": region,
                "mean_diagonal_margin": sum(margins) / len(margins),
                "p_value": p_value,
                "test_method": method,
            }
        )
    adjusted = holm_adjust([test["p_value"] for test in tests])
    for test, adjusted_p in zip(tests, adjusted):
        test["p_value_holm"] = adjusted_p
        test["rejected"] = bool(
            test["mean_diagonal_margin"] > 0.0 and adjusted_p <= alpha
        )
    return {
        "alpha": alpha,
        "family": tests,
        "passed": all(test["rejected"] for test in tests),
        "criterion": "positive joint/core/name margins after one Holm family",
    }


def summarize_multi_replica_development_gate(
    replica_reports: Sequence[Mapping],
    *,
    minimum_passing_replicas: int = 2,
) -> dict:
    if not replica_reports:
        raise ValueError("at least one replica report is required")
    if not 1 <= minimum_passing_replicas <= len(replica_reports):
        raise ValueError("minimum_passing_replicas is outside the replica count")
    passing = sum(bool(report.get("passed")) for report in replica_reports)
    return {
        "replica_count": len(replica_reports),
        "minimum_passing_replicas": minimum_passing_replicas,
        "passing_replicas": passing,
        "passed": passing >= minimum_passing_replicas,
        "replicas": list(replica_reports),
    }


def normalized_transport_recovery(
    *,
    learned_matched: float,
    learned_shuffled: float,
    oracle_matched: float,
    oracle_shuffled: float,
    text: float,
    neutral: float,
) -> dict[str, float | None]:
    """Report identity-channel and text-gain recovery without clipping ratios."""

    oracle_effect = float(oracle_matched) - float(oracle_shuffled)
    learned_effect = float(learned_matched) - float(learned_shuffled)
    text_gain = float(text) - float(neutral)
    learned_gain = float(learned_matched) - float(neutral)
    return {
        "learned_identity_effect": learned_effect,
        "oracle_identity_effect": oracle_effect,
        "identity_recovery_ratio": (
            learned_effect / oracle_effect if oracle_effect != 0.0 else None
        ),
        "learned_over_neutral_gain": learned_gain,
        "text_over_neutral_gain": text_gain,
        "text_gain_recovery_ratio": learned_gain / text_gain if text_gain != 0.0 else None,
    }
