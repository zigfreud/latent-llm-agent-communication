"""Component-aware objectives for learned LIP residual packets."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


COMPONENT_NAMES = ("core", "name", "boundary")
CONTRASTIVE_REGIONS = ("joint", "core", "name")


def build_terminal_component_masks(
    name_token_counts: torch.Tensor,
    *,
    target_positions: int,
    boundary_positions: int = 6,
) -> dict[str, torch.Tensor]:
    """Build per-task masks for terminal core, function name, and boundary."""

    if not isinstance(name_token_counts, torch.Tensor) or name_token_counts.ndim != 1:
        raise ValueError("name_token_counts must be a rank-1 tensor")
    counts = name_token_counts.to(dtype=torch.long)
    name_stop = int(target_positions) - int(boundary_positions)
    if name_stop <= 0:
        raise ValueError("boundary_positions must leave at least one non-boundary site")
    if counts.numel() == 0 or bool((counts <= 0).any()) or bool((counts >= name_stop).any()):
        raise ValueError("each function name must leave at least one terminal core site")

    positions = torch.arange(target_positions, device=counts.device)[None, :]
    name_start = name_stop - counts[:, None]
    core = positions < name_start
    name = (positions >= name_start) & (positions < name_stop)
    boundary = (positions >= name_stop).expand(counts.shape[0], -1)
    return {"core": core, "name": name, "boundary": boundary}


def _validate_packets(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if not isinstance(prediction, torch.Tensor) or prediction.ndim != 4:
        raise ValueError("prediction must have shape [batch, layers, positions, width]")
    if tuple(prediction.shape) != tuple(target.shape):
        raise ValueError("prediction and target packet shapes must match")


def _validate_masks(
    masks: Mapping[str, torch.Tensor],
    *,
    batch_size: int,
    positions: int,
) -> None:
    for component in COMPONENT_NAMES:
        if component not in masks:
            raise ValueError(f"component_masks missing {component}")
        mask = masks[component]
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError(f"component mask {component} must be boolean")
        if tuple(mask.shape) != (batch_size, positions):
            raise ValueError(
                f"component mask {component} must have shape {(batch_size, positions)}"
            )
        if bool((mask.sum(dim=1) == 0).any()):
            raise ValueError(f"component mask {component} must be non-empty per task")
    coverage = sum(masks[name].to(torch.int8) for name in COMPONENT_NAMES)
    if bool((coverage != 1).any()):
        raise ValueError("component masks must partition every target position exactly once")


def _component_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError("component-reduced values must have [batch, layers, positions]")
    expanded = mask[:, None, :].expand(-1, values.shape[1], -1)
    per_task = (values * expanded).sum(dim=(1, 2)) / expanded.sum(dim=(1, 2))
    return per_task.mean()


def _masked_flatten(packet: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is not None:
        packet = packet * mask[:, None, :, None]
    return packet.flatten(start_dim=1)


def packet_similarity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    predicted_flat = F.normalize(_masked_flatten(prediction, mask), p=2, dim=1)
    target_flat = F.normalize(_masked_flatten(target, mask), p=2, dim=1)
    return predicted_flat @ target_flat.T


def _similarity_metrics(
    similarity: torch.Tensor,
    *,
    temperature: float,
    margin_target: float,
) -> dict[str, torch.Tensor | None]:
    batch_size = similarity.shape[0]
    labels = torch.arange(batch_size, device=similarity.device)
    forward_nce = F.cross_entropy(similarity / temperature, labels)
    reverse_nce = F.cross_entropy(similarity.T / temperature, labels)
    diagonal = torch.diagonal(similarity)
    retrieval_top1 = (similarity.argmax(dim=1) == labels).float().mean()
    zero = similarity.new_zeros(())
    if batch_size < 2:
        return {
            "symmetric_nce": 0.5 * (forward_nce + reverse_nce),
            "margin_loss": zero,
            "retrieval_top1": retrieval_top1,
            "diagonal_cosine": diagonal.mean(),
            "diagonal_margin": None,
            "hard_negative_cosine": None,
        }

    diagonal_mask = torch.eye(batch_size, dtype=torch.bool, device=similarity.device)
    negatives = similarity.masked_fill(diagonal_mask, float("-inf"))
    hardest_row = negatives.max(dim=1).values
    hardest_column = negatives.max(dim=0).values
    row_margin = diagonal - hardest_row
    column_margin = diagonal - hardest_column
    margin_loss = 0.5 * (
        F.relu(margin_target - row_margin).mean()
        + F.relu(margin_target - column_margin).mean()
    )
    return {
        "symmetric_nce": 0.5 * (forward_nce + reverse_nce),
        "margin_loss": margin_loss,
        "retrieval_top1": retrieval_top1,
        "diagonal_cosine": diagonal.mean(),
        "diagonal_margin": row_margin.mean(),
        "hard_negative_cosine": hardest_row.mean(),
    }


class ComponentAwarePacketLoss(nn.Module):
    """Balance causal terminal regions while preserving task-level identity."""

    def __init__(
        self,
        *,
        temperature: float = 0.07,
        margin_target: float = 0.05,
        lambda_huber: float = 1.0,
        lambda_cosine: float = 0.25,
        lambda_symmetric_nce: float = 1.0,
        lambda_margin: float = 0.1,
        lambda_norm: float = 0.05,
        component_weights: Mapping[str, float] | None = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.margin_target = float(margin_target)
        self.lambda_huber = float(lambda_huber)
        self.lambda_cosine = float(lambda_cosine)
        self.lambda_symmetric_nce = float(lambda_symmetric_nce)
        self.lambda_margin = float(lambda_margin)
        self.lambda_norm = float(lambda_norm)
        self.eps = float(eps)
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.margin_target < 0.0:
            raise ValueError("margin_target must be non-negative")
        if any(
            value < 0.0
            for value in (
                self.lambda_huber,
                self.lambda_cosine,
                self.lambda_symmetric_nce,
                self.lambda_margin,
                self.lambda_norm,
            )
        ):
            raise ValueError("loss weights must be non-negative")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        weights = dict(component_weights or {"core": 0.45, "name": 0.45, "boundary": 0.10})
        if set(weights) != set(COMPONENT_NAMES):
            raise ValueError("component_weights must define core, name, and boundary")
        if any(float(value) < 0.0 for value in weights.values()):
            raise ValueError("component weights must be non-negative")
        total = sum(float(value) for value in weights.values())
        if total <= 0.0:
            raise ValueError("component weights must have positive total")
        self.component_weights = {
            name: float(weights[name]) / total for name in COMPONENT_NAMES
        }

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        component_masks: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | None]:
        _validate_packets(prediction, target)
        _validate_masks(
            component_masks,
            batch_size=prediction.shape[0],
            positions=prediction.shape[2],
        )

        huber_sites = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
        cosine_sites = 1.0 - F.cosine_similarity(prediction, target, dim=-1, eps=self.eps)
        huber_components = {
            name: _component_mean(huber_sites, component_masks[name])
            for name in COMPONENT_NAMES
        }
        cosine_components = {
            name: _component_mean(cosine_sites, component_masks[name])
            for name in COMPONENT_NAMES
        }
        huber_loss = sum(
            self.component_weights[name] * huber_components[name]
            for name in COMPONENT_NAMES
        )
        cosine_loss = sum(
            self.component_weights[name] * cosine_components[name]
            for name in COMPONENT_NAMES
        )

        region_metrics: dict[str, dict[str, torch.Tensor | None]] = {}
        for region in CONTRASTIVE_REGIONS:
            mask = None if region == "joint" else component_masks[region]
            similarity = packet_similarity(prediction, target, mask)
            region_metrics[region] = _similarity_metrics(
                similarity,
                temperature=self.temperature,
                margin_target=self.margin_target,
            )
        symmetric_nce = sum(
            region_metrics[name]["symmetric_nce"] for name in CONTRASTIVE_REGIONS
        ) / len(CONTRASTIVE_REGIONS)
        margin_loss = sum(
            region_metrics[name]["margin_loss"] for name in CONTRASTIVE_REGIONS
        ) / len(CONTRASTIVE_REGIONS)

        norm_components = {}
        for name in COMPONENT_NAMES:
            if self.lambda_norm == 0.0:
                norm_components[name] = prediction.new_zeros(())
                continue
            mask = component_masks[name]
            predicted_norm = torch.linalg.vector_norm(
                _masked_flatten(prediction, mask), dim=1
            )
            target_norm = torch.linalg.vector_norm(_masked_flatten(target, mask), dim=1)
            ratio = predicted_norm / (target_norm + self.eps)
            norm_components[name] = ((ratio - 1.0) ** 2).mean()
        norm_loss = sum(
            self.component_weights[name] * norm_components[name]
            for name in COMPONENT_NAMES
        )

        total_loss = (
            self.lambda_huber * huber_loss
            + self.lambda_cosine * cosine_loss
            + self.lambda_symmetric_nce * symmetric_nce
            + self.lambda_margin * margin_loss
            + self.lambda_norm * norm_loss
        )
        metrics: dict[str, torch.Tensor | None] = {
            "total_loss": total_loss,
            "huber_loss": huber_loss,
            "cosine_loss": cosine_loss,
            "symmetric_nce_loss": symmetric_nce,
            "margin_loss": margin_loss,
            "norm_loss": norm_loss,
        }
        for component in COMPONENT_NAMES:
            metrics[f"huber_{component}"] = huber_components[component]
            metrics[f"cosine_{component}"] = cosine_components[component]
            metrics[f"norm_{component}"] = norm_components[component]
        for region in CONTRASTIVE_REGIONS:
            for metric_name, value in region_metrics[region].items():
                metrics[f"{region}_{metric_name}"] = value
        return metrics
