"""Aggregate layer-by-position diagnostics for target-oracle prompt states."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


ORACLE_STATE_DIAGNOSTICS_VERSION = "lip-oracle-state-diagnostics-v1"
ORACLE_STATE_TYPES = (
    "residual_input",
    "key_pre_rope",
    "value_pre_cache",
)
ORACLE_STATE_METRICS = (
    "mean_l2_norm",
    "task_signal_fraction",
    "mean_pairwise_cosine",
    "task_effective_rank",
    "task_effective_rank_fraction",
)


def validate_state_diagnostics_contract(config: Mapping[str, Any]) -> None:
    """Validate the preregistered aggregate-only diagnostic contract."""

    expected = {
        "protocol_version": ORACLE_STATE_DIAGNOSTICS_VERSION,
        "state_types": list(ORACLE_STATE_TYPES),
        "metrics": list(ORACLE_STATE_METRICS),
        "store_raw_states": False,
    }
    if dict(config) != expected:
        raise ValueError("diagnostics must match the frozen aggregate-only contract")


def _validate_state_memories(
    state_memories: Sequence[Mapping[str, Mapping[int, torch.Tensor]]],
    *,
    layer_indices: Sequence[int],
    packet_size: int,
) -> None:
    if len(state_memories) < 2:
        raise ValueError("state diagnostics require at least two tasks")
    expected_layers = [int(layer) for layer in layer_indices]
    if not expected_layers or len(set(expected_layers)) != len(expected_layers):
        raise ValueError("layer_indices must be a non-empty unique sequence")
    reference_shapes: dict[tuple[str, int], tuple[int, int]] = {}
    for task_index, task_states in enumerate(state_memories):
        if tuple(task_states) != ORACLE_STATE_TYPES:
            raise ValueError("captured state types do not match the frozen contract")
        for state_type in ORACLE_STATE_TYPES:
            layer_states = task_states[state_type]
            if list(layer_states) != expected_layers:
                raise ValueError("captured state layers do not match the frozen order")
            for layer_idx, tensor in layer_states.items():
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
                    raise ValueError("captured oracle states must be rank-2 tensors")
                if tensor.shape[0] != int(packet_size) or tensor.shape[1] <= 0:
                    raise ValueError("captured oracle state has an invalid shape")
                if not bool(torch.isfinite(tensor).all().item()):
                    raise ValueError("captured oracle states must be finite")
                shape = (int(tensor.shape[0]), int(tensor.shape[1]))
                key = (state_type, int(layer_idx))
                if task_index == 0:
                    reference_shapes[key] = shape
                elif reference_shapes[key] != shape:
                    raise ValueError("state dimensions must match across tasks")


def _position_metrics(vectors: torch.Tensor) -> dict[str, float]:
    """Summarize one layer/position cell across tasks."""

    vectors = vectors.float()
    task_count = int(vectors.shape[0])
    norms = torch.linalg.vector_norm(vectors, dim=1)
    normalized = F.normalize(vectors, dim=1, eps=1e-12)
    cosine = normalized @ normalized.T
    off_diagonal = ~torch.eye(task_count, dtype=torch.bool, device=vectors.device)
    mean_pairwise_cosine = float(cosine.masked_select(off_diagonal).mean().item())

    centered = vectors - vectors.mean(dim=0, keepdim=True)
    total_energy = vectors.square().sum()
    task_energy = centered.square().sum()
    if float(total_energy.item()) == 0.0:
        task_signal_fraction = 0.0
    else:
        task_signal_fraction = float((task_energy / total_energy).item())

    gram = centered @ centered.T
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    eigenvalue_sum = eigenvalues.sum()
    if float(eigenvalue_sum.item()) == 0.0:
        effective_rank = 0.0
    else:
        probabilities = eigenvalues / eigenvalue_sum
        positive = probabilities > 0
        entropy = -(
            probabilities[positive] * probabilities[positive].log()
        ).sum()
        effective_rank = float(torch.exp(entropy).item())
    maximum_rank = max(1, min(task_count - 1, int(vectors.shape[1])))
    return {
        "mean_l2_norm": float(norms.mean().item()),
        "task_signal_fraction": task_signal_fraction,
        "mean_pairwise_cosine": mean_pairwise_cosine,
        "task_effective_rank": effective_rank,
        "task_effective_rank_fraction": effective_rank / maximum_rank,
    }


def summarize_state_diagnostics(
    state_memories: Sequence[Mapping[str, Mapping[int, torch.Tensor]]],
    *,
    task_ids: Sequence[str],
    layer_indices: Sequence[int],
    packet_size: int,
    run_scope: str,
) -> dict[str, Any]:
    """Create a JSON-ready state-type x layer x suffix-position grid."""

    ids = [str(task_id) for task_id in task_ids]
    if len(ids) != len(state_memories) or len(set(ids)) != len(ids):
        raise ValueError("task_ids must uniquely identify every captured task")
    layers = [int(layer) for layer in layer_indices]
    _validate_state_memories(
        state_memories,
        layer_indices=layers,
        packet_size=int(packet_size),
    )
    packet_offsets = list(range(-int(packet_size), 0))
    cells = []
    for state_type in ORACLE_STATE_TYPES:
        for layer_idx in layers:
            stacked = torch.stack(
                [task_states[state_type][layer_idx] for task_states in state_memories]
            )
            for position_index, packet_offset in enumerate(packet_offsets):
                metrics = _position_metrics(stacked[:, position_index, :])
                cells.append(
                    {
                        "state_type": state_type,
                        "layer_index": layer_idx,
                        "packet_offset": packet_offset,
                        "state_dimension": int(stacked.shape[2]),
                        **metrics,
                    }
                )
    return {
        "protocol_version": ORACLE_STATE_DIAGNOSTICS_VERSION,
        "run_scope": str(run_scope),
        "claim_eligible": str(run_scope) == "full",
        "task_count": len(ids),
        "task_ids": ids,
        "state_types": list(ORACLE_STATE_TYPES),
        "metrics": list(ORACLE_STATE_METRICS),
        "layer_indices": layers,
        "packet_offsets": packet_offsets,
        "packet_size": int(packet_size),
        "cells": cells,
        "interpretation": {
            "task_signal_fraction": (
                "task-centered energy divided by total uncentered state energy"
            ),
            "mean_pairwise_cosine": (
                "mean cosine similarity across distinct task pairs"
            ),
            "task_effective_rank": (
                "entropy effective rank of the task-centered Gram spectrum"
            ),
            "causal_status": (
                "descriptive only; functional matched-vs-shuffled controls remain "
                "the causal communication test"
            ),
        },
    }
