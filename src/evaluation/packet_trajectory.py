"""Descriptive metrics for receiver packet trajectory coherence."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


PACKET_TRAJECTORY_PROTOCOL_VERSION = "lip-receiver-trajectory-coherence-v1"


def _float_tensor(value: torch.Tensor, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    result = value.detach().float().cpu()
    if result.numel() == 0 or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must be finite and non-empty")
    return result


def tensor_alignment(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Compare two same-shaped tensors without retaining either tensor."""

    reference = _float_tensor(reference, label="reference")
    candidate = _float_tensor(candidate, label="candidate")
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate shapes differ")
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    difference = candidate_flat - reference_flat
    reference_rms = float(torch.sqrt(torch.mean(reference_flat.square())).item())
    candidate_rms = float(torch.sqrt(torch.mean(candidate_flat.square())).item())
    difference_rms = float(torch.sqrt(torch.mean(difference.square())).item())
    reference_norm = float(torch.linalg.vector_norm(reference_flat).item())
    candidate_norm = float(torch.linalg.vector_norm(candidate_flat).item())
    denominator = reference_norm * candidate_norm
    if denominator <= epsilon:
        cosine = 1.0 if reference_norm <= epsilon and candidate_norm <= epsilon else 0.0
    else:
        cosine = float(torch.dot(reference_flat, candidate_flat).item() / denominator)
    return {
        "reference_rms": reference_rms,
        "candidate_rms": candidate_rms,
        "difference_rms": difference_rms,
        "normalized_rmse": difference_rms / max(reference_rms, epsilon),
        "cosine": max(-1.0, min(1.0, cosine)),
        "candidate_to_reference_norm_ratio": candidate_norm
        / max(reference_norm, epsilon),
    }


def summarize_replay_discontinuity(
    incoming_by_layer: Mapping[int, torch.Tensor],
    scheduled_by_layer: Mapping[int, torch.Tensor],
    *,
    layer_indices: Sequence[int],
) -> dict:
    """Measure the corrective jump imposed at every replayed block boundary."""

    layers = [int(layer) for layer in layer_indices]
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("layer_indices must be a non-empty unique sequence")
    if set(layers).difference(incoming_by_layer) or set(layers).difference(
        scheduled_by_layer
    ):
        raise ValueError("incoming or scheduled packets are missing configured layers")
    cells = []
    for ordinal, layer in enumerate(layers):
        alignment = tensor_alignment(
            scheduled_by_layer[layer], incoming_by_layer[layer]
        )
        cells.append(
            {
                "layer": layer,
                "role": "carrier_entry" if ordinal == 0 else "cross_layer_transition",
                "scheduled_rms": alignment["reference_rms"],
                "incoming_rms": alignment["candidate_rms"],
                "jump_rms": alignment["difference_rms"],
                "relative_jump_rms": alignment["normalized_rmse"],
                "scheduled_incoming_cosine": alignment["cosine"],
                "incoming_to_scheduled_norm_ratio": alignment[
                    "candidate_to_reference_norm_ratio"
                ],
            }
        )
    transition_values = [
        cell["relative_jump_rms"]
        for cell in cells
        if cell["role"] == "cross_layer_transition"
    ]
    return {
        "protocol_version": PACKET_TRAJECTORY_PROTOCOL_VERSION,
        "entry": cells[0],
        "transitions": cells[1:],
        "transition_summary": {
            "count": len(transition_values),
            "mean_relative_jump_rms": (
                float(statistics.fmean(transition_values))
                if transition_values
                else math.nan
            ),
            "median_relative_jump_rms": (
                float(statistics.median(transition_values))
                if transition_values
                else math.nan
            ),
            "maximum_relative_jump_rms": (
                float(max(transition_values)) if transition_values else math.nan
            ),
        },
    }


def summarize_native_alignment(
    native_states: Mapping[str, Mapping[int, torch.Tensor]],
    candidate_states: Mapping[str, Mapping[int, torch.Tensor]],
    *,
    state_types: Sequence[str],
    layer_indices: Sequence[int],
) -> dict:
    """Compare replay-derived receiver states with the native task trajectory."""

    layers = [int(layer) for layer in layer_indices]
    cells = []
    for state_type in state_types:
        if state_type not in native_states or state_type not in candidate_states:
            raise ValueError(f"missing captured state type: {state_type}")
        for layer in layers:
            if layer not in native_states[state_type] or layer not in candidate_states[
                state_type
            ]:
                raise ValueError(f"missing {state_type} layer {layer}")
            alignment = tensor_alignment(
                native_states[state_type][layer], candidate_states[state_type][layer]
            )
            cells.append({"state_type": state_type, "layer": layer, **alignment})
    summaries = {}
    for state_type in state_types:
        rows = [row for row in cells if row["state_type"] == state_type]
        summaries[state_type] = {
            "mean_normalized_rmse": float(
                statistics.fmean(row["normalized_rmse"] for row in rows)
            ),
            "mean_cosine": float(statistics.fmean(row["cosine"] for row in rows)),
            "mean_candidate_to_native_norm_ratio": float(
                statistics.fmean(
                    row["candidate_to_reference_norm_ratio"] for row in rows
                )
            ),
        }
    return {"cells": cells, "state_summaries": summaries}


def next_token_distribution_alignment(
    native_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    position: int = -1,
) -> dict[str, float | int | bool]:
    """Compare next-token distributions at one prompt position."""

    native = _float_tensor(native_logits, label="native_logits")
    candidate = _float_tensor(candidate_logits, label="candidate_logits")
    if native.ndim != 3 or candidate.ndim != 3:
        raise ValueError("logits must have [batch, sequence, vocabulary] shape")
    if native.shape[0] != 1 or candidate.shape[0] != 1:
        raise ValueError("logit alignment requires batch size one")
    if native.shape[-1] != candidate.shape[-1]:
        raise ValueError("native and candidate vocabularies differ")
    native_row = native[0, position]
    candidate_row = candidate[0, position]
    native_log_prob = F.log_softmax(native_row, dim=-1)
    candidate_log_prob = F.log_softmax(candidate_row, dim=-1)
    native_prob = native_log_prob.exp()
    candidate_prob = candidate_log_prob.exp()
    kl = torch.sum(native_prob * (native_log_prob - candidate_log_prob))
    total_variation = 0.5 * torch.sum(torch.abs(native_prob - candidate_prob))
    native_top1 = int(torch.argmax(native_row).item())
    candidate_top1 = int(torch.argmax(candidate_row).item())
    return {
        "kl_native_to_candidate": float(kl.item()),
        "total_variation": float(total_variation.item()),
        "native_top1_token_id": native_top1,
        "candidate_top1_token_id": candidate_top1,
        "top1_agreement": native_top1 == candidate_top1,
    }

