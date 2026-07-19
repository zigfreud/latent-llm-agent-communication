"""Helpers for selecting token vectors from batched hidden states."""

from __future__ import annotations

import torch


SUPPORTED_TOKEN_POSITIONS = {"last", "last_non_padding"}


def last_non_padding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the final attended index for each row, for left or right padding."""

    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        raise ValueError("attention_mask must be a rank-2 torch.Tensor")
    valid = attention_mask.to(dtype=torch.bool)
    if not torch.all(valid.any(dim=1)):
        raise ValueError("every attention_mask row must contain at least one token")

    positions = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
    return positions.masked_fill(~valid, -1).max(dim=1).values.long()


def select_hidden_vectors(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    token_position: str = "last_non_padding",
) -> torch.Tensor:
    """Select one vector per batch row under an explicit token-position policy."""

    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
        raise ValueError("hidden_states must be a rank-3 torch.Tensor")
    if token_position not in SUPPORTED_TOKEN_POSITIONS:
        allowed = ", ".join(sorted(SUPPORTED_TOKEN_POSITIONS))
        raise ValueError(f"token_position must be one of: {allowed}")

    batch_size, sequence_length, _ = hidden_states.shape
    if sequence_length == 0:
        raise ValueError("hidden_states sequence length must be positive")

    if token_position == "last":
        token_indices = torch.full(
            (batch_size,),
            sequence_length - 1,
            dtype=torch.long,
            device=hidden_states.device,
        )
    else:
        if attention_mask is None:
            token_indices = torch.full(
                (batch_size,),
                sequence_length - 1,
                dtype=torch.long,
                device=hidden_states.device,
            )
        else:
            if tuple(attention_mask.shape) != (batch_size, sequence_length):
                raise ValueError(
                    "attention_mask shape must match hidden_states batch and sequence"
                )
            token_indices = last_non_padding_indices(
                attention_mask.to(hidden_states.device)
            )

    batch_indices = torch.arange(batch_size, device=hidden_states.device)
    return hidden_states[batch_indices, token_indices]
