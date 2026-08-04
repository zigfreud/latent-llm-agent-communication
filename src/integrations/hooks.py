from __future__ import annotations
from typing import Any, Callable, Tuple
import torch


def make_lip_hook(
    vec_injected: torch.Tensor,
    inject_pos: int,
    enable: bool = True,
    mode: str = "add",
) -> Callable[[Any, Tuple[Any, ...], Any], Any]:
    """
    Returns a forward hook that injects vec_injected into the hidden state at token position inject_pos.
    Injects at most once (first forward pass), then becomes a no-op.
    """

    if mode not in {"add", "replace"}:
        raise ValueError("injection mode must be add or replace")

    did = {"flag": False}

    def hook(module, module_in, module_out):
        if did["flag"]:
            return module_out

        if isinstance(module_out, tuple):
            hs = module_out[0]
            rest = module_out[1:]
        else:
            hs = module_out
            rest = None

        # hs: (B, T, D)
        if not enable:
            did["flag"] = True
        else:
            if not isinstance(hs, torch.Tensor) or hs.dim() != 3:
                raise ValueError("hook output must contain rank-3 hidden states")
            if hs.shape[-1] != vec_injected.shape[-1]:
                raise ValueError(
                    "injected vector width does not match hidden-state width"
                )
            if not 0 <= inject_pos < hs.shape[1]:
                raise ValueError("injection position is outside the hidden-state sequence")
            injected = vec_injected.to(device=hs.device, dtype=hs.dtype)
            if injected.ndim == 1:
                injected = injected.unsqueeze(0)
            if injected.shape[0] not in {1, hs.shape[0]}:
                raise ValueError(
                    "injected vector batch must be one or match hidden-state batch"
                )
            hs = hs.clone()
            if mode == "add":
                hs[:, inject_pos, :] = hs[:, inject_pos, :] + injected
            else:
                hs[:, inject_pos, :] = injected
            did["flag"] = True

        if rest is None:
            return hs
        return (hs,) + rest

    return hook


def make_lip_packet_hook(
    vectors: torch.Tensor,
    positions: torch.Tensor,
    enable: bool = True,
    mode: str = "replace",
) -> Callable[[Any, Tuple[Any, ...], Any], Any]:
    """Inject a position-aligned packet once during the first forward pass."""

    if mode not in {"add", "replace"}:
        raise ValueError("injection mode must be add or replace")
    if not isinstance(vectors, torch.Tensor) or vectors.ndim != 2:
        raise ValueError("packet vectors must have shape (positions, hidden_width)")
    if not isinstance(positions, torch.Tensor) or positions.ndim != 1:
        raise ValueError("packet positions must be a rank-1 tensor")
    if positions.numel() == 0 or positions.numel() != vectors.shape[0]:
        raise ValueError("packet positions and vectors must have the same non-zero length")
    if torch.unique(positions).numel() != positions.numel():
        raise ValueError("packet positions must be unique")

    did = {"flag": False}

    def hook(module, module_in, module_out):
        if did["flag"]:
            return module_out
        hidden = module_out[0] if isinstance(module_out, tuple) else module_out
        rest = module_out[1:] if isinstance(module_out, tuple) else None
        if enable:
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise ValueError("hook output must contain rank-3 hidden states")
            if hidden.shape[0] != 1:
                raise ValueError("packet injection currently requires batch size one")
            if hidden.shape[-1] != vectors.shape[-1]:
                raise ValueError("packet hidden width does not match hook output")
            selected = positions.to(device=hidden.device, dtype=torch.long)
            if int(selected.min()) < 0 or int(selected.max()) >= hidden.shape[1]:
                raise ValueError("packet position is outside the hidden-state sequence")
            injected = vectors.to(device=hidden.device, dtype=hidden.dtype)
            hidden = hidden.clone()
            if mode == "add":
                hidden[0, selected, :] = hidden[0, selected, :] + injected
            else:
                hidden[0, selected, :] = injected
        did["flag"] = True
        if rest is None:
            return hidden
        return (hidden,) + rest

    return hook


def make_lip_packet_pre_hook(
    vectors: torch.Tensor,
    positions: torch.Tensor,
    enable: bool = True,
    mode: str = "replace",
) -> Callable[[Any, Tuple[Any, ...]], Tuple[Any, ...] | None]:
    """Inject a position-aligned packet into one block input exactly once."""

    if mode not in {"add", "replace"}:
        raise ValueError("injection mode must be add or replace")
    if not isinstance(vectors, torch.Tensor) or vectors.ndim != 2:
        raise ValueError("packet vectors must have shape (positions, hidden_width)")
    if not isinstance(positions, torch.Tensor) or positions.ndim != 1:
        raise ValueError("packet positions must be a rank-1 tensor")
    if positions.numel() == 0 or positions.numel() != vectors.shape[0]:
        raise ValueError("packet positions and vectors must have the same non-zero length")
    if torch.unique(positions).numel() != positions.numel():
        raise ValueError("packet positions must be unique")

    did = {"flag": False}

    def hook(module, module_in):
        if did["flag"]:
            return None
        if not enable:
            did["flag"] = True
            return None
        if not module_in:
            raise ValueError("block pre-hook requires positional hidden states")
        hidden = module_in[0]
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("hook input must begin with rank-3 hidden states")
        if hidden.shape[0] != 1:
            raise ValueError("packet injection currently requires batch size one")
        if hidden.shape[-1] != vectors.shape[-1]:
            raise ValueError("packet hidden width does not match hook input")
        selected = positions.to(device=hidden.device, dtype=torch.long)
        if int(selected.min()) < 0 or int(selected.max()) >= hidden.shape[1]:
            raise ValueError("packet position is outside the hidden-state sequence")
        injected = vectors.to(device=hidden.device, dtype=hidden.dtype)
        hidden = hidden.clone()
        if mode == "add":
            hidden[0, selected, :] = hidden[0, selected, :] + injected
        else:
            hidden[0, selected, :] = injected
        did["flag"] = True
        return (hidden,) + tuple(module_in[1:])

    return hook
