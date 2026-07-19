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
