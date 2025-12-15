from __future__ import annotations
from typing import Callable, Any, Tuple, Optional
import torch


def make_lip_hook(
    vec_injected: torch.Tensor,
    inject_pos: int,
    enable: bool = True,
) -> Callable[[Any, Tuple[Any, ...], Any], Any]:
    """
    Returns a forward hook that injects vec_injected into the hidden state at token position inject_pos.
    Injects at most once (first forward pass), then becomes a no-op.
    """

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
        if enable and isinstance(hs, torch.Tensor) and hs.dim() == 3:
            if hs.shape[-1] == vec_injected.shape[-1] and 0 <= inject_pos < hs.shape[1]:
                hs[:, inject_pos, :] = hs[:, inject_pos, :] + vec_injected
            did["flag"] = True
        else:
            did["flag"] = True

        if rest is None:
            return hs
        return (hs,) + rest

    return hook
