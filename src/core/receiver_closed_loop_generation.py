"""Prefill-only H0-017 corrections followed by ordinary cached decoding."""
from contextlib import contextmanager

import torch


@contextmanager
def closed_loop_prefill(model, *, prompt_length, positions, protocol_code,
                        corrector, scaffold, site_scale, layer_indices):
    layers = list(layer_indices)
    if not layers or layers != list(range(len(layers))):
        raise ValueError("correction layers must be a contiguous prefix")
    if protocol_code.ndim != 3 or protocol_code.shape[0] != 1:
        raise ValueError("generation requires exactly one protocol code")
    if positions.ndim != 1 or positions.numel() == 0:
        raise ValueError("positions must be nonempty and rank one")
    if len(positions.unique()) != len(positions) or positions.min() < 0 or positions.max() >= prompt_length:
        raise ValueError("positions must be unique and inside the prompt")
    if scaffold.ndim != 3 or tuple(scaffold.shape[:2]) != (len(layers), len(positions)):
        raise ValueError("scaffold shape differs from correction sites")
    if tuple(site_scale.shape) != tuple(scaffold.shape[:2]) or not torch.isfinite(site_scale).all() or (site_scale <= 0).any():
        raise ValueError("invalid site scales")
    audit = {"incoming": {}, "corrected": {}, "decode_calls": {i: 0 for i in layers}}
    handles = []

    def make_hook(index):
        def hook(module, args):
            hidden = args[0]
            if index in audit["incoming"]:
                if hidden.shape[1] != 1:
                    raise ValueError("closed-loop generation requires cached single-token decoding")
                audit["decode_calls"][index] += 1
                return None
            if hidden.shape != (1, prompt_length, scaffold.shape[-1]):
                raise ValueError("first correction must be the full prompt prefill")
            selected = positions.to(hidden.device)
            live = hidden[:, selected, :]
            scale = site_scale[index].to(device=hidden.device, dtype=torch.float32)
            origin = scaffold[index].to(device=hidden.device, dtype=torch.float32)
            normalized = (live.float() - origin[None]) / scale[None, :, None]
            delta = corrector(protocol_code, normalized, layer_index=index)
            if delta.shape != live.shape or not torch.isfinite(delta).all():
                raise ValueError("invalid corrector delta")
            injected = live + (delta.float() * scale[None, :, None]).to(live.dtype)
            audit["incoming"][index] = live.detach()
            audit["corrected"][index] = injected.detach()
            updated = hidden.clone()
            updated[:, selected, :] = injected
            return (updated, *args[1:])
        return hook

    try:
        for index in layers:
            handles.append(model.model.layers[index].register_forward_pre_hook(make_hook(index)))
        yield audit
        if set(audit["incoming"]) != set(layers):
            raise RuntimeError("generation did not traverse every correction layer")
    finally:
        for handle in handles:
            handle.remove()
