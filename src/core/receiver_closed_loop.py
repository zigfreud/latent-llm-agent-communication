"""Differentiable frozen-receiver evolution with sequential learned corrections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


class _StopClosedLoopForward(RuntimeError):
    """Internal signal used to stop before irrelevant receiver blocks."""


def evolve_receiver_with_closed_loop_corrector(
    model,
    inputs: Mapping[str, torch.Tensor],
    *,
    positions: torch.Tensor,
    protocol_code: torch.Tensor,
    corrector,
    scaffold: torch.Tensor,
    site_scale: torch.Tensor,
    layer_indices: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Run a sequential corrector and return pre/post-correction block inputs.

    Both returned tensors have shape ``[batch, layers, positions, width]`` and
    preserve the autograd graph from later receiver states to earlier deltas.
    """

    layers = [int(layer) for layer in layer_indices]
    if not layers or layers != list(range(layers[-1] + 1)):
        raise ValueError("layer_indices must be the contiguous prefix starting at 0")
    if not isinstance(positions, torch.Tensor) or positions.ndim != 1:
        raise ValueError("positions must be a rank-1 tensor")
    if positions.numel() == 0:
        raise ValueError("positions must not be empty")
    if not isinstance(protocol_code, torch.Tensor) or protocol_code.ndim != 3:
        raise ValueError("protocol_code must have [batch, slots, width]")
    if inputs["input_ids"].ndim != 2:
        raise ValueError("receiver input_ids must have [batch, sequence]")
    if protocol_code.shape[0] != inputs["input_ids"].shape[0]:
        raise ValueError("protocol code batch must match receiver inputs")
    expected_stats = (len(layers), positions.numel())
    if scaffold.ndim != 3 or tuple(scaffold.shape[:2]) != expected_stats:
        raise ValueError("scaffold shape differs from closed-loop sites")
    if tuple(site_scale.shape) != expected_stats:
        raise ValueError("site_scale shape differs from closed-loop sites")
    if scaffold.shape[-1] <= 0 or bool((site_scale <= 0).any()):
        raise ValueError("closed-loop statistics are invalid")

    decoder_layers = model.model.layers
    stop_layer = layers[-1] + 1
    if stop_layer >= len(decoder_layers):
        raise ValueError("receiver must expose one block after the corrected prefix")
    incoming: dict[int, torch.Tensor] = {}
    corrected: dict[int, torch.Tensor] = {}
    handles = []

    def correct_input(layer_index: int):
        def hook(module, module_in):
            if not module_in:
                raise ValueError("receiver block pre-hook requires hidden states")
            hidden = module_in[0]
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise ValueError("receiver hidden state must be rank three")
            if hidden.shape[0] != protocol_code.shape[0]:
                raise ValueError("receiver and protocol batches differ")
            if hidden.shape[-1] != scaffold.shape[-1]:
                raise ValueError("receiver width differs from target statistics")
            selected = positions.to(device=hidden.device, dtype=torch.long)
            if int(selected.min()) < 0 or int(selected.max()) >= hidden.shape[1]:
                raise ValueError("closed-loop position is outside receiver sequence")
            live = hidden[:, selected, :]
            scale = site_scale[layer_index].to(
                device=live.device, dtype=torch.float32
            )
            origin = scaffold[layer_index].to(
                device=live.device, dtype=torch.float32
            )
            live_normalized = (live.float() - origin[None]) / scale[None, :, None]
            delta_normalized = corrector(
                protocol_code,
                live_normalized,
                layer_index=layer_index,
            )
            if tuple(delta_normalized.shape) != tuple(live.shape):
                raise ValueError("corrector delta shape differs from live packet")
            injected = live + (
                delta_normalized.float() * scale[None, :, None]
            ).to(dtype=live.dtype)
            incoming[layer_index] = live
            corrected[layer_index] = injected
            updated = hidden.clone()
            updated[:, selected, :] = injected
            return (updated, *module_in[1:])

        return hook

    def stop_forward(module, module_in):
        raise _StopClosedLoopForward

    for layer in layers:
        handles.append(
            decoder_layers[layer].register_forward_pre_hook(correct_input(layer))
        )
    handles.append(decoder_layers[stop_layer].register_forward_pre_hook(stop_forward))
    stopped = False
    try:
        model(
            **inputs,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
    except _StopClosedLoopForward:
        stopped = True
    finally:
        for handle in handles:
            handle.remove()
    if not stopped:
        raise RuntimeError("receiver forward did not reach the closed-loop stop layer")
    missing = set(layers).difference(incoming) | set(layers).difference(corrected)
    if missing:
        raise RuntimeError(f"failed to capture closed-loop layer(s): {sorted(missing)}")
    return {
        "incoming_before_correction": torch.stack(
            [incoming[layer] for layer in layers], dim=1
        ),
        "residual_input": torch.stack(
            [corrected[layer] for layer in layers], dim=1
        ),
    }
