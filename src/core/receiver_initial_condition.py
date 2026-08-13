"""Differentiable receiver evolution from a learned block-0 initial condition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


class _StopReceiverForward(RuntimeError):
    """Internal control signal used to avoid executing irrelevant late blocks."""


def evolve_receiver_from_entry_seed(
    model,
    inputs: Mapping[str, torch.Tensor],
    *,
    positions: torch.Tensor,
    entry_packet: torch.Tensor,
    layer_indices: Sequence[int],
) -> torch.Tensor:
    """Return differentiable block-input packets induced by one entry seed.

    The returned tensor has shape ``[batch, layers, positions, width]``. Only
    block 0 is replaced. Blocks 1 through the final requested block evolve
    normally, and execution stops before the next block to avoid unnecessary
    forward compute while preserving the autograd graph to the entry packet.
    """

    layers = [int(layer) for layer in layer_indices]
    if not layers or layers != list(range(layers[-1] + 1)):
        raise ValueError("layer_indices must be the contiguous prefix starting at 0")
    if not isinstance(entry_packet, torch.Tensor) or entry_packet.ndim != 3:
        raise ValueError("entry_packet must have shape [batch, positions, width]")
    if not isinstance(positions, torch.Tensor) or positions.ndim != 1:
        raise ValueError("positions must be a rank-1 tensor")
    if positions.numel() == 0 or entry_packet.shape[1] != positions.numel():
        raise ValueError("entry packet positions must match the replay positions")
    if inputs["input_ids"].ndim != 2:
        raise ValueError("receiver input_ids must have shape [batch, sequence]")
    if entry_packet.shape[0] != inputs["input_ids"].shape[0]:
        raise ValueError("entry packet batch must match receiver inputs")
    decoder_layers = model.model.layers
    stop_layer = layers[-1] + 1
    if stop_layer >= len(decoder_layers):
        raise ValueError("receiver must expose one block after the evolved prefix")

    captured: dict[int, torch.Tensor] = {0: entry_packet}
    handles = []

    def inject_entry(module, module_in):
        if not module_in:
            raise ValueError("block-0 pre-hook requires positional hidden states")
        hidden = module_in[0]
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("receiver block input must be rank three")
        if hidden.shape[0] != entry_packet.shape[0]:
            raise ValueError("receiver hidden batch differs from entry packet")
        if hidden.shape[-1] != entry_packet.shape[-1]:
            raise ValueError("receiver hidden width differs from entry packet")
        selected = positions.to(device=hidden.device, dtype=torch.long)
        if int(selected.min()) < 0 or int(selected.max()) >= hidden.shape[1]:
            raise ValueError("entry position is outside the receiver sequence")
        updated = hidden.clone()
        updated[:, selected, :] = entry_packet.to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        return (updated, *module_in[1:])

    def capture_input(layer: int):
        def hook(module, module_in):
            if not module_in:
                raise ValueError("receiver block pre-hook requires hidden states")
            hidden = module_in[0]
            selected = positions.to(device=hidden.device, dtype=torch.long)
            captured[layer] = hidden[:, selected, :]

        return hook

    def stop_forward(module, module_in):
        raise _StopReceiverForward

    handles.append(decoder_layers[0].register_forward_pre_hook(inject_entry))
    for layer in layers[1:]:
        handles.append(
            decoder_layers[layer].register_forward_pre_hook(capture_input(layer))
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
    except _StopReceiverForward:
        stopped = True
    finally:
        for handle in handles:
            handle.remove()
    if not stopped:
        raise RuntimeError("receiver forward did not reach the registered stop layer")
    missing = set(layers).difference(captured)
    if missing:
        raise RuntimeError(f"failed to capture receiver layer(s): {sorted(missing)}")
    return torch.stack([captured[layer] for layer in layers], dim=1)
