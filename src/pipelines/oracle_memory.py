"""Target-oracle capture and replay at transformer block-input boundaries."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from src.integrations.hooks import make_lip_packet_pre_hook


def forward_with_layer_input_capture(
    model,
    inputs: Mapping[str, torch.Tensor],
    *,
    layer_indices: Sequence[int],
    positions: torch.Tensor,
) -> tuple[Any, dict[int, torch.Tensor]]:
    """Capture exact prompt states entering each configured decoder block."""

    selected_layers = [int(layer) for layer in layer_indices]
    if not selected_layers or len(set(selected_layers)) != len(selected_layers):
        raise ValueError("layer_indices must be a non-empty unique sequence")
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def capture_hook(layer_idx: int):
        def hook(module, module_in):
            if not module_in:
                raise ValueError("block pre-hook requires positional hidden states")
            hidden = module_in[0]
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise ValueError("captured block input must be rank-3 hidden states")
            if hidden.shape[0] != 1:
                raise ValueError("oracle memory capture requires batch size one")
            selected = positions.to(device=hidden.device, dtype=torch.long)
            if selected.ndim != 1 or selected.numel() == 0:
                raise ValueError("memory positions must be a non-empty rank-1 tensor")
            if int(selected.min()) < 0 or int(selected.max()) >= hidden.shape[1]:
                raise ValueError("memory position is outside the hidden-state sequence")
            captured[layer_idx] = hidden[0, selected, :].detach().clone()

        return hook

    for layer_idx in selected_layers:
        handles.append(
            model.model.layers[layer_idx].register_forward_pre_hook(
                capture_hook(layer_idx)
            )
        )
    try:
        with torch.inference_mode():
            outputs = model(
                **inputs,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    missing = set(selected_layers).difference(captured)
    if missing:
        raise RuntimeError(
            f"failed to capture configured block inputs: {sorted(missing)}"
        )
    return outputs, captured


def forward_with_layer_input_replay(
    model,
    inputs: Mapping[str, torch.Tensor],
    *,
    positions: torch.Tensor,
    layer_packets: Mapping[int, torch.Tensor],
):
    """Run one forward pass with exact block-input prompt-state replay."""

    handles = _register_replay_hooks(model, positions, layer_packets)
    try:
        with torch.inference_mode():
            return model(
                **inputs,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()


def _register_replay_hooks(
    model,
    positions: torch.Tensor,
    layer_packets: Mapping[int, torch.Tensor],
) -> list[Any]:
    handles = []
    for layer_idx, vectors in layer_packets.items():
        hook = make_lip_packet_pre_hook(
            vectors,
            positions,
            enable=True,
            mode="replace",
        )
        handles.append(
            model.model.layers[int(layer_idx)].register_forward_pre_hook(hook)
        )
    return handles


def generate_with_layer_input_replay(
    model,
    tokenizer,
    inputs: Mapping[str, torch.Tensor],
    *,
    generation_kwargs: Mapping[str, Any],
    positions: torch.Tensor | None = None,
    layer_packets: Mapping[int, torch.Tensor] | None = None,
) -> str:
    """Generate while replaying oracle prompt states before selected blocks."""

    if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1:
        raise ValueError("oracle memory generation currently requires one prompt")
    packets = dict(layer_packets or {})
    if packets and positions is None:
        raise ValueError("positions are required with layer_packets")
    if positions is not None and not packets:
        raise ValueError("layer_packets are required with positions")
    handles = _register_replay_hooks(model, positions, packets)
    prompt_length = int(inputs["input_ids"].shape[1])
    try:
        with torch.inference_mode():
            generated = model.generate(**inputs, **dict(generation_kwargs))
    finally:
        for handle in handles:
            handle.remove()
    continuation = generated[0, prompt_length:]
    return tokenizer.decode(continuation, skip_special_tokens=True).replace(
        "</s>", ""
    ).strip()
