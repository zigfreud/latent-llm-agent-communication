"""Target-oracle capture and replay at transformer block-input boundaries."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from src.integrations.hooks import make_lip_packet_pre_hook


ORACLE_CAPTURED_STATE_TYPES = (
    "residual_input",
    "key_pre_rope",
    "value_pre_cache",
)

TRAJECTORY_CAPTURED_STATE_TYPES = (
    "incoming_before_replay",
    "residual_input",
    "query_pre_rope",
    "key_pre_rope",
    "value_pre_cache",
    "attention_output",
    "residual_output",
)


def _select_prompt_packet(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise ValueError("captured prompt state must be a rank-3 tensor")
    if hidden.shape[0] != 1:
        raise ValueError("oracle memory capture requires batch size one")
    selected = positions.to(device=hidden.device, dtype=torch.long)
    if selected.ndim != 1 or selected.numel() == 0:
        raise ValueError("memory positions must be a non-empty rank-1 tensor")
    if int(selected.min()) < 0 or int(selected.max()) >= hidden.shape[1]:
        raise ValueError("memory position is outside the hidden-state sequence")
    return hidden[0, selected, :].detach().clone()


def forward_with_layer_state_capture(
    model,
    inputs: Mapping[str, torch.Tensor],
    *,
    layer_indices: Sequence[int],
    positions: torch.Tensor,
) -> tuple[Any, dict[str, dict[int, torch.Tensor]]]:
    """Capture residual inputs and pre-cache K/V projections for each block."""

    selected_layers = [int(layer) for layer in layer_indices]
    if not selected_layers or len(set(selected_layers)) != len(selected_layers):
        raise ValueError("layer_indices must be a non-empty unique sequence")
    captured = {state_type: {} for state_type in ORACLE_CAPTURED_STATE_TYPES}
    handles = []

    def capture_block_input(layer_idx: int):
        def hook(module, module_in):
            if not module_in:
                raise ValueError("block pre-hook requires positional hidden states")
            captured["residual_input"][layer_idx] = _select_prompt_packet(
                module_in[0], positions
            )

        return hook

    def capture_projection(state_type: str, layer_idx: int):
        def hook(module, module_in, module_out):
            captured[state_type][layer_idx] = _select_prompt_packet(
                module_out, positions
            )

        return hook

    for layer_idx in selected_layers:
        layer = model.model.layers[layer_idx]
        attention = getattr(layer, "self_attn", None)
        key_projection = getattr(attention, "k_proj", None)
        value_projection = getattr(attention, "v_proj", None)
        if key_projection is None or value_projection is None:
            raise ValueError("decoder layer does not expose self_attn k_proj/v_proj")
        handles.extend(
            (
                layer.register_forward_pre_hook(capture_block_input(layer_idx)),
                key_projection.register_forward_hook(
                    capture_projection("key_pre_rope", layer_idx)
                ),
                value_projection.register_forward_hook(
                    capture_projection("value_pre_cache", layer_idx)
                ),
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
    for state_type in ORACLE_CAPTURED_STATE_TYPES:
        missing = set(selected_layers).difference(captured[state_type])
        if missing:
            raise RuntimeError(
                f"failed to capture {state_type} at layer(s): {sorted(missing)}"
            )
    return outputs, captured


def forward_with_packet_trajectory_capture(
    model,
    inputs: Mapping[str, torch.Tensor],
    *,
    layer_indices: Sequence[int],
    positions: torch.Tensor,
    layer_packets: Mapping[int, torch.Tensor] | None = None,
    replay_mode: str = "replace",
) -> tuple[Any, dict[str, dict[int, torch.Tensor]]]:
    """Capture native or replayed receiver states during one prefill forward."""

    selected_layers = [int(layer) for layer in layer_indices]
    if not selected_layers or len(set(selected_layers)) != len(selected_layers):
        raise ValueError("layer_indices must be a non-empty unique sequence")
    packets = {int(layer): value for layer, value in (layer_packets or {}).items()}
    if replay_mode not in {"replace", "add"}:
        raise ValueError("replay_mode must be replace or add")
    unknown_packets = set(packets).difference(selected_layers)
    missing_packets = set(selected_layers).difference(packets) if packets else set()
    if unknown_packets or missing_packets:
        raise ValueError("layer_packets must cover exactly the configured layers")
    captured = {state_type: {} for state_type in TRAJECTORY_CAPTURED_STATE_TYPES}
    handles = []

    def store(state_type: str, layer_idx: int, value: torch.Tensor) -> None:
        captured[state_type][layer_idx] = (
            _select_prompt_packet(value, positions).float().cpu()
        )

    def capture_and_optionally_replay(layer_idx: int):
        def hook(module, module_in):
            if not module_in:
                raise ValueError("block pre-hook requires positional hidden states")
            hidden = module_in[0]
            store("incoming_before_replay", layer_idx, hidden)
            if not packets:
                store("residual_input", layer_idx, hidden)
                return None
            selected = positions.to(device=hidden.device, dtype=torch.long)
            vectors = packets[layer_idx].to(device=hidden.device, dtype=hidden.dtype)
            if vectors.shape != (selected.numel(), hidden.shape[-1]):
                raise ValueError(
                    f"layer {layer_idx} packet shape does not match replay sites"
                )
            updated = hidden.clone()
            if replay_mode == "replace":
                injected = vectors
            else:
                injected = hidden[0, selected, :] + vectors
            captured["residual_input"][layer_idx] = (
                injected.detach().float().cpu()
            )
            updated[0, selected, :] = injected
            return (updated, *module_in[1:])

        return hook

    def capture_projection(state_type: str, layer_idx: int):
        def hook(module, module_in, module_out):
            store(state_type, layer_idx, module_out)

        return hook

    def capture_attention_output(layer_idx: int):
        def hook(module, module_in, module_out):
            value = module_out[0] if isinstance(module_out, (tuple, list)) else module_out
            store("attention_output", layer_idx, value)

        return hook

    def capture_block_output(layer_idx: int):
        def hook(module, module_in, module_out):
            value = module_out[0] if isinstance(module_out, (tuple, list)) else module_out
            store("residual_output", layer_idx, value)

        return hook

    for layer_idx in selected_layers:
        layer = model.model.layers[layer_idx]
        attention = getattr(layer, "self_attn", None)
        projections = {
            "query_pre_rope": getattr(attention, "q_proj", None),
            "key_pre_rope": getattr(attention, "k_proj", None),
            "value_pre_cache": getattr(attention, "v_proj", None),
        }
        if attention is None or any(value is None for value in projections.values()):
            raise ValueError("decoder layer does not expose self_attn q_proj/k_proj/v_proj")
        handles.append(
            layer.register_forward_pre_hook(capture_and_optionally_replay(layer_idx))
        )
        for state_type, projection in projections.items():
            handles.append(
                projection.register_forward_hook(
                    capture_projection(state_type, layer_idx)
                )
            )
        handles.append(attention.register_forward_hook(capture_attention_output(layer_idx)))
        handles.append(layer.register_forward_hook(capture_block_output(layer_idx)))
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
    for state_type in TRAJECTORY_CAPTURED_STATE_TYPES:
        missing = set(selected_layers).difference(captured[state_type])
        if missing:
            raise RuntimeError(
                f"failed to capture {state_type} at layer(s): {sorted(missing)}"
            )
    return outputs, captured



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
            captured[layer_idx] = _select_prompt_packet(module_in[0], positions)

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
    replay_mode: str = "replace",
):
    """Run one forward pass with exact block-input prompt-state replay."""

    handles = _register_replay_hooks(
        model,
        positions,
        layer_packets,
        replay_mode=replay_mode,
    )
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
    *,
    replay_mode: str = "replace",
) -> list[Any]:
    handles = []
    for layer_idx, vectors in layer_packets.items():
        hook = make_lip_packet_pre_hook(
            vectors,
            positions,
            enable=True,
            mode=replay_mode,
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
    replay_mode: str = "replace",
) -> str:
    """Generate while replaying oracle prompt states before selected blocks."""

    if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1:
        raise ValueError("oracle memory generation currently requires one prompt")
    packets = dict(layer_packets or {})
    if packets and positions is None:
        raise ValueError("positions are required with layer_packets")
    if positions is not None and not packets:
        raise ValueError("layer_packets are required with positions")
    handles = _register_replay_hooks(
        model,
        positions,
        packets,
        replay_mode=replay_mode,
    )
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
