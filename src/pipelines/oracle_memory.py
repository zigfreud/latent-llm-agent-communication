"""Target-oracle capture and replay at transformer block-input boundaries."""

from __future__ import annotations

import math
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
    replay_mode: str | Mapping[int, str] = "replace",
    replay_alpha: float | Mapping[int, float] | None = None,
) -> tuple[Any, dict[str, dict[int, torch.Tensor]]]:
    """Capture native or replayed receiver states during one prefill forward."""

    selected_layers = [int(layer) for layer in layer_indices]
    if not selected_layers or len(set(selected_layers)) != len(selected_layers):
        raise ValueError("layer_indices must be a non-empty unique sequence")
    packets = {int(layer): value for layer, value in (layer_packets or {}).items()}
    replay_modes = _normalize_replay_modes(selected_layers, replay_mode)
    replay_alphas = _normalize_replay_alphas(
        selected_layers, replay_modes, replay_alpha
    )
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
            if replay_modes[layer_idx] == "replace":
                injected = vectors
            elif replay_modes[layer_idx] == "add":
                injected = hidden[0, selected, :] + vectors
            else:
                alpha = replay_alphas[layer_idx]
                assert alpha is not None
                injected = (
                    (1.0 - alpha) * hidden[0, selected, :] + alpha * vectors
                )
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
    replay_mode: str | Mapping[int, str] = "replace",
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
    replay_mode: str | Mapping[int, str] = "replace",
    replay_alpha: float | Mapping[int, float] | None = None,
) -> list[Any]:
    replay_modes = _normalize_replay_modes(layer_packets, replay_mode)
    replay_alphas = _normalize_replay_alphas(
        layer_packets, replay_modes, replay_alpha
    )
    handles = []
    for layer_idx, vectors in layer_packets.items():
        hook = make_lip_packet_pre_hook(
            vectors,
            positions,
            enable=True,
            mode=replay_modes[int(layer_idx)],
            blend_alpha=replay_alphas[int(layer_idx)],
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
    replay_mode: str | Mapping[int, str] = "replace",
    replay_alpha: float | Mapping[int, float] | None = None,
    forced_completion_prefix_token_ids: Sequence[int] | None = None,
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
        replay_alpha=replay_alpha,
    )
    prompt_length = int(inputs["input_ids"].shape[1])
    forced_prefix = tuple(
        int(token_id) for token_id in (forced_completion_prefix_token_ids or ())
    )
    if any(token_id < 0 for token_id in forced_prefix):
        raise ValueError("forced completion prefix token IDs must be non-negative")
    generate_kwargs = dict(generation_kwargs)
    if forced_prefix:
        if generate_kwargs.get("logits_processor") is not None:
            raise ValueError(
                "forced completion prefix cannot be combined with logits_processor"
            )
        from transformers import LogitsProcessorList

        generate_kwargs["logits_processor"] = LogitsProcessorList(
            [_ForcedCompletionPrefixProcessor(prompt_length, forced_prefix)]
        )
    try:
        with torch.inference_mode():
            generated = model.generate(**inputs, **generate_kwargs)
    finally:
        for handle in handles:
            handle.remove()
    continuation = generated[0, prompt_length:]
    output = tokenizer.decode(continuation, skip_special_tokens=True).replace(
        "</s>", ""
    ).strip()
    if forced_prefix:
        decoded_prefix = tokenizer.decode(
            list(forced_prefix),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if not decoded_prefix or not output.startswith(decoded_prefix):
            raise RuntimeError("generated output does not realize the forced prefix")
    return output


class _ForcedCompletionPrefixProcessor:
    """Force a fixed token sequence at the start of a decoder-only completion."""

    def __init__(self, prompt_length: int, token_ids: Sequence[int]):
        if prompt_length <= 0:
            raise ValueError("prompt_length must be positive")
        self.prompt_length = int(prompt_length)
        self.token_ids = tuple(int(token_id) for token_id in token_ids)
        if not self.token_ids or any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("token_ids must be a non-empty non-negative sequence")

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        generated_count = int(input_ids.shape[-1]) - self.prompt_length
        if generated_count < 0:
            raise ValueError("generation input is shorter than the frozen prompt")
        if generated_count >= len(self.token_ids):
            return scores
        token_id = self.token_ids[generated_count]
        if token_id >= scores.shape[-1]:
            raise ValueError("forced prefix token is outside the model vocabulary")
        constrained = torch.full_like(scores, -torch.inf)
        constrained[:, token_id] = 0.0
        return constrained


def _normalize_replay_modes(
    layer_indices: Sequence[int] | Mapping[int, Any],
    replay_mode: str | Mapping[int, str],
) -> dict[int, str]:
    """Resolve one replay mode per layer without weakening frozen callers."""

    layers = [int(layer) for layer in layer_indices]
    if isinstance(replay_mode, str):
        modes = {layer: replay_mode for layer in layers}
    else:
        modes = {int(layer): str(mode) for layer, mode in replay_mode.items()}
        if set(modes) != set(layers):
            raise ValueError("mapped replay_mode must cover exactly the replayed layers")
    invalid = {
        mode for mode in modes.values() if mode not in {"replace", "add", "blend"}
    }
    if invalid:
        raise ValueError("replay_mode must contain only replace, add, or blend")
    return modes


def _normalize_replay_alphas(
    layer_indices: Sequence[int] | Mapping[int, Any],
    replay_modes: Mapping[int, str],
    replay_alpha: float | Mapping[int, float] | None,
) -> dict[int, float | None]:
    """Resolve convex-blend weights while rejecting silently ignored values."""

    layers = [int(layer) for layer in layer_indices]
    blend_layers = {layer for layer in layers if replay_modes[layer] == "blend"}
    if not blend_layers:
        if replay_alpha is not None:
            raise ValueError("replay_alpha is valid only when replay_mode uses blend")
        return {layer: None for layer in layers}
    if replay_alpha is None:
        raise ValueError("blend replay requires replay_alpha")
    if isinstance(replay_alpha, Mapping):
        supplied = {int(layer): float(alpha) for layer, alpha in replay_alpha.items()}
        if set(supplied) != blend_layers:
            raise ValueError("mapped replay_alpha must cover exactly the blend layers")
    else:
        supplied = {layer: float(replay_alpha) for layer in blend_layers}
    invalid = {
        alpha
        for alpha in supplied.values()
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0
    }
    if invalid:
        raise ValueError("replay_alpha must contain only finite values in [0, 1]")
    return {
        layer: supplied[layer] if layer in blend_layers else None for layer in layers
    }
