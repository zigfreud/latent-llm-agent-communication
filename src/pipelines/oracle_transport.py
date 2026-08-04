"""Shared target-oracle carrier, capture, and intervention primitives."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from src.core.prompt_protocol import (
    format_prompt,
    tokenizer_add_special_tokens,
)
from src.integrations.hooks import make_lip_hook, make_lip_packet_hook


def encode_prompt(prompt: str, tokenizer, protocol: Mapping[str, Any], device):
    formatted = format_prompt(prompt, tokenizer, protocol)
    encoded = tokenizer(
        formatted,
        return_tensors="pt",
        add_special_tokens=tokenizer_add_special_tokens(protocol),
    )
    return formatted, {key: value.to(device) for key, value in encoded.items()}


def append_reference(inputs: Mapping[str, torch.Tensor], reference_ids: torch.Tensor):
    input_ids = torch.cat((inputs["input_ids"], reference_ids.unsqueeze(0)), dim=1)
    attention_mask = torch.cat(
        (
            inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
            torch.ones(
                (1, reference_ids.numel()),
                dtype=torch.long,
                device=input_ids.device,
            ),
        ),
        dim=1,
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def build_neutral_carrier(
    neutral_inputs: Mapping[str, torch.Tensor],
    *,
    task_prompt_length: int,
    pad_token_id: int,
    mode: str,
) -> dict[str, torch.Tensor]:
    """Return the native neutral prompt or a masked, position-matched carrier."""

    input_ids = neutral_inputs["input_ids"]
    attention_mask = neutral_inputs.get("attention_mask", torch.ones_like(input_ids))
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("neutral carrier currently requires one prompt")
    native_length = int(input_ids.shape[1])
    if mode == "native":
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    if mode != "left_pad_masked_to_task_length":
        raise ValueError(f"unsupported neutral carrier mode: {mode}")
    if task_prompt_length < native_length:
        raise ValueError(
            "task prompt is shorter than the native neutral prompt; cannot left-pad"
        )
    padding = task_prompt_length - native_length
    if padding == 0:
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    pad_ids = torch.full(
        (1, padding),
        int(pad_token_id),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    pad_mask = torch.zeros(
        (1, padding),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return {
        "input_ids": torch.cat((pad_ids, input_ids), dim=1),
        "attention_mask": torch.cat((pad_mask, attention_mask), dim=1),
    }


def validate_neutral_carrier_task_lengths(
    neutral_inputs: Mapping[str, torch.Tensor],
    task_prompt_lengths: Mapping[str, int],
    *,
    mode: str,
) -> None:
    """Reject an incompatible task registry before expensive state capture."""

    if mode == "native":
        return
    if mode != "left_pad_masked_to_task_length":
        raise ValueError(f"unsupported neutral carrier mode: {mode}")
    native_length = int(neutral_inputs["input_ids"].shape[1])
    incompatible = {
        str(task_id): int(prompt_length)
        for task_id, prompt_length in task_prompt_lengths.items()
        if int(prompt_length) < native_length
    }
    if incompatible:
        details = ", ".join(
            f"{task_id}={prompt_length}"
            for task_id, prompt_length in incompatible.items()
        )
        raise ValueError(
            "neutral carrier is longer than selected task prompts "
            f"(neutral={native_length}; tasks: {details})"
        )


def forward_with_optional_replacement(
    model,
    inputs: Mapping[str, torch.Tensor],
    *,
    layer_idx: int | None = None,
    position: int | None = None,
    vector: torch.Tensor | None = None,
):
    handle = None
    if vector is not None:
        if layer_idx is None or position is None:
            raise ValueError("layer_idx and position are required for replacement")
        hook = make_lip_hook(vector, position, enable=True, mode="replace")
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            return model(
                **inputs,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        if handle is not None:
            handle.remove()


def forward_with_layer_capture(
    model,
    inputs: Mapping[str, torch.Tensor],
    *,
    layers: list[int],
    position: int,
):
    """Run one forward pass and capture actual transformer-block outputs."""

    captured: dict[int, torch.Tensor] = {}
    handles = []

    def capture_hook(layer_idx: int):
        def hook(module, module_in, module_out):
            hidden = module_out[0] if isinstance(module_out, tuple) else module_out
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise ValueError("captured layer output must contain rank-3 hidden states")
            if not 0 <= position < hidden.shape[1]:
                raise ValueError("capture position is outside the hidden-state sequence")
            captured[layer_idx] = hidden[0, position, :].detach().clone()

        return hook

    for layer in layers:
        handles.append(
            model.model.layers[layer].register_forward_hook(capture_hook(layer))
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
    missing = set(layers).difference(captured)
    if missing:
        raise RuntimeError(f"failed to capture configured layer outputs: {sorted(missing)}")
    return outputs, captured


def forward_with_packet_capture(
    model,
    inputs: Mapping[str, torch.Tensor],
    *,
    layer_idx: int,
    positions: torch.Tensor,
):
    """Run one forward pass and capture a packet from one block output."""

    captured = None

    def hook(module, module_in, module_out):
        nonlocal captured
        hidden = module_out[0] if isinstance(module_out, tuple) else module_out
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("captured layer output must contain rank-3 hidden states")
        selected = positions.to(device=hidden.device, dtype=torch.long)
        if selected.ndim != 1 or selected.numel() == 0:
            raise ValueError("packet positions must be a non-empty rank-1 tensor")
        if int(selected.min()) < 0 or int(selected.max()) >= hidden.shape[1]:
            raise ValueError("packet position is outside the hidden-state sequence")
        captured = hidden[0, selected, :].detach().clone()

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            outputs = model(
                **inputs,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        handle.remove()
    if captured is None:
        raise RuntimeError("failed to capture oracle packet")
    return outputs, captured


def forward_with_packet_replacement(
    model,
    inputs: Mapping[str, torch.Tensor],
    *,
    layer_idx: int,
    positions: torch.Tensor,
    vectors: torch.Tensor,
):
    """Run one forward pass while replacing a packet at one block output."""

    hook = make_lip_packet_hook(vectors, positions, enable=True, mode="replace")
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            return model(
                **inputs,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        handle.remove()


def generate_with_optional_packet(
    model,
    tokenizer,
    inputs: Mapping[str, torch.Tensor],
    *,
    generation_kwargs: Mapping[str, Any],
    layer_idx: int | None = None,
    positions: torch.Tensor | None = None,
    vectors: torch.Tensor | None = None,
) -> str:
    """Generate a continuation while replacing one packet during prompt prefill."""

    if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1:
        raise ValueError("oracle packet generation currently requires one prompt")
    handle = None
    if vectors is not None:
        if layer_idx is None or positions is None:
            raise ValueError("layer_idx and positions are required with packet vectors")
        hook = make_lip_packet_hook(vectors, positions, enable=True, mode="replace")
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
    elif layer_idx is not None or positions is not None:
        raise ValueError("packet layer and positions require packet vectors")
    prompt_length = int(inputs["input_ids"].shape[1])
    try:
        with torch.inference_mode():
            generated = model.generate(**inputs, **dict(generation_kwargs))
    finally:
        if handle is not None:
            handle.remove()
    continuation = generated[0, prompt_length:]
    return tokenizer.decode(continuation, skip_special_tokens=True).replace(
        "</s>", ""
    ).strip()
