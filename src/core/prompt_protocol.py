"""Shared prompt formatting contract for latent extraction and inference.

The protocol is intentionally small.  A latent pair is meaningful only when the
same formatting policy is used while building the bundle and while running a
probe, so callers should persist :func:`protocol_metadata` with their artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional


PROMPT_PROTOCOL_VERSION = "lip-prompt-v1"
SUPPORTED_PROMPT_MODES = {"raw", "chat_template"}
PROMPT_ROLES = ("source", "target")


@dataclass(frozen=True)
class PromptProtocol:
    version: str = PROMPT_PROTOCOL_VERSION
    mode: str = "raw"
    add_generation_prompt: bool = False
    system_prompt: Optional[str] = None


def parse_prompt_protocol(config: Optional[Mapping[str, Any]] = None) -> PromptProtocol:
    """Validate and normalize a prompt protocol mapping."""

    values = dict(config or {})
    unknown = sorted(
        set(values).difference(
            {"version", "mode", "add_generation_prompt", "system_prompt"}
        )
    )
    if unknown:
        raise ValueError(f"unknown prompt_protocol field(s): {', '.join(unknown)}")

    protocol = PromptProtocol(**values)
    if protocol.version != PROMPT_PROTOCOL_VERSION:
        raise ValueError(
            f"prompt_protocol.version must be {PROMPT_PROTOCOL_VERSION!r}"
        )
    if protocol.mode not in SUPPORTED_PROMPT_MODES:
        allowed = ", ".join(sorted(SUPPORTED_PROMPT_MODES))
        raise ValueError(f"prompt_protocol.mode must be one of: {allowed}")
    if not isinstance(protocol.add_generation_prompt, bool):
        raise ValueError("prompt_protocol.add_generation_prompt must be a boolean")
    if protocol.system_prompt is not None:
        if not isinstance(protocol.system_prompt, str) or not protocol.system_prompt.strip():
            raise ValueError("prompt_protocol.system_prompt must be null or non-empty text")
        if protocol.mode != "chat_template":
            raise ValueError(
                "prompt_protocol.system_prompt is supported only with mode=chat_template"
            )

    return protocol


def format_prompt(
    prompt: str,
    tokenizer: Any,
    protocol_config: Optional[Mapping[str, Any]] = None,
    *,
    add_generation_prompt: Optional[bool] = None,
) -> str:
    """Format one user prompt according to the shared protocol."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be non-empty text")

    protocol = parse_prompt_protocol(protocol_config)
    generation_marker = (
        protocol.add_generation_prompt
        if add_generation_prompt is None
        else add_generation_prompt
    )
    if not isinstance(generation_marker, bool):
        raise ValueError("add_generation_prompt override must be a boolean")

    if protocol.mode == "raw":
        return prompt

    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        raise ValueError(
            "mode=chat_template requires tokenizer.apply_chat_template"
        )

    messages = []
    if protocol.system_prompt is not None:
        messages.append({"role": "system", "content": protocol.system_prompt})
    messages.append({"role": "user", "content": prompt})
    formatted = apply_template(
        messages,
        tokenize=False,
        add_generation_prompt=generation_marker,
    )
    if not isinstance(formatted, str) or not formatted:
        raise RuntimeError("tokenizer.apply_chat_template returned no text")
    return formatted


def format_prompts(
    prompts: Iterable[str],
    tokenizer: Any,
    protocol_config: Optional[Mapping[str, Any]] = None,
    *,
    add_generation_prompt: Optional[bool] = None,
) -> list[str]:
    """Format a sequence without mutating the input prompts."""

    return [
        format_prompt(
            prompt,
            tokenizer,
            protocol_config,
            add_generation_prompt=add_generation_prompt,
        )
        for prompt in prompts
    ]


def protocol_metadata(
    protocol_config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return a JSON-serializable normalized protocol description."""

    return asdict(parse_prompt_protocol(protocol_config))


def protocol_pair_metadata(
    protocol_configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate and normalize role-specific source/target prompt protocols."""

    if not isinstance(protocol_configs, Mapping):
        raise ValueError("prompt_protocols must be a mapping")
    unknown = sorted(set(protocol_configs).difference(PROMPT_ROLES))
    missing = sorted(set(PROMPT_ROLES).difference(protocol_configs))
    if unknown:
        raise ValueError(f"unknown prompt_protocols role(s): {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing prompt_protocols role(s): {', '.join(missing)}")
    return {
        role: protocol_metadata(protocol_configs[role])
        for role in PROMPT_ROLES
    }


def resolve_role_prompt_protocol(
    config: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    """Resolve one role while retaining legacy single-protocol compatibility."""

    if role not in PROMPT_ROLES:
        raise ValueError(f"prompt role must be one of: {', '.join(PROMPT_ROLES)}")
    has_single = "prompt_protocol" in config
    has_pair = "prompt_protocols" in config
    if has_single and has_pair:
        raise ValueError("configure prompt_protocol or prompt_protocols, not both")
    if has_pair:
        return protocol_pair_metadata(config["prompt_protocols"])[role]
    return protocol_metadata(config.get("prompt_protocol"))


def protocol_contract_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the manifest fragment for a single or role-specific contract."""

    has_single = "prompt_protocol" in config
    has_pair = "prompt_protocols" in config
    if has_single and has_pair:
        raise ValueError("configure prompt_protocol or prompt_protocols, not both")
    if has_pair:
        return {"prompt_protocols": protocol_pair_metadata(config["prompt_protocols"])}
    return {"prompt_protocol": protocol_metadata(config.get("prompt_protocol"))}


def tokenizer_add_special_tokens(
    protocol_config: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Avoid duplicating special tokens already emitted by a chat template."""

    return parse_prompt_protocol(protocol_config).mode != "chat_template"
