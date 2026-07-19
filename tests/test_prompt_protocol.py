import pytest

from src.core.prompt_protocol import (
    format_prompt,
    parse_prompt_protocol,
    protocol_metadata,
    tokenizer_add_special_tokens,
)


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "<chat>" + messages[-1]["content"] + "</chat>"


def test_raw_protocol_preserves_prompt_and_uses_tokenizer_special_tokens():
    tokenizer = FakeTokenizer()
    assert format_prompt("write code", tokenizer) == "write code"
    assert tokenizer_add_special_tokens({"mode": "raw"}) is True
    assert tokenizer.calls == []


def test_chat_protocol_uses_tokenizer_template_once():
    tokenizer = FakeTokenizer()
    config = {
        "version": "lip-prompt-v1",
        "mode": "chat_template",
        "add_generation_prompt": True,
        "system_prompt": "Return Python.",
    }
    assert format_prompt("write code", tokenizer, config) == "<chat>write code</chat>"
    messages, kwargs = tokenizer.calls[0]
    assert messages == [
        {"role": "system", "content": "Return Python."},
        {"role": "user", "content": "write code"},
    ]
    assert kwargs["tokenize"] is False
    assert kwargs["add_generation_prompt"] is True
    assert tokenizer_add_special_tokens(config) is False


def test_protocol_metadata_is_normalized_and_unknown_fields_fail():
    assert protocol_metadata({"mode": "raw"}) == {
        "version": "lip-prompt-v1",
        "mode": "raw",
        "add_generation_prompt": False,
        "system_prompt": None,
    }
    with pytest.raises(ValueError, match="unknown prompt_protocol"):
        parse_prompt_protocol({"template": "legacy"})
