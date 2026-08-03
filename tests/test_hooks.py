import torch

import pytest

from src.integrations.hooks import make_lip_hook, make_lip_packet_hook


def test_hook_injects_once_without_mutating_original_output():
    original = torch.zeros(1, 3, 2)
    hook = make_lip_hook(torch.tensor([[2.0, 3.0]]), inject_pos=1)
    first = hook(None, (), original)
    second = hook(None, (), original)
    assert torch.equal(first[0, 1], torch.tensor([2.0, 3.0]))
    assert torch.equal(second, original)
    assert torch.equal(original, torch.zeros_like(original))


def test_hook_can_replace_matching_hidden_state():
    original = torch.ones(1, 2, 2)
    hook = make_lip_hook(
        torch.tensor([[4.0, 5.0]]),
        inject_pos=0,
        mode="replace",
    )
    result = hook(None, (), original)
    assert torch.equal(result[0, 0], torch.tensor([4.0, 5.0]))
    assert torch.equal(result[0, 1], torch.ones(2))


def test_packet_hook_replaces_multiple_positions_once_and_preserves_tuple():
    original = torch.zeros(1, 5, 2)
    cached = torch.tensor([9.0])
    hook = make_lip_packet_hook(
        torch.tensor([[2.0, 3.0], [4.0, 5.0]]),
        torch.tensor([1, 4]),
    )
    first = hook(None, (), (original, cached))
    second = hook(None, (), (original, cached))
    assert torch.equal(first[0][0, 1], torch.tensor([2.0, 3.0]))
    assert torch.equal(first[0][0, 4], torch.tensor([4.0, 5.0]))
    assert first[1] is cached
    assert second[0] is original
    assert torch.equal(original, torch.zeros_like(original))


def test_packet_hook_rejects_duplicate_positions():
    with pytest.raises(ValueError, match="unique"):
        make_lip_packet_hook(
            torch.zeros(2, 3),
            torch.tensor([1, 1]),
        )
