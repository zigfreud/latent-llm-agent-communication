import pytest
import torch

from src.core.hidden_states import last_non_padding_indices, select_hidden_vectors


def test_last_non_padding_handles_left_and_right_padding():
    mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    assert last_non_padding_indices(mask).tolist() == [1, 3, 3]

    hidden = torch.arange(3 * 4 * 2).reshape(3, 4, 2)
    selected = select_hidden_vectors(hidden, mask, "last_non_padding")
    assert torch.equal(selected[0], hidden[0, 1])
    assert torch.equal(selected[1], hidden[1, 3])
    assert torch.equal(selected[2], hidden[2, 3])


def test_empty_attention_row_is_rejected():
    with pytest.raises(ValueError, match="at least one token"):
        last_non_padding_indices(torch.tensor([[0, 0]]))
