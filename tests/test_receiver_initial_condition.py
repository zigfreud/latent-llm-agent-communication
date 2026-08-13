import pytest
import torch

from src.core.receiver_initial_condition import evolve_receiver_from_entry_seed


class AddLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden):
        self.calls += 1
        return hidden + 1.0


class ToyBackbone(torch.nn.Module):
    def __init__(self, layers=5):
        super().__init__()
        self.layers = torch.nn.ModuleList([AddLayer() for _ in range(layers)])


class ToyReceiver:
    def __init__(self):
        self.model = ToyBackbone()

    def __call__(
        self,
        input_ids,
        attention_mask,
        use_cache,
        output_hidden_states,
        return_dict,
    ):
        hidden = input_ids.float().unsqueeze(-1)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return type("Output", (), {"logits": hidden})()


def _inputs(batch=2):
    return {
        "input_ids": torch.tensor([[1, 2, 3]]).expand(batch, -1).clone(),
        "attention_mask": torch.ones(batch, 3, dtype=torch.long),
    }


def test_entry_seed_evolves_prefix_and_preserves_gradient():
    receiver = ToyReceiver()
    entry = torch.tensor([[[10.0]], [[20.0]]], requires_grad=True)
    trajectory = evolve_receiver_from_entry_seed(
        receiver,
        _inputs(),
        positions=torch.tensor([2]),
        entry_packet=entry,
        layer_indices=[0, 1, 2],
    )
    assert trajectory[:, :, 0, 0].tolist() == [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]
    trajectory.sum().backward()
    assert entry.grad.tolist() == [[[3.0]], [[3.0]]]
    assert [layer.calls for layer in receiver.model.layers] == [1, 1, 1, 0, 0]


def test_entry_seed_requires_contiguous_prefix():
    with pytest.raises(ValueError, match="contiguous prefix"):
        evolve_receiver_from_entry_seed(
            ToyReceiver(),
            _inputs(),
            positions=torch.tensor([2]),
            entry_packet=torch.zeros(2, 1, 1),
            layer_indices=[0, 2],
        )


def test_entry_seed_requires_matching_batch():
    with pytest.raises(ValueError, match="batch"):
        evolve_receiver_from_entry_seed(
            ToyReceiver(),
            _inputs(batch=2),
            positions=torch.tensor([2]),
            entry_packet=torch.zeros(1, 1, 1),
            layer_indices=[0, 1],
        )
