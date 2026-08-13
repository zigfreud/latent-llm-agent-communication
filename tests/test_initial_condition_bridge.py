import torch

from src.pipelines.initial_condition_bridge import (
    _entry_raw_packet,
    _induced_trajectory,
    _normalize_trajectory,
    _repeat_receiver_inputs,
)


class AddLayer(torch.nn.Module):
    def forward(self, hidden):
        return hidden + 1.0


class ToyReceiver:
    def __init__(self):
        self.model = type(
            "Backbone",
            (),
            {"layers": torch.nn.ModuleList([AddLayer() for _ in range(5)])},
        )()

    def __call__(self, input_ids, attention_mask, **kwargs):
        hidden = input_ids.float().unsqueeze(-1)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return type("Output", (), {"logits": hidden})()


def test_entry_raw_packet_reconstructs_only_layer_zero():
    normalized = torch.tensor([[[[2.0], [3.0]]]])
    scaffold = torch.tensor([[[10.0], [20.0]], [[30.0], [40.0]]])
    scale = torch.tensor([[0.5, 2.0], [3.0, 4.0]])
    raw = _entry_raw_packet(normalized, scaffold, scale)
    assert raw.tolist() == [[[11.0], [26.0]]]


def test_normalize_trajectory_uses_each_layer_site_statistics():
    scaffold = torch.tensor([[[1.0]], [[10.0]]])
    scale = torch.tensor([[2.0], [5.0]])
    raw = torch.tensor([[[[5.0]], [[20.0]]]])
    normalized = _normalize_trajectory(raw, scaffold, scale)
    assert normalized.tolist() == [[[[2.0]], [[2.0]]]]


def test_repeat_receiver_inputs_expands_one_neutral_carrier():
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    repeated = _repeat_receiver_inputs(inputs, 4)
    assert repeated["input_ids"].shape == (4, 3)
    assert repeated["input_ids"].tolist() == [[1, 2, 3]] * 4


def test_induced_trajectory_remains_differentiable_after_normalization():
    entry = torch.tensor([[[[10.0]]]], requires_grad=True)
    trajectory = _induced_trajectory(
        ToyReceiver(),
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        },
        positions=torch.tensor([2]),
        normalized_entry=entry,
        scaffold=torch.zeros(3, 1, 1),
        site_scale=torch.ones(3, 1),
        layers=[0, 1, 2],
    )
    assert trajectory[:, :, 0, 0].tolist() == [[10.0, 11.0, 12.0]]
    trajectory.sum().backward()
    assert entry.grad.tolist() == [[[[3.0]]]]
