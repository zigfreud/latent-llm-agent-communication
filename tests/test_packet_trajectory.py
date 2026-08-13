import pytest
import torch

from src.evaluation.packet_trajectory import (
    next_token_distribution_alignment,
    summarize_native_alignment,
    summarize_replay_discontinuity,
    tensor_alignment,
)
from src.pipelines.oracle_memory import forward_with_packet_trajectory_capture


class ToyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(1, 1, bias=False)
        self.k_proj = torch.nn.Linear(1, 1, bias=False)
        self.v_proj = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.q_proj.weight.fill_(2.0)
            self.k_proj.weight.fill_(3.0)
            self.v_proj.weight.fill_(4.0)

    def forward(self, hidden):
        value = self.q_proj(hidden) + self.k_proj(hidden) + self.v_proj(hidden)
        return (value, None)


class ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ToyAttention()

    def forward(self, hidden):
        return hidden + self.self_attn(hidden)[0]


class ToyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([ToyLayer(), ToyLayer()])


class ToyModel:
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


def toy_inputs():
    return {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }


def test_alignment_and_discontinuity_keep_entry_separate_from_transitions():
    alignment = tensor_alignment(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 4.0]))
    assert alignment["difference_rms"] == torch.sqrt(torch.tensor(2.0)).item()
    assert alignment["candidate_to_reference_norm_ratio"] > 1.0

    summary = summarize_replay_discontinuity(
        {0: torch.tensor([[3.0]]), 1: torch.tensor([[200.0]])},
        {0: torch.tensor([[20.0]]), 1: torch.tensor([[40.0]])},
        layer_indices=[0, 1],
    )
    assert summary["entry"]["role"] == "carrier_entry"
    assert summary["transitions"][0]["role"] == "cross_layer_transition"
    assert summary["transitions"][0]["relative_jump_rms"] == 4.0


def test_trajectory_capture_observes_incoming_state_before_each_replacement():
    packets = {0: torch.tensor([[20.0]]), 1: torch.tensor([[40.0]])}
    outputs, captured = forward_with_packet_trajectory_capture(
        ToyModel(),
        toy_inputs(),
        layer_indices=[0, 1],
        positions=torch.tensor([2]),
        layer_packets=packets,
    )
    assert captured["incoming_before_replay"][0].tolist() == [[3.0]]
    assert captured["residual_input"][0].tolist() == [[20.0]]
    assert captured["query_pre_rope"][0].tolist() == [[40.0]]
    assert captured["attention_output"][0].tolist() == [[180.0]]
    assert captured["residual_output"][0].tolist() == [[200.0]]
    assert captured["incoming_before_replay"][1].tolist() == [[200.0]]
    assert captured["residual_input"][1].tolist() == [[40.0]]
    assert captured["query_pre_rope"][1].tolist() == [[80.0]]
    assert outputs.logits[0, -1, 0].item() == 400.0


def test_native_and_logit_alignment_are_json_ready():
    native = {
        "residual_input": {0: torch.tensor([[1.0, 2.0]])},
        "query_pre_rope": {0: torch.tensor([[2.0, 4.0]])},
    }
    candidate = {
        "residual_input": {0: torch.tensor([[1.0, 2.0]])},
        "query_pre_rope": {0: torch.tensor([[1.0, 2.0]])},
    }
    state_summary = summarize_native_alignment(
        native,
        candidate,
        state_types=["residual_input", "query_pre_rope"],
        layer_indices=[0],
    )
    assert state_summary["state_summaries"]["residual_input"][
        "mean_cosine"
    ] == pytest.approx(1.0)
    assert state_summary["state_summaries"]["query_pre_rope"][
        "mean_candidate_to_native_norm_ratio"
    ] == 0.5

    logits = next_token_distribution_alignment(
        torch.tensor([[[0.0, 2.0, 1.0]]]),
        torch.tensor([[[0.0, 1.0, 2.0]]]),
    )
    assert logits["native_top1_token_id"] == 1
    assert logits["candidate_top1_token_id"] == 2
    assert logits["top1_agreement"] is False
    assert logits["kl_native_to_candidate"] > 0.0
