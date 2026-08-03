import torch
import yaml

from src.evaluation.oracle_memory import (
    ORACLE_MEMORY_CONDITIONS,
    build_condition_plan,
    design_fingerprint,
    semantic_gate,
    validate_memory_contract,
)
from src.integrations.hooks import make_lip_packet_pre_hook
from src.pipelines.oracle_memory import (
    forward_with_layer_input_capture,
    forward_with_layer_input_replay,
    generate_with_layer_input_replay,
)
from src.scripts.run_oracle_memory_functional import validate_config
from src.scripts.evaluate_oracle_packet_semantics import validate_generation_grid


class AddLayer(torch.nn.Module):
    def __init__(self, increment):
        super().__init__()
        self.increment = increment

    def forward(self, hidden):
        return hidden + self.increment


class ToyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([AddLayer(1.0), AddLayer(10.0)])


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

    def generate(self, input_ids, attention_mask, **kwargs):
        hidden = input_ids.float().unsqueeze(-1)
        for layer in self.model.layers:
            hidden = layer(hidden)
        next_token = hidden[:, -1, 0].long().unsqueeze(1)
        return torch.cat((input_ids, next_token), dim=1)


class ToyTokenizer:
    def decode(self, tokens, skip_special_tokens):
        return ",".join(str(value) for value in tokens.tolist())


def toy_inputs():
    return {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }


def test_packet_pre_hook_replaces_only_the_first_forward():
    layer = AddLayer(1.0)
    handle = layer.register_forward_pre_hook(
        make_lip_packet_pre_hook(
            torch.tensor([[9.0]]),
            torch.tensor([2]),
        )
    )
    try:
        first = layer(torch.tensor([[[1.0], [2.0], [3.0]]]))
        second = layer(torch.tensor([[[1.0], [2.0], [3.0]]]))
    finally:
        handle.remove()
    assert first[0, :, 0].tolist() == [2.0, 3.0, 10.0]
    assert second[0, :, 0].tolist() == [2.0, 3.0, 4.0]


def test_capture_records_inputs_at_each_block_boundary():
    _, packets = forward_with_layer_input_capture(
        ToyModel(),
        toy_inputs(),
        layer_indices=[0, 1],
        positions=torch.tensor([2]),
    )
    assert packets[0].tolist() == [[3.0]]
    assert packets[1].tolist() == [[4.0]]


def test_same_state_input_replay_is_an_exact_forward_identity():
    model = ToyModel()
    baseline, packets = forward_with_layer_input_capture(
        model,
        toy_inputs(),
        layer_indices=[0, 1],
        positions=torch.tensor([2]),
    )
    replayed = forward_with_layer_input_replay(
        model,
        toy_inputs(),
        positions=torch.tensor([2]),
        layer_packets=packets,
    )
    assert torch.equal(replayed.logits, baseline.logits)


def test_replay_changes_each_selected_layer_input_during_prefill():
    output = generate_with_layer_input_replay(
        ToyModel(),
        ToyTokenizer(),
        toy_inputs(),
        generation_kwargs={},
        positions=torch.tensor([2]),
        layer_packets={0: torch.tensor([[20.0]]), 1: torch.tensor([[40.0]])},
    )
    assert output == "50"


def frozen_memory_config():
    return {
        "packet_size": 32,
        "decoder_layer_count": 32,
        "self_check_tasks": 1,
        "maximum_self_logit_delta": 0.0001,
        "scopes": [
            {
                "name": "single_layer_output",
                "boundary": "block_output",
                "layers": [-16],
            },
            {
                "name": "late_half_input",
                "boundary": "block_input",
                "layers": list(range(-16, 0)),
            },
            {
                "name": "all_layer_input",
                "boundary": "block_input",
                "layers": list(range(-32, 0)),
            },
        ],
    }


def test_memory_contract_freezes_depth_and_hook_boundary():
    scopes = validate_memory_contract(frozen_memory_config())
    assert [scope["name"] for scope in scopes] == [
        "single_layer_output",
        "late_half_input",
        "all_layer_input",
    ]


def test_condition_plan_deranges_each_scope_without_changing_capacity():
    plan = build_condition_plan(
        ["a", "b", "c"],
        ORACLE_MEMORY_CONDITIONS,
        shuffle_seed=1729,
    )
    shuffled = [
        item
        for item in plan
        if item.condition.startswith("shuffled_oracle_")
    ]
    assert shuffled
    assert all(item.oracle_index != item.task_index for item in shuffled)
    assert {item.scope_name for item in shuffled} == {
        "single_layer_output",
        "late_half_input",
        "all_layer_input",
    }


def test_memory_gate_selects_smallest_task_specific_scope():
    means = {condition: 0.0 for condition in ORACLE_MEMORY_CONDITIONS}
    means["text_only_no_lip"] = 0.5
    means["oracle_late_half_input_k32"] = 0.25
    means["oracle_all_layer_input_k32"] = 0.5
    means["shuffled_oracle_all_layer_input_k32"] = 0.25
    gate = semantic_gate(means)
    assert gate["passed"] is True
    assert gate["supported_scopes"] == ["late_half_input", "all_layer_input"]
    assert gate["smallest_supported_scope"] == "late_half_input"


def test_registered_proto008_config_matches_frozen_contract():
    with open(
        "config/LIP-PROTO-008_oracle_multilayer_memory.yaml",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)
    validate_config(config)


def test_proto008_sampling_keeps_proto007_target_revision():
    with open(
        "config/LIP-PROTO-008_mbpp_test_sampling.yaml",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)
    assert config["target_model_revision"] == (
        "53346005fb0ef11d3b6a83b12c895cca40156b6c"
    )


def test_memory_generation_grid_uses_proto008_fingerprint():
    with open(
        "config/LIP-PROTO-008_oracle_multilayer_memory.yaml",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)
    task_ids = [f"task-{index}" for index in range(16)]
    records = [
        {
            "protocol_version": "lip-oracle-multilayer-memory-v1",
            "design_sha256": design_fingerprint(config),
            "task_id": task_id,
            "condition": condition,
            "generation_seed": 101,
            "task_spec": {"task_id": task_id, "test_list": ["assert True"]},
        }
        for task_id in task_ids
        for condition in ORACLE_MEMORY_CONDITIONS
    ]
    metadata = {
        "protocol_version": "lip-oracle-multilayer-memory-v1",
        "design_sha256": records[0]["design_sha256"],
        "task_ids": task_ids,
        "generation_seeds": [101],
        "run_scope": "full",
    }
    result = validate_generation_grid(
        records,
        metadata,
        config,
        allow_incomplete=False,
    )
    assert result["complete"] is True
