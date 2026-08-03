import copy

import pytest
import torch

from src.evaluation.oracle_functional import (
    ORACLE_FUNCTIONAL_CONDITIONS,
    ORACLE_FUNCTIONAL_PROTOCOL_VERSION,
    build_condition_plan,
    declares_entry_point,
    design_fingerprint,
)
from src.pipelines.oracle_experiment import load_yaml
from src.pipelines.oracle_transport import generate_with_optional_packet
from src.scripts.evaluate_oracle_packet_semantics import validate_generation_grid
from src.scripts.run_oracle_packet_functional import validate_config


CONFIG_PATH = "config/LIP-PROTO-005_oracle_packet_functional.yaml"


def frozen_config():
    from pathlib import Path

    return load_yaml(Path(CONFIG_PATH))


def test_frozen_functional_config_is_valid_and_capacity_is_not_mutable():
    config = frozen_config()
    validate_config(config)
    changed = copy.deepcopy(config)
    changed["packet"]["selected_size"] = 16
    with pytest.raises(ValueError, match="selected_size=8"):
        validate_config(changed)


def test_condition_plan_uses_same_task_packets_and_shuffled_derangement():
    task_ids = [f"t{index}" for index in range(8)]
    plan = build_condition_plan(
        task_ids,
        ORACLE_FUNCTIONAL_CONDITIONS,
        shuffle_seed=1729,
    )
    k8 = [row for row in plan if row.condition == "oracle_packet_k8"]
    shuffled = [
        row for row in plan if row.condition == "shuffled_oracle_packet_k8"
    ]
    assert all(row.packet_index == row.task_index for row in k8)
    assert all(row.packet_index != row.task_index for row in shuffled)
    assert sorted(row.packet_index for row in shuffled) == list(range(8))


def test_design_fingerprint_changes_with_generation_or_packet_contract():
    config = frozen_config()
    baseline = design_fingerprint(config)
    changed = copy.deepcopy(config)
    changed["generation"]["temperature"] = 0.3
    assert design_fingerprint(changed) != baseline


def test_entry_point_metric_requires_an_actual_function_declaration():
    assert declares_entry_point("def solve():\n    return 1", "solve") is True
    assert declares_entry_point("solve = lambda: 1", "solve") is False
    assert declares_entry_point("def other():\n    return 1", "solve") is False
    assert declares_entry_point("def solve(:\n    pass", "solve") is False
    with pytest.raises(ValueError, match="entry_point"):
        declares_entry_point("def solve():\n    pass", None)
    changed = copy.deepcopy(config)
    changed["packet"]["layer_idx"] = -8
    assert design_fingerprint(changed) != baseline


def test_generate_with_packet_replaces_prompt_prefill_once():
    class Layer(torch.nn.Module):
        def forward(self, hidden):
            return hidden

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])

    class Model:
        def __init__(self):
            self.model = Backbone()

        def generate(self, input_ids, attention_mask, **kwargs):
            hidden = input_ids.float().unsqueeze(-1)
            hidden = self.model.layers[0](hidden)
            next_token = hidden[:, -1, 0].long().unsqueeze(1)
            return torch.cat((input_ids, next_token), dim=1)

    class Tokenizer:
        def decode(self, tokens, skip_special_tokens):
            return ",".join(str(value) for value in tokens.tolist())

    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    baseline = generate_with_optional_packet(
        Model(), Tokenizer(), inputs, generation_kwargs={}
    )
    injected = generate_with_optional_packet(
        Model(),
        Tokenizer(),
        inputs,
        generation_kwargs={},
        layer_idx=-1,
        positions=torch.tensor([2]),
        vectors=torch.tensor([[9.0]]),
    )
    assert baseline == "3"
    assert injected == "9"


def test_generation_grid_validation_checks_full_factorial_design():
    config = frozen_config()
    config["data"]["functional_task_count"] = 2
    config["generation"]["seeds"] = [101]
    design = design_fingerprint(config)
    task_ids = ["a", "b"]
    metadata = {
        "protocol_version": ORACLE_FUNCTIONAL_PROTOCOL_VERSION,
        "design_sha256": design,
        "task_ids": task_ids,
        "generation_seeds": [101],
        "run_scope": "full",
    }
    records = [
        {
            "protocol_version": ORACLE_FUNCTIONAL_PROTOCOL_VERSION,
            "design_sha256": design,
            "task_id": task_id,
            "condition": condition,
            "generation_seed": 101,
            "task_spec": {"task_id": task_id, "test_list": ["assert True"]},
        }
        for task_id in task_ids
        for condition in ORACLE_FUNCTIONAL_CONDITIONS
    ]
    result = validate_generation_grid(
        records, metadata, config, allow_incomplete=False
    )
    assert result["complete"] is True
    with pytest.raises(ValueError, match="missing"):
        validate_generation_grid(
            records[:-1], metadata, config, allow_incomplete=False
        )
