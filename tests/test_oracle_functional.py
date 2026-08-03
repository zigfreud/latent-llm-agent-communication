import copy

import pytest
import torch

from src.evaluation.oracle_functional import (
    ORACLE_CAPACITY_PACKET_SIZES,
    ORACLE_CAPACITY_PROTOCOL_VERSION,
    ORACLE_FUNCTIONAL_CONDITIONS,
    ORACLE_FUNCTIONAL_PROTOCOL_VERSION,
    build_condition_plan,
    declares_entry_point,
    design_fingerprint,
    expected_functional_conditions,
    packet_contract,
    protocol_version_for_config,
    semantic_gate,
)
from src.pipelines.oracle_experiment import load_yaml
from src.pipelines.oracle_transport import generate_with_optional_packet
from src.scripts.evaluate_oracle_packet_semantics import validate_generation_grid
from src.scripts.run_oracle_packet_functional import validate_config


CONFIG_PATH = "config/LIP-PROTO-005_oracle_packet_functional.yaml"
CAPACITY_CONFIG_PATH = (
    "config/LIP-PROTO-007_oracle_packet_functional_capacity.yaml"
)


def frozen_config():
    from pathlib import Path

    return load_yaml(Path(CONFIG_PATH))


def capacity_config():
    from pathlib import Path

    return load_yaml(Path(CAPACITY_CONFIG_PATH))


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


def test_capacity_config_freezes_unused_tasks_and_all_three_packet_sizes():
    config = capacity_config()
    validate_config(config)
    assert protocol_version_for_config(config) == ORACLE_CAPACITY_PROTOCOL_VERSION
    assert packet_contract(config) == (ORACLE_CAPACITY_PACKET_SIZES, None)
    assert tuple(config["conditions"]) == expected_functional_conditions(
        ORACLE_CAPACITY_PACKET_SIZES,
        replication_size=None,
    )

    changed = copy.deepcopy(config)
    changed["data"]["functional_task_start"] = 8
    with pytest.raises(ValueError, match="tasks 16:32"):
        validate_config(changed)
    changed = copy.deepcopy(config)
    changed["packet"]["sizes"] = [8, 16, 48]
    with pytest.raises(ValueError, match=r"sizes=\[8, 16, 32\]"):
        validate_config(changed)


def test_capacity_condition_plan_pairs_each_size_with_one_derangement():
    task_ids = [f"t{index}" for index in range(16)]
    conditions = expected_functional_conditions(
        ORACLE_CAPACITY_PACKET_SIZES,
        replication_size=None,
    )
    plan = build_condition_plan(
        task_ids,
        conditions,
        shuffle_seed=1729,
        packet_sizes=ORACLE_CAPACITY_PACKET_SIZES,
        replication_size=None,
    )
    shuffled_indices = None
    for size in ORACLE_CAPACITY_PACKET_SIZES:
        matched = [row for row in plan if row.condition == f"oracle_packet_k{size}"]
        shuffled = [
            row
            for row in plan
            if row.condition == f"shuffled_oracle_packet_k{size}"
        ]
        assert all(row.packet_size == size for row in matched + shuffled)
        assert all(row.packet_index == row.task_index for row in matched)
        assert all(row.packet_index != row.task_index for row in shuffled)
        current = [row.packet_index for row in shuffled]
        assert sorted(current) == list(range(16))
        if shuffled_indices is None:
            shuffled_indices = current
        else:
            assert current == shuffled_indices


def test_design_fingerprint_changes_with_generation_or_packet_contract():
    config = frozen_config()
    baseline = design_fingerprint(config)
    changed = copy.deepcopy(config)
    changed["generation"]["temperature"] = 0.3
    assert design_fingerprint(changed) != baseline
    changed = copy.deepcopy(config)
    changed["packet"]["layer_idx"] = -8
    assert design_fingerprint(changed) != baseline


def test_entry_point_metric_requires_an_actual_function_declaration():
    assert declares_entry_point("def solve():\n    return 1", "solve") is True
    assert declares_entry_point("solve = lambda: 1", "solve") is False
    assert declares_entry_point("def other():\n    return 1", "solve") is False
    assert declares_entry_point("def solve(:\n    pass", "solve") is False
    with pytest.raises(ValueError, match="entry_point"):
        declares_entry_point("def solve():\n    pass", None)


def test_semantic_gate_requires_text_capacity_and_three_packet_improvements():
    passing = {
        "neutral_no_lip": 0.0,
        "text_only_no_lip": 0.5,
        "oracle_packet_k1": 0.0,
        "oracle_packet_k8": 0.25,
        "shuffled_oracle_packet_k8": 0.0,
    }
    assert semantic_gate(passing)["passed"] is True
    failing = {**passing, "oracle_packet_k8": 0.0}
    gate = semantic_gate(failing)
    assert gate["checks"]["text_control_nonzero"] is True
    assert gate["passed"] is False


def test_capacity_gate_selects_smallest_task_specific_packet():
    means = {
        "neutral_no_lip": 0.0,
        "text_only_no_lip": 0.5,
        "oracle_packet_k8": 0.0,
        "shuffled_oracle_packet_k8": 0.0,
        "oracle_packet_k16": 0.25,
        "shuffled_oracle_packet_k16": 0.0,
        "oracle_packet_k32": 0.5,
        "shuffled_oracle_packet_k32": 0.25,
    }
    gate = semantic_gate(
        means,
        packet_sizes=ORACLE_CAPACITY_PACKET_SIZES,
        replication_size=None,
    )
    assert gate["passed"] is True
    assert gate["supported_capacities"] == [16, 32]
    assert gate["smallest_supported_capacity"] == 16
    assert gate["capacity_checks"]["8"]["passed"] is False


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
