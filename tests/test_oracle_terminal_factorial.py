from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
import torch
import yaml

from src.evaluation.oracle_terminal_factorial import (
    ORACLE_TERMINAL_ASSIGNMENTS,
    ORACLE_TERMINAL_CONDITIONS,
    ORACLE_TERMINAL_PATTERN_CONTRACT,
    build_condition_plan,
    eligible_task_ids,
    primary_family,
    primary_gates,
    semantic_gate,
    terminal_components,
    terminal_patterns,
    validate_terminal_layout,
)
from src.scripts.evaluate_oracle_packet_semantics import (
    evaluation_contract,
    validate_generation_grid,
)
from src.scripts.materialize_oracle_terminal_candidates import (
    classify_terminal_layout,
)
from src.scripts.plot_oracle_terminal_factorial import (
    factorial_rates,
    primary_contrasts,
)
from src.scripts.run_oracle_memory_functional import (
    assemble_component_packets,
    expected_terminal_factorial_comparisons,
    experiment_contract,
    validate_config,
)


CONFIG_PATH = Path("config/LIP-PROTO-013_terminal_source_factorial.yaml")


def load_registered_config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def layout(name_token_count, task_id):
    components = terminal_components(name_token_count)
    return {
        "name_token_count": name_token_count,
        "core_offsets": list(components["core"]),
        "name_offsets": list(components["name"]),
        "boundary_offsets": list(components["boundary"]),
        "tail_offsets": list(range(-24, 0)),
        "selection_hash": hashlib.sha256(str(task_id).encode()).hexdigest(),
    }


def selected_tasks():
    return [
        {
            "task_id": f"task-{index:02d}",
            "terminal_layout": layout(2 if index < 16 else 3, index),
        }
        for index in range(32)
    ]


class TwoTokenNameTokenizer:
    def __init__(self, entry_point):
        self.entry_point = entry_point

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "x" * 40 + messages[-1]["content"] + "ABCDEF"

    def __call__(self, formatted, *, add_special_tokens, return_offsets_mapping):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        start = formatted.rfind(self.entry_point)
        midpoint = start + max(1, len(self.entry_point) // 2)
        offsets = [(index, index + 1) for index in range(start)]
        offsets.extend(
            [
                (start, midpoint),
                (midpoint, start + len(self.entry_point)),
            ]
        )
        suffix_start = start + len(self.entry_point)
        offsets.extend(
            (index, index + 1) for index in range(suffix_start, len(formatted))
        )
        return {
            "input_ids": list(range(len(offsets))),
            "offset_mapping": offsets,
        }


def test_registered_config_dispatches_the_frozen_factorial():
    config = load_registered_config()
    validate_config(config)
    contract = experiment_contract(config)
    assert contract["conditions"] == ORACLE_TERMINAL_CONDITIONS
    assert contract["plan_uses_tasks"] is True
    assert contract["position_patterns"] == terminal_patterns(config["memory"])
    protocol, fingerprint, gate = evaluation_contract(config)
    assert protocol == "lip-oracle-terminal-source-factorial-v1"
    assert len(fingerprint) == 64
    assert gate is semantic_gate
    assert config["evaluation"]["comparisons"] == (
        expected_terminal_factorial_comparisons()
    )


def test_terminal_components_are_capacity_preserving_stratified_partitions():
    for count, expected_sizes in ((2, (16, 2, 6)), (3, (15, 3, 6))):
        components = terminal_components(count)
        assert tuple(len(components[name]) for name in ("core", "name", "boundary")) == (
            expected_sizes
        )
        assert tuple(
            offset
            for name in ("core", "name", "boundary")
            for offset in components[name]
        ) == tuple(range(-24, 0))
        assert validate_terminal_layout(layout(count, count)) == count
    with pytest.raises(ValueError, match="2 or 3"):
        terminal_components(4)
    changed = deepcopy(load_registered_config()["memory"])
    changed["position_patterns"][1]["packet_offsets"][0] = -25
    with pytest.raises(ValueError, match="terminal factorial"):
        terminal_patterns(changed)
    assert tuple(item["name"] for item in ORACLE_TERMINAL_PATTERN_CONTRACT) == (
        "full_k32",
        "terminal_k24",
    )


def test_materializer_uses_render_then_tokenize_and_locates_the_final_name_span():
    entry_point = "sort_matrix"
    task = {
        "task_id": "example",
        "prompt": f"Implement the required function {entry_point}",
        "entry_point": entry_point,
    }
    protocol = load_registered_config()["prompt_protocol"]
    layout_result = classify_terminal_layout(
        task,
        TwoTokenNameTokenizer(entry_point),
        protocol,
        selection_salt="test",
    )
    assert layout_result["name_token_count"] == 2
    assert layout_result["name_offsets"] == [-8, -7]
    assert layout_result["boundary_offsets"] == [-6, -5, -4, -3, -2, -1]


def test_condition_plan_uses_one_same_stratum_donor_for_every_s_component():
    tasks = selected_tasks()
    plan = build_condition_plan(tasks, ORACLE_TERMINAL_CONDITIONS, shuffle_seed=3137)
    assert len(plan) == 32 * 12
    by_key = {(item.task_id, item.condition): item for item in plan}
    for task_index, task in enumerate(tasks):
        full = by_key[(task["task_id"], "shuffled_oracle_early_quarter_input_full_k32")]
        donor_index = full.oracle_index
        assert donor_index != task_index
        assert validate_terminal_layout(
            tasks[donor_index]["terminal_layout"]
        ) == validate_terminal_layout(task["terminal_layout"])
        for assignment in ORACLE_TERMINAL_ASSIGNMENTS:
            item = by_key[
                (
                    task["task_id"],
                    f"oracle_early_quarter_input_terminal_k24_{assignment}",
                )
            ]
            assert item.packet_offsets == tuple(range(-24, 0))
            assert sum(len(offsets) for offsets in item.component_offsets) == 24
            for source, source_index in zip(
                assignment, item.component_oracle_indices
            ):
                assert source_index == (task_index if source == "m" else donor_index)


def test_component_packet_assembly_preserves_row_order_and_scalar_capacity():
    memories = []
    for source_index in range(2):
        values = torch.arange(32, dtype=torch.float32).unsqueeze(1)
        values = values + 100 * source_index
        memories.append({"residual_input": {0: values}})
    components = terminal_components(2)
    packet, sources = assemble_component_packets(
        memories,
        layer_indices=[0],
        component_oracle_indices=(0, 1, 0),
        component_offsets=tuple(
            components[name] for name in ("core", "name", "boundary")
        ),
        capture_size=32,
    )
    assert packet[0].shape == (24, 1)
    assert packet[0][:16, 0].tolist() == list(range(8, 24))
    assert packet[0][16:18, 0].tolist() == [124.0, 125.0]
    assert packet[0][18:, 0].tolist() == list(range(26, 32))
    assert sources == [0] * 16 + [1] * 2 + [0] * 6


def test_screening_eligibility_is_hash_ordered_inside_each_stratum():
    candidates = []
    for index in range(179):
        count = 2 if index < 83 else 3
        candidates.append(
            {
                "task_id": f"task-{index:03d}",
                "terminal_layout": layout(count, 999 - index),
            }
        )
    records = [
        {
            "task_id": task["task_id"],
            "condition": "text_only_no_lip",
            "generation_seed": seed,
            "functional_pass": seed == 1423,
        }
        for task in candidates
        for seed in (1423, 1559)
    ]
    eligible = eligible_task_ids(records, list(reversed(candidates)))
    assert len(eligible[2]) == 83
    assert len(eligible[3]) == 96
    by_id = {task["task_id"]: task for task in candidates}
    for count in (2, 3):
        hashes = [
            by_id[task_id]["terminal_layout"]["selection_hash"]
            for task_id in eligible[count]
        ]
        assert hashes == sorted(hashes)


def test_evaluator_recognizes_the_179_task_terminal_screening_grid():
    config = load_registered_config()
    protocol, fingerprint, _ = evaluation_contract(config)
    task_ids = [f"task-{index:03d}" for index in range(179)]
    metadata = {
        "protocol_version": protocol,
        "design_sha256": fingerprint,
        "run_scope": "terminal_layout_capability_screening",
        "task_ids": task_ids,
        "generation_seeds": [1423, 1559],
    }
    records = [
        {
            "protocol_version": protocol,
            "design_sha256": fingerprint,
            "task_id": task_id,
            "condition": "text_only_no_lip",
            "generation_seed": seed,
            "task_spec": {"task_id": task_id},
        }
        for task_id in task_ids
        for seed in (1423, 1559)
    ]
    validation = validate_generation_grid(
        records,
        metadata,
        config,
        allow_incomplete=False,
    )
    assert validation["screening"] is True
    assert validation["complete"] is True
    assert validation["expected_record_count"] == 358


def test_semantic_gate_requires_both_replications_before_component_claims():
    means = {condition: 0.0 for condition in ORACLE_TERMINAL_CONDITIONS}
    means["text_only_no_lip"] = 0.9
    for treatment, control in (*primary_gates(), *primary_family()):
        means[treatment] = max(means[treatment], 0.8)
        means[control] = min(means[control], 0.1)
    inference = {
        "method": "two_gate_then_holm",
        "gates": [
            {
                "treatment": treatment,
                "control": control,
                "tested": True,
                "rejected": True,
            }
            for treatment, control in primary_gates()
        ],
        "family": [
            {
                "treatment": treatment,
                "control": control,
                "tested": True,
                "rejected": index == 2,
            }
            for index, (treatment, control) in enumerate(primary_family())
        ],
    }
    gate = semantic_gate(means, inference)
    assert gate["passed"] is True
    assert gate["supported_component_claims"] == ["boundary_contribution"]
    inference["gates"][1]["rejected"] = False
    assert semantic_gate(means, inference)["passed"] is False


def test_figure_extractors_preserve_factorial_order_and_global_holm_family():
    conditions = {}
    for index, assignment in enumerate(ORACLE_TERMINAL_ASSIGNMENTS):
        value = 0.8 - 0.05 * index
        conditions[
            f"oracle_early_quarter_input_terminal_k24_{assignment}"
        ] = {
            "mean": value,
            "ci_lower": value - 0.02,
            "ci_upper": value + 0.03,
        }
    summary = {
        "metrics": {"functional_pass": {"conditions": conditions}},
        "primary_inference": {
            "family": [
                {
                    "mean_difference": 0.1,
                    "ci_lower": 0.02,
                    "ci_upper": 0.18,
                    "tested": True,
                    "rejected": index == 0,
                    "p_value_holm": 0.01 + index * 0.01,
                }
                for index in range(7)
            ]
        },
    }
    means, lower, upper = factorial_rates(summary)
    assert means == pytest.approx([0.8 - 0.05 * index for index in range(8)])
    assert lower == pytest.approx([0.02] * 8)
    assert upper == pytest.approx([0.03] * 8)
    contrasts = primary_contrasts(summary)
    assert len(contrasts) == 7
    assert contrasts[0]["label"] == "core_contribution"
    assert contrasts[0]["rejected"] is True
