import yaml

from src.evaluation.oracle_layer_depth import (
    ORACLE_LAYER_DEPTH_CONDITIONS,
    ORACLE_LAYER_DEPTH_PROTOCOL_VERSION,
    build_condition_plan,
    design_fingerprint,
    primary_fixed_sequence,
    semantic_gate,
    summarize_preflight_authorization,
    validate_layer_depth_contract,
)
from src.scripts.evaluate_oracle_packet_semantics import validate_generation_grid
from src.scripts.run_oracle_memory_functional import validate_config


def frozen_depth_memory_config():
    return {
        "packet_size": 32,
        "decoder_layer_count": 32,
        "self_check_tasks": 1,
        "maximum_self_logit_delta": 0.0001,
        "scopes": [
            {
                "name": "early_quarter_input",
                "boundary": "block_input",
                "layers": list(range(-32, -24)),
            },
            {
                "name": "early_half_input",
                "boundary": "block_input",
                "layers": list(range(-32, -16)),
            },
            {
                "name": "early_three_quarters_input",
                "boundary": "block_input",
                "layers": list(range(-32, -8)),
            },
            {
                "name": "all_layer_input",
                "boundary": "block_input",
                "layers": list(range(-32, 0)),
            },
        ],
    }


def test_depth_contract_freezes_cumulative_early_prefixes():
    scopes = validate_layer_depth_contract(frozen_depth_memory_config())
    assert [len(scope["layers"]) for scope in scopes] == [8, 16, 24, 32]
    assert all(scope["layers"][0] == -32 for scope in scopes)


def test_depth_condition_plan_deranges_every_equal_capacity_control():
    plan = build_condition_plan(
        ["a", "b", "c"],
        ORACLE_LAYER_DEPTH_CONDITIONS,
        shuffle_seed=1729,
    )
    shuffled = [
        item for item in plan if item.condition.startswith("shuffled_oracle_")
    ]
    assert len(shuffled) == 3 * 4
    assert all(item.oracle_index != item.task_index for item in shuffled)


def test_primary_sequence_descends_until_minimum_supported_depth():
    assert [pair[0] for pair in primary_fixed_sequence()] == [
        "oracle_all_layer_input_k32",
        "oracle_early_three_quarters_input_k32",
        "oracle_early_half_input_k32",
        "oracle_early_quarter_input_k32",
    ]


def test_depth_gate_requires_replication_and_selects_smallest_rejected_scope():
    means = {condition: 0.0 for condition in ORACLE_LAYER_DEPTH_CONDITIONS}
    means["text_only_no_lip"] = 0.5
    means["oracle_all_layer_input_k32"] = 0.5
    means["oracle_early_three_quarters_input_k32"] = 0.5
    hypotheses = []
    for index, (treatment, control) in enumerate(primary_fixed_sequence()):
        hypotheses.append(
            {
                "treatment": treatment,
                "control": control,
                "tested": index <= 1,
                "rejected": index <= 1,
            }
        )
    gate = semantic_gate(means, {"hypotheses": hypotheses})
    assert gate["passed"] is True
    assert gate["supported_scopes"] == [
        "early_three_quarters_input",
        "all_layer_input",
    ]
    assert gate["minimum_supported_scope"] == "early_three_quarters_input"


def test_registered_proto009_config_matches_frozen_contract():
    with open(
        "config/LIP-PROTO-009_oracle_layer_depth.yaml",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)
    validate_config(config)


def test_preflight_amendment_authorizes_exact_paired_program_identity():
    task_ids = ["112", "145"]
    seed = 101
    code = {
        "112": "def perimeter(radius, height): return radius",
        "145": "def max_Abs_Diff(arr): return max(arr) - min(arr)",
    }
    rows = []
    for task_id in task_ids:
        source_id = task_ids[1 - task_ids.index(task_id)]
        for condition in ORACLE_LAYER_DEPTH_CONDITIONS:
            oracle_task_id = task_id
            extracted_code = code[task_id]
            entry_point_declared = condition != "neutral_no_lip"
            if condition.startswith("shuffled_oracle_"):
                oracle_task_id = source_id
                extracted_code = code[source_id]
                entry_point_declared = False
            rows.append(
                {
                    "experiment_id": "LIP-PROTO-009",
                    "protocol_version": ORACLE_LAYER_DEPTH_PROTOCOL_VERSION,
                    "run_scope": "preflight",
                    "design_sha256": "a" * 64,
                    "task_id": task_id,
                    "condition": condition,
                    "generation_seed": seed,
                    "oracle_task_id": oracle_task_id,
                    "entry_point_declared": entry_point_declared,
                    "extracted_code": extracted_code,
                    "functional_pass": False,
                }
            )
    metadata = {
        "experiment_id": "LIP-PROTO-009",
        "protocol_version": ORACLE_LAYER_DEPTH_PROTOCOL_VERSION,
        "run_scope": "preflight",
        "complete": True,
        "records": 20,
        "expected_records": 20,
        "task_ids": task_ids,
        "generation_seeds": [seed],
        "design_sha256": "a" * 64,
        "self_checks": [
            {"scope": scope, "maximum_absolute_logit_delta": 0.0}
            for scope in (
                "early_quarter_input",
                "early_half_input",
                "early_three_quarters_input",
                "all_layer_input",
            )
        ],
    }
    result = summarize_preflight_authorization(rows, metadata)
    assert result["passed"] is True
    assert result["claim_eligible"] is False
    assert result["functional_pass_counts"] == {
        "text_only_no_lip": 0,
        "oracle_all_layer_input_k32": 0,
    }


def test_preflight_amendment_rejects_unregistered_shuffled_program():
    task_ids = ["a", "b"]
    seed = 101
    rows = []
    for task_id in task_ids:
        source_id = task_ids[1 - task_ids.index(task_id)]
        for condition in ORACLE_LAYER_DEPTH_CONDITIONS:
            shuffled = condition.startswith("shuffled_oracle_")
            rows.append(
                {
                    "experiment_id": "LIP-PROTO-009",
                    "protocol_version": ORACLE_LAYER_DEPTH_PROTOCOL_VERSION,
                    "run_scope": "preflight",
                    "design_sha256": "b" * 64,
                    "task_id": task_id,
                    "condition": condition,
                    "generation_seed": seed,
                    "oracle_task_id": source_id if shuffled else task_id,
                    "entry_point_declared": condition not in {
                        "neutral_no_lip",
                    }
                    and not shuffled,
                    "extracted_code": "wrong" if shuffled else f"code-{task_id}",
                    "functional_pass": False,
                }
            )
    metadata = {
        "experiment_id": "LIP-PROTO-009",
        "protocol_version": ORACLE_LAYER_DEPTH_PROTOCOL_VERSION,
        "run_scope": "preflight",
        "complete": True,
        "records": 20,
        "expected_records": 20,
        "task_ids": task_ids,
        "generation_seeds": [seed],
        "design_sha256": "b" * 64,
        "self_checks": [
            {"scope": scope, "maximum_absolute_logit_delta": 0.0}
            for scope in (
                "early_quarter_input",
                "early_half_input",
                "early_three_quarters_input",
                "all_layer_input",
            )
        ],
    }
    result = summarize_preflight_authorization(rows, metadata)
    assert result["passed"] is False
    assert result["checks"]["shuffled_programs_equal_registered_source"] is False


def test_layer_depth_generation_grid_uses_proto009_fingerprint():
    with open(
        "config/LIP-PROTO-009_oracle_layer_depth.yaml",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)
    task_ids = [f"task-{index}" for index in range(16)]
    records = [
        {
            "protocol_version": ORACLE_LAYER_DEPTH_PROTOCOL_VERSION,
            "design_sha256": design_fingerprint(config),
            "task_id": task_id,
            "condition": condition,
            "generation_seed": 101,
            "task_spec": {"task_id": task_id, "test_list": ["assert True"]},
        }
        for task_id in task_ids
        for condition in ORACLE_LAYER_DEPTH_CONDITIONS
    ]
    metadata = {
        "protocol_version": ORACLE_LAYER_DEPTH_PROTOCOL_VERSION,
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
