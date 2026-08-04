from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.evaluation.oracle_block_deletion import (
    ORACLE_DELETION_CONDITIONS,
    ORACLE_DELETION_K24_PATTERN_ORDER,
    ORACLE_DELETION_OCTETS,
    ORACLE_DELETION_PATTERN_CONTRACT,
    build_condition_plan,
    deletion_patterns,
    primary_anchor,
    primary_family,
    select_eligible_holdout,
    semantic_gate,
    validate_selected_task_manifest,
    validate_deletion_design,
    validate_deletion_memory_contract,
)
from src.pipelines.oracle_experiment import (
    load_json_object,
    load_tasks,
    prompt_sha256,
    sha256_path,
    task_sha256,
    write_json,
    write_jsonl,
)
from src.scripts.evaluate_oracle_packet_semantics import evaluation_contract
from src.scripts.plot_oracle_block_deletion import (
    pattern_rates,
    position_matrix,
    primary_annotations,
)
from src.scripts.run_oracle_memory_functional import (
    expected_block_deletion_comparisons,
    experiment_contract,
    validate_config,
)
from src.scripts.select_oracle_block_deletion_tasks import select_tasks


CONFIG_PATH = Path("config/LIP-PROTO-012_block_deletion.yaml")


def load_registered_config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_registered_config_matches_frozen_contract_and_dispatch():
    config = load_registered_config()
    validate_config(config)
    runner = experiment_contract(config)
    assert runner["conditions"] == ORACLE_DELETION_CONDITIONS
    assert runner["position_patterns"] == deletion_patterns(config["memory"])
    protocol_version, design_sha256, gate = evaluation_contract(config)
    assert protocol_version == "lip-oracle-block-deletion-v1"
    assert len(design_sha256) == 64
    assert gate is semantic_gate
    assert config["evaluation"]["comparisons"] == (
        expected_block_deletion_comparisons()
    )


def frozen_memory():
    return {
        "packet_size": 32,
        "decoder_layer_count": 32,
        "self_check_tasks": 1,
        "maximum_self_logit_delta": 0.0001,
        "state_capture_layers": list(range(-32, 0)),
        "replay_scope": {
            "name": "early_quarter_input",
            "boundary": "block_input",
            "layers": list(range(-32, -24)),
        },
        "position_patterns": deepcopy(list(ORACLE_DELETION_PATTERN_CONTRACT)),
    }


def test_deletion_patterns_are_exhaustive_equal_capacity_complements():
    patterns = deletion_patterns(frozen_memory())
    assert patterns["full_k32"] == tuple(range(-32, 0))
    assert len(ORACLE_DELETION_OCTETS) == 4
    for index, octet in enumerate(ORACLE_DELETION_OCTETS, start=1):
        kept = patterns[f"drop_octet_{index}_k24"]
        assert len(kept) == 24
        assert set(kept).isdisjoint(octet)
        assert set(kept).union(octet) == set(range(-32, 0))
    with pytest.raises(ValueError, match="frozen deletion contract"):
        changed = frozen_memory()
        changed["position_patterns"][1]["packet_offsets"][0] = -32
        deletion_patterns(changed)


def test_memory_and_deletion_design_freeze_depth_and_partition():
    assert validate_deletion_memory_contract(frozen_memory()) == (
        {
            "name": "early_quarter_input",
            "boundary": "block_input",
            "layers": list(range(-32, -24)),
        },
    )
    design = {
        "method": "leave_one_contiguous_octet_out",
        "partition": [list(octet) for octet in ORACLE_DELETION_OCTETS],
        "kept_packet_size": 24,
        "interpretation_unit": "deleted_octet",
    }
    validate_deletion_design(design)
    design["kept_packet_size"] = 23
    with pytest.raises(ValueError, match="exhaustive octet partition"):
        validate_deletion_design(design)


def test_final_eligible_slice_exhausts_the_sealed_registry():
    eligible = [f"task-{index}" for index in range(81)]
    assert select_eligible_holdout(eligible) == eligible[64:81]
    with pytest.raises(ValueError, match="exactly 81 unique"):
        select_eligible_holdout(eligible[:-1])


def test_condition_plan_pairs_every_pattern_with_one_deranged_control():
    task_ids = [f"task-{index}" for index in range(17)]
    plan = build_condition_plan(
        task_ids,
        ORACLE_DELETION_CONDITIONS,
        shuffle_seed=2711,
    )
    assert len(plan) == 17 * 12
    for item in plan:
        if item.condition.startswith("oracle_"):
            assert item.oracle_index == item.task_index
            assert item.position_pattern is not None
        elif item.condition.startswith("shuffled_oracle_"):
            assert item.oracle_index != item.task_index
            assert item.position_pattern is not None
        else:
            assert item.oracle_index is None
            assert item.packet_offsets is None
    assert {
        item.position_pattern for item in plan if item.position_pattern is not None
    } == {pattern["name"] for pattern in ORACLE_DELETION_PATTERN_CONTRACT}


def primary_result(*, anchor_rejected=True):
    anchor_treatment, anchor_control = primary_anchor()
    return {
        "method": "anchor_gate_then_holm",
        "anchor": {
            "treatment": anchor_treatment,
            "control": anchor_control,
            "tested": True,
            "rejected": anchor_rejected,
        },
        "family": [
            {
                "treatment": treatment,
                "control": control,
                "tested": anchor_rejected,
                "rejected": False,
            }
            for treatment, control in primary_family()
        ],
    }


def test_semantic_gate_distinguishes_any_from_all_dispensable_octets():
    means = {condition: 0.0 for condition in ORACLE_DELETION_CONDITIONS}
    means["text_only_no_lip"] = 0.9
    means["oracle_early_quarter_input_full_k32"] = 0.85
    inference = primary_result()
    first = inference["family"][0]
    first["rejected"] = True
    means[first["treatment"]] = 0.8
    gate = semantic_gate(means, inference)
    assert gate["passed"] is True
    assert gate["block_deletion_transport_supported"] is True
    assert gate["all_octet_deletions_supported"] is False
    assert gate["dispensable_octets"] == ["octet_1"]
    assert gate["smallest_confirmed_packet_size"] == 24

    for item in inference["family"]:
        item["rejected"] = True
        means[item["treatment"]] = 0.8
    gate = semantic_gate(means, inference)
    assert gate["all_octet_deletions_supported"] is True
    assert gate["dispensable_octets"] == [
        f"octet_{index}" for index in range(1, 5)
    ]
    assert len(ORACLE_DELETION_K24_PATTERN_ORDER) == 4


def test_figure_extractors_preserve_masks_and_gatekept_holm_annotations():
    conditions = {}
    for index, pattern in enumerate(
        [item["name"] for item in ORACLE_DELETION_PATTERN_CONTRACT]
    ):
        mean = 0.8 - 0.1 * index
        conditions[f"oracle_early_quarter_input_{pattern}"] = {
            "mean": mean,
            "ci_lower": mean - 0.05,
            "ci_upper": mean + 0.05,
        }
        conditions[f"shuffled_oracle_early_quarter_input_{pattern}"] = {
            "mean": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }
    anchor_treatment, anchor_control = primary_anchor()
    summary = {
        "metrics": {"functional_pass": {"conditions": conditions}},
        "primary_inference": {
            "anchor": {
                "treatment": anchor_treatment,
                "control": anchor_control,
                "tested": True,
                "p_value": 0.001,
            },
            "family": [
                {
                    "treatment": treatment,
                    "control": control,
                    "tested": index < 2,
                    "p_value": 0.01,
                    "p_value_holm": 0.04,
                }
                for index, (treatment, control) in enumerate(primary_family())
            ],
        },
    }
    assert pattern_rates(summary)[0] == pytest.approx([0.8, 0.7, 0.6, 0.5, 0.4])
    assert pattern_rates(summary, shuffled=True)[0] == [0.0] * 5
    assert primary_annotations(summary) == [
        "p=0.001",
        "Holm p=0.04",
        "Holm p=0.04",
        "anchor failed",
        "anchor failed",
    ]
    matrix = position_matrix()
    assert [sum(row) for row in matrix] == [32, 24, 24, 24, 24]


def _ids_sha256(task_ids):
    canonical = json.dumps(task_ids, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_mock_sources(config, config_path, tmp_path):
    calibration_dir = tmp_path / "source-010"
    predecessor_dir = tmp_path / "source-011"
    calibration_dir.mkdir()
    predecessor_dir.mkdir()
    tasks = [
        {
            "task_id": str(index),
            "prompt": f"Implement function_{index}.",
            "entry_point": f"function_{index}",
            "test_list": [f"assert function_{index}() == {index}"],
            "code": f"def function_{index}(): return {index}",
        }
        for index in range(192)
    ]
    calibration_paths = {
        "candidate_tasks_jsonl": calibration_dir / "candidate_tasks.jsonl",
        "candidate_task_manifest": calibration_dir / "candidate_manifest.json",
        "screening_scored_jsonl": calibration_dir / "scored.jsonl",
        "screening_summary": calibration_dir / "summary.json",
        "selection_report": calibration_dir / "selection-report.json",
    }
    write_jsonl(calibration_paths["candidate_tasks_jsonl"], tasks)
    candidate_manifest = {
        "manifest_kind": "lip_oracle_task_manifest",
        "schema_version": 1,
        "experiment_id": "LIP-PROTO-010",
        "dataset_name": "google-research-datasets/mbpp",
        "dataset_config": "full",
        "dataset_split": "test",
        "prompt_field": "text",
        "task_count": 192,
        "sampled_ids": [task["task_id"] for task in tasks],
        "sampled_prompt_sha256": [prompt_sha256(task["prompt"]) for task in tasks],
        "sampled_task_sha256": [task_sha256(task) for task in tasks],
        "excluded_task_manifests": [],
        "excluded_task_count": 50,
        "excluded_task_ids_sha256": "a" * 64,
        "sampled_ids_disjoint_from_exclusions": True,
        "include_entry_point_in_prompt": True,
        "entry_point_resolution": "tests_then_reference_code",
        "target_model": config["models"]["target_model"],
        "target_model_revision": "5" * 40,
        "prompt_protocol": config["prompt_protocol"],
        "tasks_jsonl": str(calibration_paths["candidate_tasks_jsonl"]),
        "tasks_jsonl_sha256": sha256_path(
            calibration_paths["candidate_tasks_jsonl"]
        ),
        "sampling_config": "mock-source.yaml",
        "sampling_config_sha256": "b" * 64,
        "mock_data": False,
    }
    write_json(calibration_paths["candidate_task_manifest"], candidate_manifest)
    eligible_ids = [str(index) for index in range(81)]
    write_jsonl(
        calibration_paths["screening_scored_jsonl"],
        [
            {
                "task_id": str(index),
                "condition": "text_only_no_lip",
                "generation_seed": seed,
                "functional_pass": index < 81 and seed == 17,
            }
            for index in range(192)
            for seed in (17, 29)
        ],
    )
    write_json(calibration_paths["screening_summary"], {"complete": True})
    write_json(
        calibration_paths["selection_report"],
        {
            "experiment_id": "LIP-PROTO-010",
            "passed": True,
            "eligible_task_count": 81,
            "eligible_task_ids": eligible_ids,
            "eligible_ids_sha256": _ids_sha256(eligible_ids),
            "selected_task_ids": eligible_ids[:32],
        },
    )

    predecessor_tasks = tasks[32:64]
    predecessor_paths = {
        "selected_tasks_jsonl": predecessor_dir / "selected.jsonl",
        "selected_task_manifest": predecessor_dir / "selected-manifest.json",
        "selection_report": predecessor_dir / "selection-report.json",
        "functional_summary": predecessor_dir / "summary.json",
    }
    write_jsonl(predecessor_paths["selected_tasks_jsonl"], predecessor_tasks)
    write_json(
        predecessor_paths["selected_task_manifest"],
        {
            "experiment_id": "LIP-PROTO-011",
            "sampled_ids": eligible_ids[32:64],
        },
    )
    write_json(
        predecessor_paths["selection_report"],
        {
            "experiment_id": "LIP-PROTO-011",
            "passed": True,
            "predecessor_selected_task_ids": eligible_ids[:32],
            "selected_task_ids": eligible_ids[32:64],
        },
    )
    write_json(
        predecessor_paths["functional_summary"],
        {
            "experiment_id": "LIP-PROTO-011",
            "claim_eligible": True,
            "semantic_gate": {
                "checks": {"full_k32_replication_confirmed": True}
            },
        },
    )

    for field, path in calibration_paths.items():
        config["calibration_source"][field] = str(path)
        config["calibration_source"][f"{field}_sha256"] = sha256_path(path)
    config["calibration_source"]["artifact_manifest_sha256"] = "c" * 64
    config["calibration_source"]["eligible_ids_sha256"] = _ids_sha256(
        eligible_ids
    )
    for field, path in predecessor_paths.items():
        config["predecessor_source"][field] = str(path)
        config["predecessor_source"][f"{field}_sha256"] = sha256_path(path)
    config["predecessor_source"]["artifact_manifest_sha256"] = "d" * 64
    config["data"].update(
        {
            "tasks_jsonl": str(tmp_path / "selected-012.jsonl"),
            "task_manifest": str(tmp_path / "selected-manifest-012.json"),
        }
    )
    config["output"].update(
        {
            "selection_report_json": str(tmp_path / "selection-report-012.json"),
            "generations_jsonl": str(tmp_path / "generations.jsonl"),
            "evaluation_dir": str(tmp_path / "evaluation"),
            "state_diagnostics_json": str(tmp_path / "diagnostics.json"),
        }
    )
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )


def test_selection_exhausts_final_eligible_slice_and_binds_both_sources(tmp_path):
    config = load_registered_config()
    config_path = tmp_path / "config.yaml"
    _write_mock_sources(config, config_path, tmp_path)
    report = select_tasks(config, config_path, overwrite=False)
    assert report["selected_task_ids"] == [str(index) for index in range(64, 81)]
    assert len(report["prior_selected_task_ids"]) == 64
    selected = load_tasks(Path(config["data"]["tasks_jsonl"]))
    assert [task["task_id"] for task in selected] == report["selected_task_ids"]
    manifest_path = Path(config["data"]["task_manifest"])
    validate_selected_task_manifest(
        config,
        load_json_object(manifest_path),
        manifest_path,
    )
