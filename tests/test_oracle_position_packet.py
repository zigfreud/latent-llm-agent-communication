import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.evaluation.oracle_position_packet import (
    ORACLE_POSITION_CONDITIONS,
    ORACLE_POSITION_PATTERN_CONTRACT,
    ORACLE_POSITION_PROTOCOL_VERSION,
    build_condition_plan,
    derive_position_patterns,
    design_fingerprint,
    primary_fixed_sequence,
    select_eligible_holdout,
    semantic_gate,
    validate_selected_task_manifest,
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
from src.scripts.evaluate_oracle_packet_semantics import validate_generation_grid
from src.scripts.plot_oracle_position_packet import (
    pattern_rates,
    position_matrix,
    primary_annotations,
)
from src.scripts.run_oracle_memory_functional import (
    resolve_packet_selection,
    validate_config,
)
from src.scripts.select_oracle_position_tasks import select_tasks


CONFIG_PATH = Path("config/LIP-PROTO-011_position_sparse_packet.yaml")


def load_registered_config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _ids_sha256(task_ids):
    canonical = json.dumps(task_ids, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_registered_proto011_config_matches_frozen_contract():
    config = load_registered_config()
    validate_config(config)
    source = config["calibration_source"]
    assert source["artifact_manifest_sha256"] == (
        "80ea9320defa471a69b891affd8b58f761daffef94d64b95a5abec2419a4016c"
    )
    assert source["eligible_ids_sha256"] == (
        "b6f66368d981fefab33b39ede8f6f98c4bcf8694a84aa0b66d3ffaca680ddf43"
    )


def test_position_patterns_separate_full_ranked_window_and_suffix_packets():
    patterns = {
        item["name"]: item["packet_offsets"]
        for item in ORACLE_POSITION_PATTERN_CONTRACT
    }
    assert patterns["full_k32"] == list(range(-32, 0))
    assert patterns["diagnostic_top_k8"] == [
        -32,
        -30,
        -23,
        -22,
        -21,
        -20,
        -19,
        -18,
    ]
    assert patterns["peak_window_k8"] == list(range(-23, -15))
    assert patterns["suffix_k8"] == list(range(-8, 0))


def test_condition_plan_uses_equal_capacity_task_derangements():
    plan = build_condition_plan(
        ["a", "b", "c"],
        ORACLE_POSITION_CONDITIONS,
        shuffle_seed=2711,
    )
    shuffled = [
        item for item in plan if item.condition.startswith("shuffled_oracle_")
    ]
    assert len(shuffled) == 3 * 4
    assert all(item.oracle_index != item.task_index for item in shuffled)
    assert {len(item.packet_offsets) for item in shuffled} == {8, 32}


def test_primary_sequence_is_full_then_increasingly_conventional_k8_patterns():
    assert [pair[0] for pair in primary_fixed_sequence()] == [
        "oracle_early_quarter_input_full_k32",
        "oracle_early_quarter_input_diagnostic_top_k8",
        "oracle_early_quarter_input_peak_window_k8",
        "oracle_early_quarter_input_suffix_k8",
    ]


def test_semantic_gate_requires_full_replication_and_diagnostic_k8():
    means = {condition: 0.0 for condition in ORACLE_POSITION_CONDITIONS}
    means["text_only_no_lip"] = 0.9
    means["oracle_early_quarter_input_full_k32"] = 0.85
    means["oracle_early_quarter_input_diagnostic_top_k8"] = 0.75
    hypotheses = [
        {
            "treatment": treatment,
            "control": control,
            "tested": index < 2,
            "rejected": index < 2,
        }
        for index, (treatment, control) in enumerate(primary_fixed_sequence())
    ]
    gate = semantic_gate(means, {"hypotheses": hypotheses})
    assert gate["passed"] is True
    assert gate["position_sparse_transport_supported"] is True
    assert gate["smallest_confirmed_packet_size"] == 8
    assert gate["confirmed_patterns"] == ["full_k32", "diagnostic_top_k8"]


def test_registered_holdout_is_eligible_ranks_33_through_64():
    eligible = [f"task-{index}" for index in range(81)]
    selected = select_eligible_holdout(eligible)
    assert selected == [f"task-{index}" for index in range(32, 64)]
    with pytest.raises(ValueError):
        select_eligible_holdout(eligible[:63])


def test_end_relative_packet_offsets_map_to_positions_and_capture_rows():
    import torch

    positions, rows, offsets = resolve_packet_selection(
        prompt_length=50,
        capture_size=32,
        packet_offsets=[-32, -22, -1],
        device=torch.device("cpu"),
    )
    assert positions.tolist() == [18, 28, 49]
    assert rows == [0, 10, 31]
    assert offsets == [-32, -22, -1]


def _diagnostics_with_registered_position_peak():
    means = {offset: 0.1 for offset in range(-32, 0)}
    means.update(
        {
            -32: 0.628,
            -30: 0.614,
            -23: 0.630,
            -22: 0.663,
            -21: 0.614,
            -20: 0.611,
            -19: 0.637,
            -18: 0.616,
            -17: 0.600,
            -16: 0.590,
        }
    )
    return {
        "packet_offsets": list(range(-32, 0)),
        "layer_indices": list(range(-32, 0)),
        "cells": [
            {
                "state_type": "residual_input",
                "layer_index": layer,
                "packet_offset": offset,
                "task_signal_fraction": means[offset],
            }
            for layer in range(-32, 0)
            for offset in range(-32, 0)
        ],
    }


def test_diagnostic_rule_reproduces_frozen_ranked_and_contiguous_patterns():
    assert derive_position_patterns(_diagnostics_with_registered_position_peak()) == {
        "diagnostic_top_k8": [-32, -30, -23, -22, -21, -20, -19, -18],
        "peak_window_k8": list(range(-23, -15)),
    }


def test_generation_grid_uses_011_fingerprint_and_full_task_count():
    config = load_registered_config()
    task_ids = [f"task-{index}" for index in range(32)]
    records = [
        {
            "protocol_version": ORACLE_POSITION_PROTOCOL_VERSION,
            "design_sha256": design_fingerprint(config),
            "task_id": task_id,
            "condition": condition,
            "generation_seed": 743,
            "task_spec": {"task_id": task_id, "test_list": ["assert True"]},
        }
        for task_id in task_ids
        for condition in ORACLE_POSITION_CONDITIONS
    ]
    metadata = {
        "protocol_version": ORACLE_POSITION_PROTOCOL_VERSION,
        "design_sha256": records[0]["design_sha256"],
        "task_ids": task_ids,
        "generation_seeds": [743],
        "run_scope": "full",
    }
    result = validate_generation_grid(
        records,
        metadata,
        config,
        allow_incomplete=False,
    )
    assert result["complete"] is True
    assert result["record_count"] == 320


def test_position_figure_extractors_preserve_pattern_order_and_gate_status():
    conditions = {
        "neutral_no_lip": {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
        "text_only_no_lip": {"mean": 0.9, "ci_lower": 0.8, "ci_upper": 1.0},
    }
    for index, pattern in enumerate(
        ("full_k32", "diagnostic_top_k8", "peak_window_k8", "suffix_k8")
    ):
        mean = 0.8 - index / 10
        conditions[f"oracle_early_quarter_input_{pattern}"] = {
            "mean": mean,
            "ci_lower": mean - 0.1,
            "ci_upper": mean + 0.1,
        }
        conditions[f"shuffled_oracle_early_quarter_input_{pattern}"] = {
            "mean": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }
    summary = {
        "metrics": {"functional_pass": {"conditions": conditions}},
        "primary_inference": {
            "hypotheses": [
                {
                    "treatment": treatment,
                    "tested": index < 2,
                    "p_value": 0.01,
                }
                for index, (treatment, _) in enumerate(primary_fixed_sequence())
            ]
        },
    }
    assert pattern_rates(summary)[0] == pytest.approx([0.8, 0.7, 0.6, 0.5])
    assert primary_annotations(summary) == [
        "p=0.01",
        "p=0.01",
        "gate stopped",
        "gate stopped",
    ]
    matrix = position_matrix()
    assert [sum(row) for row in matrix] == [32, 8, 8, 8]
    assert matrix[-1][-8:] == [1] * 8


def _write_mock_source(config, config_path, tmp_path):
    source_dir = tmp_path / "source-010"
    source_dir.mkdir()
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
    paths = {
        "candidate_tasks_jsonl": source_dir / "candidate_tasks.jsonl",
        "candidate_task_manifest": source_dir / "candidate_manifest.json",
        "screening_scored_jsonl": source_dir / "scored.jsonl",
        "screening_summary": source_dir / "summary.json",
        "selection_report": source_dir / "selection-report.json",
        "state_diagnostics": source_dir / "state-diagnostics.json",
    }
    write_jsonl(paths["candidate_tasks_jsonl"], tasks)
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
        "tasks_jsonl": str(paths["candidate_tasks_jsonl"]),
        "tasks_jsonl_sha256": sha256_path(paths["candidate_tasks_jsonl"]),
        "sampling_config": "mock-source.yaml",
        "sampling_config_sha256": "b" * 64,
        "mock_data": False,
    }
    write_json(paths["candidate_task_manifest"], candidate_manifest)
    eligible_ids = [str(index) for index in range(81)]
    scored = [
        {
            "task_id": str(index),
            "condition": "text_only_no_lip",
            "generation_seed": seed,
            "functional_pass": index < 81 and seed == 17,
        }
        for index in range(192)
        for seed in (17, 29)
    ]
    write_jsonl(paths["screening_scored_jsonl"], scored)
    write_json(paths["screening_summary"], {"complete": True})
    write_json(
        paths["selection_report"],
        {
            "experiment_id": "LIP-PROTO-010",
            "passed": True,
            "eligible_task_count": 81,
            "eligible_task_ids": eligible_ids,
            "eligible_ids_sha256": _ids_sha256(eligible_ids),
            "selected_task_ids": eligible_ids[:32],
        },
    )
    write_json(paths["state_diagnostics"], _diagnostics_with_registered_position_peak())
    source = config["calibration_source"]
    for field, path in paths.items():
        source[field] = str(path)
        source[f"{field}_sha256"] = sha256_path(path)
    source["artifact_manifest_sha256"] = "c" * 64
    source["eligible_ids_sha256"] = _ids_sha256(eligible_ids)
    config["data"].update(
        {
            "tasks_jsonl": str(tmp_path / "selected.jsonl"),
            "task_manifest": str(tmp_path / "selected-manifest.json"),
        }
    )
    config["output"].update(
        {
            "selection_report_json": str(tmp_path / "selection-report-011.json"),
            "generations_jsonl": str(tmp_path / "generations.jsonl"),
            "evaluation_dir": str(tmp_path / "evaluation"),
            "state_diagnostics_json": str(tmp_path / "diagnostics.json"),
        }
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_selection_reuses_screen_but_excludes_predecessor_latent_tasks(tmp_path):
    config = load_registered_config()
    config_path = tmp_path / "config.yaml"
    _write_mock_source(config, config_path, tmp_path)
    report = select_tasks(config, config_path, overwrite=False)
    assert report["selected_task_ids"] == [str(index) for index in range(32, 64)]
    selected = load_tasks(Path(config["data"]["tasks_jsonl"]))
    assert [task["task_id"] for task in selected] == report["selected_task_ids"]
    manifest_path = Path(config["data"]["task_manifest"])
    validate_selected_task_manifest(
        config,
        load_json_object(manifest_path),
        manifest_path,
    )
