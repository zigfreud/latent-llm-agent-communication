import json
from pathlib import Path

import yaml

from src.evaluation.oracle_capability_calibration import (
    ORACLE_CAPABILITY_CONDITIONS,
    ORACLE_CAPABILITY_PROTOCOL_VERSION,
    ORACLE_CAPABILITY_SCREENING_SEEDS,
    design_fingerprint,
    eligible_task_ids,
    primary_fixed_sequence,
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
from src.scripts.run_oracle_memory_functional import validate_config
from src.scripts.select_oracle_capability_tasks import select_tasks


CONFIG_PATH = Path("config/LIP-PROTO-010_capability_calibrated_depth.yaml")


def load_registered_config():
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_registered_proto010_config_matches_frozen_contract():
    validate_config(load_registered_config())


def test_primary_sequence_starts_with_the_prospective_24_layer_hypothesis():
    assert [pair[0] for pair in primary_fixed_sequence()] == [
        "oracle_early_three_quarters_input_k32",
        "oracle_early_half_input_k32",
        "oracle_early_quarter_input_k32",
    ]


def test_eligibility_uses_any_screening_pass_and_preserves_candidate_order():
    candidate_ids = [f"task-{index}" for index in range(192)]
    passing = {"task-1", "task-7", "task-40"}
    records = [
        {
            "task_id": task_id,
            "condition": "text_only_no_lip",
            "generation_seed": seed,
            "functional_pass": task_id in passing and seed == 29,
        }
        for task_id in reversed(candidate_ids)
        for seed in ORACLE_CAPABILITY_SCREENING_SEEDS
    ]
    assert eligible_task_ids(records, candidate_ids) == [
        "task-1",
        "task-7",
        "task-40",
    ]


def test_capability_gate_does_not_make_all_layer_a_primary_family_member():
    means = {condition: 0.0 for condition in ORACLE_CAPABILITY_CONDITIONS}
    means["text_only_no_lip"] = 0.5
    means["oracle_early_three_quarters_input_k32"] = 0.5
    means["oracle_all_layer_input_k32"] = 0.4
    hypotheses = [
        {
            "treatment": treatment,
            "control": control,
            "tested": index == 0,
            "rejected": index == 0,
        }
        for index, (treatment, control) in enumerate(primary_fixed_sequence())
    ]
    gate = semantic_gate(means, {"hypotheses": hypotheses})
    assert gate["passed"] is True
    assert gate["supported_scopes"] == ["early_three_quarters_input"]
    assert gate["all_layer_descriptive_anchor"]["confirmatory_family_member"] is False


def test_screening_grid_uses_candidates_without_becoming_claim_eligible():
    config = load_registered_config()
    task_ids = [f"task-{index}" for index in range(192)]
    records = [
        {
            "protocol_version": ORACLE_CAPABILITY_PROTOCOL_VERSION,
            "design_sha256": design_fingerprint(config),
            "task_id": task_id,
            "condition": "text_only_no_lip",
            "generation_seed": seed,
            "task_spec": {"task_id": task_id, "test_list": ["assert True"]},
        }
        for task_id in task_ids
        for seed in ORACLE_CAPABILITY_SCREENING_SEEDS
    ]
    metadata = {
        "protocol_version": ORACLE_CAPABILITY_PROTOCOL_VERSION,
        "design_sha256": design_fingerprint(config),
        "task_ids": task_ids,
        "generation_seeds": list(ORACLE_CAPABILITY_SCREENING_SEEDS),
        "run_scope": "capability_screening",
    }
    result = validate_generation_grid(
        records,
        metadata,
        config,
        allow_incomplete=False,
    )
    assert result["complete"] is True
    assert result["screening"] is True


def _write_candidate_registry(config, config_path):
    tasks_path = Path(config["data"]["candidate_tasks_jsonl"])
    manifest_path = Path(config["data"]["candidate_task_manifest"])
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
    tasks_path.parent.mkdir(parents=True)
    write_jsonl(tasks_path, tasks)
    manifest = {
        "manifest_kind": "lip_oracle_task_manifest",
        "schema_version": 1,
        "experiment_id": "LIP-PROTO-010",
        "dataset_name": "google-research-datasets/mbpp",
        "dataset_config": "full",
        "dataset_split": "test",
        "prompt_field": "text",
        "sampling_seed": 1010,
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
        "tasks_jsonl": str(tasks_path),
        "tasks_jsonl_sha256": sha256_path(tasks_path),
        "sampling_config": str(config_path),
        "sampling_config_sha256": sha256_path(config_path),
        "mock_data": False,
    }
    write_json(manifest_path, manifest)
    return tasks


def test_selection_is_a_hardened_deterministic_eligible_prefix(tmp_path):
    config = load_registered_config()
    data_dir = tmp_path / "datasets"
    run_dir = tmp_path / "runs"
    config["data"].update(
        {
            "candidate_tasks_jsonl": str(data_dir / "candidates.jsonl"),
            "candidate_task_manifest": str(data_dir / "candidate-manifest.json"),
            "tasks_jsonl": str(data_dir / "selected.jsonl"),
            "task_manifest": str(data_dir / "selected-manifest.json"),
        }
    )
    config["output"].update(
        {
            "screening_generations_jsonl": str(run_dir / "generations.jsonl"),
            "screening_evaluation_dir": str(run_dir / "functional-evaluation"),
            "selection_report_json": str(run_dir / "selection-report.json"),
            "generations_jsonl": str(run_dir / "confirmation.jsonl"),
            "evaluation_dir": str(run_dir / "evaluation"),
            "state_diagnostics_json": str(run_dir / "diagnostics.json"),
        }
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    tasks = _write_candidate_registry(config, config_path)

    generations_path = Path(config["output"]["screening_generations_jsonl"])
    generations_path.parent.mkdir(parents=True)
    generations_path.write_text("screening generations\n", encoding="utf-8")
    metadata_path = generations_path.with_suffix(".metadata.json")
    metadata_path.write_text("{}\n", encoding="utf-8")
    evaluation_dir = Path(config["output"]["screening_evaluation_dir"])
    evaluation_dir.mkdir(parents=True)
    scored = [
        {
            "task_id": task["task_id"],
            "condition": "text_only_no_lip",
            "generation_seed": seed,
            "functional_pass": int(task["task_id"]) < 40 and seed == 17,
        }
        for task in tasks
        for seed in ORACLE_CAPABILITY_SCREENING_SEEDS
    ]
    scored_path = evaluation_dir / "scored_generations.jsonl"
    write_jsonl(scored_path, scored)
    summary = {
        "experiment_id": "LIP-PROTO-010",
        "protocol_version": ORACLE_CAPABILITY_PROTOCOL_VERSION,
        "execution_mode": "functional_hardened_namespace",
        "claim_eligible": False,
        "scored_jsonl_sha256": sha256_path(scored_path),
        "design_validation": {
            "design_sha256": design_fingerprint(config),
            "run_scope": "capability_screening",
            "screening": True,
            "complete": True,
        },
        "sandbox": {
            "validated": True,
            "input_sha256": {
                "config": sha256_path(config_path),
                "generations": sha256_path(generations_path),
                "metadata": sha256_path(metadata_path),
            },
        },
    }
    write_json(evaluation_dir / "summary.json", summary)

    report = select_tasks(config, config_path, overwrite=False)
    assert report["eligible_task_count"] == 40
    assert report["selected_task_ids"] == [str(index) for index in range(32)]
    assert [task["task_id"] for task in load_tasks(Path(config["data"]["tasks_jsonl"]))] == [
        str(index) for index in range(32)
    ]
    manifest_path = Path(config["data"]["task_manifest"])
    validate_selected_task_manifest(
        config,
        load_json_object(manifest_path),
        manifest_path,
    )
