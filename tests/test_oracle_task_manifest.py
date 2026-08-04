import json

import pytest
import yaml

from src.pipelines.oracle_experiment import bind_tasks_to_manifest, load_tasks
from src.scripts.materialize_oracle_tasks import materialize
from src.scripts.materialize_mbpp_prompt_configs import normalize_row


def sampling_config(tmp_path):
    return {
        "experiment_id": "LIP-PROTO-008",
        "dataset_name": "google-research-datasets/mbpp",
        "dataset_config": "full",
        "split": "test",
        "prompt_field": "text",
        "task_count": 4,
        "seed": 808,
        "max_prompt_chars": 512,
        "include_entry_point_in_prompt": True,
        "entry_point_resolution": "tests_then_reference_code",
        "target_model": "target/model",
        "target_model_revision": "a" * 40,
        "prompt_protocol": {
            "version": "lip-prompt-v1",
            "mode": "chat_template",
            "add_generation_prompt": True,
            "system_prompt": "Return Python.",
        },
        "tasks_jsonl": str(tmp_path / "tasks.jsonl"),
        "task_manifest": str(tmp_path / "manifest.json"),
    }


def test_materializer_writes_prompt_bound_task_manifest(tmp_path):
    config = sampling_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = materialize(config_path, mock_data=True)
    assert result["task_count"] == 4
    manifest = json.loads(result["task_manifest"].read_text(encoding="utf-8"))
    assert manifest["manifest_kind"] == "lip_oracle_task_manifest"
    assert manifest["sampled_ids"] == sorted(manifest["sampled_ids"])
    assert len(manifest["sampled_prompt_sha256"]) == 4
    assert len(manifest["sampled_task_sha256"]) == 4
    assert manifest["entry_point_resolution"] == "tests_then_reference_code"


def test_materializer_excludes_every_task_bound_by_prior_manifests(tmp_path):
    config = sampling_config(tmp_path)
    config["experiment_id"] = "LIP-PROTO-009"
    exclusion_path = tmp_path / "prior_manifest.json"
    exclusion_path.write_text(
        json.dumps(
            {
                "manifest_kind": "lip_oracle_task_manifest",
                "schema_version": 1,
                "experiment_id": "LIP-PROTO-008",
                "dataset_name": config["dataset_name"],
                "dataset_config": config["dataset_config"],
                "dataset_split": config["split"],
                "sampled_ids": ["mock-test-000", "mock-test-001"],
                "mock_data": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config["exclude_task_manifests"] = [str(exclusion_path)]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = materialize(config_path, mock_data=True)
    manifest = json.loads(result["task_manifest"].read_text(encoding="utf-8"))
    assert manifest["excluded_task_count"] == 2
    assert manifest["sampled_ids_disjoint_from_exclusions"] is True
    assert not {"mock-test-000", "mock-test-001"}.intersection(
        manifest["sampled_ids"]
    )


def test_builtin_shadow_entry_point_uses_reference_code_without_persisting_it():
    reference_code = "def sum(a, b):\n    return a + b\n"
    normalized = normalize_row(
        {
            "task_id": 126,
            "text": "Write the requested function.",
            "code": reference_code,
            "test_list": ["assert sum(1, 2) == 3", "assert sum(4, 5) == 9"],
        },
        "text",
        512,
        0,
        include_entry_point=True,
    )
    assert normalized["task"]["entry_point"] == "sum"
    assert "Required function name: `sum`" in normalized["prompt"]
    assert "code" not in normalized["task"]
    assert reference_code not in normalized["prompt"]


def test_task_binding_rejects_mock_registry_for_claim_runs(tmp_path):
    config = sampling_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = materialize(config_path, mock_data=True)
    tasks = load_tasks(result["tasks_jsonl"])
    audit_config = {
        "models": {"target_model": "target/model"},
        "prompt_protocol": config["prompt_protocol"],
        "data": {
            "task_manifest": str(result["task_manifest"]),
            "task_count": 4,
        },
    }
    with pytest.raises(ValueError, match="real schema-v1"):
        bind_tasks_to_manifest(audit_config, tasks)


def test_task_binding_covers_full_functional_specification(tmp_path):
    config = sampling_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = materialize(config_path, mock_data=True)
    manifest = json.loads(result["task_manifest"].read_text(encoding="utf-8"))
    manifest["mock_data"] = False
    result["task_manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit_config = {
        "models": {"target_model": "target/model"},
        "prompt_protocol": config["prompt_protocol"],
        "data": {
            "task_manifest": str(result["task_manifest"]),
            "tasks_jsonl": str(result["tasks_jsonl"]),
            "task_count": 4,
        },
    }
    tasks = load_tasks(result["tasks_jsonl"])
    bound, _, _ = bind_tasks_to_manifest(audit_config, tasks)
    assert len(bound) == 4
    tasks[0]["test_list"] = ["assert False"]
    result["tasks_jsonl"].write_text(
        "".join(json.dumps(task, sort_keys=True) + "\n" for task in tasks),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="task file digest"):
        bind_tasks_to_manifest(audit_config, load_tasks(result["tasks_jsonl"]))
