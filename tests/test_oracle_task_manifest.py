import json

import pytest
import yaml

from src.pipelines.oracle_experiment import bind_tasks_to_manifest, load_tasks
from src.scripts.materialize_oracle_tasks import materialize


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
