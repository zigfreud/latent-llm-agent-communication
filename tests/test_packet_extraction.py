import json

import pytest
import yaml

from src.pipelines.packet_extraction import (
    classify_target_terminal_layout,
    materialize_packet_bundle,
)
from src.scripts.materialize_packet_bridge_tasks import (
    materialize_packet_bridge_tasks,
)


def _config(tmp_path):
    dataset_dir = tmp_path / "datasets"
    bundle_dir = tmp_path / "bundle"
    return {
        "experiment_id": "LIP-PROTO-014",
        "predecessor": {
            "protocol": "LIP-PROTO-013",
            "sha256sums_sha256": "a" * 64,
        },
        "models": {
            "source": {"model_id": "source/test", "revision": "source-revision"},
            "target": {"model_id": "target/test", "revision": "target-revision"},
        },
        "prompt_protocols": {
            "source": {
                "version": "lip-prompt-v1",
                "mode": "raw",
                "add_generation_prompt": False,
                "system_prompt": None,
            },
            "target": {
                "version": "lip-prompt-v1",
                "mode": "raw",
                "add_generation_prompt": False,
                "system_prompt": None,
            },
        },
        "data": {
            "dataset": {
                "dataset_id": "dataset/test",
                "dataset_config": "full",
                "revision": "dataset-revision",
                "prompt_field": "text",
                "train_split": "train",
                "development_split": "validation",
            },
            "selection": {
                "salt": "packet-test-v1",
                "max_prompt_chars": 512,
                "train_count": 4,
                "development_selection_count": 2,
                "development_gate_count": 2,
            },
            "registries": {
                "train": str(dataset_dir / "train.jsonl"),
                "development_selection": str(dataset_dir / "selection.jsonl"),
                "development_gate": str(dataset_dir / "gate.jsonl"),
            },
            "registry_manifest": str(dataset_dir / "manifest.json"),
        },
        "packets": {
            "source": {
                "state_type": "residual_input",
                "layer_indices": [0, 1],
                "offsets": [-3, -2, -1],
                "width": 4,
                "dtype": "float32",
            },
            "target": {
                "state_type": "residual_input",
                "layer_indices": [0, 1],
                "offsets": [-4, -3, -2, -1],
                "width": 5,
                "dtype": "float32",
                "boundary_positions": 1,
            },
        },
        "extraction": {
            "default_bundle_dir": str(bundle_dir),
            "max_length": 64,
            "shard_size": 3,
        },
    }


def test_mock_task_materialization_and_dry_bundle_are_content_addressed(tmp_path):
    config = _config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    materialized = materialize_packet_bridge_tasks(config_path, mock_data=True)

    result = materialize_packet_bundle(config_path, dry_run=True)
    manifest = json.loads(
        (tmp_path / "bundle" / "manifest.json").read_text(encoding="utf-8")
    )

    assert materialized["counts"] == {
        "train": 4,
        "development_selection": 2,
        "development_gate": 2,
    }
    assert result["validation_status"] == "passed"
    assert result["records"] == 8
    assert result["split_counts"]["confirmation"] == 0
    assert len(manifest["shards"]) == 3
    assert manifest["registry"]["manifest_sha256"]
    assert not (tmp_path / "bundle" / "staging").exists()


def test_terminal_layout_proves_name_immediately_precedes_boundary():
    formatted = "aaaafooBBBBBB"
    metadata = {
        "input_ids": list(range(len(formatted))),
        "token_count": len(formatted),
        "offset_mapping": [[index, index + 1] for index in range(len(formatted))],
    }
    task = {"task_id": "task", "entry_point": "foo"}

    layout = classify_target_terminal_layout(
        task,
        formatted,
        metadata,
        packet_positions=10,
        boundary_positions=6,
    )

    assert layout["name_token_count"] == 3
    assert layout["name_offsets"] == [-9, -8, -7]
    assert layout["boundary_offsets"] == [-6, -5, -4, -3, -2, -1]
    assert layout["core_offsets"] == [-10]


def test_terminal_layout_rejects_nonterminal_name():
    formatted = "fooXXBBBBBB"
    metadata = {
        "input_ids": list(range(len(formatted))),
        "token_count": len(formatted),
        "offset_mapping": [[index, index + 1] for index in range(len(formatted))],
    }

    with pytest.raises(ValueError, match="immediately precede"):
        classify_target_terminal_layout(
            {"task_id": "task", "entry_point": "foo"},
            formatted,
            metadata,
            packet_positions=10,
            boundary_positions=6,
        )
