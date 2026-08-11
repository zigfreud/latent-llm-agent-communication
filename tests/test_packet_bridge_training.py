import json

import pytest
import torch
import yaml

from src.core.packet_bundle import PACKET_SPLITS, sha256_file, sha256_json
from src.pipelines.packet_bridge import (
    _resolve_training_batch_size,
    train_packet_bridge,
)


def _training_record(split, index):
    generator = torch.Generator().manual_seed(index + 100)
    source = torch.randn(2, 3, 4, generator=generator)
    target = torch.zeros(2, 4, 5)
    target[:, :, :4] = source.mean(dim=(0, 1))[None, None, :]
    target[:, :, 4] = source.mean()
    digest = f"{index + 1:064x}"
    source_ids = [index + 1, index + 2, index + 3]
    target_ids = [index + 4, index + 5, index + 6, index + 7]
    source_mask = [1, 1, 1]
    target_mask = [1, 1, 1, 1]
    return {
        "task_id": f"{split}-{index}",
        "split": split,
        "task_sha256": digest,
        "prompt_sha256": digest,
        "source_prompt_sha256": digest,
        "target_prompt_sha256": digest,
        "source_input_ids": source_ids,
        "source_attention_mask": source_mask,
        "source_input_ids_sha256": sha256_json(source_ids),
        "source_attention_mask_sha256": sha256_json(source_mask),
        "source_token_count": len(source_ids),
        "target_input_ids": target_ids,
        "target_attention_mask": target_mask,
        "target_input_ids_sha256": sha256_json(target_ids),
        "target_attention_mask_sha256": sha256_json(target_mask),
        "target_token_count": len(target_ids),
        "name_token_count": 1,
        "source_packet": source,
        "target_packet": target,
    }


def _write_training_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    shard_dir = bundle / "shards"
    shard_dir.mkdir(parents=True)
    counts = {
        "train": 8,
        "development_selection": 4,
        "development_gate": 8,
        "confirmation": 0,
    }
    records = []
    index = 0
    for split in PACKET_SPLITS:
        for _ in range(counts[split]):
            records.append(_training_record(split, index))
            index += 1
    shard_path = shard_dir / "shard_0.pt"
    torch.save(records, shard_path)
    task_keys = [(record["split"], record["task_id"]) for record in records]
    manifest = {
        "bundle_format": "lip_packet_bundle",
        "schema_version": 1,
        "trace_id": "LIP-PROTO-014-training-test",
        "extraction_mode": "dry_run",
        "extraction_scope": "full",
        "config_sha256": "b" * 64,
        "source": {
            "model_id": "source/test",
            "revision": "source-revision",
            "prompt_protocol": "raw_task_with_entry_point",
        },
        "target": {
            "model_id": "target/test",
            "revision": "target-revision",
            "prompt_protocol": "target_chat_template",
        },
        "dataset": {
            "dataset_id": "dataset/test",
            "dataset_config": "full",
            "revision": "dataset-revision",
        },
        "registry": {"manifest_sha256": "c" * 64},
        "source_packet": {
            "shape": [2, 3, 4],
            "layer_indices": [0, 1],
            "offsets": [-3, -2, -1],
            "state_type": "residual_input",
            "dtype": "float32",
        },
        "target_packet": {
            "shape": [2, 4, 5],
            "layer_indices": [0, 1],
            "offsets": [-4, -3, -2, -1],
            "state_type": "residual_input",
            "dtype": "float32",
        },
        "predecessor": {
            "protocol": "LIP-PROTO-013",
            "sha256sums_sha256": "a" * 64,
        },
        "splits": counts,
        "task_keys_sha256": sha256_json(task_keys),
        "shards": [
            {
                "path": "shards/shard_0.pt",
                "records": len(records),
                "sha256": sha256_file(shard_path),
            }
        ],
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return bundle


def test_bridge_only_training_selects_checkpoint_before_untouched_gate(tmp_path):
    bundle = _write_training_bundle(tmp_path)
    output_dir = tmp_path / "run"
    config = {
        "experiment_id": "LIP-PROTO-014-test",
        "device": "cpu",
        "output_dir": str(output_dir),
        "seed": 73,
        "data": {
            "bundle_dir": str(bundle),
            "require_real": False,
            "boundary_positions": 1,
        },
        "model": {"kind": "structured_linear"},
        "training": {
            "batch_size": 4,
            "max_updates": 2,
            "validation_interval": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "fp16_autocast": False,
            "num_workers": 0,
        },
        "loss": {
            "lambda_huber": 1.0,
            "lambda_cosine": 0.25,
            "lambda_symmetric_nce": 1.0,
            "lambda_margin": 0.1,
            "lambda_norm": 0.05,
            "component_weights": {"core": 0.45, "name": 0.45, "boundary": 0.1},
        },
        "development_gate": {"alpha": 0.05, "statistics_seed": 91},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = train_packet_bridge(config_path)

    assert result["updates_completed"] == 2
    assert result["configured_batch_size"] == 4
    assert result["effective_batch_size"] == 4
    assert result["best_step"] in {1, 2}
    assert result["bundle_validation"]["split_counts"]["confirmation"] == 0
    assert result["development_selection"]["task_count"] == 4
    assert result["development_gate_metrics"]["task_count"] == 8
    assert (output_dir / "best_checkpoint.pt").is_file()
    assert (output_dir / "target_statistics.pt").is_file()
    assert (output_dir / "run_summary.json").is_file()


def test_nonclaim_preflight_caps_only_the_effective_batch_size():
    assert _resolve_training_batch_size(
        4,
        train_count=2,
        extraction_scope="preflight",
        require_real=False,
    ) == 2

    with pytest.raises(ValueError, match="cannot exceed"):
        _resolve_training_batch_size(
            4,
            train_count=2,
            extraction_scope="full",
            require_real=False,
        )
