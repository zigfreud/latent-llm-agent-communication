import json

import pytest
import torch

from src.core.packet_bundle import (
    PACKET_SPLITS,
    PacketBundleValidationError,
    PacketRecordDataset,
    compute_target_packet_statistics,
    sha256_file,
    sha256_json,
    validate_packet_bundle,
)


def _record(split, index, *, target_value=0.0):
    digest = f"{index + 1:064x}"
    source_ids = [index + 1, index + 2, index + 3]
    target_ids = [index + 4, index + 5, index + 6, index + 7]
    source_mask = [1, 1, 1]
    target_mask = [1, 1, 1, 1]
    return {
        "task_id": f"task-{index}",
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
        "name_token_count": 2 + (index % 2),
        "source_packet": torch.full((2, 3, 4), float(index)),
        "target_packet": torch.full((2, 4, 5), float(target_value)),
    }


def _write_bundle(tmp_path):
    bundle = tmp_path / "packet-bundle"
    shards = bundle / "shards"
    shards.mkdir(parents=True)
    records = [
        _record(split, index, target_value=index)
        for index, split in enumerate(PACKET_SPLITS)
    ]
    shard_path = shards / "shard_0.pt"
    torch.save(records, shard_path)
    task_keys = [(record["split"], record["task_id"]) for record in records]
    manifest = {
        "bundle_format": "lip_packet_bundle",
        "schema_version": 1,
        "trace_id": "LIP-PROTO-014-test",
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
        "splits": {split: 1 for split in PACKET_SPLITS},
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
    return bundle, records, manifest


def test_packet_bundle_validates_shapes_hashes_provenance_and_splits(tmp_path):
    bundle, _, _ = _write_bundle(tmp_path)

    report = validate_packet_bundle(bundle)

    assert report["validation_status"] == "passed"
    assert report["records"] == 4
    assert report["source_shape"] == [2, 3, 4]
    assert report["target_shape"] == [2, 4, 5]
    assert report["split_counts"] == {split: 1 for split in PACKET_SPLITS}


def test_claim_oriented_validation_rejects_dry_run_bundle(tmp_path):
    bundle, _, _ = _write_bundle(tmp_path)

    with pytest.raises(PacketBundleValidationError, match="requires real full"):
        validate_packet_bundle(bundle, require_real=True)


def test_packet_bundle_detects_shard_tampering(tmp_path):
    bundle, records, _ = _write_bundle(tmp_path)
    records[0]["source_packet"][0, 0, 0] = 99.0
    torch.save(records, bundle / "shards" / "shard_0.pt")

    with pytest.raises(PacketBundleValidationError, match="sha256 mismatch"):
        validate_packet_bundle(bundle)


def test_training_statistics_and_dataset_normalize_only_supplied_records():
    first = _record("train", 0, target_value=2.0)
    second = _record("train", 1, target_value=6.0)

    scaffold, scale = compute_target_packet_statistics([first, second])
    dataset = PacketRecordDataset(
        [first, second],
        scaffold=scaffold,
        site_scale=scale,
    )

    assert torch.allclose(scaffold, torch.full((2, 4, 5), 4.0))
    assert torch.allclose(scale, torch.full((2, 4), 2.0), atol=1e-5)
    assert torch.allclose(dataset[0]["target_residual"], torch.full((2, 4, 5), -1.0))
    assert torch.allclose(dataset[1]["target_residual"], torch.full((2, 4, 5), 1.0))


def test_packet_statistics_reject_shape_drift():
    first = _record("train", 0)
    second = _record("train", 1)
    second["target_packet"] = torch.zeros(2, 3, 5)

    with pytest.raises(ValueError, match="share one shape"):
        compute_target_packet_statistics([first, second])
