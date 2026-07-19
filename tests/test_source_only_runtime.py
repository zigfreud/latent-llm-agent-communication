import hashlib
import json
from datetime import datetime, timezone

import pytest
import torch

from src.scripts.run_source_only_probe import (
    load_heldout_vectors,
    load_existing_results,
    random_norm_matched,
    stable_seed,
    verify_heldout_bundle,
)


def test_heldout_vectors_follow_materialized_task_ids(tmp_path):
    bundle_dir = tmp_path / "bundle"
    shards_dir = bundle_dir / "shards"
    shards_dir.mkdir(parents=True)
    torch.save(
        [
            {"src_vector": torch.tensor([1.0, 2.0]), "tgt_vector": torch.tensor([3.0])},
            {"src_vector": torch.tensor([4.0, 5.0]), "tgt_vector": torch.tensor([6.0])},
        ],
        shards_dir / "shard_0.pt",
    )
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sampled_ids": ["a", "b"],
                "shards": [{"path": "shards/shard_0.pt"}],
            }
        ),
        encoding="utf-8",
    )
    source, target = load_heldout_vectors(
        {"path": str(manifest_path)},
        [{"task_id": "b"}, {"task_id": "a"}],
    )
    assert source[0].tolist() == [[4.0, 5.0]]
    assert source[1].tolist() == [[1.0, 2.0]]
    assert target[0].tolist() == [[6.0]]


def test_heldout_bundle_verification_binds_tasks_and_model_revisions(tmp_path):
    tasks = [
        {"task_id": "eval-a", "prompt": "task a"},
        {"task_id": "eval-b", "prompt": "task b"},
    ]
    bundle_dir = tmp_path / "heldout"
    shards_dir = bundle_dir / "shards"
    shards_dir.mkdir(parents=True)
    shard_path = shards_dir / "shard_0.pt"
    torch.save(
        [
            {"src_vector": torch.ones(2), "tgt_vector": torch.ones(3)},
            {"src_vector": torch.ones(2) * 2, "tgt_vector": torch.ones(3) * 2},
        ],
        shard_path,
    )
    source_revision = "a" * 40
    target_revision = "b" * 40
    manifest = {
        "bundle_format": "lip_latent_bundle",
        "schema_version": 1,
        "trace_id": "LIP-PROTO-001-DATA-EVAL",
        "source_model": "source/model",
        "target_model": "target/model",
        "dataset_origin": "test held-out bundle",
        "input_dim": 2,
        "output_dim": 3,
        "num_samples": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "license_notes": "test only",
        "source_layer": "-1",
        "target_layer": "-2",
        "token_position": "last_non_padding",
        "prompt_protocol": {
            "version": "lip-prompt-v1",
            "mode": "raw",
            "add_generation_prompt": False,
            "system_prompt": None,
        },
        "extraction_mode": "real",
        "source_quantization": "bitsandbytes-4bit",
        "target_quantization": "bitsandbytes-4bit",
        "quantization_compute_dtype": "float16",
        "use_safetensors": True,
        "max_length": 512,
        "source_model_revision": source_revision,
        "target_model_revision": target_revision,
        "source_dataset": "dataset/id",
        "source_dataset_config": "full",
        "source_split": "validation",
        "prompt_field": "text",
        "sampling_seed": 42,
        "sampled_ids": [task["task_id"] for task in tasks],
        "sampled_prompt_sha256": [
            hashlib.sha256(task["prompt"].encode()).hexdigest() for task in tasks
        ],
        "shards": [
            {
                "path": "shards/shard_0.pt",
                "records": 2,
                "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
            }
        ],
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "models": {"source_model": "source/model", "target_model": "target/model"},
        "prompt_protocol": manifest["prompt_protocol"],
        "extraction": {
            "source_layer": -1,
            "target_layer": -2,
            "token_position": "last_non_padding",
            "max_length": 512,
        },
        "runtime": {
            "source_load_4bit": True,
            "load_4bit": True,
            "quantization_compute_dtype": "float16",
        },
        "adapter": {"input_dim": 2, "output_dim": 3},
        "data": {
            "heldout_bundle_manifest": str(manifest_path),
            "heldout_bundle_trace_id": "LIP-PROTO-001-DATA-EVAL",
            "dataset_name": "dataset/id",
            "dataset_config": "full",
            "split": "validation",
            "prompt_field": "text",
            "sampling_seed": 42,
            "task_count": 2,
        },
    }
    report = verify_heldout_bundle(
        config,
        tasks,
        {
            "source_model_revision": source_revision,
            "target_model_revision": target_revision,
            "sampled_ids": ["train-a"],
        },
    )
    assert report["records"] == 2
    assert report["sampled_ids"] == ["eval-a", "eval-b"]


def test_random_control_matches_each_reference_norm():
    reference = torch.tensor([[3.0, 4.0, 0.0]])
    control = random_norm_matched(reference, seed=12)
    assert control.norm().item() == pytest.approx(reference.norm().item())
    assert torch.equal(control, random_norm_matched(reference, seed=12))


def test_stable_seed_is_deterministic_and_order_sensitive():
    assert stable_seed(1, 2, 3) == stable_seed(1, 2, 3)
    assert stable_seed(1, 2, 3) != stable_seed(3, 2, 1)


def test_resume_reader_rejects_duplicate_design_keys(tmp_path):
    row = {
        "task_id": "a",
        "condition": "source_latent",
        "training_seed": 41,
        "generation_seed": 101,
    }
    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate result key"):
        load_existing_results(path)
