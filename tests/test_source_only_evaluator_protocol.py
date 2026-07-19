import hashlib

import pytest

from src.evaluation.source_only import design_fingerprint
from src.scripts.evaluate_source_only_semantics import validate_generation_records


CONDITIONS = [
    "neutral_no_lip",
    "text_only_no_lip",
    "source_latent",
    "shuffled_source_latent",
    "random_norm_matched",
    "oracle_target_latent",
]


def make_design():
    return {
        "experiment_id": "test",
        "neutral_target_prompt": "neutral",
        "conditions": CONDITIONS,
        "adapter": {"checkpoints": [{"training_seed": 41, "path": "unused"}]},
        "generation": {"seeds": [101]},
        "data": {"task_count": 2},
    }


def make_records(config):
    fingerprint = design_fingerprint(config)
    tasks = {
        "a": {"task_id": "a", "prompt": "task a"},
        "b": {"task_id": "b", "prompt": "task b"},
    }
    vector_kinds = {
        "neutral_no_lip": None,
        "text_only_no_lip": None,
        "source_latent": "translated_source",
        "shuffled_source_latent": "translated_source",
        "random_norm_matched": "random_norm_matched",
        "oracle_target_latent": "target_hidden",
    }
    rows = []
    for task_id, task in tasks.items():
        other = "b" if task_id == "a" else "a"
        for condition in CONDITIONS:
            target_text = task["prompt"] if condition == "text_only_no_lip" else "neutral"
            if condition in {"source_latent", "oracle_target_latent"}:
                vector_task_id = task_id
            elif condition == "shuffled_source_latent":
                vector_task_id = other
            else:
                vector_task_id = None
            rows.append(
                {
                    "protocol_version": "lip-source-only-v1",
                    "design_sha256": fingerprint,
                    "training_bundle_manifest_sha256": "a" * 64,
                    "heldout_bundle_manifest_sha256": "c" * 64,
                    "adapter_checkpoint_sha256": "b" * 64,
                    "task_id": task_id,
                    "condition": condition,
                    "training_seed": 41,
                    "generation_seed": 101,
                    "task_spec": task,
                    "target_prompt_kind": (
                        "task" if condition == "text_only_no_lip" else "neutral"
                    ),
                    "target_user_prompt_sha256": hashlib.sha256(
                        target_text.encode("utf-8")
                    ).hexdigest(),
                    "target_formatted_prompt_sha256": hashlib.sha256(
                        ("formatted:" + target_text).encode("utf-8")
                    ).hexdigest(),
                    "vector_kind": vector_kinds[condition],
                    "vector_task_id": vector_task_id,
                    "injected_vector_norm": (
                        None if vector_kinds[condition] is None else 2.0
                    ),
                }
            )
    return rows


def test_complete_generation_design_passes_protocol_validation():
    config = make_design()
    report = validate_generation_records(make_records(config), config)
    assert report["complete"] is True
    assert report["record_count"] == 12


def test_source_latent_task_text_leak_is_rejected():
    config = make_design()
    rows = make_records(config)
    source_row = next(row for row in rows if row["condition"] == "source_latent")
    source_row["target_user_prompt_sha256"] = hashlib.sha256(
        source_row["task_spec"]["prompt"].encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="non-neutral"):
        validate_generation_records(rows, config)
