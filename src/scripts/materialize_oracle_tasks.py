"""Freeze a target-oracle task registry without extracting latent bundles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from src.core.prompt_protocol import protocol_metadata
from src.pipelines.oracle_experiment import (
    prompt_sha256,
    sha256_path,
    task_sha256,
    write_json,
    write_jsonl,
)
from src.scripts.materialize_mbpp_prompt_configs import (
    load_split_rows,
    load_yaml,
    positive_int,
    required_string,
    sample_prompts,
)


DEFAULT_CONFIG = Path("config/LIP-PROTO-008_mbpp_test_sampling.yaml")
MANIFEST_KIND = "lip_oracle_task_manifest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mock-data", action="store_true")
    return parser.parse_args()


def resolve_target_revision(
    config: Mapping[str, Any], *, mock_data: bool
) -> str:
    configured = config.get("target_model_revision")
    if configured is not None:
        revision = str(configured)
    elif mock_data:
        revision = "0" * 40
    else:
        try:
            from huggingface_hub import model_info
        except ImportError as exc:
            raise RuntimeError(
                "real task materialization requires huggingface_hub"
            ) from exc
        revision = str(model_info(required_string(config, "target_model")).sha)
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("target_model_revision must be a 40-character lowercase SHA")
    return revision


def materialize(config_path: Path, *, mock_data: bool) -> dict[str, Any]:
    config = load_yaml(config_path)
    if config.get("experiment_id") != "LIP-PROTO-008":
        raise ValueError("oracle task registry must bind LIP-PROTO-008")
    split = required_string(config, "split")
    prompt_field = required_string(config, "prompt_field")
    task_count = positive_int(config, "task_count")
    seed = positive_int(config, "seed")
    max_prompt_chars = positive_int(config, "max_prompt_chars")
    include_entry_point = config.get("include_entry_point_in_prompt")
    if include_entry_point is not True:
        raise ValueError("LIP-PROTO-008 requires entry points in task prompts")
    entry_point_resolution = required_string(config, "entry_point_resolution")
    if entry_point_resolution != "tests_then_reference_code":
        raise ValueError(
            "LIP-PROTO-008 requires tests_then_reference_code entry-point resolution"
        )
    tasks_path = Path(required_string(config, "tasks_jsonl"))
    manifest_path = Path(required_string(config, "task_manifest"))
    if tasks_path == manifest_path:
        raise ValueError("tasks_jsonl and task_manifest must be different paths")

    rows = load_split_rows(config, split, mock_data)
    selected = sample_prompts(
        rows,
        task_count,
        seed,
        prompt_field,
        max_prompt_chars,
        include_entry_point=True,
    )
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(tasks_path, [item["task"] for item in selected])
    revision = resolve_target_revision(config, mock_data=mock_data)
    prompt_protocol = protocol_metadata(config.get("prompt_protocol"))
    manifest = {
        "manifest_kind": MANIFEST_KIND,
        "schema_version": 1,
        "experiment_id": "LIP-PROTO-008",
        "dataset_name": required_string(config, "dataset_name"),
        "dataset_config": config.get("dataset_config"),
        "dataset_split": split,
        "prompt_field": prompt_field,
        "sampling_seed": seed,
        "task_count": len(selected),
        "sampled_ids": [item["id"] for item in selected],
        "sampled_prompt_sha256": [
            prompt_sha256(item["prompt"]) for item in selected
        ],
        "sampled_task_sha256": [task_sha256(item["task"]) for item in selected],
        "include_entry_point_in_prompt": True,
        "entry_point_resolution": entry_point_resolution,
        "target_model": required_string(config, "target_model"),
        "target_model_revision": revision,
        "prompt_protocol": prompt_protocol,
        "tasks_jsonl": str(tasks_path),
        "tasks_jsonl_sha256": sha256_path(tasks_path),
        "sampling_config": str(config_path),
        "sampling_config_sha256": sha256_path(config_path),
        "mock_data": bool(mock_data),
    }
    write_json(manifest_path, manifest)
    return {
        "tasks_jsonl": tasks_path,
        "task_manifest": manifest_path,
        "task_count": len(selected),
        "task_ids": manifest["sampled_ids"],
        "target_model_revision": revision,
        "mock_data": bool(mock_data),
    }


def main() -> None:
    args = parse_args()
    result = materialize(args.config, mock_data=args.mock_data)
    print("Oracle task materialization passed")
    print(f"mock_data: {result['mock_data']}")
    print(f"tasks: {result['task_count']}")
    print(f"tasks_jsonl: {result['tasks_jsonl']}")
    print(f"task_manifest: {result['task_manifest']}")
    print(f"target_model_revision: {result['target_model_revision']}")


if __name__ == "__main__":
    main()
