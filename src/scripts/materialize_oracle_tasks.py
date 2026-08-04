"""Freeze a target-oracle task registry without extracting latent bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.core.prompt_protocol import protocol_metadata
from src.pipelines.oracle_experiment import (
    load_json_object,
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
SUPPORTED_EXPERIMENTS = {"LIP-PROTO-008", "LIP-PROTO-009", "LIP-PROTO-010"}


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


def _exclusion_contract(
    config: Mapping[str, Any],
) -> tuple[set[str], list[dict[str, Any]]]:
    configured = config.get("exclude_task_manifests", [])
    if not isinstance(configured, list) or any(
        not isinstance(value, str) or not value.strip() for value in configured
    ):
        raise ValueError("exclude_task_manifests must be a list of paths")
    if len(set(configured)) != len(configured):
        raise ValueError("exclude_task_manifests must not contain duplicates")
    excluded_ids: set[str] = set()
    provenance = []
    for value in configured:
        path = Path(value)
        manifest = load_json_object(path)
        if manifest.get("manifest_kind") != MANIFEST_KIND:
            raise ValueError(f"exclusion manifest has the wrong kind: {path}")
        if manifest.get("schema_version") != 1 or manifest.get("mock_data"):
            raise ValueError(
                f"exclusion manifest must be a real schema-v1 file: {path}"
            )
        if manifest.get("dataset_name") != config.get("dataset_name"):
            raise ValueError(f"exclusion manifest dataset differs: {path}")
        if manifest.get("dataset_config") != config.get("dataset_config"):
            raise ValueError(f"exclusion manifest dataset config differs: {path}")
        if manifest.get("dataset_split") != config.get("split"):
            raise ValueError(f"exclusion manifest split differs: {path}")
        sampled_ids = manifest.get("sampled_ids")
        if not isinstance(sampled_ids, list) or not sampled_ids:
            raise ValueError(f"exclusion manifest has no sampled IDs: {path}")
        normalized = {str(task_id) for task_id in sampled_ids}
        if len(normalized) != len(sampled_ids):
            raise ValueError(f"exclusion manifest has duplicate sampled IDs: {path}")
        overlap = excluded_ids.intersection(normalized)
        if overlap:
            raise ValueError(
                f"exclusion manifests overlap on {len(overlap)} task ID(s): {path}"
            )
        excluded_ids.update(normalized)
        provenance.append(
            {
                "path": str(path),
                "sha256": sha256_path(path),
                "experiment_id": manifest.get("experiment_id"),
                "task_count": len(normalized),
            }
        )
    return excluded_ids, provenance


def materialize(config_path: Path, *, mock_data: bool) -> dict[str, Any]:
    config = load_yaml(config_path)
    experiment_id = str(config.get("experiment_id", ""))
    if experiment_id not in SUPPORTED_EXPERIMENTS:
        raise ValueError(
            "oracle task registry must bind a supported protocol experiment"
        )
    split = required_string(config, "split")
    prompt_field = required_string(config, "prompt_field")
    task_count = positive_int(config, "task_count")
    seed = positive_int(config, "seed")
    max_prompt_chars = positive_int(config, "max_prompt_chars")
    include_entry_point = config.get("include_entry_point_in_prompt")
    if include_entry_point is not True:
        raise ValueError(f"{experiment_id} requires entry points in task prompts")
    entry_point_resolution = required_string(config, "entry_point_resolution")
    if entry_point_resolution != "tests_then_reference_code":
        raise ValueError(
            f"{experiment_id} requires tests_then_reference_code entry-point resolution"
        )
    tasks_path = Path(required_string(config, "tasks_jsonl"))
    manifest_path = Path(required_string(config, "task_manifest"))
    if tasks_path == manifest_path:
        raise ValueError("tasks_jsonl and task_manifest must be different paths")

    excluded_ids, exclusion_provenance = _exclusion_contract(config)
    rows = load_split_rows(config, split, mock_data)
    rows = [
        row for row in rows if str(row.get("task_id", "")) not in excluded_ids
    ]
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
        "experiment_id": experiment_id,
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
        "excluded_task_manifests": exclusion_provenance,
        "excluded_task_count": len(excluded_ids),
        "excluded_task_ids_sha256": hashlib.sha256(
            json.dumps(sorted(excluded_ids), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "sampled_ids_disjoint_from_exclusions": not bool(
            excluded_ids.intersection(item["id"] for item in selected)
        ),
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
