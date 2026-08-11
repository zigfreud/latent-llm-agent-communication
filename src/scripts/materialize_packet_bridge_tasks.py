"""Materialize deterministic train/development registries for LIP-PROTO-014."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.pipelines.oracle_experiment import load_yaml, sha256_path, task_sha256, write_jsonl
from src.scripts.materialize_mbpp_prompt_configs import normalize_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize LIP packet bridge tasks")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mock-data", action="store_true")
    return parser.parse_args()


def _load_rows(dataset_config, split, *, mock_data, minimum_count):
    if mock_data:
        return [
            {
                "task_id": f"mock-{split}-{index:04d}",
                "text": f"Return the integer {index} from a Python function.",
                "entry_point": f"solve_{index}",
                "test_list": [f"assert solve_{index}() == {index}"],
                "test_setup_code": "",
            }
            for index in range(minimum_count + 16)
        ]
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("task materialization requires the datasets package") from exc
    return list(
        load_dataset(
            dataset_config["dataset_id"],
            dataset_config["dataset_config"],
            split=split,
            revision=dataset_config["revision"],
        )
    )


def _selection_hash(*, salt: str, split: str, task_id: str) -> str:
    return hashlib.sha256(f"{salt}\0{split}\0{task_id}".encode("utf-8")).hexdigest()


def _normalize_rows(rows, *, prompt_field, max_prompt_chars, split, salt):
    normalized = []
    for index, row in enumerate(rows):
        item = normalize_row(
            row,
            prompt_field,
            max_prompt_chars,
            fallback_id=index,
            include_entry_point=True,
        )
        if item is None:
            continue
        task = item["task"]
        task_id = str(task["task_id"])
        normalized.append(
            {
                **task,
                "source_split": split,
                "selection_hash": _selection_hash(
                    salt=salt,
                    split=split,
                    task_id=task_id,
                ),
            }
        )
    normalized.sort(key=lambda task: (task["selection_hash"], task["task_id"]))
    return normalized


def _ids_sha256(tasks):
    return hashlib.sha256(
        json.dumps(
            [task["task_id"] for task in tasks], separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def materialize_packet_bridge_tasks(config_path: Path, *, mock_data: bool = False):
    config = load_yaml(config_path)
    data = config["data"]
    selection = data["selection"]
    dataset_config = data["dataset"]
    train_count = int(selection["train_count"])
    development_selection_count = int(selection["development_selection_count"])
    development_gate_count = int(selection["development_gate_count"])
    if min(train_count, development_selection_count, development_gate_count) <= 0:
        raise ValueError("all packet bridge split counts must be positive")
    salt = str(selection["salt"])
    prompt_field = str(dataset_config.get("prompt_field", "text"))
    max_prompt_chars = int(selection.get("max_prompt_chars", 4096))

    train_rows = _load_rows(
        dataset_config,
        str(dataset_config["train_split"]),
        mock_data=mock_data,
        minimum_count=train_count,
    )
    development_total = development_selection_count + development_gate_count
    development_rows = _load_rows(
        dataset_config,
        str(dataset_config["development_split"]),
        mock_data=mock_data,
        minimum_count=development_total,
    )
    train_candidates = _normalize_rows(
        train_rows,
        prompt_field=prompt_field,
        max_prompt_chars=max_prompt_chars,
        split=str(dataset_config["train_split"]),
        salt=salt,
    )
    development_candidates = _normalize_rows(
        development_rows,
        prompt_field=prompt_field,
        max_prompt_chars=max_prompt_chars,
        split=str(dataset_config["development_split"]),
        salt=salt,
    )
    if len(train_candidates) < train_count:
        raise ValueError("insufficient normalized MBPP train tasks")
    if len(development_candidates) < development_total:
        raise ValueError("insufficient normalized MBPP development tasks")

    registries = {
        "train": train_candidates[:train_count],
        "development_selection": development_candidates[:development_selection_count],
        "development_gate": development_candidates[
            development_selection_count:development_total
        ],
    }
    all_keys = [
        (split, task["task_id"])
        for split, tasks in registries.items()
        for task in tasks
    ]
    if len(all_keys) != len(set(all_keys)):
        raise RuntimeError("packet bridge task registries are not disjoint")

    output_paths = {
        split: Path(data["registries"][split]) for split in registries
    }
    for split, path in output_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(path, registries[split])
    manifest_path = Path(data["registry_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_kind": "lip_packet_bridge_task_registry",
        "schema_version": 1,
        "experiment_id": str(config["experiment_id"]),
        "mock_data": bool(mock_data),
        "dataset": {
            "dataset_id": str(dataset_config["dataset_id"]),
            "dataset_config": str(dataset_config["dataset_config"]),
            "revision": str(dataset_config["revision"]),
            "train_split": str(dataset_config["train_split"]),
            "development_split": str(dataset_config["development_split"]),
        },
        "selection": {
            "method": "sha256_order_within_dataset_split",
            "salt": salt,
            "max_prompt_chars": max_prompt_chars,
            "include_entry_point_in_prompt": True,
        },
        "splits": {
            split: {
                "count": len(tasks),
                "task_ids_sha256": _ids_sha256(tasks),
                "tasks_jsonl": str(output_paths[split]),
                "tasks_jsonl_sha256": sha256_path(output_paths[split]),
                "task_sha256": [task_sha256(task) for task in tasks],
            }
            for split, tasks in registries.items()
        },
        "task_keys_sha256": hashlib.sha256(
            json.dumps(all_keys, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"manifest": manifest_path, "registries": output_paths, "counts": {
        split: len(tasks) for split, tasks in registries.items()
    }}


def main() -> None:
    args = parse_args()
    result = materialize_packet_bridge_tasks(
        args.config,
        mock_data=args.mock_data,
    )
    print("LIP packet bridge task materialization passed")
    print(f"manifest: {result['manifest']}")
    for split, count in result["counts"].items():
        print(f"{split}: {count}")


if __name__ == "__main__":
    main()
