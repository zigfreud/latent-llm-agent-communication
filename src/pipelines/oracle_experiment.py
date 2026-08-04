"""Shared data binding and artifact I/O for target-oracle experiments."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.core.prompt_protocol import protocol_metadata


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return payload


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def task_sha256(task: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        task,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"task row {line_number} must be an object")
            task_id = str(row.get("task_id", "")).strip()
            prompt = str(row.get("prompt", "")).strip()
            if not task_id or not prompt:
                raise ValueError(f"task row {line_number} needs task_id and prompt")
            tasks.append({**row, "task_id": task_id, "prompt": prompt})
    if not tasks:
        raise ValueError("task file contains no tasks")
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("task IDs must be unique")
    return tasks


def bind_tasks_to_manifest(
    config: Mapping[str, Any],
    tasks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    """Bind task order, prompts, target, and protocol to an immutable manifest."""

    data = config["data"]
    task_manifest = data.get("task_manifest")
    manifest_value = task_manifest or data.get("heldout_bundle_manifest")
    if not isinstance(manifest_value, str) or not manifest_value.strip():
        raise ValueError("data must configure task_manifest or heldout_bundle_manifest")
    manifest_path = Path(manifest_value)
    manifest = load_json_object(manifest_path)
    if task_manifest is not None:
        if manifest.get("manifest_kind") != "lip_oracle_task_manifest":
            raise ValueError("task manifest has the wrong manifest_kind")
        if manifest.get("schema_version") != 1 or manifest.get("mock_data"):
            raise ValueError("oracle audit requires a real schema-v1 task manifest")
        tasks_path = Path(str(data["tasks_jsonl"]))
        if manifest.get("tasks_jsonl_sha256") != sha256_path(tasks_path):
            raise ValueError("task file digest does not match the task manifest")
    elif bool(data.get("require_real_bundle", True)) and manifest.get(
        "extraction_mode"
    ) != "real":
        raise ValueError("oracle audit requires a real held-out bundle")
    target_model = str(config["models"]["target_model"])
    if manifest.get("target_model") != target_model:
        raise ValueError("held-out manifest target model does not match the audit")

    expected_protocol = protocol_metadata(config.get("prompt_protocol"))
    manifest_protocol = (
        manifest.get("prompt_protocols", {}).get("target")
        if "prompt_protocols" in manifest
        else manifest.get("prompt_protocol")
    )
    if manifest_protocol != expected_protocol:
        raise ValueError("held-out manifest target prompt protocol does not match")

    revision = manifest.get("target_model_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("held-out manifest needs an immutable target model revision")
    sampled_ids = manifest.get("sampled_ids")
    prompt_hashes = manifest.get("sampled_prompt_sha256")
    if not isinstance(sampled_ids, list) or not isinstance(prompt_hashes, list):
        raise ValueError("held-out manifest needs sampled IDs and prompt hashes")
    if len(sampled_ids) != len(prompt_hashes):
        raise ValueError("held-out sampled IDs and prompt hashes have different lengths")
    task_hashes = manifest.get("sampled_task_sha256")
    if task_manifest is not None and (
        not isinstance(task_hashes, list) or len(task_hashes) != len(sampled_ids)
    ):
        raise ValueError("task manifest needs one full task hash per sampled ID")

    by_id = {task["task_id"]: task for task in tasks}
    bound = []
    for task_index, (task_id, expected_hash) in enumerate(
        zip(sampled_ids, prompt_hashes)
    ):
        task = by_id.get(str(task_id))
        if task is None:
            raise ValueError(f"held-out task is missing from task file: {task_id}")
        if prompt_sha256(task["prompt"]) != expected_hash:
            raise ValueError(f"held-out prompt digest mismatch for task {task_id}")
        if task_hashes is not None and task_sha256(task) != task_hashes[task_index]:
            raise ValueError(f"held-out task digest mismatch for task {task_id}")
        bound.append(task)

    task_count = int(data["task_count"])
    if len(bound) < task_count:
        raise ValueError("held-out bundle does not contain the configured task count")
    return bound[:task_count], manifest, manifest_path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def generation_kwargs(config: Mapping[str, Any], tokenizer) -> dict[str, Any]:
    """Build the shared deterministic/sampling contract for target generation."""

    do_sample = bool(config["do_sample"])
    kwargs = {
        "max_new_tokens": int(config["max_new_tokens"]),
        "do_sample": do_sample,
        "repetition_penalty": float(config.get("repetition_penalty", 1.0)),
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs.update(
            {
                "temperature": float(config["temperature"]),
                "top_p": float(config.get("top_p", 1.0)),
            }
        )
    return kwargs
