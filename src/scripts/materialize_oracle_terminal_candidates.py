"""Materialize the tokenizer-stratified candidate registry for LIP-PROTO-013."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.core.prompt_protocol import format_prompt, protocol_metadata
from src.evaluation.oracle_terminal_factorial import (
    ORACLE_TERMINAL_CANDIDATE_COUNT,
    ORACLE_TERMINAL_EXPERIMENT_ID,
    ORACLE_TERMINAL_PROTOCOL_VERSION,
    terminal_components,
    validate_terminal_layout,
)
from src.pipelines.oracle_experiment import (
    load_json_object,
    load_yaml,
    prompt_sha256,
    sha256_path,
    task_sha256,
    write_json,
    write_jsonl,
)
from src.scripts.materialize_mbpp_prompt_configs import normalize_row
from src.scripts.run_oracle_memory_functional import validate_config


DEFAULT_CONFIG = Path("config/LIP-PROTO-013_terminal_source_factorial.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _ids_sha256(task_ids: list[str]) -> str:
    payload = json.dumps(task_ids, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_excluded_ids(config: Mapping[str, Any]) -> tuple[set[str], dict[str, Any]]:
    source = config["population_source"]
    manifest_path = Path(str(source["candidate_task_manifest"]))
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if sha256_path(manifest_path) != source["candidate_task_manifest_sha256"]:
        raise ValueError("LIP-PROTO-010 candidate manifest hash changed")
    manifest = load_json_object(manifest_path)
    sampled = [str(task_id) for task_id in manifest.get("sampled_ids", [])]
    if (
        manifest.get("experiment_id") != "LIP-PROTO-010"
        or len(sampled) != source["candidate_manifest_task_count"]
        or len(set(sampled)) != len(sampled)
        or manifest.get("excluded_task_count")
        != source["legacy_excluded_task_count"]
        or manifest.get("excluded_task_ids_sha256")
        != source["legacy_excluded_ids_sha256"]
    ):
        raise ValueError("LIP-PROTO-010 manifest violates the frozen population source")

    legacy: set[str] = set()
    legacy_sources = []
    for item in manifest.get("excluded_task_manifests", []):
        if not isinstance(item, Mapping):
            raise ValueError("010 exclusion provenance must contain mappings")
        path = Path(str(item.get("path", "")))
        if not path.is_file():
            raise FileNotFoundError(path)
        legacy_manifest = load_json_object(path)
        ids = [str(task_id) for task_id in legacy_manifest.get("sampled_ids", [])]
        if not ids or len(set(ids)) != len(ids) or legacy.intersection(ids):
            raise ValueError("legacy exclusion manifests must be non-empty and disjoint")
        legacy.update(ids)
        legacy_sources.append(
            {
                "path": str(path),
                "sha256": sha256_path(path),
                "experiment_id": legacy_manifest.get("experiment_id"),
                "task_count": len(ids),
            }
        )
    if len(legacy) != source["legacy_excluded_task_count"]:
        raise ValueError("legacy exclusion count changed")
    if _ids_sha256(sorted(legacy)) != source["legacy_excluded_ids_sha256"]:
        raise ValueError("legacy exclusion identity hash changed")
    excluded = set(sampled).union(legacy)
    if len(excluded) != source["total_excluded_task_count"]:
        raise ValueError("010 candidates and legacy exclusions must total 242 tasks")
    return excluded, {
        "candidate_manifest": str(manifest_path),
        "candidate_manifest_sha256": sha256_path(manifest_path),
        "candidate_ids_sha256": _ids_sha256(sampled),
        "legacy_sources": legacy_sources,
        "legacy_ids_sha256": _ids_sha256(sorted(legacy)),
        "excluded_task_count": len(excluded),
        "excluded_task_ids_sha256": _ids_sha256(sorted(excluded)),
        "dataset_name": manifest["dataset_name"],
        "dataset_config": manifest.get("dataset_config"),
        "dataset_split": manifest["dataset_split"],
        "prompt_field": manifest["prompt_field"],
    }


def classify_terminal_layout(
    task: Mapping[str, Any],
    tokenizer,
    prompt_protocol: Mapping[str, Any],
    *,
    selection_salt: str,
) -> dict[str, Any] | None:
    """Locate the final required-function-name span in token coordinates."""

    prompt = str(task["prompt"])
    entry_point = str(task.get("entry_point", ""))
    if not entry_point:
        raise ValueError("task has no entry point")
    formatted = format_prompt(prompt, tokenizer, prompt_protocol)
    name_start = formatted.rfind(entry_point)
    if name_start < 0:
        raise ValueError("formatted prompt does not contain the required entry point")
    name_stop = name_start + len(entry_point)
    encoded = tokenizer(
        formatted,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = [int(value) for value in encoded["input_ids"]]
    char_offsets = [tuple(int(value) for value in pair) for pair in encoded["offset_mapping"]]
    token_indices = [
        index
        for index, (start, stop) in enumerate(char_offsets)
        if stop > name_start and start < name_stop
    ]
    count = len(token_indices)
    if count not in (2, 3):
        return None
    relative = tuple(index - len(input_ids) for index in token_indices)
    components = terminal_components(count)
    if relative != components["name"]:
        return None
    if len(input_ids) < 32:
        return None
    selection_hash = hashlib.sha256(
        f"{selection_salt}\0{count}\0{task['task_id']}".encode("utf-8")
    ).hexdigest()
    layout = {
        "name_token_count": count,
        "core_offsets": list(components["core"]),
        "name_offsets": list(components["name"]),
        "boundary_offsets": list(components["boundary"]),
        "tail_offsets": list(range(-24, 0)),
        "selection_hash": selection_hash,
        "formatted_prompt_token_count": len(input_ids),
        "name_token_ids": [input_ids[index] for index in token_indices],
        "boundary_token_ids": input_ids[-6:],
    }
    validate_terminal_layout(layout)
    return layout


def materialize_candidates(
    config: dict[str, Any],
    config_path: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    validate_config(config)
    source = config["population_source"]
    preflight_path = Path(str(source["preflight_report"]))
    if not preflight_path.is_file():
        raise FileNotFoundError(preflight_path)
    if sha256_path(preflight_path) != source["preflight_report_sha256"]:
        raise ValueError("terminal-layout preflight report hash changed")
    excluded, provenance = _source_excluded_ids(config)

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "candidate materialization requires datasets and transformers"
        ) from exc

    dataset = load_dataset(
        provenance["dataset_name"],
        provenance["dataset_config"],
        split=provenance["dataset_split"],
    )
    rows = [row for row in dataset if str(row.get("task_id", "")) not in excluded]
    if len(rows) != source["remaining_task_count"]:
        raise ValueError("remaining MBPP population changed")
    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["target_model"],
        revision=config["models"]["target_model_revision"],
        use_fast=True,
    )
    protocol = protocol_metadata(config["prompt_protocol"])
    candidates = []
    for index, row in enumerate(rows):
        normalized = normalize_row(
            row,
            provenance["prompt_field"],
            512,
            index,
            include_entry_point=True,
        )
        if normalized is None:
            continue
        task = normalized["task"]
        layout = classify_terminal_layout(
            task,
            tokenizer,
            protocol,
            selection_salt=source["candidate_order_salt"],
        )
        if layout is not None:
            candidates.append({**task, "terminal_layout": layout})
    candidates.sort(
        key=lambda task: (
            int(task["terminal_layout"]["name_token_count"]),
            str(task["terminal_layout"]["selection_hash"]),
            str(task["task_id"]),
        )
    )
    strata = {
        str(count): sum(
            task["terminal_layout"]["name_token_count"] == count
            for task in candidates
        )
        for count in (2, 3)
    }
    if len(candidates) != ORACLE_TERMINAL_CANDIDATE_COUNT or strata != source[
        "structural_strata"
    ]:
        raise ValueError("materialized tokenizer strata differ from the frozen audit")

    data = config["data"]
    output = config["output"]
    tasks_path = Path(str(data["candidate_tasks_jsonl"]))
    manifest_path = Path(str(data["candidate_task_manifest"]))
    report_path = Path(str(output["candidate_registry_report_json"]))
    for path in (tasks_path, manifest_path, report_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"candidate registry output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(tasks_path, candidates)
    task_ids = [str(task["task_id"]) for task in candidates]
    manifest = {
        "manifest_kind": "lip_oracle_task_manifest",
        "schema_version": 1,
        "experiment_id": ORACLE_TERMINAL_EXPERIMENT_ID,
        "selection_kind": "terminal_layout_structural_candidates",
        "dataset_name": provenance["dataset_name"],
        "dataset_config": provenance["dataset_config"],
        "dataset_split": provenance["dataset_split"],
        "prompt_field": provenance["prompt_field"],
        "task_count": len(candidates),
        "sampled_ids": task_ids,
        "sampled_prompt_sha256": [
            prompt_sha256(str(task["prompt"])) for task in candidates
        ],
        "sampled_task_sha256": [task_sha256(task) for task in candidates],
        "population_source": provenance,
        "preflight_report": str(preflight_path),
        "preflight_report_sha256": sha256_path(preflight_path),
        "preflight_source_commit": source["preflight_source_commit"],
        "candidate_order_method": source["candidate_order_method"],
        "candidate_order_salt": source["candidate_order_salt"],
        "candidate_ids_sha256": _ids_sha256(task_ids),
        "structural_strata": strata,
        "sampled_ids_disjoint_from_exclusions": not bool(
            set(task_ids).intersection(excluded)
        ),
        "include_entry_point_in_prompt": True,
        "entry_point_resolution": "tests_then_reference_code",
        "target_model": config["models"]["target_model"],
        "target_model_revision": config["models"]["target_model_revision"],
        "prompt_protocol": protocol,
        "tasks_jsonl": str(tasks_path),
        "tasks_jsonl_sha256": sha256_path(tasks_path),
        "sampling_config": str(config_path),
        "sampling_config_sha256": sha256_path(config_path),
        "mock_data": False,
    }
    write_json(manifest_path, manifest)
    report = {
        "experiment_id": ORACLE_TERMINAL_EXPERIMENT_ID,
        "protocol_version": ORACLE_TERMINAL_PROTOCOL_VERSION,
        "claim_eligible": False,
        "remaining_task_count": len(rows),
        "candidate_task_count": len(candidates),
        "candidate_task_ids_sha256": _ids_sha256(task_ids),
        "structural_strata": strata,
        "candidate_tasks_jsonl": str(tasks_path),
        "candidate_tasks_jsonl_sha256": sha256_path(tasks_path),
        "candidate_task_manifest": str(manifest_path),
        "candidate_task_manifest_sha256": sha256_path(manifest_path),
        "passed": True,
    }
    write_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    report = materialize_candidates(
        config,
        args.config,
        overwrite=args.overwrite,
    )
    print("Terminal-layout candidate registry completed")
    print(f"candidates: {report['candidate_task_count']}")
    print(f"strata: {report['structural_strata']}")
    print(f"report: {config['output']['candidate_registry_report_json']}")


if __name__ == "__main__":
    main()
