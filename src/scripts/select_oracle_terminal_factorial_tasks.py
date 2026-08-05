"""Select the balanced capability-confirmation registry for LIP-PROTO-013."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.evaluation.oracle_terminal_factorial import (
    ORACLE_TERMINAL_ELIGIBILITY_RULE,
    ORACLE_TERMINAL_EXPERIMENT_ID,
    ORACLE_TERMINAL_PROTOCOL_VERSION,
    ORACLE_TERMINAL_SCREENING_SCOPE,
    ORACLE_TERMINAL_SCREENING_SEEDS,
    ORACLE_TERMINAL_SELECTED_COUNT,
    ORACLE_TERMINAL_SELECTED_PER_STRATUM,
    candidate_binding_config,
    design_fingerprint,
    eligible_task_ids,
    validate_terminal_layout,
)
from src.pipelines.oracle_experiment import (
    bind_tasks_to_manifest,
    load_json_object,
    load_tasks,
    load_yaml,
    prompt_sha256,
    sha256_path,
    task_sha256,
    write_json,
    write_jsonl,
)
from src.scripts.evaluate_oracle_packet_semantics import read_jsonl
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


def _validate_screening_summary(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    config_path: Path,
    generations_path: Path,
) -> None:
    validation = summary.get("design_validation", {})
    sandbox = summary.get("sandbox", {})
    input_hashes = sandbox.get("input_sha256", {})
    metadata_path = generations_path.with_suffix(".metadata.json")
    scored_path = (
        Path(str(config["output"]["screening_evaluation_dir"]))
        / "scored_generations.jsonl"
    )
    checks = {
        "experiment": summary.get("experiment_id") == ORACLE_TERMINAL_EXPERIMENT_ID,
        "protocol": summary.get("protocol_version")
        == ORACLE_TERMINAL_PROTOCOL_VERSION,
        "design": validation.get("design_sha256") == design_fingerprint(config),
        "scope": validation.get("run_scope") == ORACLE_TERMINAL_SCREENING_SCOPE,
        "screening": validation.get("screening") is True,
        "complete": validation.get("complete") is True,
        "not_claim_eligible": summary.get("claim_eligible") is False,
        "hardened_execution": summary.get("execution_mode")
        == "functional_hardened_namespace",
        "sandbox_validated": sandbox.get("validated") is True,
        "scored_hash": summary.get("scored_jsonl_sha256") == sha256_path(scored_path),
        "config_hash": input_hashes.get("config") == sha256_path(config_path),
        "generations_hash": input_hashes.get("generations")
        == sha256_path(generations_path),
        "metadata_hash": input_hashes.get("metadata") == sha256_path(metadata_path),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "terminal screening summary failed provenance: " + ", ".join(failed)
        )


def _selected_manifest(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    selected_tasks: list[Mapping[str, Any]],
    selected_tasks_path: Path,
    candidate_manifest: Mapping[str, Any],
    candidate_manifest_path: Path,
    screening_summary_path: Path,
    scored_path: Path,
    eligible_by_stratum: Mapping[int, list[str]],
) -> dict[str, Any]:
    task_ids = [str(task["task_id"]) for task in selected_tasks]
    selected_by_stratum = {
        str(count): ids[:ORACLE_TERMINAL_SELECTED_PER_STRATUM]
        for count, ids in eligible_by_stratum.items()
    }
    return {
        "manifest_kind": "lip_oracle_task_manifest",
        "schema_version": 1,
        "experiment_id": ORACLE_TERMINAL_EXPERIMENT_ID,
        "selection_kind": "terminal_layout_stratified_capability_confirmation",
        "dataset_name": candidate_manifest["dataset_name"],
        "dataset_config": candidate_manifest.get("dataset_config"),
        "dataset_split": candidate_manifest["dataset_split"],
        "prompt_field": candidate_manifest["prompt_field"],
        "task_count": len(selected_tasks),
        "sampled_ids": task_ids,
        "sampled_prompt_sha256": [
            prompt_sha256(str(task["prompt"])) for task in selected_tasks
        ],
        "sampled_task_sha256": [task_sha256(task) for task in selected_tasks],
        "candidate_manifest": str(candidate_manifest_path),
        "candidate_manifest_sha256": sha256_path(candidate_manifest_path),
        "candidate_task_count": int(config["data"]["candidate_task_count"]),
        "candidate_ids_sha256": _ids_sha256(
            [str(task_id) for task_id in candidate_manifest["sampled_ids"]]
        ),
        "screening_summary": str(screening_summary_path),
        "screening_summary_sha256": sha256_path(screening_summary_path),
        "screening_scored_jsonl": str(scored_path),
        "screening_scored_jsonl_sha256": sha256_path(scored_path),
        "screening_seeds": list(ORACLE_TERMINAL_SCREENING_SEEDS),
        "eligibility_rule": ORACLE_TERMINAL_ELIGIBILITY_RULE,
        "eligible_task_count_by_name_token_count": {
            str(count): len(ids) for count, ids in eligible_by_stratum.items()
        },
        "eligible_ids_sha256_by_name_token_count": {
            str(count): _ids_sha256(ids) for count, ids in eligible_by_stratum.items()
        },
        "selection_order": "sha256_within_tokenizer_stratum",
        "selected_task_count": ORACLE_TERMINAL_SELECTED_COUNT,
        "selected_task_count_per_stratum": ORACLE_TERMINAL_SELECTED_PER_STRATUM,
        "selected_task_ids_by_name_token_count": selected_by_stratum,
        "sampled_ids_disjoint_from_exclusions": candidate_manifest.get(
            "sampled_ids_disjoint_from_exclusions"
        )
        is True,
        "include_entry_point_in_prompt": True,
        "entry_point_resolution": candidate_manifest["entry_point_resolution"],
        "target_model": candidate_manifest["target_model"],
        "target_model_revision": candidate_manifest["target_model_revision"],
        "prompt_protocol": candidate_manifest["prompt_protocol"],
        "tasks_jsonl": str(selected_tasks_path),
        "tasks_jsonl_sha256": sha256_path(selected_tasks_path),
        "sampling_config": str(config_path),
        "sampling_config_sha256": sha256_path(config_path),
        "mock_data": False,
    }


def select_tasks(
    config: dict[str, Any],
    config_path: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    validate_config(config)
    data = config["data"]
    output = config["output"]
    candidates, candidate_manifest, candidate_manifest_path = bind_tasks_to_manifest(
        candidate_binding_config(config),
        load_tasks(Path(str(data["candidate_tasks_jsonl"]))),
    )
    generations_path = Path(str(output["screening_generations_jsonl"]))
    summary_path = Path(str(output["screening_evaluation_dir"])) / "summary.json"
    scored_path = (
        Path(str(output["screening_evaluation_dir"]))
        / "scored_generations.jsonl"
    )
    summary = load_json_object(summary_path)
    _validate_screening_summary(
        summary,
        config,
        config_path=config_path,
        generations_path=generations_path,
    )
    eligible_by_stratum = eligible_task_ids(read_jsonl(scored_path), candidates)
    selected_ids_by_stratum = {
        count: ids[:ORACLE_TERMINAL_SELECTED_PER_STRATUM]
        for count, ids in eligible_by_stratum.items()
    }
    passed = all(
        len(ids) == ORACLE_TERMINAL_SELECTED_PER_STRATUM
        for ids in selected_ids_by_stratum.values()
    )
    selected_ids = [
        *selected_ids_by_stratum[2],
        *selected_ids_by_stratum[3],
    ]
    report_path = Path(str(output["selection_report_json"]))
    tasks_path = Path(str(data["tasks_jsonl"]))
    manifest_path = Path(str(data["task_manifest"]))
    for path in (report_path, tasks_path, manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"selection output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment_id": ORACLE_TERMINAL_EXPERIMENT_ID,
        "protocol_version": ORACLE_TERMINAL_PROTOCOL_VERSION,
        "design_sha256": design_fingerprint(config),
        "selection_kind": "terminal_layout_stratified_capability_confirmation",
        "claim_eligible": False,
        "candidate_manifest": str(candidate_manifest_path),
        "candidate_manifest_sha256": sha256_path(candidate_manifest_path),
        "screening_summary": str(summary_path),
        "screening_summary_sha256": sha256_path(summary_path),
        "screening_scored_jsonl": str(scored_path),
        "screening_scored_jsonl_sha256": sha256_path(scored_path),
        "eligibility_rule": ORACLE_TERMINAL_ELIGIBILITY_RULE,
        "eligible_task_ids_by_name_token_count": eligible_by_stratum,
        "eligible_task_count_by_name_token_count": {
            str(count): len(ids) for count, ids in eligible_by_stratum.items()
        },
        "selected_task_ids_by_name_token_count": selected_ids_by_stratum,
        "selected_task_ids": selected_ids,
        "selection_order": "sha256_within_tokenizer_stratum",
        "passed": passed,
    }
    if not passed:
        write_json(report_path, report)
        counts = {count: len(ids) for count, ids in eligible_by_stratum.items()}
        raise RuntimeError(
            "terminal strata lack 16 text-capable tasks each: " + str(counts)
        )

    by_id = {str(task["task_id"]): task for task in candidates}
    selected_tasks = [by_id[task_id] for task_id in selected_ids]
    if [validate_terminal_layout(task["terminal_layout"]) for task in selected_tasks] != (
        [2] * ORACLE_TERMINAL_SELECTED_PER_STRATUM
        + [3] * ORACLE_TERMINAL_SELECTED_PER_STRATUM
    ):
        raise RuntimeError("selected task order is not balanced by tokenizer stratum")
    write_jsonl(tasks_path, selected_tasks)
    manifest = _selected_manifest(
        config=config,
        config_path=config_path,
        selected_tasks=selected_tasks,
        selected_tasks_path=tasks_path,
        candidate_manifest=candidate_manifest,
        candidate_manifest_path=candidate_manifest_path,
        screening_summary_path=summary_path,
        scored_path=scored_path,
        eligible_by_stratum=eligible_by_stratum,
    )
    write_json(manifest_path, manifest)
    report.update(
        {
            "selected_tasks_jsonl": str(tasks_path),
            "selected_tasks_jsonl_sha256": sha256_path(tasks_path),
            "selected_task_manifest": str(manifest_path),
            "selected_task_manifest_sha256": sha256_path(manifest_path),
        }
    )
    write_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    report = select_tasks(config, args.config, overwrite=args.overwrite)
    print("Terminal-layout confirmation selection completed")
    print(f"selected: {len(report['selected_task_ids'])}")
    print(f"report: {config['output']['selection_report_json']}")


if __name__ == "__main__":
    main()
