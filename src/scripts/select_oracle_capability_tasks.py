"""Select the frozen LIP-PROTO-010 confirmation tasks from hardened screening."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.evaluation.oracle_capability_calibration import (
    ORACLE_CAPABILITY_ELIGIBILITY_RULE,
    ORACLE_CAPABILITY_EXPERIMENT_ID,
    ORACLE_CAPABILITY_PROTOCOL_VERSION,
    ORACLE_CAPABILITY_SCREENING_SCOPE,
    ORACLE_CAPABILITY_SCREENING_SEEDS,
    ORACLE_CAPABILITY_SELECTED_COUNT,
    candidate_binding_config,
    design_fingerprint,
    eligible_task_ids,
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


DEFAULT_CONFIG = Path("config/LIP-PROTO-010_capability_calibrated_depth.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _ids_sha256(task_ids: list[str]) -> str:
    canonical = json.dumps(task_ids, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_screening_summary(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    config_path: Path,
    generations_path: Path,
) -> None:
    design_validation = summary.get("design_validation", {})
    sandbox = summary.get("sandbox", {})
    input_hashes = sandbox.get("input_sha256", {})
    metadata_path = generations_path.with_suffix(".metadata.json")
    checks = {
        "experiment": summary.get("experiment_id") == ORACLE_CAPABILITY_EXPERIMENT_ID,
        "protocol": summary.get("protocol_version")
        == ORACLE_CAPABILITY_PROTOCOL_VERSION,
        "design": design_validation.get("design_sha256")
        == design_fingerprint(config),
        "scope": design_validation.get("run_scope")
        == ORACLE_CAPABILITY_SCREENING_SCOPE,
        "screening": design_validation.get("screening") is True,
        "complete": design_validation.get("complete") is True,
        "not_claim_eligible": summary.get("claim_eligible") is False,
        "hardened_execution": summary.get("execution_mode")
        == "functional_hardened_namespace",
        "sandbox_validated": sandbox.get("validated") is True,
        "scored_hash": summary.get("scored_jsonl_sha256")
        == sha256_path(
            Path(str(config["output"]["screening_evaluation_dir"]))
            / "scored_generations.jsonl"
        ),
        "config_hash": input_hashes.get("config") == sha256_path(config_path),
        "generations_hash": input_hashes.get("generations")
        == sha256_path(generations_path),
        "metadata_hash": input_hashes.get("metadata") == sha256_path(metadata_path),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "screening summary failed provenance checks: " + ", ".join(failed)
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
    eligible_ids: list[str],
) -> dict[str, Any]:
    task_ids = [str(task["task_id"]) for task in selected_tasks]
    return {
        "manifest_kind": "lip_oracle_task_manifest",
        "schema_version": 1,
        "experiment_id": ORACLE_CAPABILITY_EXPERIMENT_ID,
        "selection_kind": "capability_calibrated_confirmation",
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
        "screening_seeds": list(ORACLE_CAPABILITY_SCREENING_SEEDS),
        "eligibility_rule": ORACLE_CAPABILITY_ELIGIBILITY_RULE,
        "eligible_task_count": len(eligible_ids),
        "eligible_ids_sha256": _ids_sha256(eligible_ids),
        "selection_order": "candidate_manifest_order",
        "selected_task_count": ORACLE_CAPABILITY_SELECTED_COUNT,
        "selected_ids_are_eligible_prefix": task_ids
        == eligible_ids[:ORACLE_CAPABILITY_SELECTED_COUNT],
        "excluded_task_manifests": candidate_manifest.get(
            "excluded_task_manifests", []
        ),
        "excluded_task_count": candidate_manifest.get("excluded_task_count"),
        "excluded_task_ids_sha256": candidate_manifest.get(
            "excluded_task_ids_sha256"
        ),
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
    candidate_tasks_path = Path(str(data["candidate_tasks_jsonl"]))
    candidate_tasks = load_tasks(candidate_tasks_path)
    bound_candidates, candidate_manifest, candidate_manifest_path = (
        bind_tasks_to_manifest(candidate_binding_config(config), candidate_tasks)
    )
    generations_path = Path(str(output["screening_generations_jsonl"]))
    summary_path = Path(str(output["screening_evaluation_dir"])) / "summary.json"
    scored_path = Path(str(output["screening_evaluation_dir"])) / (
        "scored_generations.jsonl"
    )
    summary = load_json_object(summary_path)
    _validate_screening_summary(
        summary,
        config,
        config_path=config_path,
        generations_path=generations_path,
    )
    scored = read_jsonl(scored_path)
    candidate_ids = [str(task["task_id"]) for task in bound_candidates]
    eligible_ids = eligible_task_ids(scored, candidate_ids)
    selected_ids = eligible_ids[:ORACLE_CAPABILITY_SELECTED_COUNT]
    report_path = Path(str(output["selection_report_json"]))
    selected_tasks_path = Path(str(data["tasks_jsonl"]))
    selected_manifest_path = Path(str(data["task_manifest"]))
    for path in (report_path, selected_tasks_path, selected_manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"selection output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    passed = len(selected_ids) == ORACLE_CAPABILITY_SELECTED_COUNT
    report = {
        "experiment_id": ORACLE_CAPABILITY_EXPERIMENT_ID,
        "protocol_version": ORACLE_CAPABILITY_PROTOCOL_VERSION,
        "design_sha256": design_fingerprint(config),
        "selection_kind": "capability_calibrated_confirmation",
        "claim_eligible": False,
        "candidate_manifest": str(candidate_manifest_path),
        "candidate_manifest_sha256": sha256_path(candidate_manifest_path),
        "screening_summary": str(summary_path),
        "screening_summary_sha256": sha256_path(summary_path),
        "screening_scored_jsonl": str(scored_path),
        "screening_scored_jsonl_sha256": sha256_path(scored_path),
        "eligibility_rule": ORACLE_CAPABILITY_ELIGIBILITY_RULE,
        "candidate_task_count": len(candidate_ids),
        "eligible_task_count": len(eligible_ids),
        "eligible_task_ids": eligible_ids,
        "eligible_ids_sha256": _ids_sha256(eligible_ids),
        "requested_task_count": ORACLE_CAPABILITY_SELECTED_COUNT,
        "selected_task_ids": selected_ids,
        "selection_order": "candidate_manifest_order",
        "passed": passed,
    }
    if not passed:
        write_json(report_path, report)
        raise RuntimeError(
            f"only {len(eligible_ids)} candidates passed screening; "
            f"{ORACLE_CAPABILITY_SELECTED_COUNT} were required"
        )

    selected_set = set(selected_ids)
    selected_tasks = [
        task for task in bound_candidates if str(task["task_id"]) in selected_set
    ]
    write_jsonl(selected_tasks_path, selected_tasks)
    manifest = _selected_manifest(
        config=config,
        config_path=config_path,
        selected_tasks=selected_tasks,
        selected_tasks_path=selected_tasks_path,
        candidate_manifest=candidate_manifest,
        candidate_manifest_path=candidate_manifest_path,
        screening_summary_path=summary_path,
        scored_path=scored_path,
        eligible_ids=eligible_ids,
    )
    if manifest["sampled_ids"] != selected_ids:
        raise RuntimeError("selected task order changed while writing the manifest")
    write_json(selected_manifest_path, manifest)
    report.update(
        {
            "selected_tasks_jsonl": str(selected_tasks_path),
            "selected_tasks_jsonl_sha256": sha256_path(selected_tasks_path),
            "selected_task_manifest": str(selected_manifest_path),
            "selected_task_manifest_sha256": sha256_path(selected_manifest_path),
        }
    )
    write_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    report = select_tasks(config, args.config, overwrite=args.overwrite)
    print("Capability-calibrated task selection completed")
    print(f"eligible: {report['eligible_task_count']}")
    print(f"selected: {len(report['selected_task_ids'])}")
    print(f"report: {config['output']['selection_report_json']}")


if __name__ == "__main__":
    main()
