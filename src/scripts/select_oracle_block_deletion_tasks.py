"""Select the final latent-unseen holdout for LIP-PROTO-012."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.oracle_block_deletion import (
    ORACLE_DELETION_CALIBRATION_EXPERIMENT,
    ORACLE_DELETION_ELIGIBLE_START,
    ORACLE_DELETION_ELIGIBLE_STOP,
    ORACLE_DELETION_EXPERIMENT_ID,
    ORACLE_DELETION_PREDECESSOR,
    ORACLE_DELETION_PROTOCOL_VERSION,
    ORACLE_DELETION_SELECTED_COUNT,
    design_fingerprint,
    select_eligible_holdout,
)
from src.evaluation.oracle_capability_calibration import (
    ORACLE_CAPABILITY_ELIGIBILITY_RULE,
    ORACLE_CAPABILITY_SCREENING_SEEDS,
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


DEFAULT_CONFIG = Path("config/LIP-PROTO-012_block_deletion.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _ids_sha256(task_ids: Sequence[str]) -> str:
    canonical = json.dumps(list(task_ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validated_paths(
    source: Mapping[str, Any],
    fields: Sequence[str],
    *,
    label: str,
) -> dict[str, Path]:
    paths = {}
    failed = []
    for field in fields:
        value = source.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label}.{field} must be a path")
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[field] = path
        if sha256_path(path) != source.get(f"{field}_sha256"):
            failed.append(field)
    if failed:
        raise ValueError(f"{label} hash mismatch: " + ", ".join(sorted(failed)))
    return paths


def validate_source_artifacts(
    config: Mapping[str, Any],
) -> tuple[dict[str, Path], dict[str, Path]]:
    calibration = config.get("calibration_source")
    predecessor = config.get("predecessor_source")
    if not isinstance(calibration, Mapping) or not isinstance(
        predecessor, Mapping
    ):
        raise ValueError("calibration_source and predecessor_source must be mappings")
    calibration_paths = _validated_paths(
        calibration,
        (
            "candidate_tasks_jsonl",
            "candidate_task_manifest",
            "screening_scored_jsonl",
            "screening_summary",
            "selection_report",
        ),
        label="calibration_source",
    )
    predecessor_paths = _validated_paths(
        predecessor,
        (
            "selected_tasks_jsonl",
            "selected_task_manifest",
            "selection_report",
            "functional_summary",
        ),
        label="predecessor_source",
    )
    return calibration_paths, predecessor_paths


def _candidate_binding_config(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    return {
        **config,
        "experiment_id": ORACLE_DELETION_CALIBRATION_EXPERIMENT,
        "data": {
            "tasks_jsonl": str(paths["candidate_tasks_jsonl"]),
            "task_manifest": str(paths["candidate_task_manifest"]),
            "task_count": 192,
        },
    }


def _validate_prior_partitions(
    *,
    config: Mapping[str, Any],
    calibration_paths: Mapping[str, Path],
    predecessor_paths: Mapping[str, Path],
    eligible_ids: list[str],
) -> list[str]:
    calibration = config["calibration_source"]
    source_report = load_json_object(calibration_paths["selection_report"])
    first_ids = [str(task_id) for task_id in source_report.get("selected_task_ids", [])]
    predecessor_manifest = load_json_object(
        predecessor_paths["selected_task_manifest"]
    )
    predecessor_report = load_json_object(predecessor_paths["selection_report"])
    predecessor_summary = load_json_object(predecessor_paths["functional_summary"])
    predecessor_tasks = load_tasks(predecessor_paths["selected_tasks_jsonl"])
    second_ids = [
        str(task_id) for task_id in predecessor_manifest.get("sampled_ids", [])
    ]
    full_replication = (
        predecessor_summary.get("semantic_gate", {})
        .get("checks", {})
        .get("full_k32_replication_confirmed")
    )
    checks = {
        "eligible_count": len(eligible_ids) == calibration.get("eligible_task_count"),
        "eligible_hash": _ids_sha256(eligible_ids)
        == calibration.get("eligible_ids_sha256"),
        "calibration_experiment": source_report.get("experiment_id")
        == ORACLE_DELETION_CALIBRATION_EXPERIMENT,
        "calibration_passed": source_report.get("passed") is True,
        "first_slice": first_ids == eligible_ids[:32],
        "predecessor_manifest_experiment": predecessor_manifest.get("experiment_id")
        == ORACLE_DELETION_PREDECESSOR,
        "predecessor_manifest_slice": second_ids == eligible_ids[32:64],
        "predecessor_task_file": [
            str(task["task_id"]) for task in predecessor_tasks
        ]
        == second_ids,
        "predecessor_report_experiment": predecessor_report.get("experiment_id")
        == ORACLE_DELETION_PREDECESSOR,
        "predecessor_report_passed": predecessor_report.get("passed") is True,
        "predecessor_report_slice": predecessor_report.get("selected_task_ids")
        == second_ids,
        "predecessor_report_first_slice": predecessor_report.get(
            "predecessor_selected_task_ids"
        )
        == first_ids,
        "predecessor_summary_experiment": predecessor_summary.get("experiment_id")
        == ORACLE_DELETION_PREDECESSOR,
        "predecessor_claim_eligible": predecessor_summary.get("claim_eligible")
        is True,
        "predecessor_full_replication": full_replication is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "prior latent partition provenance failed: " + ", ".join(failed)
        )
    prior_ids = first_ids + second_ids
    if len(prior_ids) != 64 or len(set(prior_ids)) != 64:
        raise ValueError("prior selections must form 64 unique tasks")
    return prior_ids


def _selected_manifest(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    calibration_paths: Mapping[str, Path],
    predecessor_paths: Mapping[str, Path],
    candidate_manifest: Mapping[str, Any],
    selected_tasks: list[Mapping[str, Any]],
    selected_tasks_path: Path,
    eligible_ids: list[str],
    prior_ids: list[str],
) -> dict[str, Any]:
    calibration = config["calibration_source"]
    predecessor = config["predecessor_source"]
    task_ids = [str(task["task_id"]) for task in selected_tasks]
    return {
        "manifest_kind": "lip_oracle_task_manifest",
        "schema_version": 1,
        "experiment_id": ORACLE_DELETION_EXPERIMENT_ID,
        "selection_kind": "capability_calibrated_final_latent_unseen_holdout",
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
        "calibration_source_experiment": ORACLE_DELETION_CALIBRATION_EXPERIMENT,
        "calibration_artifact_manifest_sha256": calibration[
            "artifact_manifest_sha256"
        ],
        "predecessor_source_experiment": ORACLE_DELETION_PREDECESSOR,
        "predecessor_artifact_manifest_sha256": predecessor[
            "artifact_manifest_sha256"
        ],
        "candidate_manifest": str(calibration_paths["candidate_task_manifest"]),
        "candidate_manifest_sha256": calibration[
            "candidate_task_manifest_sha256"
        ],
        "screening_scored_jsonl": str(
            calibration_paths["screening_scored_jsonl"]
        ),
        "screening_scored_jsonl_sha256": calibration[
            "screening_scored_jsonl_sha256"
        ],
        "screening_summary": str(calibration_paths["screening_summary"]),
        "screening_summary_sha256": calibration["screening_summary_sha256"],
        "calibration_selection_report": str(
            calibration_paths["selection_report"]
        ),
        "calibration_selection_report_sha256": calibration[
            "selection_report_sha256"
        ],
        "predecessor_selected_tasks_jsonl": str(
            predecessor_paths["selected_tasks_jsonl"]
        ),
        "predecessor_selected_tasks_jsonl_sha256": predecessor[
            "selected_tasks_jsonl_sha256"
        ],
        "predecessor_selected_task_manifest": str(
            predecessor_paths["selected_task_manifest"]
        ),
        "predecessor_selected_task_manifest_sha256": predecessor[
            "selected_task_manifest_sha256"
        ],
        "predecessor_selection_report": str(
            predecessor_paths["selection_report"]
        ),
        "predecessor_selection_report_sha256": predecessor[
            "selection_report_sha256"
        ],
        "predecessor_functional_summary": str(
            predecessor_paths["functional_summary"]
        ),
        "predecessor_functional_summary_sha256": predecessor[
            "functional_summary_sha256"
        ],
        "screening_seeds": list(ORACLE_CAPABILITY_SCREENING_SEEDS),
        "eligibility_rule": ORACLE_CAPABILITY_ELIGIBILITY_RULE,
        "eligible_task_count": len(eligible_ids),
        "eligible_ids_sha256": _ids_sha256(eligible_ids),
        "eligible_rank_start_zero_based": ORACLE_DELETION_ELIGIBLE_START,
        "eligible_rank_stop_exclusive": ORACLE_DELETION_ELIGIBLE_STOP,
        "selection_order": "candidate_manifest_order",
        "selected_task_count": ORACLE_DELETION_SELECTED_COUNT,
        "selected_ids_are_registered_holdout_slice": task_ids
        == eligible_ids[
            ORACLE_DELETION_ELIGIBLE_START:ORACLE_DELETION_ELIGIBLE_STOP
        ],
        "prior_selected_task_ids_sha256": _ids_sha256(prior_ids),
        "sampled_ids_disjoint_from_prior_latent": not bool(
            set(task_ids).intersection(prior_ids)
        ),
        "deletion_design": dict(config["deletion_design"]),
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
    calibration_paths, predecessor_paths = validate_source_artifacts(config)
    candidate_tasks = load_tasks(calibration_paths["candidate_tasks_jsonl"])
    bound_candidates, candidate_manifest, _ = bind_tasks_to_manifest(
        _candidate_binding_config(config, calibration_paths), candidate_tasks
    )
    scored = read_jsonl(calibration_paths["screening_scored_jsonl"])
    candidate_ids = [str(task["task_id"]) for task in bound_candidates]
    eligible_ids = eligible_task_ids(scored, candidate_ids)
    prior_ids = _validate_prior_partitions(
        config=config,
        calibration_paths=calibration_paths,
        predecessor_paths=predecessor_paths,
        eligible_ids=eligible_ids,
    )
    selected_ids = select_eligible_holdout(eligible_ids)
    if set(selected_ids).intersection(prior_ids):
        raise RuntimeError("final holdout overlaps an earlier latent experiment")

    selected_by_id = {str(task["task_id"]): task for task in bound_candidates}
    selected_tasks = [selected_by_id[task_id] for task_id in selected_ids]
    data = config["data"]
    output = config["output"]
    tasks_path = Path(str(data["tasks_jsonl"]))
    manifest_path = Path(str(data["task_manifest"]))
    report_path = Path(str(output["selection_report_json"]))
    for path in (tasks_path, manifest_path, report_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"selection output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(tasks_path, selected_tasks)
    manifest = _selected_manifest(
        config=config,
        config_path=config_path,
        calibration_paths=calibration_paths,
        predecessor_paths=predecessor_paths,
        candidate_manifest=candidate_manifest,
        selected_tasks=selected_tasks,
        selected_tasks_path=tasks_path,
        eligible_ids=eligible_ids,
        prior_ids=prior_ids,
    )
    write_json(manifest_path, manifest)
    report = {
        "experiment_id": ORACLE_DELETION_EXPERIMENT_ID,
        "protocol_version": ORACLE_DELETION_PROTOCOL_VERSION,
        "design_sha256": design_fingerprint(config),
        "selection_kind": manifest["selection_kind"],
        "claim_eligible": False,
        "calibration_artifact_manifest_sha256": config["calibration_source"][
            "artifact_manifest_sha256"
        ],
        "predecessor_artifact_manifest_sha256": config["predecessor_source"][
            "artifact_manifest_sha256"
        ],
        "eligible_task_count": len(eligible_ids),
        "eligible_ids_sha256": _ids_sha256(eligible_ids),
        "eligible_rank_start_zero_based": ORACLE_DELETION_ELIGIBLE_START,
        "eligible_rank_stop_exclusive": ORACLE_DELETION_ELIGIBLE_STOP,
        "prior_selected_task_ids": prior_ids,
        "selected_task_ids": selected_ids,
        "selected_tasks_jsonl": str(tasks_path),
        "selected_tasks_jsonl_sha256": sha256_path(tasks_path),
        "selected_task_manifest": str(manifest_path),
        "selected_task_manifest_sha256": sha256_path(manifest_path),
        "deletion_design": dict(config["deletion_design"]),
        "passed": True,
    }
    write_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    report = select_tasks(config, args.config, overwrite=args.overwrite)
    print("Block-deletion holdout selection completed")
    print(f"eligible: {report['eligible_task_count']}")
    print(f"selected: {len(report['selected_task_ids'])}")
    print(f"report: {config['output']['selection_report_json']}")


if __name__ == "__main__":
    main()
