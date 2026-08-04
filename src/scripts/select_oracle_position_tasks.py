"""Select the latent-unseen LIP-PROTO-011 holdout from the sealed 010 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.evaluation.oracle_capability_calibration import (
    ORACLE_CAPABILITY_ELIGIBILITY_RULE,
    ORACLE_CAPABILITY_SCREENING_SEEDS,
    eligible_task_ids,
)
from src.evaluation.oracle_position_packet import (
    ORACLE_POSITION_ELIGIBLE_START,
    ORACLE_POSITION_ELIGIBLE_STOP,
    ORACLE_POSITION_EXPERIMENT_ID,
    ORACLE_POSITION_PREDECESSOR,
    ORACLE_POSITION_PROTOCOL_VERSION,
    ORACLE_POSITION_SELECTED_COUNT,
    derive_position_patterns,
    design_fingerprint,
    select_eligible_holdout,
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


DEFAULT_CONFIG = Path("config/LIP-PROTO-011_position_sparse_packet.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _ids_sha256(task_ids: list[str]) -> str:
    canonical = json.dumps(task_ids, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_path(source: Mapping[str, Any], field: str) -> Path:
    value = source.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"calibration_source.{field} must be a path")
    return Path(value)


def validate_source_artifacts(config: Mapping[str, Any]) -> dict[str, Path]:
    """Bind every reused input to the sealed LIP-PROTO-010 payload hashes."""

    source = config.get("calibration_source")
    if not isinstance(source, Mapping):
        raise ValueError("calibration_source must be a mapping")
    file_fields = (
        "candidate_tasks_jsonl",
        "candidate_task_manifest",
        "screening_scored_jsonl",
        "screening_summary",
        "selection_report",
        "state_diagnostics",
    )
    paths = {field: _source_path(source, field) for field in file_fields}
    failed = []
    for field, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = str(source.get(f"{field}_sha256", ""))
        if sha256_path(path) != expected:
            failed.append(field)
    if failed:
        raise ValueError(
            "calibration source hash mismatch: " + ", ".join(sorted(failed))
        )
    return paths


def _source_binding_config(
    config: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    return {
        **config,
        "experiment_id": ORACLE_POSITION_PREDECESSOR,
        "data": {
            "tasks_jsonl": str(paths["candidate_tasks_jsonl"]),
            "task_manifest": str(paths["candidate_task_manifest"]),
            "task_count": 192,
        },
    }


def _validate_predecessor_selection(
    source_report: Mapping[str, Any],
    source: Mapping[str, Any],
    eligible_ids: list[str],
) -> list[str]:
    predecessor_ids = [
        str(task_id) for task_id in source_report.get("selected_task_ids", [])
    ]
    checks = {
        "experiment": source_report.get("experiment_id")
        == ORACLE_POSITION_PREDECESSOR,
        "passed": source_report.get("passed") is True,
        "eligible_count": source_report.get("eligible_task_count")
        == source.get("eligible_task_count"),
        "eligible_ids": source_report.get("eligible_task_ids") == eligible_ids,
        "eligible_hash": source_report.get("eligible_ids_sha256")
        == source.get("eligible_ids_sha256")
        == _ids_sha256(eligible_ids),
        "selected_prefix": predecessor_ids == eligible_ids[:32],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "LIP-PROTO-010 selection provenance failed: " + ", ".join(failed)
        )
    return predecessor_ids


def _selected_manifest(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    source_paths: Mapping[str, Path],
    candidate_manifest: Mapping[str, Any],
    selected_tasks: list[Mapping[str, Any]],
    selected_tasks_path: Path,
    eligible_ids: list[str],
    predecessor_ids: list[str],
    derived_patterns: Mapping[str, list[int]],
) -> dict[str, Any]:
    source = config["calibration_source"]
    task_ids = [str(task["task_id"]) for task in selected_tasks]
    return {
        "manifest_kind": "lip_oracle_task_manifest",
        "schema_version": 1,
        "experiment_id": ORACLE_POSITION_EXPERIMENT_ID,
        "selection_kind": "capability_calibrated_latent_unseen_holdout",
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
        "calibration_source_experiment": ORACLE_POSITION_PREDECESSOR,
        "calibration_artifact_manifest_sha256": source[
            "artifact_manifest_sha256"
        ],
        "candidate_manifest": str(source_paths["candidate_task_manifest"]),
        "candidate_manifest_sha256": source["candidate_task_manifest_sha256"],
        "screening_scored_jsonl": str(source_paths["screening_scored_jsonl"]),
        "screening_scored_jsonl_sha256": source[
            "screening_scored_jsonl_sha256"
        ],
        "screening_summary": str(source_paths["screening_summary"]),
        "screening_summary_sha256": source["screening_summary_sha256"],
        "predecessor_selection_report": str(source_paths["selection_report"]),
        "predecessor_selection_report_sha256": source[
            "selection_report_sha256"
        ],
        "source_state_diagnostics": str(source_paths["state_diagnostics"]),
        "source_state_diagnostics_sha256": source[
            "state_diagnostics_sha256"
        ],
        "screening_seeds": list(ORACLE_CAPABILITY_SCREENING_SEEDS),
        "eligibility_rule": ORACLE_CAPABILITY_ELIGIBILITY_RULE,
        "eligible_task_count": len(eligible_ids),
        "eligible_ids_sha256": _ids_sha256(eligible_ids),
        "eligible_rank_start_zero_based": ORACLE_POSITION_ELIGIBLE_START,
        "eligible_rank_stop_exclusive": ORACLE_POSITION_ELIGIBLE_STOP,
        "selection_order": "candidate_manifest_order",
        "selected_task_count": ORACLE_POSITION_SELECTED_COUNT,
        "selected_ids_are_registered_holdout_slice": task_ids
        == eligible_ids[
            ORACLE_POSITION_ELIGIBLE_START:ORACLE_POSITION_ELIGIBLE_STOP
        ],
        "sampled_ids_disjoint_from_predecessor_selection": not bool(
            set(task_ids).intersection(predecessor_ids)
        ),
        "position_selection": dict(config["position_selection"]),
        "derived_position_patterns": dict(derived_patterns),
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
    source_paths = validate_source_artifacts(config)
    source = config["calibration_source"]
    candidate_tasks = load_tasks(source_paths["candidate_tasks_jsonl"])
    bound_candidates, candidate_manifest, _ = bind_tasks_to_manifest(
        _source_binding_config(config, source_paths), candidate_tasks
    )
    scored = read_jsonl(source_paths["screening_scored_jsonl"])
    candidate_ids = [str(task["task_id"]) for task in bound_candidates]
    eligible_ids = eligible_task_ids(scored, candidate_ids)
    source_report = load_json_object(source_paths["selection_report"])
    predecessor_ids = _validate_predecessor_selection(
        source_report,
        source,
        eligible_ids,
    )
    selected_ids = select_eligible_holdout(eligible_ids)
    if set(selected_ids).intersection(predecessor_ids):
        raise RuntimeError("registered holdout overlaps LIP-PROTO-010 confirmation")

    diagnostics = load_json_object(source_paths["state_diagnostics"])
    derived_patterns = derive_position_patterns(diagnostics)
    expected_patterns = {
        "diagnostic_top_k8": list(
            config["position_selection"]["expected_diagnostic_top_offsets"]
        ),
        "peak_window_k8": list(
            config["position_selection"]["expected_peak_window_offsets"]
        ),
    }
    if derived_patterns != expected_patterns:
        raise ValueError("source diagnostics do not reproduce frozen position choices")

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
        source_paths=source_paths,
        candidate_manifest=candidate_manifest,
        selected_tasks=selected_tasks,
        selected_tasks_path=tasks_path,
        eligible_ids=eligible_ids,
        predecessor_ids=predecessor_ids,
        derived_patterns=derived_patterns,
    )
    write_json(manifest_path, manifest)
    report = {
        "experiment_id": ORACLE_POSITION_EXPERIMENT_ID,
        "protocol_version": ORACLE_POSITION_PROTOCOL_VERSION,
        "design_sha256": design_fingerprint(config),
        "selection_kind": manifest["selection_kind"],
        "claim_eligible": False,
        "calibration_artifact_manifest_sha256": source[
            "artifact_manifest_sha256"
        ],
        "eligible_task_count": len(eligible_ids),
        "eligible_ids_sha256": _ids_sha256(eligible_ids),
        "eligible_rank_start_zero_based": ORACLE_POSITION_ELIGIBLE_START,
        "eligible_rank_stop_exclusive": ORACLE_POSITION_ELIGIBLE_STOP,
        "predecessor_selected_task_ids": predecessor_ids,
        "selected_task_ids": selected_ids,
        "selected_tasks_jsonl": str(tasks_path),
        "selected_tasks_jsonl_sha256": sha256_path(tasks_path),
        "selected_task_manifest": str(manifest_path),
        "selected_task_manifest_sha256": sha256_path(manifest_path),
        "derived_position_patterns": derived_patterns,
        "passed": True,
    }
    write_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    report = select_tasks(config, args.config, overwrite=args.overwrite)
    print("Position-sparse holdout selection completed")
    print(f"eligible: {report['eligible_task_count']}")
    print(f"selected: {len(report['selected_task_ids'])}")
    print(f"report: {config['output']['selection_report_json']}")


if __name__ == "__main__":
    main()
