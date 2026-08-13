"""Open the sealed LIP-PROTO-014 confirmation cohort after its dev gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.evaluation.oracle_terminal_factorial import (
    ORACLE_TERMINAL_SELECTED_PER_STRATUM,
    candidate_binding_config,
    eligible_task_ids,
    validate_selected_task_manifest,
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
from src.pipelines.packet_extraction import load_bound_packet_tasks
from src.scripts.evaluate_oracle_packet_semantics import read_jsonl


DEFAULT_CONFIG = Path("config/LIP-PROTO-014_source_conditioned_residual_packet.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--matrix-summary", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _ids_sha256(task_ids) -> str:
    payload = json.dumps(list(task_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_matrix_gate(
    summary: dict,
    *,
    config_path: Path,
) -> None:
    checks = {
        "experiment": summary.get("experiment_id") == "LIP-PROTO-014",
        "contract": summary.get("contract_config_sha256") == sha256_path(config_path),
        "full_matrix": summary.get("full_registered_matrix") is True,
        "ready": summary.get("ready_for_confirmation") is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "confirmation remains sealed because the matrix gate failed: "
            + ", ".join(failed)
        )


def select_confirmation_tasks(
    config_path: Path | str,
    matrix_summary_path: Path | str,
    *,
    overwrite: bool = False,
) -> dict:
    config_path = Path(config_path)
    matrix_summary_path = Path(matrix_summary_path)
    config = load_yaml(config_path)
    confirmation = config["confirmation"]
    matrix_summary = load_json_object(matrix_summary_path)
    _validate_matrix_gate(matrix_summary, config_path=config_path)

    predecessor_config_path = Path(str(confirmation["predecessor_config"]))
    predecessor_config = load_yaml(predecessor_config_path)
    candidate_tasks, candidate_manifest, candidate_manifest_path = (
        bind_tasks_to_manifest(
            candidate_binding_config(predecessor_config),
            load_tasks(Path(str(confirmation["candidate_tasks_jsonl"]))),
        )
    )
    predecessor_selected_path = Path(
        str(confirmation["predecessor_selected_manifest"])
    )
    predecessor_selected = load_json_object(predecessor_selected_path)
    validate_selected_task_manifest(
        predecessor_config,
        predecessor_selected,
        predecessor_selected_path,
    )
    eligible = eligible_task_ids(
        read_jsonl(Path(str(confirmation["screening_scored_jsonl"]))),
        candidate_tasks,
    )

    start, stop = [
        int(value) for value in confirmation["rank_slice_within_capable_stratum"]
    ]
    if start != ORACLE_TERMINAL_SELECTED_PER_STRATUM or stop - start != 16:
        raise ValueError("confirmation rank slice must remain [16, 32]")
    predecessor_by_stratum = predecessor_selected.get(
        "selected_task_ids_by_name_token_count", {}
    )
    for count in (2, 3):
        predecessor_ids = [
            str(task_id) for task_id in predecessor_by_stratum.get(str(count), [])
        ]
        if predecessor_ids != eligible[count][:start]:
            raise ValueError(
                f"LIP-PROTO-013 selected IDs no longer equal eligible prefix for {count}"
            )
        if len(eligible[count]) < stop:
            raise RuntimeError(
                f"tokenizer stratum {count} has fewer than {stop} capable tasks"
            )

    selected_by_stratum = {
        count: eligible[count][start:stop] for count in (2, 3)
    }
    selected_ids = [*selected_by_stratum[2], *selected_by_stratum[3]]
    if len(selected_ids) != int(confirmation["task_count"]):
        raise RuntimeError("confirmation task count differs from the frozen contract")
    predecessor_ids = {str(value) for value in predecessor_selected["sampled_ids"]}
    if predecessor_ids.intersection(selected_ids):
        raise RuntimeError("confirmation tasks overlap LIP-PROTO-013 tasks")

    training_tasks, training_manifest, training_manifest_path = (
        load_bound_packet_tasks(config)
    )
    training_ids = {str(task["task_id"]) for task in training_tasks}
    if training_ids.intersection(selected_ids):
        raise RuntimeError("confirmation tasks overlap bridge train/development tasks")

    by_id = {str(task["task_id"]): task for task in candidate_tasks}
    selected_tasks = [by_id[task_id] for task_id in selected_ids]
    observed_strata = [
        validate_terminal_layout(task["terminal_layout"])
        for task in selected_tasks
    ]
    if observed_strata != [2] * 16 + [3] * 16:
        raise RuntimeError("confirmation task order is not tokenizer-balanced")

    tasks_path = Path(str(confirmation["tasks_jsonl"]))
    manifest_path = Path(str(confirmation["task_manifest"]))
    report_path = Path(str(confirmation["selection_report"]))
    for path in (tasks_path, manifest_path, report_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"confirmation output already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(tasks_path, selected_tasks)
    manifest = {
        "manifest_kind": "lip_packet_confirmation_task_manifest",
        "schema_version": 1,
        "experiment_id": "LIP-PROTO-014",
        "selection_kind": "sealed_capable_rank_slice_by_tokenizer_stratum",
        "task_count": len(selected_tasks),
        "sampled_ids": selected_ids,
        "sampled_prompt_sha256": [
            prompt_sha256(str(task["prompt"])) for task in selected_tasks
        ],
        "sampled_task_sha256": [task_sha256(task) for task in selected_tasks],
        "selected_task_ids_by_name_token_count": {
            str(count): selected_by_stratum[count] for count in (2, 3)
        },
        "rank_slice_within_capable_stratum": [start, stop],
        "candidate_manifest": str(candidate_manifest_path),
        "candidate_manifest_sha256": sha256_path(candidate_manifest_path),
        "screening_scored_jsonl": str(confirmation["screening_scored_jsonl"]),
        "screening_scored_jsonl_sha256": sha256_path(
            Path(str(confirmation["screening_scored_jsonl"]))
        ),
        "predecessor_selected_manifest": str(predecessor_selected_path),
        "predecessor_selected_manifest_sha256": sha256_path(
            predecessor_selected_path
        ),
        "predecessor_selected_ids_sha256": _ids_sha256(
            predecessor_selected["sampled_ids"]
        ),
        "training_registry_manifest": str(training_manifest_path),
        "training_registry_manifest_sha256": sha256_path(training_manifest_path),
        "training_task_keys_sha256": training_manifest["task_keys_sha256"],
        "matrix_summary": str(matrix_summary_path),
        "matrix_summary_sha256": sha256_path(matrix_summary_path),
        "contract_config": str(config_path),
        "contract_config_sha256": sha256_path(config_path),
        "tasks_jsonl": str(tasks_path),
        "tasks_jsonl_sha256": sha256_path(tasks_path),
        "target_model": candidate_manifest["target_model"],
        "target_model_revision": candidate_manifest["target_model_revision"],
        "prompt_protocol": candidate_manifest["prompt_protocol"],
        "mock_data": False,
    }
    write_json(manifest_path, manifest)
    report = {
        "experiment_id": "LIP-PROTO-014",
        "claim_eligible": False,
        "matrix_gate_passed": True,
        "selected_task_count": len(selected_tasks),
        "selected_task_ids_by_name_token_count": manifest[
            "selected_task_ids_by_name_token_count"
        ],
        "selected_ids_sha256": _ids_sha256(selected_ids),
        "disjoint_from_predecessor": True,
        "disjoint_from_training_and_development": True,
        "task_manifest": str(manifest_path),
        "task_manifest_sha256": sha256_path(manifest_path),
        "passed": True,
    }
    write_json(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    report = select_confirmation_tasks(
        args.config,
        args.matrix_summary,
        overwrite=args.overwrite,
    )
    print("LIP packet confirmation selection passed")
    print(f"selected: {report['selected_task_count']}")
    print(f"manifest: {report['task_manifest']}")


if __name__ == "__main__":
    main()
