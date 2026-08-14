"""Validate and harden-score the post-hoc LIP-EVAL-034 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.alias_normalized_diagnostic import (
    ALIAS_DIAGNOSTIC_EXPERIMENT_ID,
    ALIAS_DIAGNOSTIC_PROTOCOL_VERSION,
    ALIAS_DIAGNOSTIC_SOURCE_EXPERIMENT_ID,
    ALIAS_DIAGNOSTIC_SOURCE_PROTOCOL_VERSION,
    alias_diagnostic_design_fingerprint,
    build_single_function_alias,
    summarize_alias_diagnostic,
    validate_alias_diagnostic_contract,
)
from src.evaluation.functional_bridge_screen import (
    FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS,
    FUNCTIONAL_BRIDGE_SCREEN_EXPECTED_RECORDS,
    FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS,
    FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS,
    expected_functional_bridge_screen_keys,
)
from src.evaluation.oracle_functional import declares_entry_point
from src.evaluation.semantics import (
    CandidateProcessPolicy,
    check_syntax,
    extract_code,
    run_functional_tests,
)
from src.evaluation.statistics import summarize_metric
from src.pipelines.oracle_experiment import (
    load_json_object,
    load_yaml,
    prepare_output_dir,
    sha256_path,
    write_json,
    write_jsonl,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON constant {value}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line, parse_constant=reject_constant)
            if not isinstance(row, dict):
                raise ValueError(f"row {line_number} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError("EVAL-034 source generation file is empty")
    return rows


def _key(row: Mapping) -> tuple[str, str, int, int]:
    return (
        str(row.get("task_id", "")),
        str(row.get("condition", "")),
        int(row.get("generation_seed", -1)),
        int(row.get("training_seed", -1)),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_alias_diagnostic_source(
    rows: Sequence[Mapping],
    metadata: Mapping,
    config: Mapping,
    generations_path: Path,
    *,
    allow_incomplete: bool,
    security_context: Mapping[str, Any] | None = None,
) -> dict:
    """Bind the diagnostic to the exact completed EVAL-033 artifacts."""

    validate_alias_diagnostic_contract(config)
    source = config["source"]
    metadata_path = generations_path.with_suffix(".metadata.json")
    observed_hashes = {
        "generations": sha256_path(generations_path),
        "metadata": sha256_path(metadata_path),
    }
    expected_hashes = {
        "generations": source["generations_sha256"],
        "metadata": source["metadata_sha256"],
    }
    if observed_hashes != expected_hashes:
        raise ValueError("EVAL-034 source artifact hashes do not match EVAL-033")
    if security_context is not None:
        sandbox_hashes = security_context.get("input_sha256", {})
        if sandbox_hashes.get("generations") != observed_hashes["generations"]:
            raise ValueError("sandbox generation hash is not bound to EVAL-033")
        if sandbox_hashes.get("metadata") != observed_hashes["metadata"]:
            raise ValueError("sandbox metadata hash is not bound to EVAL-033")

    metadata_checks = {
        "experiment": metadata.get("experiment_id")
        == ALIAS_DIAGNOSTIC_SOURCE_EXPERIMENT_ID,
        "protocol": metadata.get("protocol_version")
        == ALIAS_DIAGNOSTIC_SOURCE_PROTOCOL_VERSION,
        "design": metadata.get("design_sha256") == source["design_sha256"],
        "config": metadata.get("config_sha256") == source["config_sha256"],
        "claim": metadata.get("claim_eligible") is False,
        "complete": metadata.get("complete") is True,
        "tasks": metadata.get("task_count") == source["task_count"],
        "records": metadata.get("records") == source["expected_records"],
        "expected_records": metadata.get("expected_records")
        == source["expected_records"],
        "conditions": tuple(metadata.get("conditions", ()))
        == FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS,
        "training_seeds": tuple(metadata.get("training_seeds", ()))
        == FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS,
        "generation_seeds": tuple(metadata.get("generation_seeds", ()))
        == FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS,
    }
    failed = [name for name, passed in metadata_checks.items() if not passed]
    if failed:
        raise ValueError("EVAL-034 source metadata drifted: " + ", ".join(failed))
    if allow_incomplete:
        raise ValueError("EVAL-034 does not permit an incomplete post-hoc source")
    if len(rows) != FUNCTIONAL_BRIDGE_SCREEN_EXPECTED_RECORDS:
        raise ValueError("EVAL-034 requires all 576 EVAL-033 rows")
    task_ids = [str(value) for value in metadata.get("task_ids", ())]
    expected = expected_functional_bridge_screen_keys(task_ids)
    observed = [_key(row) for row in rows]
    if len(set(observed)) != len(observed):
        raise ValueError("EVAL-034 source contains duplicate EVAL-033 rows")
    if set(observed) != expected:
        raise ValueError("EVAL-034 source grid is not the complete EVAL-033 grid")

    exact_entry_point_declarations = 0
    for row in rows:
        task = row.get("task_spec")
        if not isinstance(task, Mapping):
            raise ValueError("EVAL-034 source row has no task_spec")
        entry_point = task.get("entry_point")
        code = extract_code(row.get("output_text"))
        row_checks = {
            "experiment": row.get("experiment_id")
            == ALIAS_DIAGNOSTIC_SOURCE_EXPERIMENT_ID,
            "protocol": row.get("protocol_version")
            == ALIAS_DIAGNOSTIC_SOURCE_PROTOCOL_VERSION,
            "design": row.get("design_sha256") == source["design_sha256"],
            "config": row.get("config_sha256") == source["config_sha256"],
            "claim": row.get("claim_eligible") is False,
            "condition": row.get("condition")
            in FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS,
            "task_identity": str(task.get("task_id", ""))
            == str(row.get("task_id", "")),
            "entry_point": isinstance(entry_point, str) and bool(entry_point.strip()),
        }
        failed_row = [name for name, passed in row_checks.items() if not passed]
        if failed_row:
            raise ValueError(
                f"EVAL-034 source row {_key(row)} drifted: "
                + ", ".join(failed_row)
            )
        exact_entry_point_declarations += int(
            declares_entry_point(code, str(entry_point))
        )
    if exact_entry_point_declarations != int(
        source["exact_entry_point_declarations"]
    ):
        raise ValueError("EVAL-033 exact entry-point count changed")
    return {
        "source_experiment_id": ALIAS_DIAGNOSTIC_SOURCE_EXPERIMENT_ID,
        "complete": True,
        "claim_eligible": False,
        "task_count": len(task_ids),
        "record_count": len(rows),
        "exact_entry_point_declarations": exact_entry_point_declarations,
        "input_sha256": observed_hashes,
    }


def evaluate(
    config: dict[str, Any],
    generations_path: Path,
    output_dir: Path,
    *,
    functional: bool,
    allow_incomplete: bool,
    overwrite: bool,
    candidate_process_policy: CandidateProcessPolicy | None = None,
    security_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = load_json_object(generations_path.with_suffix(".metadata.json"))
    rows = _read_jsonl(generations_path)
    source_validation = validate_alias_diagnostic_source(
        rows,
        metadata,
        config,
        generations_path,
        allow_incomplete=allow_incomplete,
        security_context=security_context,
    )
    prepare_output_dir(output_dir, overwrite=overwrite)
    scored = []
    reasons: Counter[str] = Counter()
    for row in rows:
        code = extract_code(row["output_text"])
        syntax = check_syntax(code)
        entry_point = str(row["task_spec"]["entry_point"])
        alias = build_single_function_alias(code, entry_point)
        reasons[alias["reason"]] += 1
        scored_row = dict(row)
        scored_row.update(
            {
                "diagnostic_experiment_id": ALIAS_DIAGNOSTIC_EXPERIMENT_ID,
                "diagnostic_protocol_version": ALIAS_DIAGNOSTIC_PROTOCOL_VERSION,
                "diagnostic_design_sha256": alias_diagnostic_design_fingerprint(
                    config
                ),
                "diagnostic_claim_eligible": False,
                "extracted_code": code,
                **syntax,
                "entry_point_declared_original": declares_entry_point(
                    code, entry_point
                ),
                "alias_eligible": alias["eligible"],
                "alias_reason": alias["reason"],
                "top_level_function_count": alias["top_level_function_count"],
                "top_level_function_names": alias["top_level_function_names"],
                "generated_function_name": alias["generated_function_name"],
                "alias_binding_applied": alias["alias_binding_applied"],
                "alias_normalized_code": alias["normalized_code"],
                "alias_normalized_code_sha256": (
                    _sha256_text(alias["normalized_code"])
                    if alias["normalized_code"] is not None
                    else None
                ),
            }
        )
        if functional and alias["eligible"]:
            functional_result = run_functional_tests(
                alias["normalized_code"],
                row["task_spec"],
                timeout_seconds=5.0,
                memory_mb=512,
                process_policy=candidate_process_policy,
            )
            scored_row["alias_functional_pass"] = functional_result[
                "functional_pass"
            ]
            scored_row["alias_functional_error_type"] = functional_result[
                "functional_error_type"
            ]
            scored_row["alias_functional_error"] = functional_result[
                "functional_error"
            ]
        elif functional:
            scored_row["alias_functional_pass"] = False
            scored_row["alias_functional_error_type"] = "AliasIneligible"
            scored_row["alias_functional_error"] = alias["reason"]
        else:
            scored_row["alias_functional_pass"] = None
            scored_row["alias_functional_error_type"] = None
            scored_row["alias_functional_error"] = None
        scored.append(scored_row)

    policy = config["evaluation"]
    statistics_kwargs = {
        "bootstrap_iterations": int(policy["bootstrap_iterations"]),
        "confidence": float(policy["confidence"]),
        "seed": int(policy["statistics_seed"]),
    }
    descriptive = {
        "syntax_pass": summarize_metric(
            scored,
            "syntax_pass",
            FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS,
            [("learned_matched", "learned_shuffled")],
            **statistics_kwargs,
        ),
        "alias_eligible": summarize_metric(
            scored,
            "alias_eligible",
            FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS,
            [("learned_matched", "learned_shuffled")],
            **statistics_kwargs,
        ),
    }
    sandbox_validated = bool(security_context and security_context.get("validated"))
    inference = None
    if functional:
        descriptive["alias_functional_pass"] = summarize_metric(
            scored,
            "alias_functional_pass",
            FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS,
            [("learned_matched", "learned_shuffled")],
            **statistics_kwargs,
        )
        inference = summarize_alias_diagnostic(scored, config)
    diagnostic_route = (
        inference["diagnostic_route"]
        if functional and sandbox_validated and inference
        else "not_scored_in_hardened_namespace"
    )
    summary = {
        "experiment_id": ALIAS_DIAGNOSTIC_EXPERIMENT_ID,
        "protocol_version": ALIAS_DIAGNOSTIC_PROTOCOL_VERSION,
        "design_sha256": alias_diagnostic_design_fingerprint(config),
        "execution_mode": (
            "functional_hardened_namespace"
            if functional and sandbox_validated
            else "functional_subprocess"
            if functional
            else "syntax_only"
        ),
        "claim_eligible": False,
        "can_upgrade_EVAL_033": False,
        "subprocess_is_security_sandbox": (
            sandbox_validated if functional else None
        ),
        "diagnostic_route": diagnostic_route,
        "source_validation": source_validation,
        "normalization_reason_counts": dict(sorted(reasons.items())),
        "inference": inference,
        "metrics": descriptive,
        "artifact_provenance": {
            "source_drive_root": config["source"]["drive_root"],
            "source_generations_sha256": config["source"][
                "generations_sha256"
            ],
            "source_metadata_sha256": config["source"]["metadata_sha256"],
            "source_functional_summary_sha256": config["source"][
                "functional_summary_sha256"
            ],
            "source_run_commit": config["source"]["run_commit"],
        },
    }
    if security_context is not None:
        summary["sandbox"] = dict(security_context)
    scored_path = output_dir / "scored_generations.jsonl"
    write_jsonl(scored_path, scored)
    summary["scored_jsonl"] = str(scored_path)
    summary["scored_jsonl_sha256"] = sha256_path(scored_path)
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/LIP-EVAL-034_alias_normalized_functional_diagnostic.yaml"
        ),
    )
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--functional", action="store_true")
    parser.add_argument("--allow-unsafe-execution", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.functional and not args.allow_unsafe_execution:
        raise RuntimeError("use the hardened namespace runner for functional scoring")
    summary = evaluate(
        load_yaml(args.config),
        args.generations,
        args.output_dir,
        functional=args.functional,
        allow_incomplete=args.allow_incomplete,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "execution_mode": summary["execution_mode"],
                "claim_eligible": summary["claim_eligible"],
                "diagnostic_route": summary["diagnostic_route"],
            }
        )
    )


if __name__ == "__main__":
    main()
