"""Validate and harden-score the sequential LIP-EVAL-037 blend screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.alias_normalized_diagnostic import build_single_function_alias
from src.evaluation.constant_entry_point_screen import declares_top_level_function
from src.evaluation.oracle_functional import stable_seed
from src.evaluation.oracle_native_packet_blend_screen import (
    ORACLE_BLEND_CONDITIONS,
    ORACLE_BLEND_CONFIRMATION_GENERATION_SEEDS,
    ORACLE_BLEND_EXPERIMENT_ID,
    ORACLE_BLEND_PROTOCOL_VERSION,
    ORACLE_BLEND_SCREEN_ALPHAS,
    ORACLE_BLEND_SCREEN_GENERATION_SEED,
    expected_oracle_blend_keys,
    oracle_blend_design_fingerprint,
    summarize_oracle_blend_confirmation,
    summarize_oracle_blend_screen,
    validate_oracle_blend_contract,
)
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
        raise ValueError("EVAL-037 generation file is empty")
    return rows


def _key(row: Mapping) -> tuple[str, str, int, float]:
    return (
        str(row.get("task_id", "")),
        str(row.get("condition", "")),
        int(row.get("generation_seed", -1)),
        float(row.get("blend_alpha", -1.0)),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _observed_layout(rows: Sequence[Mapping]) -> tuple[list[str], list[tuple[str, str]]]:
    observed_conditions = [
        condition
        for condition in ORACLE_BLEND_CONDITIONS
        if any(row.get("condition") == condition for row in rows)
    ]
    task_sets = {
        condition: {
            str(row.get("task_id"))
            for row in rows
            if row.get("condition") == condition
        }
        for condition in observed_conditions
    }
    matched, shuffled = ORACLE_BLEND_CONDITIONS
    comparisons = (
        [(matched, shuffled)]
        if matched in task_sets
        and shuffled in task_sets
        and bool(task_sets[matched].intersection(task_sets[shuffled]))
        else []
    )
    return observed_conditions, comparisons


def validate_oracle_blend_grid(
    rows: Sequence[Mapping],
    metadata: Mapping,
    config: Mapping,
    *,
    allow_incomplete: bool,
) -> dict:
    validate_oracle_blend_contract(config)
    design_sha = oracle_blend_design_fingerprint(config)
    prefix = str(config["decoding_interface"]["prefix"])
    metadata_checks = {
        "experiment": metadata.get("experiment_id") == ORACLE_BLEND_EXPERIMENT_ID,
        "protocol": metadata.get("protocol_version") == ORACLE_BLEND_PROTOCOL_VERSION,
        "design": metadata.get("design_sha256") == design_sha,
        "scope": metadata.get("run_scope")
        == "development_only_reused_open_P014_cohort",
        "claim": metadata.get("claim_eligible") is False,
        "task_count": metadata.get("task_count") == 32,
        "conditions": tuple(metadata.get("conditions", ()))
        == ORACLE_BLEND_CONDITIONS,
        "screen_alphas": tuple(float(value) for value in metadata.get("screen_alphas", ()))
        == ORACLE_BLEND_SCREEN_ALPHAS,
        "screen_seed": metadata.get("screen_generation_seed")
        == ORACLE_BLEND_SCREEN_GENERATION_SEED,
        "confirmation_seeds": tuple(metadata.get("confirmation_generation_seeds", ()))
        == ORACLE_BLEND_CONFIRMATION_GENERATION_SEEDS,
        "canonical_entry": metadata.get("canonical_entry_point") == "f_0",
        "forced_prefix": metadata.get("forced_completion_prefix") == prefix,
        "forced_prefix_hash": isinstance(
            metadata.get("forced_completion_prefix_token_ids_sha256"), str
        ),
        "position_separation": metadata.get("receiver_position_audit", {}).get(
            "positionally_separated"
        )
        is True,
    }
    failed = [name for name, passed in metadata_checks.items() if not passed]
    if failed:
        raise ValueError("EVAL-037 metadata drifted: " + ", ".join(failed))

    task_ids = [str(value) for value in metadata.get("task_ids", ())]
    screen_expected = expected_oracle_blend_keys(task_ids, "screen")
    selected_alpha = metadata.get("selected_alpha")
    confirmation_expected = (
        expected_oracle_blend_keys(
            task_ids, "confirm", selected_alpha=float(selected_alpha)
        )
        if selected_alpha is not None
        else set()
    )
    expected = screen_expected | confirmation_expected
    donors = {
        str(target): str(donor)
        for target, donor in metadata.get("donor_task_ids", {}).items()
    }
    if set(donors) != set(task_ids) or any(
        target == donor or donor not in task_ids for target, donor in donors.items()
    ):
        raise ValueError("EVAL-037 donor map is invalid")

    observed = []
    tasks = {}
    salt = int(config["conditions"]["effective_generation_seed_salt"])
    for row in rows:
        key = _key(row)
        observed.append(key)
        task_id, condition, generation_seed, alpha = key
        task = row.get("task_spec")
        if not isinstance(task, Mapping) or str(task.get("task_id", "")) != task_id:
            raise ValueError("EVAL-037 canonical task identity changed")
        previous = tasks.setdefault(task_id, dict(task))
        if previous != dict(task):
            raise ValueError("EVAL-037 canonical task varies across rows")
        phase = str(row.get("phase", ""))
        if phase == "screen":
            phase_shape = (
                generation_seed == ORACLE_BLEND_SCREEN_GENERATION_SEED
                and alpha in ORACLE_BLEND_SCREEN_ALPHAS
            )
        else:
            phase_shape = (
                phase == "confirm"
                and selected_alpha is not None
                and generation_seed in ORACLE_BLEND_CONFIRMATION_GENERATION_SEEDS
                and alpha == float(selected_alpha)
            )
        source_expected = (
            task_id if condition == "oracle_blend_matched" else donors.get(task_id)
        )
        output = row.get("output_text")
        packet_norm = row.get("packet_frobenius_norm")
        row_checks = {
            "experiment": row.get("experiment_id") == ORACLE_BLEND_EXPERIMENT_ID,
            "protocol": row.get("protocol_version") == ORACLE_BLEND_PROTOCOL_VERSION,
            "design": row.get("design_sha256") == design_sha,
            "config": row.get("config_sha256") == metadata.get("config_sha256"),
            "scope": row.get("run_scope")
            == "development_only_reused_open_P014_cohort",
            "claim": row.get("claim_eligible") is False,
            "condition": condition in ORACLE_BLEND_CONDITIONS,
            "phase_shape": phase_shape,
            "effective_seed": row.get("effective_generation_seed")
            == stable_seed(generation_seed, task_ids.index(task_id), salt),
            "prompt_kind": row.get("target_prompt_kind")
            == "constant_opaque_entry_point_with_constrained_prefix",
            "prompt": row.get("target_user_prompt_sha256")
            == metadata.get("receiver_user_prompt_sha256"),
            "formatted_prompt": row.get("target_formatted_prompt_sha256")
            == metadata.get("receiver_formatted_prompt_sha256"),
            "input": row.get("target_input_ids_sha256")
            == metadata.get("receiver_input_ids_sha256"),
            "attention": row.get("target_attention_mask_sha256")
            == metadata.get("receiver_attention_mask_sha256"),
            "canonical_entry": row.get("canonical_entry_point")
            == task.get("entry_point")
            == "f_0",
            "position_separation": row.get(
                "canonical_entry_point_positionally_separated"
            )
            is True,
            "forced_prefix": row.get("forced_completion_prefix") == prefix,
            "forced_prefix_hash": row.get(
                "forced_completion_prefix_token_ids_sha256"
            )
            == metadata.get("forced_completion_prefix_token_ids_sha256"),
            "forced_prefix_realized": row.get("forced_prefix_realized") is True
            and isinstance(output, str)
            and output.startswith(prefix),
            "packet": row.get("packet_present") is True
            and isinstance(row.get("packet_sha256"), str)
            and len(row["packet_sha256"]) == 64
            and packet_norm is not None
            and math.isfinite(float(packet_norm)),
            "layers": row.get("packet_layer_indices")
            == list(config["packets"]["oracle_layer_indices"]),
            "offsets": row.get("packet_offsets")
            == list(config["packets"]["offsets"]),
            "replay": row.get("packet_replay_mode") == "blend",
            "alpha": row.get("blend_alpha") == alpha,
            "native_weight": math.isclose(
                float(row.get("native_weight", -1.0)), 1.0 - alpha
            ),
            "packet_weight": math.isclose(
                float(row.get("packet_weight", -1.0)), alpha
            ),
            "source": row.get("source_task_id") == source_expected,
            "donor": row.get("donor_task_id")
            == (donors[task_id] if condition == "oracle_blend_shuffled" else None),
        }
        failed_row = [name for name, passed in row_checks.items() if not passed]
        if failed_row:
            raise ValueError(f"EVAL-037 row {key} drifted: " + ", ".join(failed_row))
    if len(set(observed)) != len(observed):
        raise ValueError("EVAL-037 contains duplicate rows")
    unexpected = set(observed).difference(expected)
    missing = expected.difference(observed)
    if unexpected:
        raise ValueError("EVAL-037 contains unexpected rows")
    if missing and not allow_incomplete:
        raise ValueError(f"EVAL-037 is missing {len(missing)} rows")
    screen_complete = screen_expected.issubset(observed)
    confirmation_complete = bool(confirmation_expected) and confirmation_expected.issubset(
        observed
    )
    complete = screen_complete and confirmation_complete
    if bool(metadata.get("screen_phase_complete")) != screen_complete:
        raise ValueError("EVAL-037 screen completeness disagrees with grid")
    if bool(metadata.get("confirmation_phase_complete")) != confirmation_complete:
        raise ValueError("EVAL-037 confirmation completeness disagrees with grid")
    if bool(metadata.get("complete")) != complete:
        raise ValueError("EVAL-037 total completeness disagrees with grid")
    if metadata.get("records") != len(rows):
        raise ValueError("EVAL-037 metadata record count disagrees with grid")
    return {
        "complete": complete,
        "screen_phase_complete": screen_complete,
        "confirmation_phase_complete": confirmation_complete,
        "run_scope": "development_only_reused_open_P014_cohort",
        "claim_eligible": False,
        "task_count": len(task_ids),
        "record_count": len(rows),
        "expected_record_count": len(expected),
        "missing_record_count": len(missing),
        "design_sha256": design_sha,
        "selected_alpha": selected_alpha,
        "canonical_entry_point": "f_0",
        "forced_completion_prefix": prefix,
        "cluster_unit": "task_id",
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
    design_validation = validate_oracle_blend_grid(
        rows, metadata, config, allow_incomplete=allow_incomplete
    )
    prepare_output_dir(output_dir, overwrite=overwrite)
    reasons: Counter[str] = Counter()
    scored = []
    for row in rows:
        code = extract_code(row["output_text"])
        syntax = check_syntax(code)
        entry_point = str(row["task_spec"]["entry_point"])
        exact_declared = declares_top_level_function(code, entry_point)
        alias = build_single_function_alias(code, entry_point)
        reasons[alias["reason"]] += 1
        scored_row = dict(row)
        scored_row.update(
            {
                "extracted_code": code,
                **syntax,
                "entry_point_declared_top_level": exact_declared,
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
        if functional and syntax["syntax_pass"]:
            exact_result = run_functional_tests(
                code,
                row["task_spec"],
                timeout_seconds=5.0,
                memory_mb=512,
                process_policy=candidate_process_policy,
            )
        elif functional:
            exact_result = {
                "functional_pass": False,
                "functional_error_type": "SyntaxError",
                "functional_error": syntax["syntax_error"],
            }
        else:
            exact_result = {
                "functional_pass": None,
                "functional_error_type": None,
                "functional_error": None,
            }
        scored_row.update(exact_result)
        if functional and alias["eligible"]:
            alias_result = (
                exact_result
                if not alias["alias_binding_applied"]
                else run_functional_tests(
                    alias["normalized_code"],
                    row["task_spec"],
                    timeout_seconds=5.0,
                    memory_mb=512,
                    process_policy=candidate_process_policy,
                )
            )
            scored_row["alias_functional_pass"] = alias_result["functional_pass"]
            scored_row["alias_functional_error_type"] = alias_result[
                "functional_error_type"
            ]
            scored_row["alias_functional_error"] = alias_result["functional_error"]
        elif functional:
            scored_row["alias_functional_pass"] = False
            scored_row["alias_functional_error_type"] = "AliasIneligible"
            scored_row["alias_functional_error"] = alias["reason"]
        else:
            scored_row["alias_functional_pass"] = None
            scored_row["alias_functional_error_type"] = None
            scored_row["alias_functional_error"] = None
        if functional:
            scored_row["core_functional_pass"] = bool(
                scored_row["functional_pass"]
                or scored_row["alias_functional_pass"]
            )
            scored_row["binding_gap"] = bool(
                scored_row["core_functional_pass"]
                and not scored_row["functional_pass"]
            )
        else:
            scored_row["core_functional_pass"] = None
            scored_row["binding_gap"] = None
        scored.append(scored_row)

    policy = config["evaluation"]
    statistics_kwargs = {
        "bootstrap_iterations": int(policy["bootstrap_iterations"]),
        "confidence": float(policy["confidence"]),
        "seed": int(policy["statistics_seed"]),
    }
    metrics_by_alpha = {}
    for phase in ("screen", "confirm"):
        for alpha in sorted({float(row["blend_alpha"]) for row in scored if row["phase"] == phase}):
            subset = [
                row
                for row in scored
                if row["phase"] == phase and float(row["blend_alpha"]) == alpha
            ]
            observed_conditions, comparisons = _observed_layout(subset)
            metrics_by_alpha[f"{phase}:{alpha}"] = {
                metric: summarize_metric(
                    subset,
                    metric,
                    observed_conditions,
                    comparisons,
                    **statistics_kwargs,
                )
                for metric in (
                    "syntax_pass",
                    "entry_point_declared_top_level",
                    "alias_eligible",
                    "forced_prefix_realized",
                    *(
                        (
                            "functional_pass",
                            "alias_functional_pass",
                            "core_functional_pass",
                            "binding_gap",
                        )
                        if functional
                        else ()
                    ),
                )
            }
    inference = None
    if functional and design_validation["screen_phase_complete"]:
        inference = (
            summarize_oracle_blend_confirmation(scored, config)
            if design_validation["complete"]
            else summarize_oracle_blend_screen(
                [row for row in scored if row["phase"] == "screen"], config
            )
        )
    sandbox_validated = bool(security_context and security_context.get("validated"))
    diagnostic_route = (
        inference["diagnostic_route"]
        if functional and sandbox_validated and inference
        else "not_scored_in_hardened_namespace"
    )
    summary = {
        "experiment_id": ORACLE_BLEND_EXPERIMENT_ID,
        "protocol_version": ORACLE_BLEND_PROTOCOL_VERSION,
        "execution_mode": (
            "functional_hardened_namespace"
            if functional and sandbox_validated
            else "functional_subprocess"
            if functional
            else "syntax_only"
        ),
        "phase": (
            "complete"
            if design_validation["complete"]
            else "screen"
            if design_validation["screen_phase_complete"]
            else "partial"
        ),
        "claim_eligible": False,
        "can_upgrade_EVAL_033": False,
        "can_upgrade_EVAL_034": False,
        "can_upgrade_EVAL_035": False,
        "can_upgrade_EVAL_036": False,
        "subprocess_is_security_sandbox": sandbox_validated if functional else None,
        "diagnostic_route": diagnostic_route,
        "design_validation": design_validation,
        "normalization_reason_counts": dict(sorted(reasons.items())),
        "inference": inference,
        "metrics_by_alpha": metrics_by_alpha,
        "artifact_provenance": {
            key: metadata.get(key)
            for key in (
                "predecessor_registry_sha256",
                "predecessor_diagnostic_route",
                "source_screen_config_sha256",
                "source_screen_run_commit",
                "P014_generations_sha256",
                "P014_metadata_sha256",
                "confirmation_bundle_manifest_sha256",
                "source_model_revision",
                "target_model_revision",
                "receiver_user_prompt_sha256",
                "receiver_formatted_prompt_sha256",
                "receiver_input_ids_sha256",
                "receiver_position_audit",
                "forced_completion_prefix_sha256",
                "forced_completion_prefix_token_ids_sha256",
                "screen_phase_generations_sha256",
                "screen_lock_sha256",
            )
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
        default=Path("config/LIP-EVAL-037_oracle_native_packet_blend_screen.yaml"),
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
                "phase": summary["phase"],
                "claim_eligible": summary["claim_eligible"],
                "diagnostic_route": summary["diagnostic_route"],
            }
        )
    )


if __name__ == "__main__":
    main()
