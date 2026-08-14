"""Frozen alias policy and exploratory statistics for LIP-EVAL-034."""

from __future__ import annotations

import ast
import hashlib
import json
import keyword
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean

from src.evaluation.functional_bridge_screen import (
    FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS,
    FUNCTIONAL_BRIDGE_SCREEN_EXPECTED_RECORDS,
    FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS,
    FUNCTIONAL_BRIDGE_SCREEN_TASK_COUNT,
    FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS,
)
from src.evaluation.statistics import bootstrap_mean_ci, sign_flip_p_value


ALIAS_DIAGNOSTIC_EXPERIMENT_ID = "LIP-EVAL-034"
ALIAS_DIAGNOSTIC_PROTOCOL_VERSION = (
    "lip-alias-normalized-functional-diagnostic-v1"
)
ALIAS_DIAGNOSTIC_SOURCE_EXPERIMENT_ID = "LIP-EVAL-033"
ALIAS_DIAGNOSTIC_SOURCE_PROTOCOL_VERSION = "lip-functional-bridge-screen-v1"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_alias_diagnostic_contract(config: Mapping) -> None:
    source = config.get("source", {})
    normalization = config.get("normalization", {})
    eligibility = normalization.get("eligibility", {})
    evaluation = config.get("evaluation", {})
    decision = config.get("decision", {})
    checks = {
        "experiment": config.get("experiment_id")
        == ALIAS_DIAGNOSTIC_EXPERIMENT_ID,
        "protocol": config.get("protocol_version")
        == ALIAS_DIAGNOSTIC_PROTOCOL_VERSION,
        "claim_status": config.get("claim_status")
        == "post_hoc_development_only_nonclaim",
        "source_experiment": source.get("experiment_id")
        == ALIAS_DIAGNOSTIC_SOURCE_EXPERIMENT_ID,
        "source_protocol": source.get("protocol_version")
        == ALIAS_DIAGNOSTIC_SOURCE_PROTOCOL_VERSION,
        "source_complete": source.get("complete") is True,
        "source_nonclaim": source.get("claim_eligible") is False,
        "source_records": int(source.get("expected_records", -1))
        == FUNCTIONAL_BRIDGE_SCREEN_EXPECTED_RECORDS,
        "source_tasks": int(source.get("task_count", -1))
        == FUNCTIONAL_BRIDGE_SCREEN_TASK_COUNT,
        "source_conditions": tuple(source.get("conditions", ()))
        == FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS,
        "source_training_seeds": tuple(source.get("training_seeds", ()))
        == FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS,
        "source_generation_seeds": tuple(source.get("generation_seeds", ()))
        == FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS,
        "source_exact_names_zero": int(
            source.get("exact_entry_point_declarations", -1)
        )
        == 0,
        "normalization_method": normalization.get("method")
        == "single_top_level_function_alias_binding",
        "syntax_required": eligibility.get("syntax_valid") is True,
        "one_top_level_function": int(
            eligibility.get("exact_top_level_function_count", -1)
        )
        == 1,
        "preserve_code": normalization.get("preserve_original_code") is True,
        "no_rewrites": all(
            normalization.get(field) is False
            for field in (
                "body_rewrite",
                "argument_rewrite",
                "control_flow_rewrite",
                "test_rewrite",
            )
        ),
        "metric": evaluation.get("metric") == "alias_functional_pass",
        "replicates": int(evaluation.get("replicates_per_task_condition", -1))
        == 9,
        "comparison": evaluation.get("primary_treatment") == "learned_matched"
        and evaluation.get("primary_control") == "learned_shuffled",
        "hardened": evaluation.get("require_hardened_namespace") is True,
        "ineligible_is_failure": evaluation.get("ineligible_counts_as_failure")
        is True,
        "nonclaim": evaluation.get("claim_eligible") is False,
        "cannot_upgrade": decision.get("can_upgrade_EVAL_033") is False,
        "no_holdout": decision.get("fresh_holdout_spend_authorized") is False,
        "no_proto": decision.get("proto_015_execution_authorized") is False,
        "no_generation": config.get("compute", {}).get("generation_required")
        is False,
        "no_gpu": config.get("compute", {}).get("gpu_required") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("LIP-EVAL-034 contract drifted: " + ", ".join(failed))


def alias_diagnostic_design_fingerprint(config: Mapping) -> str:
    validate_alias_diagnostic_contract(config)
    return _canonical_sha256(config)


def build_single_function_alias(code: str, expected_entry_point: str) -> dict:
    """Expose one existing top-level function under the expected name.

    The transformation appends one name binding. It never renames or edits the
    function itself, which preserves recursive references to its original name.
    """

    if not isinstance(expected_entry_point, str):
        raise TypeError("expected_entry_point must be text")
    expected = expected_entry_point.strip()
    if (
        not expected
        or not expected.isidentifier()
        or keyword.iskeyword(expected)
    ):
        raise ValueError("expected_entry_point must be a non-keyword identifier")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "eligible": False,
            "reason": "syntax_invalid",
            "syntax_error": f"{exc.msg} (line {exc.lineno})",
            "top_level_function_count": 0,
            "top_level_function_names": [],
            "generated_function_name": None,
            "alias_binding_applied": False,
            "normalized_code": None,
        }
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    names = [node.name for node in functions]
    if len(functions) != 1:
        return {
            "eligible": False,
            "reason": (
                "missing_top_level_function"
                if not functions
                else "ambiguous_multiple_top_level_functions"
            ),
            "syntax_error": None,
            "top_level_function_count": len(functions),
            "top_level_function_names": names,
            "generated_function_name": None,
            "alias_binding_applied": False,
            "normalized_code": None,
        }
    generated_name = names[0]
    already_exact = generated_name == expected
    normalized = code.rstrip()
    if not already_exact:
        normalized += f"\n\n{expected} = {generated_name}\n"
    return {
        "eligible": True,
        "reason": "already_exact" if already_exact else "single_function_aliased",
        "syntax_error": None,
        "top_level_function_count": 1,
        "top_level_function_names": names,
        "generated_function_name": generated_name,
        "alias_binding_applied": not already_exact,
        "normalized_code": normalized,
    }


def summarize_alias_diagnostic(records: Sequence[Mapping], config: Mapping) -> dict:
    """Summarize all 576 rows, counting alias-ineligible rows as failures."""

    validate_alias_diagnostic_contract(config)
    if len(records) != FUNCTIONAL_BRIDGE_SCREEN_EXPECTED_RECORDS:
        raise ValueError("LIP-EVAL-034 requires all 576 frozen source rows")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    eligible: dict[str, int] = defaultdict(int)
    passed: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    by_seed: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in records:
        task_id = str(row["task_id"])
        condition = str(row["condition"])
        if condition not in FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS:
            raise ValueError(f"unknown condition: {condition}")
        value = float(bool(row["alias_functional_pass"]))
        grouped[(task_id, condition)].append(value)
        by_seed[(int(row["training_seed"]), task_id, condition)].append(value)
        totals[condition] += 1
        eligible[condition] += int(bool(row["alias_eligible"]))
        passed[condition] += int(bool(row["alias_functional_pass"]))
    task_ids = sorted({task_id for task_id, _ in grouped})
    if len(task_ids) != FUNCTIONAL_BRIDGE_SCREEN_TASK_COUNT:
        raise ValueError("LIP-EVAL-034 requires exactly 32 tasks")
    differences = []
    for task_id in task_ids:
        treatment = grouped[(task_id, "learned_matched")]
        control = grouped[(task_id, "learned_shuffled")]
        if len(treatment) != 9 or len(control) != 9:
            raise ValueError("each task-condition requires nine fixed replicates")
        differences.append(mean(treatment) - mean(control))

    evaluation = config["evaluation"]
    lower, upper = bootstrap_mean_ci(
        differences,
        iterations=int(evaluation["bootstrap_iterations"]),
        confidence=float(evaluation["confidence"]),
        seed=int(evaluation["statistics_seed"]),
    )
    p_value, method = sign_flip_p_value(
        differences,
        alternative="greater",
        seed=int(evaluation["statistics_seed"]) + 1,
    )
    mean_difference = mean(differences)
    seed_results = {}
    for seed in FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS:
        seed_differences = []
        for task_id in task_ids:
            treatment = by_seed[(seed, task_id, "learned_matched")]
            control = by_seed[(seed, task_id, "learned_shuffled")]
            if len(treatment) != 3 or len(control) != 3:
                raise ValueError(
                    "each seed-task-condition requires three generations"
                )
            seed_differences.append(mean(treatment) - mean(control))
        seed_mean = mean(seed_differences)
        seed_results[str(seed)] = {
            "task_count": len(seed_differences),
            "mean_difference": seed_mean,
            "positive": seed_mean > 0.0,
        }
    positive_seeds = sum(row["positive"] for row in seed_results.values())
    matched_passes = passed["learned_matched"]
    if matched_passes == 0:
        route_key = "zero_matched_passes"
    elif (
        mean_difference > 0.0
        and positive_seeds
        >= int(evaluation["minimum_positive_bridge_seeds"])
    ):
        route_key = "positive_matched_task_difference_with_seed_guardrail"
    else:
        route_key = "otherwise"
    route = config["decision"][route_key]
    condition_results = {}
    for condition in FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS:
        condition_results[condition] = {
            "records": totals[condition],
            "alias_eligible": eligible[condition],
            "alias_eligible_rate": eligible[condition] / totals[condition],
            "alias_functional_passes": passed[condition],
            "alias_functional_rate_all_rows": passed[condition]
            / totals[condition],
            "alias_functional_rate_eligible_only": (
                passed[condition] / eligible[condition]
                if eligible[condition]
                else 0.0
            ),
        }
    return {
        "condition_results": condition_results,
        "primary_diagnostic": {
            "treatment": "learned_matched",
            "control": "learned_shuffled",
            "task_count": len(task_ids),
            "nonzero_task_count": sum(abs(value) > 1e-15 for value in differences),
            "mean_difference": mean_difference,
            "ci_lower": lower,
            "ci_upper": upper,
            "p_value": p_value,
            "p_value_method": method,
            "alternative": "greater",
            "exploratory_only": True,
        },
        "seed_guardrail": {
            "minimum_positive_bridge_seeds": int(
                evaluation["minimum_positive_bridge_seeds"]
            ),
            "positive_bridge_seeds": positive_seeds,
            "by_seed": seed_results,
            "passed": positive_seeds
            >= int(evaluation["minimum_positive_bridge_seeds"]),
        },
        "diagnostic_route": route["route"],
        "recommended_action": route["action"],
        "can_upgrade_EVAL_033": False,
        "claim_eligible": False,
    }
