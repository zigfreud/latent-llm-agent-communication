"""Frozen design and decision statistics for LIP-EVAL-035."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import keyword
import tokenize
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from statistics import mean

from src.evaluation.functional_bridge_screen import (
    FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS,
    FUNCTIONAL_BRIDGE_SCREEN_TASK_COUNT,
    FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS,
)
from src.evaluation.statistics import bootstrap_mean_ci, sign_flip_p_value


CONSTANT_ENTRY_POINT_EXPERIMENT_ID = "LIP-EVAL-035"
CONSTANT_ENTRY_POINT_PROTOCOL_VERSION = (
    "lip-constant-opaque-entry-point-screen-v1"
)
CONSTANT_ENTRY_POINT_SHARED_CONDITIONS = (
    "canonical_no_packet",
    "oracle_teacher_matched",
    "oracle_teacher_shuffled",
)
CONSTANT_ENTRY_POINT_REPLICA_CONDITIONS = (
    "learned_matched",
    "learned_shuffled",
)
CONSTANT_ENTRY_POINT_CONDITIONS = (
    *CONSTANT_ENTRY_POINT_SHARED_CONDITIONS,
    *CONSTANT_ENTRY_POINT_REPLICA_CONDITIONS,
)
CONSTANT_ENTRY_POINT_EXPECTED_RECORDS = 864


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_constant_entry_point_contract(config: Mapping) -> None:
    predecessor = config.get("predecessor", {})
    source = config.get("source_screen", {})
    interface = config.get("receiver_interface", {})
    conditions = config.get("conditions", {})
    packets = config.get("packets", {})
    generation = config.get("generation", {})
    evaluation = config.get("evaluation", {})
    decision = config.get("decision", {})
    entry_point = interface.get("entry_point")
    prompt = interface.get("prompt")
    checks = {
        "experiment": config.get("experiment_id")
        == CONSTANT_ENTRY_POINT_EXPERIMENT_ID,
        "protocol": config.get("protocol_version")
        == CONSTANT_ENTRY_POINT_PROTOCOL_VERSION,
        "scope": config.get("claim_status")
        == "development_only_open_cohort_mechanism_screen",
        "predecessor": predecessor.get("experiment_id") == "LIP-EVAL-034",
        "predecessor_decision": predecessor.get("required_decision")
        == "LIP_EVAL_035_design_authorized",
        "predecessor_route": predecessor.get("required_route")
        == "matched_specific_alias_recovery_candidate",
        "source": source.get("experiment_id") == "LIP-EVAL-033",
        "reuse": all(
            source.get(field) is True
            for field in (
                "reuse_models",
                "reuse_checkpoints",
                "reuse_open_P014_cohort",
                "reuse_generation_seeds",
                "reuse_effective_generation_seed_schedule",
            )
        ),
        "entry_identifier": isinstance(entry_point, str)
        and entry_point.isidentifier()
        and not keyword.iskeyword(entry_point),
        "opaque_entry": entry_point == "f_0",
        "prompt": isinstance(prompt, str)
        and prompt.count(str(entry_point)) == 1,
        "constant_prompt": interface.get("same_prompt_for_every_task") is True,
        "no_semantic_text": interface.get("semantic_task_text_in_prompt") is False,
        "no_original_name": interface.get("original_entry_point_in_prompt")
        is False,
        "position_separation": interface.get(
            "require_entry_point_outside_intervention_suffix"
        )
        is True
        and int(interface.get("minimum_tokens_after_entry_point", -1))
        > int(interface.get("intervention_suffix_tokens", -1)),
        "shared_conditions": tuple(conditions.get("shared", ()))
        == CONSTANT_ENTRY_POINT_SHARED_CONDITIONS,
        "replica_conditions": tuple(conditions.get("replicated", ()))
        == CONSTANT_ENTRY_POINT_REPLICA_CONDITIONS,
        "training_seeds": tuple(conditions.get("training_seeds", ()))
        == FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS,
        "generation_seeds": tuple(conditions.get("generation_seeds", ()))
        == FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS,
        "records": int(conditions.get("expected_records", -1))
        == CONSTANT_ENTRY_POINT_EXPECTED_RECORDS,
        "offsets": tuple(packets.get("offsets", ())) == tuple(range(-24, 0)),
        "learned_layer": packets.get("learned_layer_indices") == [0],
        "oracle_layers": packets.get("oracle_layer_indices")
        == list(range(8)),
        "sampling": int(generation.get("max_new_tokens", -1)) == 256
        and generation.get("do_sample") is True
        and float(generation.get("temperature", -1)) == 0.2
        and float(generation.get("top_p", -1)) == 0.95
        and float(generation.get("repetition_penalty", -1)) == 1.0,
        "metrics": evaluation.get("exact_metric") == "functional_pass"
        and evaluation.get("core_metric") == "core_functional_pass"
        and evaluation.get("alias_backoff_metric") == "alias_functional_pass"
        and evaluation.get("binding_gap_metric") == "binding_gap",
        "primary": evaluation.get("primary_treatment") == "learned_matched"
        and evaluation.get("primary_control") == "learned_shuffled",
        "statistics": float(evaluation.get("alpha", -1)) == 0.05
        and evaluation.get("alternative") == "greater"
        and float(evaluation.get("confidence", -1)) == 0.95
        and int(evaluation.get("bootstrap_iterations", -1)) == 10000
        and int(evaluation.get("statistics_seed", -1)) == 4507,
        "guardrail": int(evaluation.get("minimum_positive_bridge_seeds", -1))
        == 2,
        "capacity": float(evaluation.get("oracle_core_capacity_minimum", -1))
        == 0.75,
        "specificity": float(evaluation.get("oracle_shuffled_core_maximum", -1))
        == 0.10
        and float(evaluation.get("no_packet_core_maximum", -1)) == 0.10,
        "hardened": evaluation.get("require_hardened_namespace") is True,
        "nonclaim": evaluation.get("claim_eligible") is False,
        "no_upgrades": decision.get("can_upgrade_EVAL_033") is False
        and decision.get("can_upgrade_EVAL_034") is False,
        "no_holdout": decision.get("fresh_holdout_spend_authorized") is False,
        "no_proto": decision.get("proto_015_execution_authorized") is False,
        "L4": config.get("compute", {}).get("preferred_accelerator") == "L4",
        "no_fallback": config.get("compute", {}).get("allow_silent_fallback")
        is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("LIP-EVAL-035 contract drifted: " + ", ".join(failed))


def constant_entry_point_design_fingerprint(config: Mapping) -> str:
    validate_constant_entry_point_contract(config)
    return _canonical_sha256(config)


def expected_constant_entry_point_keys(
    task_ids: Sequence[str],
) -> set[tuple[str, str, int, int | None]]:
    if len(task_ids) != FUNCTIONAL_BRIDGE_SCREEN_TASK_COUNT:
        raise ValueError("LIP-EVAL-035 requires exactly 32 tasks")
    if len(set(map(str, task_ids))) != len(task_ids):
        raise ValueError("LIP-EVAL-035 task IDs must be unique")
    shared = {
        (str(task_id), condition, generation_seed, None)
        for task_id in task_ids
        for generation_seed in FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS
        for condition in CONSTANT_ENTRY_POINT_SHARED_CONDITIONS
    }
    replicated = {
        (str(task_id), condition, generation_seed, training_seed)
        for task_id in task_ids
        for generation_seed in FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS
        for training_seed in FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS
        for condition in CONSTANT_ENTRY_POINT_REPLICA_CONDITIONS
    }
    return shared | replicated


def replace_identifier(source: str, old: str, new: str) -> tuple[str, int]:
    """Replace direct call targets without touching strings or comments."""

    if not all(
        isinstance(value, str)
        and value.isidentifier()
        and not keyword.iskeyword(value)
        for value in (old, new)
    ):
        raise ValueError("identifier replacement requires valid identifiers")
    tree = ast.parse(source)
    all_positions = {
        (node.lineno, node.col_offset)
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == old
    }
    call_positions = {
        (node.func.lineno, node.func.col_offset)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == old
    }
    if all_positions.difference(call_positions):
        raise ValueError(
            "source task uses its entry_point outside a direct function call"
        )
    tokens = []
    replacements = 0
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if (
            token.type == tokenize.NAME
            and token.string == old
            and token.start in call_positions
        ):
            token = tokenize.TokenInfo(
                token.type, new, token.start, token.end, token.line
            )
            replacements += 1
        tokens.append(token)
    return tokenize.untokenize(tokens), replacements


def canonicalize_task(task: Mapping, entry_point: str) -> dict:
    """Return an evaluation-only task whose tests call one opaque symbol."""

    result = deepcopy(dict(task))
    original = result.get("entry_point")
    if not isinstance(original, str) or not original.isidentifier():
        raise ValueError("source task entry_point must be an identifier")
    tests = result.get("test_list", result.get("tests", []))
    if not isinstance(tests, list) or not tests:
        raise ValueError("source task must provide a non-empty test_list")
    rewritten_tests = []
    replacements = 0
    for test in tests:
        if not isinstance(test, str):
            raise ValueError("source task tests must be text")
        rewritten, count = replace_identifier(test, original, entry_point)
        compile(rewritten, "<canonical-test>", "exec")
        rewritten_tests.append(rewritten)
        replacements += count
    if replacements == 0:
        raise ValueError("source task tests never reference their entry_point")
    setup = result.get("test_setup_code", "")
    if setup:
        setup, _ = replace_identifier(str(setup), original, entry_point)
        compile(setup, "<canonical-test-setup>", "exec")
    prompt = str(result.get("prompt", ""))
    result.update(
        {
            "entry_point": entry_point,
            "original_entry_point": original,
            "prompt": prompt.replace(f"`{original}`", f"`{entry_point}`"),
            "test_list": rewritten_tests,
            "test_setup_code": setup,
            "canonical_test_identifier_replacements": replacements,
        }
    )
    result.pop("tests", None)
    return result


def declares_top_level_function(code: str, entry_point: str) -> bool:
    if not isinstance(entry_point, str) or not entry_point.isidentifier():
        raise ValueError("entry_point must be an identifier")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_point
        for node in tree.body
    )


def _endpoint_summary(
    records: Sequence[Mapping], config: Mapping, metric: str
) -> dict:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_seed: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in records:
        condition = str(row["condition"])
        task_id = str(row["task_id"])
        value = float(bool(row[metric]))
        if condition in CONSTANT_ENTRY_POINT_REPLICA_CONDITIONS:
            grouped[(task_id, condition)].append(value)
            by_seed[(int(row["training_seed"]), task_id, condition)].append(value)
    task_ids = sorted({task_id for task_id, _ in grouped})
    differences = []
    for task_id in task_ids:
        matched = grouped[(task_id, "learned_matched")]
        shuffled = grouped[(task_id, "learned_shuffled")]
        if len(matched) != 9 or len(shuffled) != 9:
            raise ValueError("each learned task-condition requires nine replicates")
        differences.append(mean(matched) - mean(shuffled))
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
    primary_passed = bool(
        mean_difference > 0.0 and p_value <= float(evaluation["alpha"])
    )
    seed_results = {}
    for seed in FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS:
        seed_differences = []
        for task_id in task_ids:
            matched = by_seed[(seed, task_id, "learned_matched")]
            shuffled = by_seed[(seed, task_id, "learned_shuffled")]
            if len(matched) != 3 or len(shuffled) != 3:
                raise ValueError(
                    "each learned seed-task-condition requires three generations"
                )
            seed_differences.append(mean(matched) - mean(shuffled))
        seed_mean = mean(seed_differences)
        seed_results[str(seed)] = {
            "task_count": len(seed_differences),
            "mean_difference": seed_mean,
            "positive": seed_mean > 0.0,
        }
    positive_seeds = sum(row["positive"] for row in seed_results.values())
    guardrail_passed = positive_seeds >= int(
        evaluation["minimum_positive_bridge_seeds"]
    )
    return {
        "metric": metric,
        "task_count": len(task_ids),
        "nonzero_task_count": sum(abs(value) > 1e-15 for value in differences),
        "mean_difference": mean_difference,
        "ci_lower": lower,
        "ci_upper": upper,
        "p_value": p_value,
        "p_value_method": method,
        "alternative": "greater",
        "primary_passed": primary_passed,
        "seed_guardrail": {
            "minimum_positive_bridge_seeds": int(
                evaluation["minimum_positive_bridge_seeds"]
            ),
            "positive_bridge_seeds": positive_seeds,
            "by_seed": seed_results,
            "passed": guardrail_passed,
        },
        "signal_detected": bool(primary_passed and guardrail_passed),
    }


def summarize_constant_entry_point_screen(
    records: Sequence[Mapping], config: Mapping
) -> dict:
    """Apply exact-binding, core-recovery, capacity, and specificity gates."""

    validate_constant_entry_point_contract(config)
    if len(records) != CONSTANT_ENTRY_POINT_EXPECTED_RECORDS:
        raise ValueError("LIP-EVAL-035 requires all 864 frozen cells")
    condition_counts: dict[str, dict[str, int]] = {
        condition: {
            "records": 0,
            "exact_functional_passes": 0,
            "alias_functional_passes": 0,
            "core_functional_passes": 0,
            "binding_gap_passes": 0,
            "alias_eligible": 0,
        }
        for condition in CONSTANT_ENTRY_POINT_CONDITIONS
    }
    for row in records:
        condition = str(row["condition"])
        if condition not in condition_counts:
            raise ValueError(f"unknown EVAL-035 condition: {condition}")
        counts = condition_counts[condition]
        counts["records"] += 1
        counts["exact_functional_passes"] += int(bool(row["functional_pass"]))
        counts["alias_functional_passes"] += int(
            bool(row["alias_functional_pass"])
        )
        counts["core_functional_passes"] += int(
            bool(row["core_functional_pass"])
        )
        counts["binding_gap_passes"] += int(bool(row["binding_gap"]))
        counts["alias_eligible"] += int(bool(row["alias_eligible"]))
    condition_results = {}
    for condition, counts in condition_counts.items():
        total = counts["records"]
        condition_results[condition] = {
            **counts,
            "exact_functional_rate": counts["exact_functional_passes"] / total,
            "core_functional_rate": counts["core_functional_passes"] / total,
            "binding_gap_rate": counts["binding_gap_passes"] / total,
            "alias_eligible_rate": counts["alias_eligible"] / total,
        }
    exact = _endpoint_summary(records, config, "functional_pass")
    core = _endpoint_summary(records, config, "core_functional_pass")
    evaluation = config["evaluation"]
    oracle_capacity = condition_results["oracle_teacher_matched"][
        "core_functional_rate"
    ] >= float(evaluation["oracle_core_capacity_minimum"])
    oracle_specificity = condition_results["oracle_teacher_shuffled"][
        "core_functional_rate"
    ] <= float(evaluation["oracle_shuffled_core_maximum"])
    no_packet_specificity = condition_results["canonical_no_packet"][
        "core_functional_rate"
    ] <= float(evaluation["no_packet_core_maximum"])
    if not oracle_capacity:
        decision_key = "oracle_capacity_failure"
    elif not (oracle_specificity and no_packet_specificity):
        decision_key = "specificity_control_failure"
    elif exact["signal_detected"]:
        decision_key = "exact_signal"
    elif core["signal_detected"]:
        decision_key = "core_only_signal"
    else:
        decision_key = "no_signal"
    decision = config["decision"][decision_key]
    return {
        "condition_results": condition_results,
        "exact_binding_endpoint": exact,
        "core_recovery_endpoint": core,
        "control_gates": {
            "oracle_core_capacity_minimum": float(
                evaluation["oracle_core_capacity_minimum"]
            ),
            "oracle_capacity_passed": oracle_capacity,
            "oracle_shuffled_core_maximum": float(
                evaluation["oracle_shuffled_core_maximum"]
            ),
            "oracle_specificity_passed": oracle_specificity,
            "no_packet_core_maximum": float(
                evaluation["no_packet_core_maximum"]
            ),
            "no_packet_specificity_passed": no_packet_specificity,
        },
        "diagnostic_route": decision["route"],
        "recommended_action": decision["action"],
        "can_upgrade_EVAL_033": False,
        "can_upgrade_EVAL_034": False,
        "claim_eligible": False,
    }
