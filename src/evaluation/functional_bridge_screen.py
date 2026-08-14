"""Frozen design and statistics for the development-only LIP-EVAL-033."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean

from src.evaluation.statistics import bootstrap_mean_ci, sign_flip_p_value


FUNCTIONAL_BRIDGE_SCREEN_EXPERIMENT_ID = "LIP-EVAL-033"
FUNCTIONAL_BRIDGE_SCREEN_PROTOCOL_VERSION = "lip-functional-bridge-screen-v1"
FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS = ("learned_matched", "learned_shuffled")
FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS = (4001, 4003, 4007)
FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS = (4127, 4241, 4357)
FUNCTIONAL_BRIDGE_SCREEN_TASK_COUNT = 32
FUNCTIONAL_BRIDGE_SCREEN_EXPECTED_RECORDS = 576


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_functional_bridge_screen_contract(config: Mapping) -> None:
    cohort = config.get("cohort", {})
    systems = config.get("systems", {})
    generation = config.get("generation", {})
    evaluation = config.get("evaluation", {})
    decision = config.get("decision", {})
    checks = {
        "experiment": config.get("experiment_id")
        == FUNCTIONAL_BRIDGE_SCREEN_EXPERIMENT_ID,
        "protocol": config.get("protocol_version")
        == FUNCTIONAL_BRIDGE_SCREEN_PROTOCOL_VERSION,
        "claim_status": config.get("claim_status")
        == "development_only_reused_open_P014_functional_cohort",
        "predecessor": config.get("predecessor", {}).get("protocol")
        == "LIP-H0-016",
        "cohort": cohort.get("protocol") == "LIP-PROTO-014",
        "already_open": cohort.get("already_open") is True,
        "claim_scope": cohort.get("claim_scope") == "development_only",
        "task_count": int(cohort.get("task_count", -1))
        == FUNCTIONAL_BRIDGE_SCREEN_TASK_COUNT,
        "strata": cohort.get("tokenizer_strata") == {2: 16, 3: 16},
        "generation_seeds": tuple(cohort.get("generation_seeds", ()))
        == FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS,
        "derangement_seed": int(cohort.get("derangement_seed", -1)) == 4513,
        "training_seeds": tuple(systems.get("training_seeds", ()))
        == FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS,
        "conditions": tuple(systems.get("conditions", ()))
        == FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS,
        "checkpoint_seeds": set(map(int, systems.get("checkpoints", {})))
        == set(FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS),
        "expected_records": int(generation.get("expected_records", -1))
        == FUNCTIONAL_BRIDGE_SCREEN_EXPECTED_RECORDS,
        "max_new_tokens": int(generation.get("max_new_tokens", -1)) == 256,
        "sampling": generation.get("do_sample") is True
        and float(generation.get("temperature", -1)) == 0.2
        and float(generation.get("top_p", -1)) == 0.95
        and float(generation.get("repetition_penalty", -1)) == 1.0,
        "primary": evaluation.get("primary_treatment") == "learned_matched"
        and evaluation.get("primary_control") == "learned_shuffled",
        "alpha": float(evaluation.get("alpha", -1)) == 0.05,
        "alternative": evaluation.get("alternative") == "greater",
        "guardrail": int(evaluation.get("minimum_positive_bridge_seeds", -1))
        == 2,
        "not_claim_eligible": evaluation.get("claim_eligible") is False,
        "proto_blocked": decision.get("proto_015_execution_authorized") is False,
        "L4": config.get("compute", {}).get("preferred_accelerator") == "L4",
        "no_fallback": config.get("compute", {}).get("allow_silent_fallback")
        is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("LIP-EVAL-033 contract drifted: " + ", ".join(failed))


def functional_bridge_screen_design_fingerprint(config: Mapping) -> str:
    validate_functional_bridge_screen_contract(config)
    return _canonical_sha256(config)


def expected_functional_bridge_screen_keys(
    task_ids: Sequence[str],
) -> set[tuple[str, str, int, int]]:
    if len(task_ids) != FUNCTIONAL_BRIDGE_SCREEN_TASK_COUNT:
        raise ValueError("LIP-EVAL-033 requires exactly 32 tasks")
    if len(set(map(str, task_ids))) != len(task_ids):
        raise ValueError("LIP-EVAL-033 task IDs must be unique")
    return {
        (str(task_id), condition, generation_seed, training_seed)
        for task_id in task_ids
        for generation_seed in FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS
        for training_seed in FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS
        for condition in FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS
    }


def summarize_functional_bridge_screen(
    records: Sequence[Mapping], config: Mapping
) -> dict:
    """Apply the single task-clustered endpoint and seed guardrail."""

    validate_functional_bridge_screen_contract(config)
    metric = str(config["evaluation"]["metric"])
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_seed: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in records:
        task_id = str(row["task_id"])
        condition = str(row["condition"])
        value = float(bool(row[metric]))
        grouped[(task_id, condition)].append(value)
        by_seed[(int(row["training_seed"]), task_id, condition)].append(value)
    task_ids = sorted({task_id for task_id, _ in grouped})
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
    primary_passed = bool(
        mean_difference > 0.0 and p_value <= float(evaluation["alpha"])
    )
    seed_results = {}
    for seed in FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS:
        seed_differences = []
        for task_id in task_ids:
            treatment = by_seed[(seed, task_id, "learned_matched")]
            control = by_seed[(seed, task_id, "learned_shuffled")]
            if len(treatment) != 3 or len(control) != 3:
                raise ValueError("each seed-task-condition requires three generations")
            seed_differences.append(mean(treatment) - mean(control))
        seed_results[str(seed)] = {
            "task_count": len(seed_differences),
            "mean_difference": mean(seed_differences),
            "positive": mean(seed_differences) > 0.0,
        }
    positive_seeds = sum(row["positive"] for row in seed_results.values())
    guardrail_passed = bool(
        positive_seeds >= int(evaluation["minimum_positive_bridge_seeds"])
    )
    return {
        "primary_endpoint": {
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
            "passed": primary_passed,
        },
        "seed_guardrail": {
            "minimum_positive_bridge_seeds": int(
                evaluation["minimum_positive_bridge_seeds"]
            ),
            "positive_bridge_seeds": positive_seeds,
            "by_seed": seed_results,
            "passed": guardrail_passed,
        },
        "development_functional_signal_detected": bool(
            primary_passed and guardrail_passed
        ),
        "claim_eligible": False,
    }
