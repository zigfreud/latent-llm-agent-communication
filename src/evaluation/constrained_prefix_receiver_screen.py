"""Frozen design and decisions for LIP-EVAL-036 constrained-prefix screen."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from src.evaluation.constant_entry_point_screen import _endpoint_summary
from src.evaluation.functional_bridge_screen import (
    FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS,
    FUNCTIONAL_BRIDGE_SCREEN_TASK_COUNT,
    FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS,
)


CONSTRAINED_PREFIX_EXPERIMENT_ID = "LIP-EVAL-036"
CONSTRAINED_PREFIX_PROTOCOL_VERSION = "lip-constrained-prefix-receiver-screen-v1"
CONSTRAINED_PREFIX_CONTROL_CONDITIONS = (
    "canonical_no_packet",
    "oracle_teacher_matched",
    "oracle_teacher_shuffled",
)
CONSTRAINED_PREFIX_LEARNED_CONDITIONS = (
    "learned_matched",
    "learned_shuffled",
)
CONSTRAINED_PREFIX_CONDITIONS = (
    *CONSTRAINED_PREFIX_CONTROL_CONDITIONS,
    *CONSTRAINED_PREFIX_LEARNED_CONDITIONS,
)
CONSTRAINED_PREFIX_CONTROL_RECORDS = 288
CONSTRAINED_PREFIX_LEARNED_RECORDS = 576
CONSTRAINED_PREFIX_EXPECTED_RECORDS = 864


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_constrained_prefix_contract(config: Mapping) -> None:
    predecessor = config.get("predecessor", {})
    source = config.get("source_screen", {})
    interface = config.get("receiver_interface", {})
    decoding = config.get("decoding_interface", {})
    conditions = config.get("conditions", {})
    packets = config.get("packets", {})
    generation = config.get("generation", {})
    evaluation = config.get("evaluation", {})
    decision = config.get("decision", {})
    checks = {
        "experiment": config.get("experiment_id")
        == CONSTRAINED_PREFIX_EXPERIMENT_ID,
        "protocol": config.get("protocol_version")
        == CONSTRAINED_PREFIX_PROTOCOL_VERSION,
        "scope": config.get("claim_status")
        == "development_only_open_cohort_sequential_mechanism_screen",
        "predecessor": predecessor.get("experiment_id") == "LIP-EVAL-035",
        "predecessor_route": predecessor.get("required_route")
        == "constant_carrier_oracle_capacity_failure",
        "source": source.get("experiment_id") == "LIP-EVAL-035",
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
        "constant_entry": interface.get("entry_point") == "f_0",
        "constant_prompt": interface.get("same_prompt_for_every_task") is True,
        "no_semantic_text": interface.get("semantic_task_text_in_prompt") is False,
        "position_separation": interface.get(
            "require_entry_point_outside_intervention_suffix"
        )
        is True,
        "forced_prefix": decoding.get("mode") == "forced_completion_prefix"
        and decoding.get("prefix") == "def f_0"
        and decoding.get("same_prefix_for_every_task") is True
        and decoding.get("semantic_task_text_in_prefix") is False
        and decoding.get("original_entry_point_in_prefix") is False
        and decoding.get("require_exact_output_start") is True,
        "control_conditions": tuple(conditions.get("control_phase", ()))
        == CONSTRAINED_PREFIX_CONTROL_CONDITIONS,
        "learned_conditions": tuple(conditions.get("learned_phase", ()))
        == CONSTRAINED_PREFIX_LEARNED_CONDITIONS,
        "training_seeds": tuple(conditions.get("training_seeds", ()))
        == FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS,
        "generation_seeds": tuple(conditions.get("generation_seeds", ()))
        == FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS,
        "record_counts": int(conditions.get("control_expected_records", -1))
        == CONSTRAINED_PREFIX_CONTROL_RECORDS
        and int(conditions.get("learned_expected_records", -1))
        == CONSTRAINED_PREFIX_LEARNED_RECORDS
        and int(conditions.get("expected_records", -1))
        == CONSTRAINED_PREFIX_EXPECTED_RECORDS,
        "offsets": tuple(packets.get("offsets", ())) == tuple(range(-24, 0)),
        "layers": packets.get("learned_layer_indices") == [0]
        and packets.get("oracle_layer_indices") == list(range(8)),
        "replay": packets.get("replay_mode") == "replace"
        and float(packets.get("replay_gain", -1)) == 1.0,
        "sampling": int(generation.get("max_new_tokens", -1)) == 256
        and generation.get("do_sample") is True
        and float(generation.get("temperature", -1)) == 0.2
        and float(generation.get("top_p", -1)) == 0.95
        and float(generation.get("repetition_penalty", -1)) == 1.0,
        "statistics": evaluation.get("exact_metric") == "functional_pass"
        and evaluation.get("core_metric") == "core_functional_pass"
        and evaluation.get("primary_treatment") == "learned_matched"
        and evaluation.get("primary_control") == "learned_shuffled"
        and float(evaluation.get("alpha", -1)) == 0.05
        and evaluation.get("alternative") == "greater"
        and float(evaluation.get("confidence", -1)) == 0.95
        and int(evaluation.get("bootstrap_iterations", -1)) == 10000
        and int(evaluation.get("statistics_seed", -1)) == 4607,
        "control_gates": float(
            evaluation.get("oracle_core_capacity_minimum", -1)
        )
        == 0.75
        and float(evaluation.get("oracle_shuffled_core_maximum", -1)) == 0.10
        and float(evaluation.get("no_packet_core_maximum", -1)) == 0.10
        and float(evaluation.get("forced_prefix_realization_minimum", -1))
        == 1.0,
        "guardrail": int(evaluation.get("minimum_positive_bridge_seeds", -1))
        == 2,
        "hardened": evaluation.get("require_hardened_namespace") is True,
        "nonclaim": evaluation.get("claim_eligible") is False,
        "sequential": decision.get("learned_phase_requires_control_lock") is True,
        "no_upgrades": decision.get("can_upgrade_EVAL_033") is False
        and decision.get("can_upgrade_EVAL_034") is False
        and decision.get("can_upgrade_EVAL_035") is False,
        "no_holdout": decision.get("fresh_holdout_spend_authorized") is False,
        "no_proto": decision.get("proto_015_execution_authorized") is False,
        "L4": config.get("compute", {}).get("preferred_accelerator") == "L4",
        "no_fallback": config.get("compute", {}).get("allow_silent_fallback")
        is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("LIP-EVAL-036 contract drifted: " + ", ".join(failed))


def constrained_prefix_design_fingerprint(config: Mapping) -> str:
    validate_constrained_prefix_contract(config)
    return _canonical_sha256(config)


def expected_constrained_prefix_keys(
    task_ids: Sequence[str], phase: str = "all"
) -> set[tuple[str, str, int, int | None]]:
    if len(task_ids) != FUNCTIONAL_BRIDGE_SCREEN_TASK_COUNT:
        raise ValueError("LIP-EVAL-036 requires exactly 32 tasks")
    if len(set(map(str, task_ids))) != len(task_ids):
        raise ValueError("LIP-EVAL-036 task IDs must be unique")
    controls = {
        (str(task_id), condition, generation_seed, None)
        for task_id in task_ids
        for generation_seed in FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS
        for condition in CONSTRAINED_PREFIX_CONTROL_CONDITIONS
    }
    learned = {
        (str(task_id), condition, generation_seed, training_seed)
        for task_id in task_ids
        for generation_seed in FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS
        for training_seed in FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS
        for condition in CONSTRAINED_PREFIX_LEARNED_CONDITIONS
    }
    if phase == "controls":
        return controls
    if phase == "learned":
        return learned
    if phase != "all":
        raise ValueError("phase must be controls, learned, or all")
    return controls | learned


def _condition_results(records: Sequence[Mapping]) -> dict[str, dict]:
    results = {}
    for condition in CONSTRAINED_PREFIX_CONDITIONS:
        rows = [row for row in records if row.get("condition") == condition]
        if not rows:
            continue
        total = len(rows)
        exact = sum(bool(row.get("functional_pass")) for row in rows)
        alias = sum(bool(row.get("alias_functional_pass")) for row in rows)
        core = sum(bool(row.get("core_functional_pass")) for row in rows)
        gaps = sum(bool(row.get("binding_gap")) for row in rows)
        realized = sum(bool(row.get("forced_prefix_realized")) for row in rows)
        results[condition] = {
            "records": total,
            "exact_functional_passes": exact,
            "alias_functional_passes": alias,
            "core_functional_passes": core,
            "binding_gap_passes": gaps,
            "forced_prefix_realized": realized,
            "exact_functional_rate": exact / total,
            "core_functional_rate": core / total,
            "binding_gap_rate": gaps / total,
            "forced_prefix_realization_rate": realized / total,
        }
    return results


def summarize_constrained_prefix_controls(
    records: Sequence[Mapping], config: Mapping
) -> dict:
    validate_constrained_prefix_contract(config)
    if len(records) != CONSTRAINED_PREFIX_CONTROL_RECORDS:
        raise ValueError("LIP-EVAL-036 control phase requires exactly 288 rows")
    results = _condition_results(records)
    if set(results) != set(CONSTRAINED_PREFIX_CONTROL_CONDITIONS):
        raise ValueError("LIP-EVAL-036 control conditions are incomplete")
    evaluation = config["evaluation"]
    oracle_capacity = results["oracle_teacher_matched"][
        "core_functional_rate"
    ] >= float(evaluation["oracle_core_capacity_minimum"])
    oracle_specificity = results["oracle_teacher_shuffled"][
        "core_functional_rate"
    ] <= float(evaluation["oracle_shuffled_core_maximum"])
    no_packet_specificity = results["canonical_no_packet"][
        "core_functional_rate"
    ] <= float(evaluation["no_packet_core_maximum"])
    prefix_realization = all(
        result["forced_prefix_realization_rate"]
        >= float(evaluation["forced_prefix_realization_minimum"])
        for result in results.values()
    )
    if not prefix_realization:
        decision_key = "prefix_realization_failure"
    elif not oracle_capacity:
        decision_key = "oracle_capacity_failure"
    elif not (oracle_specificity and no_packet_specificity):
        decision_key = "specificity_control_failure"
    else:
        decision_key = "controls_passed"
    decision = config["decision"][decision_key]
    return {
        "condition_results": results,
        "control_gates": {
            "forced_prefix_realization_minimum": float(
                evaluation["forced_prefix_realization_minimum"]
            ),
            "forced_prefix_realization_passed": prefix_realization,
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
        "controls_passed": bool(
            prefix_realization
            and oracle_capacity
            and oracle_specificity
            and no_packet_specificity
        ),
        "diagnostic_route": decision["route"],
        "recommended_action": decision["action"],
        "learned_phase_authorized_by_frozen_gate": bool(
            decision_key == "controls_passed"
        ),
        "claim_eligible": False,
    }


def summarize_constrained_prefix_screen(
    records: Sequence[Mapping], config: Mapping
) -> dict:
    validate_constrained_prefix_contract(config)
    if len(records) != CONSTRAINED_PREFIX_EXPECTED_RECORDS:
        raise ValueError("LIP-EVAL-036 full screen requires exactly 864 rows")
    controls = summarize_constrained_prefix_controls(
        [
            row
            for row in records
            if row.get("condition") in CONSTRAINED_PREFIX_CONTROL_CONDITIONS
        ],
        config,
    )
    exact = _endpoint_summary(records, config, "functional_pass")
    core = _endpoint_summary(records, config, "core_functional_pass")
    if not controls["controls_passed"]:
        route = controls["diagnostic_route"]
        action = controls["recommended_action"]
    elif exact["signal_detected"]:
        route = config["decision"]["exact_signal"]["route"]
        action = config["decision"]["exact_signal"]["action"]
    elif core["signal_detected"]:
        route = config["decision"]["core_only_signal"]["route"]
        action = config["decision"]["core_only_signal"]["action"]
    else:
        route = config["decision"]["no_signal"]["route"]
        action = config["decision"]["no_signal"]["action"]
    return {
        **controls,
        "condition_results": _condition_results(records),
        "exact_binding_endpoint": exact,
        "core_recovery_endpoint": core,
        "diagnostic_route": route,
        "recommended_action": action,
        "can_upgrade_EVAL_033": False,
        "can_upgrade_EVAL_034": False,
        "can_upgrade_EVAL_035": False,
        "claim_eligible": False,
    }
