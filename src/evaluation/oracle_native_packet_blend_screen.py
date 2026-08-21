"""Frozen design and decisions for LIP-EVAL-037 oracle blend screen."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence


ORACLE_BLEND_EXPERIMENT_ID = "LIP-EVAL-037"
ORACLE_BLEND_PROTOCOL_VERSION = "lip-oracle-native-packet-blend-screen-v1"
ORACLE_BLEND_CONDITIONS = (
    "oracle_blend_matched",
    "oracle_blend_shuffled",
)
ORACLE_BLEND_SCREEN_ALPHAS = (0.25, 0.50, 0.75)
ORACLE_BLEND_SCREEN_GENERATION_SEED = 4127
ORACLE_BLEND_CONFIRMATION_GENERATION_SEEDS = (4241, 4357)
ORACLE_BLEND_SCREEN_RECORDS = 192
ORACLE_BLEND_CONFIRMATION_RECORDS = 128
ORACLE_BLEND_EXPECTED_RECORDS = 320


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_oracle_blend_contract(config: Mapping) -> None:
    predecessor = config.get("predecessor", {})
    source = config.get("source_screen", {})
    interface = config.get("receiver_interface", {})
    decoding = config.get("decoding_interface", {})
    conditions = config.get("conditions", {})
    packets = config.get("packets", {})
    anchors = config.get("reused_anchors", {})
    evaluation = config.get("evaluation", {})
    selection = config.get("selection", {})
    decision = config.get("decision", {})
    checks = {
        "experiment": config.get("experiment_id") == ORACLE_BLEND_EXPERIMENT_ID,
        "protocol": config.get("protocol_version")
        == ORACLE_BLEND_PROTOCOL_VERSION,
        "scope": config.get("claim_status")
        == "development_only_open_cohort_sequential_oracle_mechanism_screen",
        "predecessor": predecessor.get("experiment_id") == "LIP-EVAL-036",
        "predecessor_route": predecessor.get("required_route")
        == "constrained_prefix_oracle_capacity_failure",
        "source": source.get("experiment_id") == "LIP-EVAL-036",
        "reuse": all(
            source.get(field) is True
            for field in (
                "reuse_models",
                "reuse_open_P014_cohort",
                "reuse_generation_seeds",
                "reuse_donor_map",
                "reuse_constrained_prefix",
            )
        ),
        "constant_entry": interface.get("entry_point") == "f_0"
        and interface.get("same_prompt_for_every_task") is True
        and interface.get("semantic_task_text_in_prompt") is False,
        "forced_prefix": decoding.get("mode") == "forced_completion_prefix"
        and decoding.get("prefix") == "def f_0"
        and decoding.get("argument_list_forced") is False
        and decoding.get("body_tokens_forced") is False,
        "conditions": tuple(conditions.get("names", ()))
        == ORACLE_BLEND_CONDITIONS,
        "alphas": tuple(float(value) for value in conditions.get("screen_alphas", ()))
        == ORACLE_BLEND_SCREEN_ALPHAS,
        "screen_seed": int(conditions.get("screen_generation_seed", -1))
        == ORACLE_BLEND_SCREEN_GENERATION_SEED,
        "confirmation_seeds": tuple(
            int(value) for value in conditions.get("confirmation_generation_seeds", ())
        )
        == ORACLE_BLEND_CONFIRMATION_GENERATION_SEEDS,
        "record_counts": int(conditions.get("screen_expected_records", -1))
        == ORACLE_BLEND_SCREEN_RECORDS
        and int(conditions.get("confirmation_expected_records", -1))
        == ORACLE_BLEND_CONFIRMATION_RECORDS
        and int(conditions.get("expected_records", -1))
        == ORACLE_BLEND_EXPECTED_RECORDS,
        "blend": packets.get("replay_mode") == "blend"
        and packets.get("blend_formula")
        == "(1-alpha)*native_residual + alpha*oracle_packet"
        and tuple(packets.get("oracle_layer_indices", ())) == tuple(range(8))
        and tuple(packets.get("offsets", ())) == tuple(range(-24, 0)),
        "anchors": anchors.get("experiment_id") == "LIP-EVAL-036"
        and anchors.get("generations_sha256")
        == "bfef48894fd8f007abd3defdf546eaec10985e175871c7eeae477e2095416871"
        and anchors.get("control_summary_sha256")
        == "336dbd99b588551ee56917216625dc5c0813e27c63e5692abb479da93cb5b600"
        and anchors.get("alpha_0_screen_seed_core_passes") == 0
        and anchors.get("alpha_1_screen_seed_matched_core_passes") == 23
        and anchors.get("alpha_1_screen_seed_shuffled_core_passes") == 0,
        "gates": float(evaluation.get("oracle_core_capacity_minimum", -1))
        == 0.75
        and float(evaluation.get("oracle_shuffled_core_maximum", -1)) == 0.10
        and float(evaluation.get("forced_prefix_realization_minimum", -1)) == 1.0
        and evaluation.get("require_hardened_namespace") is True
        and evaluation.get("claim_eligible") is False,
        "selection": selection.get("phase") == "screen_seed_only"
        and selection.get("objective") == "maximum_matched_core_rate"
        and selection.get("requires_specificity_gate") is True
        and selection.get("tie_break") == "smallest_alpha"
        and selection.get("confirmation_excludes_screen_seed") is True,
        "sequential": decision.get("confirmation_requires_screen_lock") is True,
        "no_learned": decision.get("learned_execution_authorized") is False,
        "no_upgrades": all(
            decision.get(key) is False
            for key in (
                "can_upgrade_EVAL_033",
                "can_upgrade_EVAL_034",
                "can_upgrade_EVAL_035",
                "can_upgrade_EVAL_036",
            )
        ),
        "no_holdout": decision.get("fresh_holdout_spend_authorized") is False,
        "no_proto": decision.get("proto_015_execution_authorized") is False,
        "L4": config.get("compute", {}).get("preferred_accelerator") == "L4",
        "no_fallback": config.get("compute", {}).get("allow_silent_fallback")
        is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("LIP-EVAL-037 contract drifted: " + ", ".join(failed))


def oracle_blend_design_fingerprint(config: Mapping) -> str:
    validate_oracle_blend_contract(config)
    return _canonical_sha256(config)


def expected_oracle_blend_keys(
    task_ids: Sequence[str],
    phase: str,
    *,
    selected_alpha: float | None = None,
) -> set[tuple[str, str, int, float]]:
    if len(task_ids) != 32 or len(set(map(str, task_ids))) != 32:
        raise ValueError("LIP-EVAL-037 requires exactly 32 unique tasks")
    if phase == "screen":
        alphas = ORACLE_BLEND_SCREEN_ALPHAS
        seeds = (ORACLE_BLEND_SCREEN_GENERATION_SEED,)
    elif phase == "confirm":
        if selected_alpha not in ORACLE_BLEND_SCREEN_ALPHAS:
            raise ValueError("confirmation requires one frozen screen alpha")
        alphas = (float(selected_alpha),)
        seeds = ORACLE_BLEND_CONFIRMATION_GENERATION_SEEDS
    else:
        raise ValueError("phase must be screen or confirm")
    return {
        (str(task_id), condition, generation_seed, float(alpha))
        for task_id in task_ids
        for generation_seed in seeds
        for alpha in alphas
        for condition in ORACLE_BLEND_CONDITIONS
    }


def _condition_result(rows: Sequence[Mapping]) -> dict:
    total = len(rows)
    if total == 0:
        raise ValueError("condition result cannot summarize zero rows")
    exact = sum(bool(row.get("functional_pass")) for row in rows)
    alias = sum(bool(row.get("alias_functional_pass")) for row in rows)
    core = sum(bool(row.get("core_functional_pass")) for row in rows)
    gaps = sum(bool(row.get("binding_gap")) for row in rows)
    realized = sum(bool(row.get("forced_prefix_realized")) for row in rows)
    return {
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


def _alpha_results(
    records: Sequence[Mapping], *, phase: str, alpha: float
) -> dict[str, dict]:
    return {
        condition: _condition_result(
            [
                row
                for row in records
                if row.get("phase") == phase
                and row.get("condition") == condition
                and float(row.get("blend_alpha", -1)) == float(alpha)
            ]
        )
        for condition in ORACLE_BLEND_CONDITIONS
    }


def summarize_oracle_blend_screen(
    records: Sequence[Mapping], config: Mapping
) -> dict:
    validate_oracle_blend_contract(config)
    if len(records) != ORACLE_BLEND_SCREEN_RECORDS:
        raise ValueError("LIP-EVAL-037 screen phase requires exactly 192 rows")
    policy = config["evaluation"]
    candidates = []
    alpha_results = {}
    for alpha in ORACLE_BLEND_SCREEN_ALPHAS:
        results = _alpha_results(records, phase="screen", alpha=alpha)
        alpha_results[str(alpha)] = results
        matched = results["oracle_blend_matched"]
        shuffled = results["oracle_blend_shuffled"]
        prefix_passed = all(
            result["forced_prefix_realization_rate"]
            >= float(policy["forced_prefix_realization_minimum"])
            for result in results.values()
        )
        capacity_passed = matched["core_functional_rate"] >= float(
            policy["oracle_core_capacity_minimum"]
        )
        specificity_passed = shuffled["core_functional_rate"] <= float(
            policy["oracle_shuffled_core_maximum"]
        )
        candidates.append(
            {
                "alpha": alpha,
                "matched_core_rate": matched["core_functional_rate"],
                "shuffled_core_rate": shuffled["core_functional_rate"],
                "prefix_realization_passed": prefix_passed,
                "capacity_passed": capacity_passed,
                "specificity_passed": specificity_passed,
                "selection_eligible": bool(
                    prefix_passed and capacity_passed and specificity_passed
                ),
            }
        )
    eligible = [item for item in candidates if item["selection_eligible"]]
    selected = (
        sorted(
            eligible,
            key=lambda item: (-item["matched_core_rate"], item["alpha"]),
        )[0]
        if eligible
        else None
    )
    decision_key = "screen_candidate" if selected else "screen_no_candidate"
    decision = config["decision"][decision_key]
    return {
        "reused_anchor_results": dict(config["reused_anchors"]),
        "alpha_results": alpha_results,
        "screen_candidates": candidates,
        "selected_alpha": selected["alpha"] if selected else None,
        "screen_passed": selected is not None,
        "confirmation_authorized_by_frozen_gate": selected is not None,
        "diagnostic_route": decision["route"],
        "recommended_action": decision["action"],
        "claim_eligible": False,
    }


def summarize_oracle_blend_confirmation(
    records: Sequence[Mapping], config: Mapping
) -> dict:
    validate_oracle_blend_contract(config)
    if len(records) != ORACLE_BLEND_EXPECTED_RECORDS:
        raise ValueError("LIP-EVAL-037 confirmation requires exactly 320 rows")
    screen_rows = [row for row in records if row.get("phase") == "screen"]
    screen = summarize_oracle_blend_screen(screen_rows, config)
    alpha = screen["selected_alpha"]
    if alpha is None:
        raise ValueError("confirmation rows exist without a selected alpha")
    confirm = _alpha_results(records, phase="confirm", alpha=float(alpha))
    policy = config["evaluation"]
    matched = confirm["oracle_blend_matched"]
    shuffled = confirm["oracle_blend_shuffled"]
    prefix_passed = all(
        result["forced_prefix_realization_rate"]
        >= float(policy["forced_prefix_realization_minimum"])
        for result in confirm.values()
    )
    capacity_passed = matched["core_functional_rate"] >= float(
        policy["oracle_core_capacity_minimum"]
    )
    specificity_passed = shuffled["core_functional_rate"] <= float(
        policy["oracle_shuffled_core_maximum"]
    )
    passed = bool(prefix_passed and capacity_passed and specificity_passed)
    decision_key = "confirmation_passed" if passed else "confirmation_failed"
    decision = config["decision"][decision_key]
    return {
        **screen,
        "confirmation_results": confirm,
        "confirmation_gates": {
            "forced_prefix_realization_passed": prefix_passed,
            "oracle_capacity_passed": capacity_passed,
            "oracle_specificity_passed": specificity_passed,
            "oracle_core_capacity_minimum": float(
                policy["oracle_core_capacity_minimum"]
            ),
            "oracle_shuffled_core_maximum": float(
                policy["oracle_shuffled_core_maximum"]
            ),
        },
        "confirmation_passed": passed,
        "diagnostic_route": decision["route"],
        "recommended_action": decision["action"],
        "can_upgrade_EVAL_033": False,
        "can_upgrade_EVAL_034": False,
        "can_upgrade_EVAL_035": False,
        "can_upgrade_EVAL_036": False,
        "claim_eligible": False,
    }
