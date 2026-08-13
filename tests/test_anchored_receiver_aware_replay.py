import pytest

from src.evaluation.anchored_receiver_aware_replay import (
    ANCHORED_OPERATOR,
    summarize_anchored_gate,
)


def _row(task_id, variant, seed, operator, value):
    return {
        "task_id": str(task_id),
        "variant": variant,
        "training_seed": seed,
        "operator": operator,
        "intervention_jump": {
            "transition_summary": {"mean_relative_jump_rms": value}
        },
        "oracle_replay_alignment": {
            "state_summaries": {
                "attention_output": {"mean_normalized_rmse": value},
                "residual_output": {"mean_normalized_rmse": value},
            }
        },
    }


def test_anchored_gate_requires_oracle_origin_and_two_primary_replicas():
    rows = []
    for task in range(8):
        rows.append(_row(task, "oracle", None, ANCHORED_OPERATOR, 0.5))
    for variant in ("component_contrastive", "structured_linear_regression"):
        for seed in (4001, 4003, 4007):
            for task in range(8):
                rows.append(_row(task, variant, seed, "absolute_replace", 1.0))
                anchored = 0.5
                if variant == "component_contrastive" and seed == 4007:
                    anchored = 1.5
                rows.append(_row(task, variant, seed, ANCHORED_OPERATOR, anchored))

    result = summarize_anchored_gate(
        rows,
        variants=("component_contrastive", "structured_linear_regression"),
        seeds=(4001, 4003, 4007),
        primary_variant="component_contrastive",
        minimum_taskwise_improvements=6,
        minimum_passing_replicas=2,
        oracle_unanchored_reference={
            "transition_jump_nrmse": 1.0,
            "attention_output_nrmse": 1.0,
            "residual_output_nrmse": 1.0,
        },
        oracle_maximum_fraction={
            "transition_jump_nrmse": 1.0,
            "attention_output_nrmse": 0.75,
            "residual_output_nrmse": 0.75,
        },
    )
    assert result["oracle_entry_origin"]["passed"] is True
    assert result["learned_operator"]["passing_primary_replicas"] == 2
    assert result["advance_to_functional_identity_test"] is True


def test_anchored_gate_blocks_when_oracle_origin_is_not_repaired():
    rows = [_row(0, "oracle", None, ANCHORED_OPERATOR, 0.9)]
    rows.extend(
        (
            _row(0, "component_contrastive", 4001, "absolute_replace", 1.0),
            _row(0, "component_contrastive", 4001, ANCHORED_OPERATOR, 0.5),
        )
    )
    result = summarize_anchored_gate(
        rows,
        variants=("component_contrastive",),
        seeds=(4001,),
        primary_variant="component_contrastive",
        minimum_taskwise_improvements=1,
        minimum_passing_replicas=1,
        oracle_unanchored_reference={
            "transition_jump_nrmse": 1.0,
            "attention_output_nrmse": 1.0,
            "residual_output_nrmse": 1.0,
        },
        oracle_maximum_fraction={
            "transition_jump_nrmse": 1.0,
            "attention_output_nrmse": 0.75,
            "residual_output_nrmse": 0.75,
        },
    )
    assert result["learned_operator"]["passed"] is True
    assert result["oracle_entry_origin"]["passed"] is False
    assert result["advance_to_functional_identity_test"] is False


def test_anchored_gate_rejects_incomplete_reference():
    with pytest.raises(ValueError, match="oracle reference"):
        summarize_anchored_gate(
            [_row(0, "oracle", None, ANCHORED_OPERATOR, 0.5)],
            variants=("component_contrastive",),
            seeds=(4001,),
            primary_variant="component_contrastive",
            minimum_taskwise_improvements=1,
            minimum_passing_replicas=1,
            oracle_unanchored_reference={"transition_jump_nrmse": 1.0},
            oracle_maximum_fraction={
                "transition_jump_nrmse": 1.0,
                "attention_output_nrmse": 0.75,
                "residual_output_nrmse": 0.75,
            },
        )
