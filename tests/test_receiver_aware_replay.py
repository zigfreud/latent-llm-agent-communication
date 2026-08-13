import pytest

from src.evaluation.receiver_aware_replay import summarize_receiver_aware_gate


def _row(task_id, variant, seed, operator, jump, attention, residual):
    return {
        "task_id": str(task_id),
        "variant": variant,
        "training_seed": seed,
        "operator": operator,
        "intervention_jump": {
            "transition_summary": {"mean_relative_jump_rms": jump}
        },
        "oracle_replay_alignment": {
            "state_summaries": {
                "attention_output": {"mean_normalized_rmse": attention},
                "residual_output": {"mean_normalized_rmse": residual},
            }
        },
    }


def test_gate_requires_all_metrics_and_two_primary_replicas():
    rows = []
    variants = ("component_contrastive", "structured_linear_regression")
    seeds = (4001, 4003, 4007)
    for variant in variants:
        for seed in seeds:
            for task in range(8):
                rows.append(
                    _row(
                        task,
                        variant,
                        seed,
                        "absolute_replace",
                        1.0,
                        1.0,
                        1.0,
                    )
                )
                add_value = 0.5
                if variant == "component_contrastive" and seed == 4007:
                    add_value = 1.5
                rows.append(
                    _row(
                        task,
                        variant,
                        seed,
                        "live_task_delta_add",
                        add_value,
                        add_value,
                        add_value,
                    )
                )

    result = summarize_receiver_aware_gate(
        rows,
        variants=variants,
        seeds=seeds,
        primary_variant="component_contrastive",
        minimum_taskwise_improvements=6,
        minimum_passing_replicas=2,
    )
    assert result["passing_primary_replicas"] == 2
    assert result["advance_to_functional_identity_test"] is True


def test_gate_rejects_incomplete_operator_pair():
    rows = [
        _row(
            0,
            "component_contrastive",
            4001,
            "absolute_replace",
            1.0,
            1.0,
            1.0,
        )
    ]
    with pytest.raises(ValueError, match="missing paired operator row"):
        summarize_receiver_aware_gate(
            rows,
            variants=("component_contrastive",),
            seeds=(4001,),
            primary_variant="component_contrastive",
            minimum_taskwise_improvements=1,
            minimum_passing_replicas=1,
        )
