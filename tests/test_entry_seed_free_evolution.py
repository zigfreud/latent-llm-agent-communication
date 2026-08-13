from src.evaluation.entry_seed_free_evolution import (
    FREE_EVOLUTION_OPERATOR,
    summarize_entry_seed_gate,
)


def _row(task, variant, seed, operator, attention, residual):
    return {
        "task_id": str(task),
        "variant": variant,
        "training_seed": seed,
        "operator": operator,
        "oracle_replay_alignment": {
            "state_summaries": {
                "attention_output": {"mean_normalized_rmse": attention},
                "residual_output": {"mean_normalized_rmse": residual},
            }
        },
    }


def test_entry_seed_gate_requires_oracle_and_two_primary_replicas():
    rows = [_row(task, "oracle", None, FREE_EVOLUTION_OPERATOR, 0.5, 0.5) for task in range(8)]
    for variant in ("component_contrastive", "structured_linear_regression"):
        for seed in (4001, 4003, 4007):
            for task in range(8):
                rows.append(_row(task, variant, seed, "absolute_replace", 1.0, 1.0))
                value = 0.5
                if variant == "component_contrastive" and seed == 4007:
                    value = 1.5
                rows.append(
                    _row(task, variant, seed, FREE_EVOLUTION_OPERATOR, value, value)
                )
    result = summarize_entry_seed_gate(
        rows,
        variants=("component_contrastive", "structured_linear_regression"),
        seeds=(4001, 4003, 4007),
        primary_variant="component_contrastive",
        minimum_taskwise_improvements=6,
        minimum_passing_replicas=2,
        oracle_anchored_delta_reference={
            "attention_output_nrmse": 1.0,
            "residual_output_nrmse": 1.0,
        },
        oracle_maximum_fraction=0.75,
    )
    assert result["oracle_free_evolution"]["passed"] is True
    assert result["learned_operator"]["passing_primary_replicas"] == 2
    assert result["advance_to_functional_identity_test"] is True


def test_entry_seed_gate_blocks_when_oracle_does_not_improve_enough():
    rows = [_row(0, "oracle", None, FREE_EVOLUTION_OPERATOR, 0.9, 0.9)]
    rows.extend(
        (
            _row(0, "component_contrastive", 4001, "absolute_replace", 1.0, 1.0),
            _row(0, "component_contrastive", 4001, FREE_EVOLUTION_OPERATOR, 0.5, 0.5),
        )
    )
    result = summarize_entry_seed_gate(
        rows,
        variants=("component_contrastive",),
        seeds=(4001,),
        primary_variant="component_contrastive",
        minimum_taskwise_improvements=1,
        minimum_passing_replicas=1,
        oracle_anchored_delta_reference={
            "attention_output_nrmse": 1.0,
            "residual_output_nrmse": 1.0,
        },
        oracle_maximum_fraction=0.75,
    )
    assert result["learned_operator"]["passed"] is True
    assert result["oracle_free_evolution"]["passed"] is False
    assert result["advance_to_functional_identity_test"] is False
