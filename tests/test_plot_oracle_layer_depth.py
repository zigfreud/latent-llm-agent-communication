from src.scripts.plot_oracle_layer_depth import (
    control_interval,
    depth_rates,
    primary_annotations,
)


def functional_summary():
    conditions = {
        "neutral_no_lip": {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
        "text_only_no_lip": {"mean": 0.5, "ci_lower": 0.25, "ci_upper": 0.75},
    }
    scopes = (
        "early_quarter_input",
        "early_half_input",
        "early_three_quarters_input",
        "all_layer_input",
    )
    for index, scope in enumerate(scopes, start=1):
        conditions[f"oracle_{scope}_k32"] = {
            "mean": index / 10,
            "ci_lower": index / 20,
            "ci_upper": index / 5,
        }
        conditions[f"shuffled_oracle_{scope}_k32"] = {
            "mean": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }
    return {
        "metrics": {"functional_pass": {"conditions": conditions}},
        "primary_inference": {
            "hypotheses": [
                {
                    "treatment": f"oracle_{scope}_k32",
                    "tested": index < 2,
                    "p_value": 0.03125 if index == 0 else 1.0,
                }
                for index, scope in enumerate(reversed(scopes))
            ]
        },
    }


def test_depth_rates_preserve_registered_early_prefix_order():
    depths, means, lower, upper = depth_rates(functional_summary())
    assert depths == [8, 16, 24, 32]
    assert means == [0.1, 0.2, 0.3, 0.4]
    assert lower == [0.05, 0.1, 0.15, 0.2]
    assert upper == [0.1, 0.2, 0.3, 0.4]


def test_depth_rates_read_matched_capacity_controls():
    _, means, _, _ = depth_rates(functional_summary(), shuffled=True)
    assert means == [0.0, 0.0, 0.0, 0.0]


def test_layer_depth_controls_and_fixed_sequence_annotations():
    summary = functional_summary()
    assert control_interval(summary, "text_only_no_lip") == (0.5, 0.25, 0.75)
    assert primary_annotations(summary) == {
        32: "p=0.0312",
        24: "p=1",
        16: "gate stopped",
        8: "gate stopped",
    }
