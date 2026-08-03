from src.scripts.plot_oracle_functional_capacity import (
    control_interval,
    functional_rates,
    prefix_recovery,
)


def position_summary():
    return {
        "by_packet_size": {
            str(packet_size): {
                split: {
                    "windows": {
                        "first_8_tokens": {
                            "recovery_fraction": value,
                        }
                    }
                }
                for split, value in (
                    ("selection", packet_size / 40),
                    ("confirmation", packet_size / 50),
                )
            }
            for packet_size in (8, 16, 32)
        }
    }


def functional_summary():
    conditions = {
        "neutral_no_lip": {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
        "text_only_no_lip": {"mean": 0.4, "ci_lower": 0.2, "ci_upper": 0.6},
    }
    for packet_size in (8, 16, 32):
        conditions[f"oracle_packet_k{packet_size}"] = {
            "mean": packet_size / 100,
            "ci_lower": packet_size / 200,
            "ci_upper": packet_size / 50,
        }
        conditions[f"shuffled_oracle_packet_k{packet_size}"] = {
            "mean": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }
    return {"metrics": {"functional_pass": {"conditions": conditions}}}


def test_prefix_recovery_uses_registered_functional_capacities():
    packet_sizes, values = prefix_recovery(position_summary(), split="selection")
    assert packet_sizes == [8, 16, 32]
    assert values == [0.2, 0.4, 0.8]


def test_functional_rates_return_asymmetric_errors():
    packet_sizes, means, lower, upper = functional_rates(
        functional_summary(), condition_prefix="oracle_packet"
    )
    assert packet_sizes == [8, 16, 32]
    assert means == [0.08, 0.16, 0.32]
    assert lower == [0.04, 0.08, 0.16]
    assert upper == [0.08, 0.16, 0.32]


def test_control_interval_preserves_task_clustered_bounds():
    assert control_interval(functional_summary(), "text_only_no_lip") == (
        0.4,
        0.2,
        0.6,
    )
