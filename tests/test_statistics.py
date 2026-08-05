import pytest

from src.evaluation.statistics import (
    holm_adjust,
    sign_flip_p_value,
    summarize_fixed_sequence,
    summarize_gatekept_holm,
    summarize_metric,
    summarize_two_gate_holm,
    task_means,
)


def records_for_four_tasks():
    rows = []
    for task_id in ("a", "b", "c", "d"):
        for seed in (1, 2):
            rows.append(
                {
                    "task_id": task_id,
                    "condition": "source_latent",
                    "generation_seed": seed,
                    "training_seed": 41,
                    "functional_pass": True,
                }
            )
            rows.append(
                {
                    "task_id": task_id,
                    "condition": "neutral_no_lip",
                    "generation_seed": seed,
                    "training_seed": 41,
                    "functional_pass": False,
                }
            )
    return rows


def test_replicates_are_averaged_within_task():
    rows = [
        {"task_id": "a", "condition": "x", "score": 0},
        {"task_id": "a", "condition": "x", "score": 1},
        {"task_id": "b", "condition": "x", "score": 1},
    ]
    assert task_means(rows, "x", "score") == {"a": 0.5, "b": 1.0}


def test_summary_uses_paired_task_differences():
    summary = summarize_metric(
        records_for_four_tasks(),
        "functional_pass",
        ["source_latent", "neutral_no_lip"],
        [["source_latent", "neutral_no_lip"]],
        bootstrap_iterations=100,
        seed=9,
    )
    assert summary["conditions"]["source_latent"]["mean"] == 1.0
    comparison = summary["comparisons"][0]
    assert comparison["mean_difference"] == 1.0
    assert comparison["task_count"] == 4
    assert comparison["nonzero_task_count"] == 4
    assert comparison["p_value_two_sided"] == pytest.approx(0.125)
    assert comparison["p_value_holm"] == pytest.approx(0.125)
    assert comparison["p_value_method"] == "exact"
    assert comparison["by_training_seed"]["41"]["mean_difference"] == 1.0
    assert summary["conditions"]["source_latent"]["by_generation_seed"] == {
        "1": {"task_count": 4, "mean": 1.0},
        "2": {"task_count": 4, "mean": 1.0},
    }
    assert comparison["by_generation_seed"] == {
        "1": {"task_count": 4, "mean_difference": 1.0},
        "2": {"task_count": 4, "mean_difference": 1.0},
    }


def test_sign_flip_requires_paired_tasks():
    with pytest.raises(ValueError, match="at least one"):
        sign_flip_p_value([])


def test_one_sided_sign_flip_uses_registered_direction():
    p_value, method = sign_flip_p_value([1.0] * 8, alternative="greater")
    assert p_value == pytest.approx(1.0 / 256.0)
    assert method == "exact"


def test_sign_flip_ignores_zero_difference_clusters_for_exact_enumeration():
    p_value, method = sign_flip_p_value(
        [1.0] * 5 + [0.0] * 27,
        alternative="greater",
    )
    assert p_value == pytest.approx(1.0 / 32.0)
    assert method == "exact"


def test_holm_adjust_preserves_input_order_and_step_down_monotonicity():
    assert holm_adjust([0.04, 0.01, 0.03]) == pytest.approx([0.06, 0.03, 0.06])
    with pytest.raises(ValueError, match="between zero and one"):
        holm_adjust([1.1])


def test_gatekept_holm_opens_family_only_after_anchor_rejection():
    records = []
    for task_index in range(8):
        task_id = f"task-{task_index}"
        for condition, value in (
            ("anchor", 1),
            ("anchor_control", 0),
            ("a", 1),
            ("a_control", 0),
            ("b", int(task_index < 2)),
            ("b_control", 0),
        ):
            records.append(
                {"task_id": task_id, "condition": condition, "score": value}
            )
    summary = summarize_gatekept_holm(
        records,
        "score",
        ["anchor", "anchor_control"],
        [["a", "a_control"], ["b", "b_control"]],
        bootstrap_iterations=100,
    )
    assert summary["anchor"]["rejected"] is True
    assert summary["family"][0]["tested"] is True
    assert summary["family"][0]["p_value_holm"] == pytest.approx(2 / 256)
    assert summary["family"][0]["rejected"] is True
    assert summary["family"][1]["rejected"] is False

    failed_anchor = summarize_gatekept_holm(
        [
            {
                "task_id": record["task_id"],
                "condition": record["condition"],
                "score": (
                    0 if record["condition"] == "anchor" else record["score"]
                ),
            }
            for record in records
        ],
        "score",
        ["anchor", "anchor_control"],
        [["a", "a_control"]],
        bootstrap_iterations=100,
    )
    assert failed_anchor["anchor"]["rejected"] is False
    assert failed_anchor["family"][0]["tested"] is False
    assert failed_anchor["family"][0]["rejected"] is False


def test_two_gate_holm_opens_one_global_family_only_after_both_gates():
    records = []
    for task_index in range(10):
        task_id = f"task-{task_index}"
        for condition, value in (
            ("gate_1", 1),
            ("gate_1_control", 0),
            ("gate_2", 1),
            ("gate_2_control", 0),
            ("component_a", 1),
            ("component_a_control", 0),
            ("component_b", int(task_index < 2)),
            ("component_b_control", 0),
        ):
            records.append(
                {"task_id": task_id, "condition": condition, "score": value}
            )
    summary = summarize_two_gate_holm(
        records,
        "score",
        [
            ["gate_1", "gate_1_control"],
            ["gate_2", "gate_2_control"],
        ],
        [
            ["component_a", "component_a_control"],
            ["component_b", "component_b_control"],
        ],
        bootstrap_iterations=100,
    )
    assert [item["rejected"] for item in summary["gates"]] == [True, True]
    assert summary["family"][0]["tested"] is True
    assert summary["family"][0]["p_value_holm"] == pytest.approx(2 / 1024)
    assert summary["family"][0]["rejected"] is True
    assert summary["family"][1]["rejected"] is False

    failed = summarize_two_gate_holm(
        [
            {
                **record,
                "score": (
                    0 if record["condition"] == "gate_2" else record["score"]
                ),
            }
            for record in records
        ],
        "score",
        [
            ["gate_1", "gate_1_control"],
            ["gate_2", "gate_2_control"],
        ],
        [["component_a", "component_a_control"]],
        bootstrap_iterations=100,
    )
    assert failed["gates"][0]["rejected"] is True
    assert failed["gates"][1]["rejected"] is False
    assert failed["family"][0]["tested"] is False
    assert failed["family"][0]["rejected"] is False


def test_fixed_sequence_stops_confirmatory_rejection_after_first_failure():
    rows = []
    for task_id in tuple("abcdefgh"):
        rows.extend(
            [
                {"task_id": task_id, "condition": "depth_32", "score": 1},
                {"task_id": task_id, "condition": "shuffle_32", "score": 0},
                {"task_id": task_id, "condition": "depth_24", "score": 0},
                {"task_id": task_id, "condition": "shuffle_24", "score": 0},
                {"task_id": task_id, "condition": "depth_16", "score": 1},
                {"task_id": task_id, "condition": "shuffle_16", "score": 0},
            ]
        )
    summary = summarize_fixed_sequence(
        rows,
        "score",
        [
            ["depth_32", "shuffle_32"],
            ["depth_24", "shuffle_24"],
            ["depth_16", "shuffle_16"],
        ],
        bootstrap_iterations=100,
        seed=9,
    )
    hypotheses = summary["hypotheses"]
    assert hypotheses[1]["nonzero_task_count"] == 0
    assert hypotheses[0]["tested"] is True
    assert hypotheses[0]["rejected"] is True
    assert hypotheses[1]["tested"] is True
    assert hypotheses[1]["rejected"] is False
    assert hypotheses[2]["p_value"] < 0.05
    assert hypotheses[2]["tested"] is False
    assert hypotheses[2]["rejected"] is False
