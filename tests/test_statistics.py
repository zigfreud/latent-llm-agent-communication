import pytest

from src.evaluation.statistics import (
    sign_flip_p_value,
    summarize_fixed_sequence,
    summarize_metric,
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
    assert comparison["p_value_two_sided"] == pytest.approx(0.125)
    assert comparison["p_value_holm"] == pytest.approx(0.125)
    assert comparison["p_value_method"] == "exact"
    assert comparison["by_training_seed"]["41"]["mean_difference"] == 1.0


def test_sign_flip_requires_paired_tasks():
    with pytest.raises(ValueError, match="at least one"):
        sign_flip_p_value([])


def test_one_sided_sign_flip_uses_registered_direction():
    p_value, method = sign_flip_p_value([1.0] * 8, alternative="greater")
    assert p_value == pytest.approx(1.0 / 256.0)
    assert method == "exact"


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
    assert hypotheses[0]["tested"] is True
    assert hypotheses[0]["rejected"] is True
    assert hypotheses[1]["tested"] is True
    assert hypotheses[1]["rejected"] is False
    assert hypotheses[2]["p_value"] < 0.05
    assert hypotheses[2]["tested"] is False
    assert hypotheses[2]["rejected"] is False
