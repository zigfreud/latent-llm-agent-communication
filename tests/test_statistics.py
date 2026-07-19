import pytest

from src.evaluation.statistics import sign_flip_p_value, summarize_metric, task_means


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
