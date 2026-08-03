import pytest

from src.evaluation.source_only import (
    build_condition_plan,
    design_fingerprint,
    derangement_indices,
    infer_entry_point_from_tests,
    task_prompt_with_entry_point,
    target_prompt_for_condition,
)


CONDITIONS = [
    "neutral_no_lip",
    "text_only_no_lip",
    "source_latent",
    "shuffled_source_latent",
    "random_norm_matched",
    "oracle_target_latent",
]


def test_shuffled_control_is_a_reproducible_derangement():
    first = derangement_indices(8, 123)
    assert first == derangement_indices(8, 123)
    assert sorted(first) == list(range(8))
    assert all(index != value for index, value in enumerate(first))


def test_target_prompt_policy_prevents_task_text_in_source_latent_condition():
    task = "SECRET TASK PROMPT"
    neutral = "Return Python code."
    assert target_prompt_for_condition("source_latent", task, neutral) == neutral
    assert target_prompt_for_condition("text_only_no_lip", task, neutral) == task


def test_condition_plan_records_mismatched_vector_source():
    plan = build_condition_plan(["a", "b", "c"], CONDITIONS, seed=5)
    shuffled = [item for item in plan if item.condition == "shuffled_source_latent"]
    assert len(shuffled) == 3
    assert all(item.task_index != item.vector_index for item in shuffled)
    source = [item for item in plan if item.condition == "source_latent"]
    assert all(item.task_index == item.vector_index for item in source)


def test_shuffled_control_rejects_single_task():
    with pytest.raises(ValueError, match="at least two"):
        build_condition_plan(
            ["only"],
            ["neutral_no_lip", "source_latent", "shuffled_source_latent"],
            seed=1,
        )


def test_design_fingerprint_changes_with_target_text_policy():
    first = {"conditions": CONDITIONS, "neutral_target_prompt": "neutral"}
    second = {"conditions": CONDITIONS, "neutral_target_prompt": "different"}
    assert design_fingerprint(first) == design_fingerprint(dict(first))
    assert design_fingerprint(first) != design_fingerprint(second)


def test_entry_point_is_inferred_and_bound_into_transmitted_prompt():
    tests = [
        "assert sum_elements((1, 2, 3)) == 6",
        "assert sum_elements((4, 5)) == 9",
    ]
    entry_point = infer_entry_point_from_tests(tests)
    assert entry_point == "sum_elements"
    prompt = task_prompt_with_entry_point("Sum tuple elements.", entry_point)
    assert prompt.endswith("Required function name: `sum_elements`.")


def test_entry_point_inference_rejects_inconsistent_tests():
    with pytest.raises(ValueError, match="one common entry point"):
        infer_entry_point_from_tests(["assert foo()", "assert bar()"])


def test_entry_point_inference_ignores_nested_argument_constructors():
    tests = [
        "assert max_chain_length([Pair(5, 24), Pair(15, 25)]) == 2",
        "assert max_chain_length([Pair(1, 2)]) == 1",
    ]
    assert infer_entry_point_from_tests(tests) == "max_chain_length"
