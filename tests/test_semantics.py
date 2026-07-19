from src.evaluation.semantics import (
    check_syntax,
    evaluate_generation,
    extract_code,
)


def test_extract_code_handles_python_fence():
    text = "Result:\n```python\ndef add(a, b):\n    return a + b\n```"
    assert extract_code(text).startswith("def add")
    assert check_syntax(extract_code(text))["syntax_pass"] is True


def test_syntax_failure_is_recorded_without_execution():
    result = evaluate_generation(
        {"task_id": "x", "condition": "source_latent", "output_text": "def broken("},
        {"test_list": ["assert True"]},
    )
    assert result["syntax_pass"] is False
    assert result["functional_pass"] is None


def test_functional_subprocess_passes_and_fails_candidate_tests():
    task = {"test_list": ["assert add(2, 3) == 5"]}
    passed = evaluate_generation(
        {"task_id": "x", "condition": "source_latent", "output_text": "def add(a, b): return a + b"},
        task,
        run_functional=True,
    )
    failed = evaluate_generation(
        {"task_id": "x", "condition": "source_latent", "output_text": "def add(a, b): return a - b"},
        task,
        run_functional=True,
    )
    assert passed["functional_pass"] is True
    assert failed["functional_pass"] is False
    assert failed["functional_error_type"] == "AssertionError"
