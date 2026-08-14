import hashlib
import json
from pathlib import Path

import pytest

from src.evaluation.alias_normalized_diagnostic import (
    alias_diagnostic_design_fingerprint,
    build_single_function_alias,
    summarize_alias_diagnostic,
    validate_alias_diagnostic_contract,
)
from src.evaluation.functional_bridge_screen import (
    FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS,
    FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS,
    FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS,
)
from src.pipelines.oracle_experiment import load_yaml
from src.scripts.evaluate_alias_normalized_diagnostic import evaluate
from src.scripts.run_hardened_oracle_evaluation import evaluator_for_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "config" / "LIP-EVAL-034_alias_normalized_functional_diagnostic.yaml"
)


def _config():
    return load_yaml(CONFIG_PATH)


def _scored(*, matched_positive_seeds: set[int], shuffled_pass: bool = False):
    rows = []
    for task_id in map(str, range(32)):
        for generation_seed in FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS:
            for training_seed in FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS:
                rows.extend(
                    [
                        {
                            "task_id": task_id,
                            "condition": "learned_matched",
                            "generation_seed": generation_seed,
                            "training_seed": training_seed,
                            "alias_eligible": True,
                            "alias_functional_pass": training_seed
                            in matched_positive_seeds,
                        },
                        {
                            "task_id": task_id,
                            "condition": "learned_shuffled",
                            "generation_seed": generation_seed,
                            "training_seed": training_seed,
                            "alias_eligible": True,
                            "alias_functional_pass": shuffled_pass,
                        },
                    ]
                )
    return rows


def test_alias_contract_is_post_hoc_and_cannot_upgrade_eval033():
    config = _config()
    validate_alias_diagnostic_contract(config)
    assert len(alias_diagnostic_design_fingerprint(config)) == 64
    config["decision"]["can_upgrade_EVAL_033"] = True
    with pytest.raises(ValueError, match="cannot_upgrade"):
        validate_alias_diagnostic_contract(config)


def test_single_function_alias_preserves_recursive_self_reference():
    code = "def factorial_alt(n):\n    return 1 if n < 2 else n * factorial_alt(n - 1)"
    result = build_single_function_alias(code, "factorial")
    assert result["eligible"] is True
    assert result["generated_function_name"] == "factorial_alt"
    assert result["normalized_code"].endswith(
        "factorial = factorial_alt\n"
    )
    namespace = {}
    exec(result["normalized_code"], namespace, namespace)
    assert namespace["factorial"](5) == 120


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("answer = 1", "missing_top_level_function"),
        (
            "def first():\n    return 1\n\ndef second():\n    return 2",
            "ambiguous_multiple_top_level_functions",
        ),
        ("def broken(:\n    pass", "syntax_invalid"),
    ],
)
def test_alias_policy_rejects_missing_ambiguous_and_invalid_code(code, reason):
    result = build_single_function_alias(code, "answer")
    assert result["eligible"] is False
    assert result["reason"] == reason
    assert result["normalized_code"] is None


def test_diagnostic_routes_zero_matched_passes_to_dynamic_bridge():
    summary = summarize_alias_diagnostic(
        _scored(matched_positive_seeds=set()), _config()
    )
    assert summary["diagnostic_route"] == "no_alias_normalized_core_recovery"
    assert summary["recommended_action"] == (
        "design_dynamic_closed_loop_trajectory_bridge"
    )
    assert summary["claim_eligible"] is False


def test_diagnostic_routes_two_positive_seeds_to_name_readout_candidate():
    summary = summarize_alias_diagnostic(
        _scored(matched_positive_seeds={4001, 4003}), _config()
    )
    assert summary["primary_diagnostic"]["mean_difference"] > 0
    assert summary["seed_guardrail"]["positive_bridge_seeds"] == 2
    assert summary["diagnostic_route"] == (
        "matched_specific_alias_recovery_candidate"
    )


def test_diagnostic_routes_one_positive_seed_to_seed_sensitive_ambiguity():
    summary = summarize_alias_diagnostic(
        _scored(matched_positive_seeds={4001}), _config()
    )
    assert summary["diagnostic_route"] == (
        "non_specific_or_seed_sensitive_alias_recovery"
    )


def test_hardened_dispatch_selects_eval034_evaluator():
    evaluator = evaluator_for_config(_config())
    assert evaluator.__module__ == (
        "src.scripts.evaluate_alias_normalized_diagnostic"
    )


def test_syntax_only_evaluation_binds_complete_frozen_grid(tmp_path):
    config = _config()
    source = config["source"]
    task_ids = [str(index) for index in range(32)]
    rows = []
    for task_id in task_ids:
        task = {
            "task_id": task_id,
            "entry_point": f"answer_{task_id}",
            "test_list": [f"assert answer_{task_id}() == {int(task_id)}"],
        }
        for generation_seed in FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS:
            for training_seed in FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS:
                for condition in FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS:
                    rows.append(
                        {
                            "experiment_id": "LIP-EVAL-033",
                            "protocol_version": "lip-functional-bridge-screen-v1",
                            "design_sha256": source["design_sha256"],
                            "config_sha256": source["config_sha256"],
                            "claim_eligible": False,
                            "task_id": task_id,
                            "condition": condition,
                            "generation_seed": generation_seed,
                            "training_seed": training_seed,
                            "output_text": (
                                f"def generated_{task_id}():\n    return {task_id}"
                            ),
                            "task_spec": task,
                        }
                    )
    metadata = {
        "experiment_id": "LIP-EVAL-033",
        "protocol_version": "lip-functional-bridge-screen-v1",
        "design_sha256": source["design_sha256"],
        "config_sha256": source["config_sha256"],
        "claim_eligible": False,
        "complete": True,
        "task_ids": task_ids,
        "task_count": 32,
        "records": 576,
        "expected_records": 576,
        "conditions": list(FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS),
        "training_seeds": list(FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS),
        "generation_seeds": list(FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS),
    }
    generations = tmp_path / "generations.jsonl"
    generations.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    metadata_path = generations.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    source["generations_sha256"] = hashlib.sha256(
        generations.read_bytes()
    ).hexdigest()
    source["metadata_sha256"] = hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()

    summary = evaluate(
        config,
        generations,
        tmp_path / "evaluation",
        functional=False,
        allow_incomplete=False,
        overwrite=False,
    )

    assert summary["execution_mode"] == "syntax_only"
    assert summary["diagnostic_route"] == "not_scored_in_hardened_namespace"
    assert summary["source_validation"]["record_count"] == 576
    assert summary["normalization_reason_counts"] == {
        "single_function_aliased": 576
    }
    assert summary["claim_eligible"] is False
