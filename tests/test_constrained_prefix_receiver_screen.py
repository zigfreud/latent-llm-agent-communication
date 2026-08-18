import hashlib
import json
from pathlib import Path
from statistics import mean

import pytest
import torch

from src.evaluation.constrained_prefix_receiver_screen import (
    CONSTRAINED_PREFIX_CONTROL_CONDITIONS,
    CONSTRAINED_PREFIX_LEARNED_CONDITIONS,
    constrained_prefix_design_fingerprint,
    expected_constrained_prefix_keys,
    summarize_constrained_prefix_controls,
    summarize_constrained_prefix_screen,
    validate_constrained_prefix_contract,
)
from src.evaluation.functional_bridge_screen import (
    FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS,
    FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS,
)
from src.pipelines.constrained_prefix_receiver_screen import (
    _prefix_token_ids,
    _validate_control_lock,
    validate_constrained_prefix_runtime_contract,
)
from src.pipelines.oracle_experiment import load_yaml
from src.pipelines.oracle_memory import _ForcedCompletionPrefixProcessor
from src.scripts.run_hardened_oracle_evaluation import evaluator_for_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "config" / "LIP-EVAL-036_constrained_prefix_receiver_screen.yaml"
)


def _config():
    return load_yaml(CONFIG_PATH)


@pytest.fixture(autouse=True)
def _fast_deterministic_statistics(monkeypatch):
    monkeypatch.setattr(
        "src.evaluation.constant_entry_point_screen.bootstrap_mean_ci",
        lambda values, **kwargs: (mean(values), mean(values)),
    )
    monkeypatch.setattr(
        "src.evaluation.constant_entry_point_screen.sign_flip_p_value",
        lambda values, **kwargs: (
            0.001 if mean(values) > 0 else 1.0,
            "test_exact",
        ),
    )


def _scored(
    *,
    include_learned: bool = False,
    oracle_capacity: bool = True,
    oracle_shuffled: bool = False,
    no_packet: bool = False,
    prefix_realized: bool = True,
    exact_positive_seeds: set[int] = frozenset(),
    core_positive_seeds: set[int] = frozenset(),
):
    rows = []
    for task_id in map(str, range(32)):
        for generation_seed in FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS:
            shared_values = {
                "canonical_no_packet": no_packet,
                "oracle_teacher_matched": oracle_capacity,
                "oracle_teacher_shuffled": oracle_shuffled,
            }
            for condition, core_pass in shared_values.items():
                exact_pass = core_pass and condition == "oracle_teacher_matched"
                rows.append(
                    {
                        "task_id": task_id,
                        "condition": condition,
                        "generation_seed": generation_seed,
                        "training_seed": None,
                        "functional_pass": exact_pass,
                        "alias_functional_pass": core_pass,
                        "core_functional_pass": core_pass,
                        "binding_gap": core_pass and not exact_pass,
                        "forced_prefix_realized": prefix_realized,
                    }
                )
            if not include_learned:
                continue
            for training_seed in FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS:
                exact_pass = training_seed in exact_positive_seeds
                core_pass = exact_pass or training_seed in core_positive_seeds
                for condition in CONSTRAINED_PREFIX_LEARNED_CONDITIONS:
                    matched = condition == "learned_matched"
                    rows.append(
                        {
                            "task_id": task_id,
                            "condition": condition,
                            "generation_seed": generation_seed,
                            "training_seed": training_seed,
                            "functional_pass": exact_pass if matched else False,
                            "alias_functional_pass": core_pass if matched else False,
                            "core_functional_pass": core_pass if matched else False,
                            "binding_gap": (
                                core_pass and not exact_pass if matched else False
                            ),
                            "forced_prefix_realized": prefix_realized,
                        }
                    )
    return rows


def test_contract_and_runtime_predecessor_validate():
    config = _config()
    validate_constrained_prefix_contract(config)
    assert len(constrained_prefix_design_fingerprint(config)) == 64
    source, predecessor, registry = validate_constrained_prefix_runtime_contract(
        config, config_path=CONFIG_PATH
    )
    assert source["experiment_id"] == "LIP-EVAL-033"
    assert predecessor["decision"]["diagnostic_route"] == (
        "constant_carrier_oracle_capacity_failure"
    )
    assert registry["execution"]["complete"] is True


def test_contract_rejects_semantic_or_variable_prefix():
    config = _config()
    config["decoding_interface"]["same_prefix_for_every_task"] = False
    with pytest.raises(ValueError, match="forced_prefix"):
        validate_constrained_prefix_contract(config)


def test_sequential_grid_has_288_controls_and_576_learned_cells():
    task_ids = [str(index) for index in range(32)]
    controls = expected_constrained_prefix_keys(task_ids, "controls")
    learned = expected_constrained_prefix_keys(task_ids, "learned")
    assert len(controls) == 288
    assert len(learned) == 576
    assert len(expected_constrained_prefix_keys(task_ids)) == 864
    assert all(key[1] in CONSTRAINED_PREFIX_CONTROL_CONDITIONS for key in controls)


def test_forced_prefix_processor_constrains_only_initial_completion_tokens():
    processor = _ForcedCompletionPrefixProcessor(4, [2, 5])
    scores = torch.randn(1, 8)
    first = processor(torch.ones(1, 4, dtype=torch.long), scores)
    second = processor(torch.ones(1, 5, dtype=torch.long), scores)
    released = processor(torch.ones(1, 6, dtype=torch.long), scores)
    assert torch.isfinite(first[0, 2])
    assert torch.isneginf(first[0, [0, 1, 3, 4, 5, 6, 7]]).all()
    assert torch.isfinite(second[0, 5])
    assert torch.equal(released, scores)


class _PrefixTokenizer:
    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [1, 2, 3]}

    def decode(self, token_ids, **kwargs):
        assert list(token_ids) == [1, 2, 3]
        return "def f_0"


def test_prefix_tokenization_round_trips_exactly():
    token_ids, decoded = _prefix_token_ids(_PrefixTokenizer(), "def f_0")
    assert token_ids == [1, 2, 3]
    assert decoded == "def f_0"


def test_learned_phase_lock_binds_control_inputs(tmp_path):
    generations = tmp_path / "generations.jsonl"
    generations.write_text(
        "".join(
            json.dumps({"condition": "canonical_no_packet", "row": index}) + "\n"
            for index in range(288)
        ),
        encoding="utf-8",
    )
    metadata = generations.with_suffix(".metadata.json")
    metadata.write_text("{}\n", encoding="utf-8")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    lock = tmp_path / "summary.json"
    lock.write_text(
        json.dumps(
            {
                "experiment_id": "LIP-EVAL-036",
                "protocol_version": "lip-constrained-prefix-receiver-screen-v1",
                "diagnostic_route": "constrained_prefix_controls_passed",
                "subprocess_is_security_sandbox": True,
                "inference": {"controls_passed": True},
                "sandbox": {
                    "input_sha256": {
                        "config": digest(CONFIG_PATH),
                        "generations": digest(generations),
                        "metadata": digest(metadata),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    validated = _validate_control_lock(
        lock,
        config_path=CONFIG_PATH,
        output_path=generations,
        metadata_path=metadata,
        metadata={},
        learned_rows_exist=False,
    )
    assert validated["inference"]["controls_passed"] is True

    metadata.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="metadata hash differs"):
        _validate_control_lock(
            lock,
            config_path=CONFIG_PATH,
            output_path=generations,
            metadata_path=metadata,
            metadata={},
            learned_rows_exist=False,
        )


def test_control_gate_authorizes_learned_phase_only_after_valid_controls():
    passed = summarize_constrained_prefix_controls(_scored(), _config())
    assert passed["controls_passed"] is True
    assert passed["learned_phase_authorized_by_frozen_gate"] is True
    assert passed["diagnostic_route"] == "constrained_prefix_controls_passed"

    failed = summarize_constrained_prefix_controls(
        _scored(oracle_capacity=False), _config()
    )
    assert failed["controls_passed"] is False
    assert failed["diagnostic_route"] == (
        "constrained_prefix_oracle_capacity_failure"
    )


def test_prefix_realization_failure_precedes_oracle_gate():
    summary = summarize_constrained_prefix_controls(
        _scored(prefix_realized=False, oracle_capacity=False), _config()
    )
    assert summary["diagnostic_route"] == "forced_prefix_mechanism_failure"


def test_full_exact_signal_selects_constrained_binding_route():
    summary = summarize_constrained_prefix_screen(
        _scored(include_learned=True, exact_positive_seeds={4001, 4003}),
        _config(),
    )
    assert summary["controls_passed"] is True
    assert summary["exact_binding_endpoint"]["signal_detected"] is True
    assert summary["diagnostic_route"] == (
        "constrained_binding_preserves_learned_transport_candidate"
    )


def test_hardened_dispatch_selects_eval036_evaluator():
    evaluator = evaluator_for_config(_config())
    assert evaluator.__module__ == (
        "src.scripts.evaluate_constrained_prefix_receiver_screen"
    )
