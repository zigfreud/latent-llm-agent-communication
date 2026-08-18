from pathlib import Path
from statistics import mean

import pytest
import torch

from src.evaluation.constant_entry_point_screen import (
    CONSTANT_ENTRY_POINT_REPLICA_CONDITIONS,
    CONSTANT_ENTRY_POINT_SHARED_CONDITIONS,
    canonicalize_task,
    constant_entry_point_design_fingerprint,
    expected_constant_entry_point_keys,
    replace_identifier,
    summarize_constant_entry_point_screen,
    validate_constant_entry_point_contract,
)
from src.evaluation.functional_bridge_screen import (
    FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS,
    FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS,
)
from src.evaluation.oracle_functional import stable_seed
from src.pipelines.constant_entry_point_screen import (
    _constant_inputs,
    validate_constant_entry_point_runtime_contract,
)
from src.pipelines.oracle_experiment import load_yaml
from src.scripts.evaluate_constant_entry_point_screen import (
    validate_constant_entry_point_grid,
)
from src.scripts.run_hardened_oracle_evaluation import evaluator_for_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "config"
    / "LIP-EVAL-035_constant_opaque_entry_point_receiver_screen.yaml"
)
SOURCE_CONFIG_PATH = ROOT / "config" / "LIP-EVAL-033_functional_bridge_screen.yaml"


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


def _task(task_id: str = "0") -> dict:
    return {
        "task_id": task_id,
        "prompt": "Return the answer.\n\nRequired function name: `answer`.",
        "entry_point": "answer",
        "test_list": [
            f"assert answer() == {int(task_id)}",
            "assert answer('answer') != 'answer'",
        ],
        "test_setup_code": "",
    }


def _scored(
    *,
    exact_positive_seeds: set[int] = frozenset(),
    core_positive_seeds: set[int] = frozenset(),
    oracle_capacity: bool = True,
    oracle_shuffled: bool = False,
    no_packet: bool = False,
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
                        "alias_eligible": True,
                    }
                )
            for training_seed in FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS:
                exact_pass = training_seed in exact_positive_seeds
                core_pass = exact_pass or training_seed in core_positive_seeds
                for condition in CONSTANT_ENTRY_POINT_REPLICA_CONDITIONS:
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
                            "alias_eligible": True,
                        }
                    )
    return rows


def test_contract_and_predecessor_runtime_validate():
    config = _config()
    validate_constant_entry_point_contract(config)
    assert len(constant_entry_point_design_fingerprint(config)) == 64
    source, predecessor, registry = validate_constant_entry_point_runtime_contract(
        config, config_path=CONFIG_PATH
    )
    assert source["experiment_id"] == "LIP-EVAL-033"
    assert predecessor["decision"]["LIP_EVAL_035_design_authorized"] is True
    assert registry["execution"]["complete"] is True


def test_contract_rejects_holdout_or_claim_upgrade():
    config = _config()
    config["decision"]["fresh_holdout_spend_authorized"] = True
    with pytest.raises(ValueError, match="no_holdout"):
        validate_constant_entry_point_contract(config)


def test_expected_grid_contains_864_cells():
    task_ids = [str(index) for index in range(32)]
    keys = expected_constant_entry_point_keys(task_ids)
    assert len(keys) == 864
    assert sum(key[3] is None for key in keys) == 288
    assert sum(key[3] is not None for key in keys) == 576


def test_identifier_rewrite_changes_names_but_not_strings():
    rewritten, count = replace_identifier(
        "assert answer('answer') == answer()  # answer", "answer", "f_0"
    )
    assert count == 2
    assert "f_0('answer')" in rewritten
    assert "f_0()" in rewritten
    assert "# answer" in rewritten


def test_identifier_rewrite_rejects_non_call_uses():
    with pytest.raises(ValueError, match="outside a direct function call"):
        replace_identifier("candidate = answer", "answer", "f_0")


def test_canonical_task_preserves_original_name_for_audit():
    task = canonicalize_task(_task(), "f_0")
    assert task["entry_point"] == "f_0"
    assert task["original_entry_point"] == "answer"
    assert task["canonical_test_identifier_replacements"] == 2
    assert all("f_0" in test for test in task["test_list"])
    assert "'answer'" in task["test_list"][1]


class _CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        return "SYSTEM\n" + "\n".join(message["content"] for message in messages) + "\nASSISTANT"

    def __call__(self, text, **kwargs):
        ids = torch.arange(1, len(text) + 1).unsqueeze(0)
        result = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = torch.tensor(
                [[(index, index + 1) for index in range(len(text))]]
            )
        return result


def test_constant_symbol_is_outside_the_intervention_suffix():
    formatted, inputs, audit = _constant_inputs(
        _config(), load_yaml(SOURCE_CONFIG_PATH), _CharacterTokenizer(), torch.device("cpu")
    )
    assert "f_0" in formatted
    assert inputs["input_ids"].shape[1] > 24
    assert audit["positionally_separated"] is True
    assert max(audit["entry_point_token_positions"]) < audit[
        "intervention_position_min"
    ]


def test_exact_signal_selects_binding_recovery_route():
    summary = summarize_constant_entry_point_screen(
        _scored(exact_positive_seeds={4001, 4003}), _config()
    )
    assert summary["control_gates"]["oracle_capacity_passed"] is True
    assert summary["exact_binding_endpoint"]["signal_detected"] is True
    assert summary["diagnostic_route"] == (
        "constant_binding_recovers_learned_transport_candidate"
    )


def test_core_only_signal_selects_learned_readout_route():
    summary = summarize_constant_entry_point_screen(
        _scored(core_positive_seeds={4001, 4003}), _config()
    )
    assert summary["exact_binding_endpoint"]["signal_detected"] is False
    assert summary["core_recovery_endpoint"]["signal_detected"] is True
    assert summary["diagnostic_route"] == (
        "core_survives_but_prompt_binding_fails"
    )


def test_failed_oracle_capacity_precedes_learned_route():
    summary = summarize_constant_entry_point_screen(
        _scored(
            exact_positive_seeds={4001, 4003},
            oracle_capacity=False,
        ),
        _config(),
    )
    assert summary["diagnostic_route"] == (
        "constant_carrier_oracle_capacity_failure"
    )


def test_non_specific_control_precedes_learned_route():
    summary = summarize_constant_entry_point_screen(
        _scored(
            exact_positive_seeds={4001, 4003},
            no_packet=True,
        ),
        _config(),
    )
    assert summary["diagnostic_route"] == (
        "non_specific_constant_prompt_or_oracle_control_failure"
    )


def test_no_signal_selects_dynamic_trajectory_route():
    summary = summarize_constant_entry_point_screen(_scored(), _config())
    assert summary["diagnostic_route"] == (
        "carrier_reconditioning_erases_alias_recovery"
    )


def test_incomplete_grid_validator_checks_constant_prompt_and_packet_absence():
    config = _config()
    task_ids = [str(index) for index in range(32)]
    donors = {
        task_id: str((index + 1) % 32)
        for index, task_id in enumerate(task_ids)
    }
    metadata = {
        "experiment_id": "LIP-EVAL-035",
        "protocol_version": "lip-constant-opaque-entry-point-screen-v1",
        "design_sha256": constant_entry_point_design_fingerprint(config),
        "config_sha256": "c" * 64,
        "run_scope": "development_only_reused_open_P014_cohort",
        "claim_eligible": False,
        "task_ids": task_ids,
        "task_count": 32,
        "donor_task_ids": donors,
        "shared_conditions": list(CONSTANT_ENTRY_POINT_SHARED_CONDITIONS),
        "replica_conditions": list(CONSTANT_ENTRY_POINT_REPLICA_CONDITIONS),
        "generation_seeds": list(FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS),
        "training_seeds": list(FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS),
        "expected_records": 864,
        "records": 1,
        "complete": False,
        "canonical_entry_point": "f_0",
        "receiver_user_prompt_sha256": "u" * 64,
        "receiver_formatted_prompt_sha256": "p" * 64,
        "receiver_input_ids_sha256": "i" * 64,
        "receiver_attention_mask_sha256": "a" * 64,
        "receiver_position_audit": {"positionally_separated": True},
    }
    task = canonicalize_task(_task("0"), "f_0")
    row = {
        "experiment_id": "LIP-EVAL-035",
        "protocol_version": "lip-constant-opaque-entry-point-screen-v1",
        "design_sha256": metadata["design_sha256"],
        "config_sha256": metadata["config_sha256"],
        "run_scope": "development_only_reused_open_P014_cohort",
        "claim_eligible": False,
        "task_id": "0",
        "condition": "canonical_no_packet",
        "generation_seed": 4127,
        "effective_generation_seed": stable_seed(4127, 0, 14014),
        "training_seed": None,
        "target_prompt_kind": "constant_opaque_entry_point",
        "target_user_prompt_sha256": "u" * 64,
        "target_formatted_prompt_sha256": "p" * 64,
        "target_input_ids_sha256": "i" * 64,
        "target_attention_mask_sha256": "a" * 64,
        "canonical_entry_point": "f_0",
        "canonical_entry_point_positionally_separated": True,
        "packet_present": False,
        "packet_sha256": None,
        "packet_frobenius_norm": None,
        "packet_layer_indices": [],
        "packet_offsets": [],
        "source_task_id": None,
        "donor_task_id": None,
        "output_text": "def f_0():\n    return 0",
        "task_spec": task,
    }
    validation = validate_constant_entry_point_grid(
        [row], metadata, config, allow_incomplete=True
    )
    assert validation["record_count"] == 1
    assert validation["complete"] is False


def test_hardened_dispatch_selects_eval035_evaluator():
    evaluator = evaluator_for_config(_config())
    assert evaluator.__module__ == (
        "src.scripts.evaluate_constant_entry_point_screen"
    )
