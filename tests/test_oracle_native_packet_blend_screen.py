import hashlib
import json
from pathlib import Path

import pytest
import torch

from src.evaluation.oracle_native_packet_blend_screen import (
    ORACLE_BLEND_SCREEN_ALPHAS,
    expected_oracle_blend_keys,
    oracle_blend_design_fingerprint,
    summarize_oracle_blend_confirmation,
    summarize_oracle_blend_screen,
    validate_oracle_blend_contract,
)
from src.integrations.hooks import make_lip_packet_pre_hook
from src.pipelines.oracle_experiment import load_yaml
from src.pipelines.oracle_native_packet_blend_screen import (
    _validate_screen_lock,
    validate_oracle_blend_runtime_contract,
)
from src.scripts.run_hardened_oracle_evaluation import evaluator_for_config
from src.scripts.evaluate_oracle_native_packet_blend_screen import _observed_layout


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "config" / "LIP-EVAL-037_oracle_native_packet_blend_screen.yaml"
)


def _config():
    return load_yaml(CONFIG_PATH)


def _screen_rows(passes_by_alpha=None):
    passes_by_alpha = passes_by_alpha or {0.25: 20, 0.5: 25, 0.75: 25}
    rows = []
    for task_index in range(32):
        for alpha in ORACLE_BLEND_SCREEN_ALPHAS:
            for condition in ("oracle_blend_matched", "oracle_blend_shuffled"):
                passed = (
                    condition == "oracle_blend_matched"
                    and task_index < passes_by_alpha[alpha]
                )
                rows.append(
                    {
                        "phase": "screen",
                        "task_id": str(task_index),
                        "condition": condition,
                        "generation_seed": 4127,
                        "blend_alpha": alpha,
                        "functional_pass": passed,
                        "alias_functional_pass": passed,
                        "core_functional_pass": passed,
                        "binding_gap": False,
                        "forced_prefix_realized": True,
                    }
                )
    return rows


def _confirmation_rows(alpha=0.5, matched_passes_per_seed=25):
    rows = []
    for task_index in range(32):
        for seed in (4241, 4357):
            for condition in ("oracle_blend_matched", "oracle_blend_shuffled"):
                passed = (
                    condition == "oracle_blend_matched"
                    and task_index < matched_passes_per_seed
                )
                rows.append(
                    {
                        "phase": "confirm",
                        "task_id": str(task_index),
                        "condition": condition,
                        "generation_seed": seed,
                        "blend_alpha": alpha,
                        "functional_pass": passed,
                        "alias_functional_pass": passed,
                        "core_functional_pass": passed,
                        "binding_gap": False,
                        "forced_prefix_realized": True,
                    }
                )
    return rows


def test_contract_runtime_predecessor_and_grid_validate():
    config = _config()
    validate_oracle_blend_contract(config)
    assert len(oracle_blend_design_fingerprint(config)) == 64
    source, predecessor, source_registry = validate_oracle_blend_runtime_contract(
        config, config_path=CONFIG_PATH
    )
    assert source["experiment_id"] == "LIP-EVAL-033"
    assert predecessor["decision"]["diagnostic_route"] == (
        "constrained_prefix_oracle_capacity_failure"
    )
    assert source_registry["execution"]["complete"] is True
    task_ids = [str(index) for index in range(32)]
    assert len(expected_oracle_blend_keys(task_ids, "screen")) == 192
    assert (
        len(expected_oracle_blend_keys(task_ids, "confirm", selected_alpha=0.5))
        == 128
    )


def test_screen_selects_smallest_alpha_on_equal_best_rate():
    summary = summarize_oracle_blend_screen(_screen_rows(), _config())
    assert summary["screen_passed"] is True
    assert summary["selected_alpha"] == 0.5
    assert summary["diagnostic_route"] == "oracle_blend_screen_candidate_selected"


def test_screen_stops_when_no_alpha_reaches_capacity():
    summary = summarize_oracle_blend_screen(
        _screen_rows({0.25: 23, 0.5: 23, 0.75: 23}), _config()
    )
    assert summary["screen_passed"] is False
    assert summary["selected_alpha"] is None
    assert summary["diagnostic_route"] == "oracle_blend_screen_no_candidate"


def test_confirmation_uses_only_locked_alpha_and_unseen_seeds():
    rows = _screen_rows() + _confirmation_rows()
    passed = summarize_oracle_blend_confirmation(rows, _config())
    assert passed["selected_alpha"] == 0.5
    assert passed["confirmation_passed"] is True
    assert passed["diagnostic_route"] == "oracle_blend_capacity_restored_candidate"

    failed = summarize_oracle_blend_confirmation(
        _screen_rows() + _confirmation_rows(matched_passes_per_seed=23), _config()
    )
    assert failed["confirmation_passed"] is False
    assert failed["diagnostic_route"] == "oracle_blend_candidate_does_not_replicate"


def test_blend_hook_interpolates_native_and_packet_exactly():
    native = torch.tensor([[[1.0, 3.0], [2.0, 4.0], [9.0, 11.0]]])
    packet = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    positions = torch.tensor([0, 2])
    hook = make_lip_packet_pre_hook(
        packet, positions, mode="blend", blend_alpha=0.25
    )
    (result,) = hook(None, (native,))
    expected = native.clone()
    expected[0, positions, :] = 0.75 * native[0, positions, :] + 0.25 * packet
    assert torch.equal(result, expected)
    assert torch.equal(native, torch.tensor([[[1.0, 3.0], [2.0, 4.0], [9.0, 11.0]]]))
    assert hook(None, (native,)) is None


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), None])
def test_blend_hook_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="blend_alpha"):
        make_lip_packet_pre_hook(
            torch.ones(1, 2),
            torch.tensor([0]),
            mode="blend",
            blend_alpha=alpha,
        )


def test_screen_lock_binds_screen_inputs(tmp_path):
    generations = tmp_path / "generations.jsonl"
    generations.write_text(
        "".join(
            json.dumps({"phase": "screen", "row": index}) + "\n"
            for index in range(192)
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
                "experiment_id": "LIP-EVAL-037",
                "protocol_version": "lip-oracle-native-packet-blend-screen-v1",
                "diagnostic_route": "oracle_blend_screen_candidate_selected",
                "subprocess_is_security_sandbox": True,
                "inference": {
                    "screen_passed": True,
                    "confirmation_authorized_by_frozen_gate": True,
                    "selected_alpha": 0.5,
                },
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
    validated = _validate_screen_lock(
        lock,
        config_path=CONFIG_PATH,
        output_path=generations,
        metadata_path=metadata,
        metadata={},
        confirmation_rows_exist=False,
    )
    assert validated["inference"]["selected_alpha"] == 0.5


def test_hardened_dispatch_selects_eval037_evaluator():
    evaluator = evaluator_for_config(_config())
    assert evaluator.__module__ == (
        "src.scripts.evaluate_oracle_native_packet_blend_screen"
    )


def test_partial_metric_layout_waits_for_a_paired_control():
    matched_only = [{"task_id": "0", "condition": "oracle_blend_matched"}]
    conditions, comparisons = _observed_layout(matched_only)
    assert conditions == ["oracle_blend_matched"]
    assert comparisons == []

    paired = matched_only + [
        {"task_id": "0", "condition": "oracle_blend_shuffled"}
    ]
    conditions, comparisons = _observed_layout(paired)
    assert conditions == ["oracle_blend_matched", "oracle_blend_shuffled"]
    assert comparisons == [
        ("oracle_blend_matched", "oracle_blend_shuffled")
    ]
