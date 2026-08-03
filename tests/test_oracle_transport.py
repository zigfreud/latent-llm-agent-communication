import hashlib
import json

import pytest
import torch

from src.evaluation.oracle_transport import (
    continuation_token_metrics,
    continuation_token_profile,
    normalize_layer_indices,
    recovery_fraction,
    summarize_packet_capacity,
    summarize_packet_position_recovery,
    summarize_oracle_transport,
)
from src.pipelines.oracle_transport import (
    build_neutral_carrier,
    forward_with_layer_capture,
    forward_with_packet_capture,
    forward_with_packet_replacement,
)
from src.scripts.run_oracle_packet_audit import validate_config as validate_packet_config
from src.scripts.run_oracle_transport_audit import (
    bind_tasks_to_manifest,
    validate_config,
)


def protocol_config():
    return {
        "experiment_id": "LIP-PROTO-002",
        "source_protocol_experiment": "LIP-PROTO-001",
        "models": {"target_model": "target/model"},
        "prompt_protocol": {
            "version": "lip-prompt-v1",
            "mode": "chat_template",
            "add_generation_prompt": True,
            "system_prompt": "Return Python.",
        },
        "runtime": {"device": "auto", "load_4bit": True},
        "data": {
            "tasks_jsonl": "tasks.jsonl",
            "heldout_bundle_manifest": "manifest.json",
            "require_real_bundle": True,
            "task_count": 4,
            "selection_task_count": 2,
            "preflight_task_count": 2,
        },
        "neutral_target_prompt": "Use the latent signal.",
        "audit": {
            "layers": [-1, -2],
            "injection_mode": "replace",
            "reference_max_new_tokens": 8,
            "minimum_reference_tokens": 2,
            "self_check_tasks": 1,
            "minimum_task_advantage_nll": 0.05,
            "minimum_informative_tasks_per_split": 1,
            "minimum_confirmation_recovery": 0.1,
            "maximum_self_nll_delta": 0.0001,
        },
        "output": {"directory": "runs/test"},
    }


def test_continuation_metrics_use_prompt_boundary_alignment():
    logits = torch.full((1, 5, 4), -10.0)
    reference = torch.tensor([2, 1])
    logits[0, 2, 2] = 10.0
    logits[0, 3, 1] = 10.0
    result = continuation_token_metrics(logits, reference, prompt_length=3)
    assert result["token_count"] == 2
    assert result["top1_accuracy"] == 1.0
    assert result["nll"] == pytest.approx(0.0, abs=1e-6)


def test_continuation_profile_preserves_relative_token_positions():
    logits = torch.full((1, 6, 4), -10.0)
    reference = torch.tensor([2, 1, 3])
    logits[0, 2, 2] = 10.0
    logits[0, 3, 0] = 10.0
    logits[0, 4, 3] = 10.0
    result = continuation_token_profile(logits, reference, prompt_length=3)
    assert result["token_count"] == 3
    assert result["token_top1_correct"] == [True, False, True]
    assert result["token_nlls"][0] == pytest.approx(0.0, abs=1e-6)
    assert result["token_nlls"][1] > 10.0
    assert result["token_nlls"][2] == pytest.approx(0.0, abs=1e-6)


def test_recovery_fraction_requires_task_prompt_advantage():
    recovery, advantage, informative = recovery_fraction(
        1.0, 2.0, 1.5, minimum_task_advantage=0.05
    )
    assert informative is True
    assert advantage == pytest.approx(1.0)
    assert recovery == pytest.approx(0.5)

    recovery, advantage, informative = recovery_fraction(
        1.0, 1.01, 0.9, minimum_task_advantage=0.05
    )
    assert recovery is None
    assert advantage == pytest.approx(0.01)
    assert informative is False


def test_layer_indices_are_negative_unique_and_in_range():
    assert normalize_layer_indices([-1, -2, -4], 4) == [-1, -2, -4]
    with pytest.raises(ValueError, match="unique"):
        normalize_layer_indices([-1, -1], 4)
    with pytest.raises(ValueError, match="negative"):
        normalize_layer_indices([0], 4)
    with pytest.raises(ValueError, match="negative"):
        normalize_layer_indices([-5], 4)


def test_summary_selects_on_first_split_and_gates_on_confirmation():
    task_ids = ["s1", "s2", "c1", "c2"]
    rows = []
    recoveries = {
        -1: {"s1": 0.8, "s2": 0.6, "c1": 0.2, "c2": 0.2},
        -2: {"s1": 0.3, "s2": 0.3, "c1": 0.9, "c2": 0.9},
    }
    for layer, by_task in recoveries.items():
        for task_id, recovery in by_task.items():
            rows.append(
                {
                    "task_id": task_id,
                    "layer_idx": layer,
                    "informative": True,
                    "task_advantage_nll": 1.0,
                    "recovery_fraction": recovery,
                    "self_nll_delta": 0.0 if task_id == "s1" else None,
                }
            )
    summary = summarize_oracle_transport(
        rows,
        task_ids=task_ids,
        layers=[-1, -2],
        selection_task_count=2,
        minimum_informative_tasks_per_split=2,
        minimum_confirmation_recovery=0.1,
        maximum_self_nll_delta=0.0001,
        run_scope="full",
    )
    assert summary["selected_layer"] == -1
    assert summary["confirmation"]["mean_recovery_fraction"] == pytest.approx(0.2)
    assert summary["gate"]["passed"] is True


def test_preflight_summary_is_never_claim_eligible():
    rows = [
        {
            "task_id": task_id,
            "layer_idx": -1,
            "informative": True,
            "task_advantage_nll": 1.0,
            "recovery_fraction": 1.0,
            "self_nll_delta": 0.0,
        }
        for task_id in ("a", "b")
    ]
    summary = summarize_oracle_transport(
        rows,
        task_ids=["a", "b"],
        layers=[-1],
        selection_task_count=1,
        minimum_informative_tasks_per_split=1,
        minimum_confirmation_recovery=0.1,
        maximum_self_nll_delta=0.0001,
        run_scope="preflight",
    )
    assert summary["claim_eligible"] is False
    assert summary["gate"]["passed"] is False


def test_manifest_binding_checks_real_target_protocol_and_prompt_hashes(tmp_path):
    config = protocol_config()
    tasks = [
        {"task_id": "a", "prompt": "prompt a"},
        {"task_id": "b", "prompt": "prompt b"},
        {"task_id": "c", "prompt": "prompt c"},
        {"task_id": "d", "prompt": "prompt d"},
    ]
    manifest = {
        "extraction_mode": "real",
        "target_model": "target/model",
        "target_model_revision": "a" * 40,
        "prompt_protocols": {"target": config["prompt_protocol"]},
        "sampled_ids": [task["task_id"] for task in tasks],
        "sampled_prompt_sha256": [
            hashlib.sha256(task["prompt"].encode()).hexdigest() for task in tasks
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config["data"]["heldout_bundle_manifest"] = str(manifest_path)
    bound, loaded, path = bind_tasks_to_manifest(config, list(reversed(tasks)))
    assert [task["task_id"] for task in bound] == ["a", "b", "c", "d"]
    assert loaded == manifest
    assert path == manifest_path


def test_config_rejects_factorial_expansion_of_injection_mode():
    config = protocol_config()
    validate_config(config)
    config["audit"]["injection_mode"] = "add"
    with pytest.raises(ValueError, match="fixes injection_mode=replace"):
        validate_config(config)


def test_layer_capture_reads_module_output_before_model_postprocessing():
    class Layer(torch.nn.Module):
        def __init__(self, increment):
            super().__init__()
            self.increment = increment

        def forward(self, hidden):
            return hidden + self.increment

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(1.0), Layer(2.0)])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()

        def forward(
            self,
            input_ids,
            attention_mask,
            use_cache,
            output_hidden_states,
            return_dict,
        ):
            hidden = input_ids.float().unsqueeze(-1)
            for layer in self.model.layers:
                hidden = layer(hidden)
            return type("Output", (), {"logits": hidden * 10.0})()

    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    outputs, captured = forward_with_layer_capture(
        Model(), inputs, layers=[-1, -2], position=2
    )
    assert captured[-2].item() == 4.0
    assert captured[-1].item() == 6.0
    assert outputs.logits[0, 2, 0].item() == 60.0


def test_masked_left_padding_matches_task_length_without_visible_tokens():
    neutral = {
        "input_ids": torch.tensor([[7, 8, 9]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    carrier = build_neutral_carrier(
        neutral,
        task_prompt_length=5,
        pad_token_id=0,
        mode="left_pad_masked_to_task_length",
    )
    assert carrier["input_ids"].tolist() == [[0, 0, 7, 8, 9]]
    assert carrier["attention_mask"].tolist() == [[0, 0, 1, 1, 1]]


def test_length_controlled_experiment_requires_masked_carrier():
    config = protocol_config()
    config["experiment_id"] = "LIP-PROTO-003"
    config["carrier"] = {"mode": "left_pad_masked_to_task_length"}
    validate_config(config)
    config["carrier"]["mode"] = "native"
    with pytest.raises(ValueError, match="requires carrier.mode"):
        validate_config(config)


def test_packet_summary_selects_smallest_crossing_and_confirms_it():
    task_ids = ["s1", "s2", "c1", "c2"]
    rows = []
    recoveries = {
        1: {task_id: 0.01 for task_id in task_ids},
        2: {"s1": 0.11, "s2": 0.13, "c1": 0.14, "c2": 0.12},
        4: {task_id: 0.9 for task_id in task_ids},
    }
    for packet_size, by_task in recoveries.items():
        for task_id, recovery in by_task.items():
            rows.append(
                {
                    "task_id": task_id,
                    "packet_size": packet_size,
                    "informative": True,
                    "task_advantage_nll": 1.0,
                    "recovery_fraction": recovery,
                    "self_nll_delta": 0.0 if task_id == "s1" else None,
                }
            )
    summary = summarize_packet_capacity(
        rows,
        task_ids=task_ids,
        packet_sizes=[1, 2, 4],
        selection_task_count=2,
        minimum_informative_tasks_per_split=2,
        minimum_recovery=0.1,
        maximum_self_nll_delta=0.0001,
        run_scope="full",
    )
    assert summary["selected_packet_size"] == 2
    assert summary["confirmation"]["mean_recovery_fraction"] == pytest.approx(0.13)
    assert summary["gate"]["passed"] is True


def test_packet_summary_without_selection_crossing_has_no_confirmation_gate():
    task_ids = ["s", "c"]
    rows = [
        {
            "task_id": task_id,
            "packet_size": size,
            "informative": True,
            "task_advantage_nll": 1.0,
            "recovery_fraction": 0.05,
            "self_nll_delta": 0.0,
        }
        for size in (1, 2)
        for task_id in task_ids
    ]
    summary = summarize_packet_capacity(
        rows,
        task_ids=task_ids,
        packet_sizes=[1, 2],
        selection_task_count=1,
        minimum_informative_tasks_per_split=1,
        minimum_recovery=0.1,
        maximum_self_nll_delta=0.0001,
        run_scope="full",
    )
    assert summary["selected_packet_size"] is None
    assert summary["confirmation"] is None
    assert summary["gate"]["passed"] is False


def test_position_summary_selects_on_prefix_instead_of_late_tokens():
    task_ids = ["s1", "s2", "c1", "c2"]
    rows = []
    for packet_size in (1, 8):
        for task_id in task_ids:
            task = [1.0, 1.0, 1.0, 1.0]
            neutral = [2.0, 2.0, 2.0, 2.0]
            injected = (
                [1.95, 1.95, 1.0, 1.0]
                if packet_size == 1
                else [1.5, 1.5, 1.5, 1.5]
            )
            rows.append(
                {
                    "task_id": task_id,
                    "packet_size": packet_size,
                    "task_token_nlls": task,
                    "neutral_token_nlls": neutral,
                    "injected_token_nlls": injected,
                    "self_nll_delta": 0.0 if task_id == "s1" else None,
                }
            )
    summary = summarize_packet_position_recovery(
        rows,
        task_ids=task_ids,
        packet_sizes=[1, 8],
        selection_task_count=2,
        prefix_token_counts=[1, 2],
        gate_prefix_token_count=2,
        minimum_task_support_per_split=2,
        minimum_task_advantage=0.05,
        minimum_recovery=0.1,
        maximum_self_nll_delta=0.0001,
        run_scope="full",
    )
    assert summary["selected_packet_size"] == 8
    assert summary["confirmation"]["recovery_fraction"] == pytest.approx(0.5)
    assert summary["by_packet_size"]["1"]["selection"]["windows"][
        "full_sequence"
    ]["recovery_fraction"] > 0.1
    assert summary["by_packet_size"]["1"]["selection"]["windows"][
        "first_2_tokens"
    ]["recovery_fraction"] == pytest.approx(0.05)
    assert summary["gate"]["passed"] is True


def test_position_summary_tracks_support_at_each_token_position():
    rows = [
        {
            "task_id": task_id,
            "packet_size": 1,
            "task_token_nlls": [1.0] * length,
            "neutral_token_nlls": [2.0] * length,
            "injected_token_nlls": [1.5] * length,
            "self_nll_delta": 0.0,
        }
        for task_id, length in (("s", 3), ("c", 2))
    ]
    summary = summarize_packet_position_recovery(
        rows,
        task_ids=["s", "c"],
        packet_sizes=[1],
        selection_task_count=1,
        prefix_token_counts=[1, 2],
        gate_prefix_token_count=2,
        minimum_task_support_per_split=1,
        minimum_task_advantage=0.05,
        minimum_recovery=0.1,
        maximum_self_nll_delta=0.0001,
        run_scope="preflight",
    )
    selection_positions = summary["by_packet_size"]["1"]["selection"][
        "by_token_position"
    ]
    assert selection_positions["3"]["task_support"] == 1
    assert selection_positions["3"]["recovery_fraction"] == pytest.approx(0.5)
    assert summary["claim_eligible"] is False
    assert summary["gate"]["passed"] is False


def test_packet_config_freezes_layer_capacity_axis_and_replication_anchor():
    config = protocol_config()
    config["experiment_id"] = "LIP-PROTO-004"
    config["layer_selection_experiment"] = "LIP-PROTO-003"
    config["carrier"] = {"mode": "left_pad_masked_to_task_length"}
    config["audit"] = {
        "layer_idx": -16,
        "packet_sizes": [1, 2, 4],
        "injection_mode": "replace",
        "reference_max_new_tokens": 8,
        "minimum_reference_tokens": 2,
        "self_check_tasks": 1,
        "minimum_task_advantage_nll": 0.05,
        "minimum_informative_tasks_per_split": 1,
        "minimum_recovery": 0.1,
        "maximum_self_nll_delta": 0.0001,
    }
    validate_packet_config(config)
    config["audit"]["packet_sizes"] = [2, 4]
    with pytest.raises(ValueError, match="start at 1"):
        validate_packet_config(config)


def test_position_config_binds_prior_protocols_and_prefix_gate():
    config = protocol_config()
    config.update(
        {
            "experiment_id": "LIP-PROTO-006",
            "layer_selection_experiment": "LIP-PROTO-003",
            "capacity_source_experiment": "LIP-PROTO-004",
            "functional_source_experiment": "LIP-PROTO-005",
            "carrier": {"mode": "left_pad_masked_to_task_length"},
            "position_analysis": {
                "prefix_token_counts": [1, 4, 8],
                "gate_prefix_token_count": 8,
                "minimum_task_support_per_split": 1,
                "estimator": "pooled_nll_ratio",
            },
        }
    )
    config["audit"] = {
        "layer_idx": -16,
        "packet_sizes": [1, 8],
        "injection_mode": "replace",
        "reference_max_new_tokens": 8,
        "minimum_reference_tokens": 8,
        "self_check_tasks": 1,
        "minimum_task_advantage_nll": 0.05,
        "minimum_informative_tasks_per_split": 1,
        "minimum_recovery": 0.1,
        "maximum_self_nll_delta": 0.0001,
    }
    validate_packet_config(config)
    config["position_analysis"]["gate_prefix_token_count"] = 2
    with pytest.raises(ValueError, match="must be a configured prefix"):
        validate_packet_config(config)


def test_packet_capture_and_replacement_share_block_output_boundary():
    class Layer(torch.nn.Module):
        def forward(self, hidden):
            return hidden + 1.0

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()

        def forward(
            self,
            input_ids,
            attention_mask,
            use_cache,
            output_hidden_states,
            return_dict,
        ):
            hidden = self.model.layers[0](input_ids.float().unsqueeze(-1))
            return type("Output", (), {"logits": hidden})()

    model = Model()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    positions = torch.tensor([1, 3])
    baseline, packet = forward_with_packet_capture(
        model, inputs, layer_idx=-1, positions=positions
    )
    replaced = forward_with_packet_replacement(
        model,
        inputs,
        layer_idx=-1,
        positions=positions,
        vectors=packet,
    )
    assert packet[:, 0].tolist() == [3.0, 5.0]
    assert torch.equal(replaced.logits, baseline.logits)
