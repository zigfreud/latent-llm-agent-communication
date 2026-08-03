import hashlib
import json

import pytest
import torch

from src.evaluation.oracle_transport import (
    continuation_token_metrics,
    normalize_layer_indices,
    recovery_fraction,
    summarize_oracle_transport,
)
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
