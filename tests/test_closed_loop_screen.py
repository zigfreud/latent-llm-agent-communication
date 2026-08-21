from copy import deepcopy
from pathlib import Path

import pytest

from src.pipelines.closed_loop_screen import (
    aggregate_closed_loop_screen,
    validate_closed_loop_screen_contract,
)
from src.pipelines.oracle_experiment import load_yaml


ROOT = Path(__file__).resolve().parents[1]
AGGREGATION = ROOT / "config" / "LIP-H0-017_paired_screen_aggregation.yaml"
EXPERIMENT = ROOT / "config" / "LIP-H0-017_closed_loop_trajectory_corrector.yaml"


def _metrics(*, rmse: float, margin: float, core_retrieval: float = 0.5):
    regions = {}
    for region in ("joint", "core", "name"):
        retrieval = core_retrieval if region == "core" else 0.5
        regions[region] = {
            "retrieval_top1": retrieval,
            "tasks": [
                {"task_id": str(index), "diagonal_margin": margin + index * 1e-4}
                for index in range(32)
            ],
        }
    return {"normalized_residual_rmse": rmse, "regions": regions}


def _pilot():
    return {
        "experiment_id": "LIP-H0-017",
        "protocol_version": "lip-closed-loop-trajectory-corrector-v1",
        "stage": "pilot",
        "variant": "closed_loop_live",
        "run_commit": "5ccc26098ffa226a0969ed9efd82a6b6ecf890ac",
        "complete": True,
        "pilot_gate": {"passed": True},
        "provenance": {
            "experiment_config_sha256": (
                "c0df4a8db4c672a6bdd7437fe9fb0bc2c4a157a9d12914acb0d2b076872881ea"
            )
        },
    }


def _cell(variant: str, *, rmse: float, margin: float, core_retrieval: float = 0.5):
    provenance = {
        "experiment_config_sha256": (
            "c0df4a8db4c672a6bdd7437fe9fb0bc2c4a157a9d12914acb0d2b076872881ea"
        ),
        "parent_config_sha256": "1" * 64,
        "learned_registry_sha256": "2" * 64,
        "functional_registry_sha256": "3" * 64,
        "source_registry_sha256": "4" * 64,
        "bundle_manifest_sha256": "5" * 64,
        "source_encoder_checkpoint_sha256": "6" * 64,
        "candidate_bank_sha256": "7" * 64,
    }
    return {
        "experiment_id": "LIP-H0-017",
        "protocol_version": "lip-closed-loop-trajectory-corrector-v1",
        "stage": "paired_screen_cell",
        "variant": variant,
        "seed": 4007,
        "run_commit": "a" * 40,
        "complete": True,
        "pilot_gate": None,
        "provenance": provenance,
        "training": {
            "updates_completed": 128,
            "batch_size": 16,
            "best_step": 120,
            "resolved_stage": {
                "max_updates": 128,
                "batch_size": 16,
                "variant": variant,
            },
            "batch_policy": {"batches": [[0, 1]], "partition_seed": 4007},
        },
        "development_gate_metrics": {
            "incoming_trajectory": _metrics(
                rmse=rmse, margin=margin, core_retrieval=core_retrieval
            )
        },
    }


def _aggregate(control, treatment):
    return aggregate_closed_loop_screen(
        load_yaml(AGGREGATION),
        load_yaml(EXPERIMENT),
        _pilot(),
        control,
        treatment,
        aggregation_path=AGGREGATION,
        experiment_path=EXPERIMENT,
    )


def test_frozen_closed_loop_screen_contract_validates():
    validate_closed_loop_screen_contract(
        load_yaml(AGGREGATION),
        load_yaml(EXPERIMENT),
        aggregation_path=AGGREGATION,
        experiment_path=EXPERIMENT,
    )


def test_paired_screen_passes_only_the_complete_frozen_gate():
    result = _aggregate(
        _cell("open_loop_zero_live", rmse=1.0, margin=0.0),
        _cell("closed_loop_live", rmse=0.85, margin=0.2),
    )

    assert result["aggregate_gate"]["rmse_reduction_passed"] is True
    assert result["aggregate_gate"]["holm_margin_family_passed"] is True
    assert result["aggregate_gate"]["passed"] is True
    assert result["decision"]["eval_038_design_authorized"] is True
    assert result["decision"]["eval_038_execution_authorized"] is False
    assert result["decision"]["functional_transport_supported"] is False


def test_paired_screen_fails_below_the_frozen_rmse_effect_size():
    result = _aggregate(
        _cell("open_loop_zero_live", rmse=1.0, margin=0.0),
        _cell("closed_loop_live", rmse=0.91, margin=0.2),
    )

    assert result["aggregate_gate"]["rmse_reduction_passed"] is False
    assert result["aggregate_gate"]["holm_margin_family_passed"] is True
    assert result["aggregate_gate"]["passed"] is False


def test_paired_screen_fails_when_core_retrieval_is_lower():
    result = _aggregate(
        _cell("open_loop_zero_live", rmse=1.0, margin=0.0, core_retrieval=0.5),
        _cell("closed_loop_live", rmse=0.8, margin=0.2, core_retrieval=0.49),
    )

    assert result["aggregate_gate"]["core_retrieval_not_lower"] is False
    assert result["aggregate_gate"]["passed"] is False


def test_paired_screen_rejects_batch_plan_or_task_identity_drift():
    control = _cell("open_loop_zero_live", rmse=1.0, margin=0.0)
    treatment = _cell("closed_loop_live", rmse=0.8, margin=0.2)
    drifted_plan = deepcopy(treatment)
    drifted_plan["training"]["batch_policy"]["partition_seed"] = 4001
    with pytest.raises(ValueError, match="training batch plan"):
        _aggregate(control, drifted_plan)

    drifted_tasks = deepcopy(treatment)
    drifted_tasks["development_gate_metrics"]["incoming_trajectory"]["regions"][
        "core"
    ]["tasks"][0]["task_id"] = "different"
    with pytest.raises(ValueError, match="task identities differ"):
        _aggregate(control, drifted_tasks)
