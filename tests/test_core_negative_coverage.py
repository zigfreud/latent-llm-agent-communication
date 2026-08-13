from pathlib import Path

import pytest

from src.pipelines.core_negative_coverage import (
    validate_core_negative_coverage_contract,
)
from src.pipelines.oracle_experiment import load_yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "config" / "LIP-H0-011_core_negative_coverage.yaml"
PARENT = ROOT / "config" / "LIP-PROTO-014_source_conditioned_residual_packet.yaml"
PREDECESSOR = (
    ROOT
    / "experiments"
    / "registry"
    / "LIP-H0-010_initial_condition_bridge_matrix_v3.json"
)


def _validate(experiment):
    validate_core_negative_coverage_contract(
        experiment,
        load_yaml(PARENT),
        experiment_path=EXPERIMENT,
        parent_path=PARENT,
        predecessor_registry_path=PREDECESSOR,
    )


def test_frozen_core_negative_coverage_contract_validates():
    _validate(load_yaml(EXPERIMENT))


def test_contract_rejects_loss_weight_drift():
    experiment = load_yaml(EXPERIMENT)
    experiment["loss"]["induced_trajectory"]["lambda_margin"] = 1.0

    with pytest.raises(ValueError, match="margin pressure"):
        _validate(experiment)


def test_contract_rejects_example_budget_drift():
    experiment = load_yaml(EXPERIMENT)
    experiment["training"]["full_matrix"]["max_updates"] = 127

    with pytest.raises(ValueError, match="2,048-example"):
        _validate(experiment)
