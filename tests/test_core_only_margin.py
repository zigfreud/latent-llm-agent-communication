from pathlib import Path

import pytest

from src.pipelines.core_only_margin import validate_core_only_margin_contract
from src.pipelines.oracle_experiment import load_yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "config" / "LIP-H0-014_core_only_margin.yaml"
PARENT = ROOT / "config" / "LIP-PROTO-014_source_conditioned_residual_packet.yaml"
PREDECESSOR = (
    ROOT / "experiments" / "registry" / "LIP-EVAL-032_gradient_geometry.json"
)


def _validate(experiment):
    validate_core_only_margin_contract(
        experiment,
        load_yaml(PARENT),
        experiment_path=EXPERIMENT,
        parent_path=PARENT,
        predecessor_registry_path=PREDECESSOR,
    )


def test_frozen_core_only_margin_contract_validates():
    _validate(load_yaml(EXPERIMENT))


def test_core_only_margin_rejects_soft_blend_drift():
    experiment = load_yaml(EXPERIMENT)
    experiment["loss"]["induced_trajectory"]["margin_region_weights"] = {
        "joint": 0.05,
        "core": 0.90,
        "name": 0.05,
    }
    with pytest.raises(ValueError, match="core-only"):
        _validate(experiment)


def test_core_only_margin_rejects_scalar_drift():
    experiment = load_yaml(EXPERIMENT)
    experiment["loss"]["induced_trajectory"]["lambda_margin"] = 1.5
    with pytest.raises(ValueError, match="non-regional induced loss drifted"):
        _validate(experiment)


def test_core_only_margin_rejects_predecessor_hash_drift():
    experiment = load_yaml(EXPERIMENT)
    experiment["predecessor"]["registry_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="registry differs"):
        _validate(experiment)


def test_core_only_margin_rejects_training_budget_drift():
    experiment = load_yaml(EXPERIMENT)
    experiment["training"]["full_matrix"]["max_updates"] = 129
    with pytest.raises(ValueError, match="drifted from H0-013"):
        _validate(experiment)
