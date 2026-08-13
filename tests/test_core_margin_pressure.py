from pathlib import Path

import pytest

from src.pipelines.core_margin_pressure import validate_core_margin_pressure_contract
from src.pipelines.oracle_experiment import load_yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "config" / "LIP-H0-013_core_margin_pressure.yaml"
PARENT = ROOT / "config" / "LIP-PROTO-014_source_conditioned_residual_packet.yaml"
PREDECESSOR = (
    ROOT / "experiments" / "registry" / "LIP-H0-012_core_negative_replication.json"
)


def _validate(experiment):
    validate_core_margin_pressure_contract(
        experiment,
        load_yaml(PARENT),
        experiment_path=EXPERIMENT,
        parent_path=PARENT,
        predecessor_registry_path=PREDECESSOR,
    )


def test_frozen_core_margin_pressure_contract_validates():
    _validate(load_yaml(EXPERIMENT))


def test_core_margin_pressure_rejects_margin_drift():
    experiment = load_yaml(EXPERIMENT)
    experiment["loss"]["induced_trajectory"]["lambda_margin"] = 0.10

    with pytest.raises(ValueError, match="lambda_margin drifted"):
        _validate(experiment)


def test_core_margin_pressure_rejects_nonmargin_loss_drift():
    experiment = load_yaml(EXPERIMENT)
    experiment["loss"]["induced_trajectory"]["lambda_symmetric_nce"] = 0.5

    with pytest.raises(ValueError, match="lambda_symmetric_nce drifted"):
        _validate(experiment)


def test_core_margin_pressure_rejects_predecessor_hash_drift():
    experiment = load_yaml(EXPERIMENT)
    experiment["predecessor"]["registry_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="registry differs"):
        _validate(experiment)
