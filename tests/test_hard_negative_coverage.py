from pathlib import Path

import pytest

from src.pipelines.hard_negative_coverage import (
    validate_hard_negative_coverage_contract,
)
from src.pipelines.oracle_experiment import load_yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "config" / "LIP-H0-015_hard_negative_batches.yaml"
PARENT = ROOT / "config" / "LIP-PROTO-014_source_conditioned_residual_packet.yaml"
PREDECESSOR = ROOT / "experiments" / "registry" / "LIP-H0-014_core_only_margin.json"


def _validate(experiment):
    validate_hard_negative_coverage_contract(
        experiment,
        load_yaml(PARENT),
        experiment_path=EXPERIMENT,
        parent_path=PARENT,
        predecessor_registry_path=PREDECESSOR,
    )


def test_frozen_hard_negative_coverage_contract_validates():
    _validate(load_yaml(EXPERIMENT))


def test_hard_negative_coverage_rejects_loss_drift():
    experiment = load_yaml(EXPERIMENT)
    experiment["loss"]["induced_trajectory"]["lambda_margin"] = 0.5
    with pytest.raises(ValueError, match="loss drifted from H0-013"):
        _validate(experiment)


def test_hard_negative_coverage_rejects_plan_drift():
    experiment = load_yaml(EXPERIMENT)
    experiment["training"]["batch_policy"][
        "expected_global_hardest_coverage"
    ] = 0.50
    with pytest.raises(ValueError, match="batch policy drifted"):
        _validate(experiment)


def test_hard_negative_coverage_rejects_predecessor_hash_drift():
    experiment = load_yaml(EXPERIMENT)
    experiment["predecessor"]["registry_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="registry differs"):
        _validate(experiment)
