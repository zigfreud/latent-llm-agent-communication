from pathlib import Path

import pytest

from src.pipelines.core_negative_replication import (
    validate_core_negative_replication_contract,
)
from src.pipelines.oracle_experiment import load_yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "config" / "LIP-H0-012_core_negative_replication.yaml"
PARENT = ROOT / "config" / "LIP-PROTO-014_source_conditioned_residual_packet.yaml"
PREDECESSOR = (
    ROOT / "experiments" / "registry" / "LIP-H0-011_core_negative_coverage.json"
)


def _validate(experiment):
    validate_core_negative_replication_contract(
        experiment,
        load_yaml(PARENT),
        experiment_path=EXPERIMENT,
        parent_path=PARENT,
        predecessor_registry_path=PREDECESSOR,
    )


def test_frozen_replication_contract_validates():
    _validate(load_yaml(EXPERIMENT))


def test_replication_rejects_seed_drift():
    experiment = load_yaml(EXPERIMENT)
    experiment["training"]["seeds"] = [4001, 4007]

    with pytest.raises(ValueError, match="seeds drifted"):
        _validate(experiment)


def test_replication_rejects_predecessor_hash_drift():
    experiment = load_yaml(EXPERIMENT)
    experiment["predecessor"]["screen_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="artifact differs"):
        _validate(experiment)
