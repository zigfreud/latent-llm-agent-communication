import json
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
REGISTRY = ROOT / "experiments" / "registry" / "LIP-H0-015_hard_negative_batches.json"


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


def test_hard_negative_registry_authorizes_only_exact_replication():
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert registry["paired_to_H0_013_seed_4007"]["directional_success"] is True
    assert registry["paired_to_H0_013_seed_4007"]["strong_success"] is True
    assert registry["screen"]["holm_family_passed"] is True
    assert registry["decision"]["exact_replication_authorized"] is True
    assert registry["decision"]["replication_seeds"] == [4001, 4003]
    assert registry["decision"]["functional_confirmation_authorized"] is False
    assert registry["decision"]["proto_015_status"] == "premature"
