import json
from pathlib import Path

import pytest

from src.pipelines.hard_negative_replication import (
    aggregate_hard_negative_replication,
    validate_hard_negative_replication_contract,
)
from src.pipelines.oracle_experiment import load_json_object, load_yaml


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "config" / "LIP-H0-016_hard_negative_replication.yaml"
PARENT = ROOT / "config" / "LIP-PROTO-014_source_conditioned_residual_packet.yaml"
PREDECESSOR = ROOT / "experiments" / "registry" / "LIP-H0-015_hard_negative_batches.json"
REGISTRY = ROOT / "experiments" / "registry" / "LIP-H0-016_hard_negative_replication.json"


def _validate(experiment):
    validate_hard_negative_replication_contract(
        experiment,
        load_yaml(PARENT),
        experiment_path=EXPERIMENT,
        parent_path=PARENT,
        predecessor_registry_path=PREDECESSOR,
    )


def _summary(seed: int, *, passed: bool, plan_hash: str):
    family = []
    regions = {}
    for index, region in enumerate(("joint", "core", "name")):
        regions[region] = {"retrieval_top1": 0.75 + index * 0.05}
        family.append(
            {
                "region": region,
                "mean_diagonal_margin": 0.01 + index * 0.01,
                "p_value_holm": 0.01 if passed else 0.2,
                "rejected": passed,
            }
        )
    return {
        "experiment_id": "LIP-H0-016",
        "seed": seed,
        "training": {"best_step": 120},
        "development_gate_metrics": {
            "induced_trajectory": {
                "normalized_residual_rmse": 1.4,
                "regions": regions,
            }
        },
        "development_gate": {"family": family, "passed": passed},
        "provenance": {"training_batch_plan_sha256": plan_hash},
    }


def test_frozen_hard_negative_replication_contract_validates():
    _validate(load_yaml(EXPERIMENT))


def test_replication_rejects_seed_drift():
    experiment = load_yaml(EXPERIMENT)
    experiment["training"]["seeds"] = [4001, 4007]
    with pytest.raises(ValueError, match="seeds drifted"):
        _validate(experiment)


def test_replication_rejects_batch_policy_drift():
    experiment = load_yaml(EXPERIMENT)
    experiment["training"]["batch_policy"]["partition_seed"] = 4003
    with pytest.raises(ValueError, match="batch_policy drifted"):
        _validate(experiment)


def test_aggregate_passes_with_one_new_strong_seed():
    experiment = load_yaml(EXPERIMENT)
    predecessor = load_json_object(PREDECESSOR)
    plan_hash = predecessor["artifacts"]["screen"]["training_batch_plan"]["sha256"]
    aggregate = aggregate_hard_negative_replication(
        experiment,
        predecessor,
        {
            4001: _summary(4001, passed=True, plan_hash=plan_hash),
            4003: _summary(4003, passed=False, plan_hash=plan_hash),
        },
    )
    assert aggregate["aggregate_gate"]["strong_replication_passed"] is True
    assert aggregate["aggregate_gate"]["all_new_seeds_passed"] is False
    assert aggregate["decision"]["functional_confirmation_execution_authorized"] is False


def test_aggregate_rejects_batch_plan_drift():
    experiment = load_yaml(EXPERIMENT)
    predecessor = load_json_object(PREDECESSOR)
    plan_hash = predecessor["artifacts"]["screen"]["training_batch_plan"]["sha256"]
    with pytest.raises(ValueError, match="batch plan differs"):
        aggregate_hard_negative_replication(
            experiment,
            predecessor,
            {
                4001: _summary(4001, passed=True, plan_hash="0" * 64),
                4003: _summary(4003, passed=True, plan_hash=plan_hash),
            },
        )


def test_replication_registry_preserves_threshold_level_decision():
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    assert registry["aggregate_gate"]["strong_replication_passed"] is True
    assert registry["aggregate_gate"]["strong_seed_ids"] == [4003, 4007]
    assert registry["aggregate_gate"]["all_new_seeds_passed"] is False
    assert registry["cells"]["4001"]["holm_family_passed"] is False
    assert registry["cells"]["4003"]["holm_family_passed"] is True
    assert registry["decision"]["LIP_EVAL_033_design_authorized"] is True
    assert registry["decision"]["LIP_EVAL_033_execution_authorized"] is False
    assert registry["decision"]["proto_015_status"] == "premature"
