from copy import deepcopy
from pathlib import Path

import pytest
import torch

from src.pipelines.gradient_geometry import (
    _candidate_diagnostics_from_similarity,
    _gradient_geometry,
    route_gradient_geometry,
    validate_gradient_geometry_contract,
)
from src.pipelines.oracle_experiment import load_yaml
from src.pipelines.oracle_experiment import load_json_object


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "LIP-EVAL-032_gradient_geometry.yaml"
REGISTRY = ROOT / "experiments" / "registry" / "LIP-EVAL-032_gradient_geometry.json"


def test_frozen_gradient_geometry_contract_validates():
    validate_gradient_geometry_contract(load_yaml(CONFIG))


def test_registered_result_selects_only_the_scale_intervention():
    registry = load_json_object(REGISTRY)
    assert registry["routing"]["selected"] == "scale_limited"
    assert registry["decision"]["H0_014_development_intervention_authorized"] is True
    assert registry["decision"]["replication_authorized"] is False
    assert registry["decision"]["functional_confirmation_authorized"] is False


def test_gradient_geometry_contract_rejects_gate_access():
    config = load_yaml(CONFIG)
    config["data"]["allowed_splits"].append("development_gate")
    with pytest.raises(ValueError, match="only the train split"):
        validate_gradient_geometry_contract(config)


def test_gradient_geometry_reports_norms_and_direction():
    entry = torch.tensor([1.0, 2.0], requires_grad=True)
    objectives = {
        "core_margin": entry[0],
        "symmetric_nce": entry[1],
        "reconstruction": -entry[0],
        "non_margin": entry[1] - entry[0],
        "configured_total": entry[1],
    }
    norms, cosines = _gradient_geometry(objectives, entry)
    assert norms["core_margin"] == pytest.approx(1.0)
    assert cosines["core_margin__reconstruction"] == pytest.approx(-1.0)
    assert cosines["core_margin__symmetric_nce"] == pytest.approx(0.0)


def test_candidate_bank_detects_missing_global_hardest_negative():
    similarity = torch.tensor(
        [
            [1.0, 0.2, 0.9, 0.1],
            [0.2, 1.0, 0.1, 0.8],
            [0.9, 0.1, 1.0, 0.2],
            [0.1, 0.8, 0.2, 1.0],
        ]
    )
    report = _candidate_diagnostics_from_similarity(
        similarity, ["a", "b", "c", "d"], batch_size=2
    )
    assert report["global_hardest_coverage"] == 0.0
    assert all(row["local_minus_global_margin"] > 0.0 for row in report["rows"])


def _routing_rows(*, ratio: float, cosine: float):
    return [
        {
            "checkpoint": "H0_013",
            "effective_core_to_nonmargin_gradient_ratio": ratio,
            "gradient_cosines": {"core_margin__non_margin": cosine},
        }
        for _ in range(16)
    ]


def _candidate_bank(coverage: bool):
    return {
        "H0_013": {
            "rows": [
                {"global_hardest_in_assigned_batch": coverage} for _ in range(256)
            ]
        }
    }


def _routing_config():
    config = deepcopy(load_yaml(CONFIG))
    config["bootstrap"]["resamples"] = 100
    return config


def test_routing_identifies_scale_limited_geometry():
    result = route_gradient_geometry(
        _routing_rows(ratio=0.05, cosine=0.2),
        _candidate_bank(coverage=True),
        _routing_config(),
    )
    assert result["route"] == "scale_limited"


def test_routing_identifies_conflict_limited_geometry():
    result = route_gradient_geometry(
        _routing_rows(ratio=0.3, cosine=-0.5),
        _candidate_bank(coverage=True),
        _routing_config(),
    )
    assert result["route"] == "conflict_limited"


def test_routing_identifies_coverage_limited_geometry():
    result = route_gradient_geometry(
        _routing_rows(ratio=0.3, cosine=0.2),
        _candidate_bank(coverage=False),
        _routing_config(),
    )
    assert result["route"] == "coverage_limited"
