import hashlib
import json
from pathlib import Path

import pytest

from src.evaluation.functional_bridge_screen import (
    FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS,
    FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS,
    FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS,
    expected_functional_bridge_screen_keys,
    functional_bridge_screen_design_fingerprint,
    summarize_functional_bridge_screen,
    validate_functional_bridge_screen_contract,
)
from src.evaluation.oracle_functional import stable_seed
from src.pipelines.functional_bridge_screen import (
    validate_functional_bridge_screen_runtime_contract,
)
from src.pipelines.oracle_experiment import load_yaml
from src.scripts.evaluate_functional_bridge_screen import (
    evaluate,
    validate_functional_bridge_screen_grid,
)
from src.scripts.run_hardened_oracle_evaluation import evaluator_for_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "LIP-EVAL-033_functional_bridge_screen.yaml"


def _config():
    return load_yaml(CONFIG_PATH)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _task(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "prompt": f"Return {task_id}",
        "entry_point": f"answer_{task_id}",
        "test_list": [f"assert answer_{task_id}() == {int(task_id)}"],
        "terminal_layout": {
            "function_name": f"answer_{task_id}",
            "function_name_token_count": 2 if int(task_id) < 16 else 3,
        },
    }


def _grid(*, complete: bool = True):
    config = _config()
    task_ids = [str(index) for index in range(32)]
    donors = {
        task_id: str((index + 1) % 16 + (16 if index >= 16 else 0))
        for index, task_id in enumerate(task_ids)
    }
    # Correct the second stratum's local rotation.
    donors.update(
        {
            str(index): str(16 + ((index - 16 + 1) % 16))
            for index in range(16, 32)
        }
    )
    rows = []
    for task_index, task_id in enumerate(task_ids):
        for generation_seed in FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS:
            for training_seed in FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS:
                for condition in FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS:
                    source_id = (
                        task_id if condition == "learned_matched" else donors[task_id]
                    )
                    rows.append(
                        {
                            "experiment_id": "LIP-EVAL-033",
                            "protocol_version": "lip-functional-bridge-screen-v1",
                            "design_sha256": functional_bridge_screen_design_fingerprint(
                                config
                            ),
                            "config_sha256": "c" * 64,
                            "run_scope": "development_only_reused_open_P014_cohort",
                            "claim_eligible": False,
                            "task_id": task_id,
                            "condition": condition,
                            "generation_seed": generation_seed,
                            "effective_generation_seed": stable_seed(
                                generation_seed, task_index, 14014
                            ),
                            "training_seed": training_seed,
                            "target_prompt_kind": "neutral",
                            "target_input_ids_sha256": "i" * 64,
                            "target_attention_mask_sha256": "a" * 64,
                            "packet_present": True,
                            "packet_kind": condition,
                            "packet_layer_indices": [0],
                            "packet_offsets": config["packets"]["target"]["offsets"],
                            "packet_sha256": _sha(f"{source_id}-{training_seed}"),
                            "packet_frobenius_norm": 1.0,
                            "packet_layer_norms": [1.0],
                            "packet_residual_layer_norms": [1.0],
                            "source_task_id": source_id,
                            "donor_task_id": donors[task_id]
                            if condition == "learned_shuffled"
                            else None,
                            "P014_generations_sha256": config["cohort"][
                                "source_artifacts"
                            ]["generations_sha256"],
                            "confirmation_bundle_manifest_sha256": config["cohort"][
                                "source_artifacts"
                            ]["confirmation_bundle_manifest_sha256"],
                            "target_model_revision": config["models"]["target"][
                                "revision"
                            ],
                            "output_text": "def answer():\n    return 1",
                            "task_spec": _task(task_id),
                        }
                    )
    if not complete:
        rows.pop()
    metadata = {
        "experiment_id": "LIP-EVAL-033",
        "protocol_version": "lip-functional-bridge-screen-v1",
        "design_sha256": functional_bridge_screen_design_fingerprint(config),
        "config_sha256": "c" * 64,
        "run_scope": "development_only_reused_open_P014_cohort",
        "claim_eligible": False,
        "task_ids": task_ids,
        "task_count": 32,
        "donor_task_ids": donors,
        "conditions": list(FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS),
        "generation_seeds": list(FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS),
        "training_seeds": list(FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS),
        "expected_records": 576,
        "records": len(rows),
        "complete": complete,
        "P014_generations_sha256": config["cohort"]["source_artifacts"][
            "generations_sha256"
        ],
        "confirmation_bundle_manifest_sha256": config["cohort"][
            "source_artifacts"
        ]["confirmation_bundle_manifest_sha256"],
        "neutral_input_ids_sha256": "i" * 64,
        "neutral_attention_mask_sha256": "a" * 64,
        "target_model_revision": config["models"]["target"]["revision"],
    }
    return config, rows, metadata


def test_functional_bridge_screen_contract_and_runtime_authorization_validate():
    config = _config()
    validate_functional_bridge_screen_contract(config)
    predecessor, h015, p014 = validate_functional_bridge_screen_runtime_contract(
        config, config_path=CONFIG_PATH
    )
    assert predecessor["decision"]["LIP_EVAL_033_design_authorized"] is True
    assert h015["screen"]["holm_family_passed"] is True
    assert p014["experiment_id"] == "LIP-PROTO-014"


def test_functional_bridge_screen_contract_rejects_claim_upgrade():
    config = _config()
    config["evaluation"]["claim_eligible"] = True
    with pytest.raises(ValueError, match="not_claim_eligible"):
        validate_functional_bridge_screen_contract(config)


def test_expected_grid_contains_576_cells():
    task_ids = [str(index) for index in range(32)]
    assert len(expected_functional_bridge_screen_keys(task_ids)) == 576


def test_grid_validator_binds_entry_packets_to_same_stratum_donors():
    config, rows, metadata = _grid()
    validation = validate_functional_bridge_screen_grid(
        rows, metadata, config, allow_incomplete=False
    )
    assert validation["complete"] is True
    assert validation["claim_eligible"] is False
    shuffled = next(row for row in rows if row["condition"] == "learned_shuffled")
    shuffled["packet_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen donor"):
        validate_functional_bridge_screen_grid(
            rows, metadata, config, allow_incomplete=False
        )


def _scored(*, positive_seeds: set[int]):
    rows = []
    for task_id in map(str, range(32)):
        for generation_seed in FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS:
            for training_seed in FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS:
                rows.extend(
                    [
                        {
                            "task_id": task_id,
                            "condition": "learned_matched",
                            "generation_seed": generation_seed,
                            "training_seed": training_seed,
                            "functional_pass": training_seed in positive_seeds,
                        },
                        {
                            "task_id": task_id,
                            "condition": "learned_shuffled",
                            "generation_seed": generation_seed,
                            "training_seed": training_seed,
                            "functional_pass": False,
                        },
                    ]
                )
    return rows


def test_primary_endpoint_requires_the_two_seed_guardrail():
    passed = summarize_functional_bridge_screen(
        _scored(positive_seeds={4001, 4003}), _config()
    )
    assert passed["primary_endpoint"]["passed"] is True
    assert passed["seed_guardrail"]["passed"] is True
    assert passed["development_functional_signal_detected"] is True

    guarded = summarize_functional_bridge_screen(
        _scored(positive_seeds={4001}), _config()
    )
    assert guarded["primary_endpoint"]["passed"] is True
    assert guarded["seed_guardrail"]["passed"] is False
    assert guarded["development_functional_signal_detected"] is False


def test_hardened_dispatch_selects_the_eval033_evaluator():
    evaluator = evaluator_for_config(_config())
    assert evaluator.__module__ == "src.scripts.evaluate_functional_bridge_screen"


def test_functional_summary_exposes_validated_sandbox_marker(tmp_path, monkeypatch):
    config, rows, metadata = _grid(complete=False)
    rows = rows[:2]
    generations = tmp_path / "generations.jsonl"
    generations.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    generations.with_suffix(".metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    def fake_evaluate_generation(row, task, **kwargs):
        return {
            **row,
            "syntax_pass": True,
            "functional_pass": False,
            "extracted_code": "def answer():\n    return 1",
        }

    monkeypatch.setattr(
        "src.scripts.evaluate_functional_bridge_screen.evaluate_generation",
        fake_evaluate_generation,
    )
    summary = evaluate(
        config,
        generations,
        tmp_path / "evaluation",
        functional=True,
        allow_incomplete=True,
        overwrite=False,
        security_context={"validated": True},
    )

    assert summary["execution_mode"] == "functional_hardened_namespace"
    assert summary["subprocess_is_security_sandbox"] is True
    assert summary["claim_eligible"] is False
