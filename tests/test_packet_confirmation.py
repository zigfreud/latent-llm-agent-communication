import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from src.evaluation.oracle_functional import stable_seed
from src.evaluation.oracle_terminal_factorial import terminal_components
from src.evaluation.packet_bridge_confirmation import (
    PACKET_CONFIRMATION_CONDITIONS,
    PACKET_CONFIRMATION_EVALUATION_POLICY,
    PACKET_CONFIRMATION_GENERATION_SEEDS,
    PACKET_CONFIRMATION_PROTOCOL_VERSION,
    PACKET_CONFIRMATION_REPLICA_CONDITIONS,
    PACKET_CONFIRMATION_SHARED_CONDITIONS,
    PACKET_CONFIRMATION_TRAINING_SEEDS,
    expected_confirmation_generation_keys,
    isotropic_residual_with_matched_layer_norms,
    packet_confirmation_design_fingerprint,
    stratified_confirmation_donors,
    validate_packet_confirmation_contract,
)
from src.pipelines.packet_confirmation import _validate_existing_metadata
from src.scripts.evaluate_packet_bridge_confirmation import (
    validate_confirmation_generation_grid,
)
from src.scripts.run_hardened_oracle_evaluation import evaluator_for_config


CONFIG_PATH = Path(
    "config/LIP-PROTO-014_source_conditioned_residual_packet.yaml"
)


def load_registered_config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _layout(name_token_count: int, task_id: str) -> dict:
    components = terminal_components(name_token_count)
    return {
        "name_token_count": name_token_count,
        "core_offsets": list(components["core"]),
        "name_offsets": list(components["name"]),
        "boundary_offsets": list(components["boundary"]),
        "tail_offsets": list(range(-24, 0)),
        "selection_hash": _sha(task_id),
    }


def _tasks() -> list[dict]:
    return [
        {
            "task_id": f"task-{index:02d}",
            "prompt": f"Write function_{index}",
            "entry_point": f"function_{index}",
            "terminal_layout": _layout(2 if index < 16 else 3, f"task-{index:02d}"),
        }
        for index in range(32)
    ]


def test_registered_confirmation_contract_has_one_task_clustered_1344_cell_grid():
    config = load_registered_config()
    validate_packet_confirmation_contract(config)
    tasks = _tasks()
    task_ids = [task["task_id"] for task in tasks]
    donors = stratified_confirmation_donors(tasks)
    keys = expected_confirmation_generation_keys(task_ids)

    assert len(keys) == 32 * (5 * 3 + 3 * 3 * 3) == 1344
    assert len(packet_confirmation_design_fingerprint(config)) == 64
    assert set(donors) == set(range(32))
    assert all(target != donor for target, donor in donors.items())
    assert all((target < 16) == (donor < 16) for target, donor in donors.items())


def test_isotropic_random_control_matches_each_receiver_layer_norm():
    generator = torch.Generator(device="cpu").manual_seed(4513)
    reference = torch.randn(8, 24, 32, generator=generator)
    random_generator = torch.Generator(device="cpu").manual_seed(991)
    residual = isotropic_residual_with_matched_layer_norms(
        reference,
        generator=random_generator,
    )

    expected = torch.linalg.vector_norm(reference.flatten(1), dim=1)
    observed = torch.linalg.vector_norm(residual.flatten(1), dim=1)
    assert torch.isfinite(residual).all()
    assert torch.allclose(observed, expected, rtol=5e-6, atol=1e-6)


def _synthetic_generation_grid():
    config = load_registered_config()
    tasks = _tasks()
    task_ids = [task["task_id"] for task in tasks]
    donors = stratified_confirmation_donors(tasks)
    donor_ids = {
        task_ids[target]: task_ids[source] for target, source in donors.items()
    }
    design = packet_confirmation_design_fingerprint(config)
    config_hash = "d" * 64
    scaffold_hash = "a" * 64
    neutral_ids_hash = "b" * 64
    neutral_mask_hash = "c" * 64
    layers = [int(value) for value in config["packets"]["target"]["layer_indices"]]
    offsets = [int(value) for value in config["packets"]["target"]["offsets"]]
    records = []
    for task_index, task in enumerate(tasks):
        task_id = task_ids[task_index]
        donor_id = donor_ids[task_id]
        for generation_seed in PACKET_CONFIRMATION_GENERATION_SEEDS:
            cells = [
                (condition, None)
                for condition in PACKET_CONFIRMATION_SHARED_CONDITIONS
            ] + [
                (condition, training_seed)
                for training_seed in PACKET_CONFIRMATION_TRAINING_SEEDS
                for condition in PACKET_CONFIRMATION_REPLICA_CONDITIONS
            ]
            for condition, training_seed in cells:
                packet_present = condition not in {
                    "neutral_no_lip",
                    "text_only_no_lip",
                }
                if condition == "mean_scaffold":
                    packet_hash = scaffold_hash
                    residual_norms = [0.0] * len(layers)
                elif condition == "oracle_teacher_matched":
                    packet_hash = _sha(f"teacher:{task_id}")
                    residual_norms = [1.0] * len(layers)
                elif condition == "oracle_teacher_shuffled":
                    packet_hash = _sha(f"teacher:{donor_id}")
                    residual_norms = [1.0] * len(layers)
                elif condition == "learned_matched":
                    packet_hash = _sha(f"learned:{task_id}:{training_seed}")
                    residual_norms = [1.0] * len(layers)
                elif condition == "learned_shuffled":
                    packet_hash = _sha(f"learned:{donor_id}:{training_seed}")
                    residual_norms = [1.0] * len(layers)
                elif condition == "random_residual_norm_matched":
                    packet_hash = _sha(
                        f"random:{task_id}:{training_seed}:{generation_seed}"
                    )
                    residual_norms = [1.0] * len(layers)
                else:
                    packet_hash = None
                    residual_norms = None

                shuffled = "shuffled" in condition
                matched = condition in {
                    "oracle_teacher_matched",
                    "learned_matched",
                    "random_residual_norm_matched",
                }
                random_control = condition == "random_residual_norm_matched"
                records.append(
                    {
                        "protocol_version": PACKET_CONFIRMATION_PROTOCOL_VERSION,
                        "design_sha256": design,
                        "experiment_id": "LIP-PROTO-014",
                        "config_sha256": config_hash,
                        "run_scope": "confirmation",
                        "claim_eligible": True,
                        "task_id": task_id,
                        "condition": condition,
                        "generation_seed": generation_seed,
                        "effective_generation_seed": stable_seed(
                            generation_seed, task_index, 14014
                        ),
                        "training_seed": training_seed,
                        "target_prompt_kind": (
                            "task" if condition == "text_only_no_lip" else "neutral"
                        ),
                        "target_input_ids_sha256": (
                            _sha(f"text-ids:{task_id}")
                            if condition == "text_only_no_lip"
                            else neutral_ids_hash
                        ),
                        "target_attention_mask_sha256": (
                            _sha(f"text-mask:{task_id}")
                            if condition == "text_only_no_lip"
                            else neutral_mask_hash
                        ),
                        "packet_present": packet_present,
                        "packet_kind": condition if packet_present else None,
                        "packet_layer_indices": layers if packet_present else [],
                        "packet_offsets": offsets if packet_present else [],
                        "packet_sha256": packet_hash,
                        "packet_frobenius_norm": 1.0 if packet_present else None,
                        "packet_layer_norms": (
                            [1.0] * len(layers) if packet_present else None
                        ),
                        "packet_residual_layer_norms": residual_norms,
                        "source_task_id": (
                            donor_id
                            if shuffled
                            else task_id
                            if matched
                            else None
                        ),
                        "donor_task_id": donor_id if shuffled else None,
                        "random_residual_seed": (
                            stable_seed(
                                generation_seed,
                                task_index,
                                int(training_seed),
                                14017,
                            )
                            if random_control
                            else None
                        ),
                        "random_reference_residual_layer_norms": (
                            [1.0] * len(layers) if random_control else None
                        ),
                        "random_norm_match_maximum_absolute_delta": (
                            0.0 if random_control else None
                        ),
                        "random_norm_match_maximum_relative_delta": (
                            0.0 if random_control else None
                        ),
                        "confirmation_manifest_sha256": "1" * 64,
                        "confirmation_bundle_manifest_sha256": "2" * 64,
                        "training_bundle_manifest_sha256": "3" * 64,
                        "matrix_summary_sha256": "4" * 64,
                        "target_model_revision": config["models"]["target"][
                            "revision"
                        ],
                        "output_text": "def answer():\n    return 1",
                        "task_spec": task,
                    }
                )
    metadata = {
        "experiment_id": "LIP-PROTO-014",
        "protocol_version": PACKET_CONFIRMATION_PROTOCOL_VERSION,
        "design_sha256": design,
        "config_sha256": config_hash,
        "run_scope": "confirmation",
        "task_ids": task_ids,
        "task_count": 32,
        "conditions": list(PACKET_CONFIRMATION_CONDITIONS),
        "shared_conditions": list(PACKET_CONFIRMATION_SHARED_CONDITIONS),
        "replica_conditions": list(PACKET_CONFIRMATION_REPLICA_CONDITIONS),
        "generation_seeds": list(PACKET_CONFIRMATION_GENERATION_SEEDS),
        "training_seeds": list(PACKET_CONFIRMATION_TRAINING_SEEDS),
        "evaluation_policy": dict(PACKET_CONFIRMATION_EVALUATION_POLICY),
        "donor_task_ids": donor_ids,
        "expected_records": 1344,
        "records": 1344,
        "complete": True,
        "claim_eligible": True,
        "task_manifest_sha256": "1" * 64,
        "confirmation_bundle_manifest_sha256": "2" * 64,
        "training_bundle_manifest_sha256": "3" * 64,
        "matrix_summary_sha256": "4" * 64,
        "target_model_revision": config["models"]["target"]["revision"],
        "training_scaffold_sha256": scaffold_hash,
        "training_site_scale_sha256": "5" * 64,
        "neutral_input_ids_sha256": neutral_ids_hash,
        "neutral_attention_mask_sha256": neutral_mask_hash,
    }
    return config, records, metadata


def test_confirmation_grid_validator_binds_donors_replicas_and_random_norms():
    config, records, metadata = _synthetic_generation_grid()
    validation = validate_confirmation_generation_grid(
        records,
        metadata,
        config,
        allow_incomplete=False,
    )
    assert validation["complete"] is True
    assert validation["record_count"] == 1344
    assert validation["cluster_unit"] == "task_id"

    changed = deepcopy(records)
    random_row = next(
        row
        for row in changed
        if row["condition"] == "random_residual_norm_matched"
    )
    random_row["random_residual_seed"] += 1
    with pytest.raises(ValueError, match="random residual seed"):
        validate_confirmation_generation_grid(
            changed,
            metadata,
            config,
            allow_incomplete=False,
        )


def test_hardened_dispatch_uses_the_packet_confirmation_evaluator():
    evaluator = evaluator_for_config(load_registered_config())
    assert evaluator.__module__ == "src.scripts.evaluate_packet_bridge_confirmation"


def test_resume_accepts_lagging_metadata_after_the_last_atomic_jsonl_row(tmp_path):
    task_ids = [f"task-{index:02d}" for index in range(32)]
    expected = expected_confirmation_generation_keys(task_ids)
    metadata_path = tmp_path / "generations.metadata.json"
    metadata = {
        "experiment_id": "LIP-PROTO-014",
        "protocol_version": PACKET_CONFIRMATION_PROTOCOL_VERSION,
        "design_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "run_scope": "confirmation",
        "task_ids": task_ids,
        "conditions": list(PACKET_CONFIRMATION_CONDITIONS),
        "generation_seeds": list(PACKET_CONFIRMATION_GENERATION_SEEDS),
        "training_seeds": list(PACKET_CONFIRMATION_TRAINING_SEEDS),
        "expected_records": len(expected),
        "records": len(expected) - 1,
        "task_manifest_sha256": "c" * 64,
        "confirmation_bundle_manifest_sha256": "d" * 64,
        "training_bundle_manifest_sha256": "e" * 64,
        "matrix_summary_sha256": "f" * 64,
        "training_scaffold_sha256": "1" * 64,
        "training_site_scale_sha256": "2" * 64,
        "complete": False,
        "claim_eligible": False,
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    loaded = _validate_existing_metadata(
        metadata_path,
        design_sha256="a" * 64,
        config_sha256="b" * 64,
        task_ids=task_ids,
        existing_keys=expected,
        expected_keys=expected,
        confirmation_manifest_sha256="c" * 64,
        confirmation_bundle_manifest_sha256="d" * 64,
        training_bundle_manifest_sha256="e" * 64,
        matrix_summary_sha256="f" * 64,
    )
    assert loaded["complete"] is False
