"""Score and test the frozen LIP-PROTO-014 functional confirmation."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.oracle_functional import declares_entry_point, stable_seed
from src.evaluation.oracle_terminal_factorial import validate_terminal_layout
from src.evaluation.packet_bridge import normalized_transport_recovery
from src.evaluation.packet_bridge_confirmation import (
    PACKET_CONFIRMATION_CONDITIONS,
    PACKET_CONFIRMATION_EVALUATION_POLICY,
    PACKET_CONFIRMATION_EXPERIMENT_ID,
    PACKET_CONFIRMATION_GENERATION_SEEDS,
    PACKET_CONFIRMATION_PROTOCOL_VERSION,
    PACKET_CONFIRMATION_REPLICA_CONDITIONS,
    PACKET_CONFIRMATION_SHARED_CONDITIONS,
    PACKET_CONFIRMATION_TASK_COUNT,
    PACKET_CONFIRMATION_TRAINING_SEEDS,
    expected_confirmation_generation_keys,
    packet_confirmation_design_fingerprint,
    stratified_confirmation_donors,
    validate_packet_confirmation_contract,
)
from src.evaluation.semantics import CandidateProcessPolicy, evaluate_generation
from src.evaluation.statistics import summarize_gatekept_holm, summarize_metric
from src.pipelines.oracle_experiment import (
    load_json_object,
    load_yaml,
    prepare_output_dir,
    sha256_path,
    write_json,
    write_jsonl,
)


DEFAULT_CONFIG = Path(
    "config/LIP-PROTO-014_source_conditioned_residual_packet.yaml"
)
PRIMARY_ANCHOR = ("oracle_teacher_matched", "oracle_teacher_shuffled")
PRIMARY_FAMILY = (
    ("learned_matched", "learned_shuffled"),
    ("learned_matched", "mean_scaffold"),
    ("learned_matched", "random_residual_norm_matched"),
)
DESCRIPTIVE_COMPARISONS = (
    PRIMARY_ANCHOR,
    *PRIMARY_FAMILY,
    ("learned_matched", "neutral_no_lip"),
    ("text_only_no_lip", "neutral_no_lip"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--functional", action="store_true")
    parser.add_argument("--allow-unsafe-execution", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    def reject_constant(value: str):
        raise ValueError(f"generation JSON contains non-finite constant {value}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line, parse_constant=reject_constant)
            if not isinstance(row, dict):
                raise ValueError(f"generation row {line_number} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError("confirmation generation file contains no records")
    return rows


def _record_key(row: Mapping) -> tuple[str, str, int, int | None]:
    training_seed = row.get("training_seed")
    return (
        str(row.get("task_id", "")),
        str(row.get("condition", "")),
        int(row.get("generation_seed")),
        None if training_seed is None else int(training_seed),
    )


def _finite_float_sequence(value: object, *, label: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain {length} values")
    values = [float(item) for item in value]
    if any(not math.isfinite(item) or item < 0.0 for item in values):
        raise ValueError(f"{label} must contain finite non-negative values")
    return values


def _validate_packet_artifact_semantics(
    records: Sequence[Mapping],
    metadata: Mapping,
    config: Mapping,
    *,
    task_ids: Sequence[str],
    donor_task_ids: Mapping[str, str],
) -> None:
    """Cross-check packet identity, donor reuse, and random-norm controls."""

    target_layers = [int(value) for value in config["packets"]["target"]["layer_indices"]]
    target_offsets = [int(value) for value in config["packets"]["target"]["offsets"]]
    layer_count = len(target_layers)
    scaffold_sha256 = metadata.get("training_scaffold_sha256")
    if not isinstance(scaffold_sha256, str) or len(scaffold_sha256) != 64:
        raise ValueError("confirmation metadata lacks the training scaffold hash")

    teacher_hashes: dict[str, set[str]] = {}
    learned_hashes: dict[tuple[str, int], set[str]] = {}
    learned_norms: dict[tuple[str, int], list[float]] = {}
    for row in records:
        task_id, condition, generation_seed, training_seed = _record_key(row)
        packet_present = condition not in {"neutral_no_lip", "text_only_no_lip"}
        if packet_present:
            _finite_float_sequence(
                row.get("packet_layer_norms"),
                label=f"{condition} packet layer norms",
                length=layer_count,
            )
            residual_norms = _finite_float_sequence(
                row.get("packet_residual_layer_norms"),
                label=f"{condition} residual layer norms",
                length=layer_count,
            )
            packet_norm = float(row.get("packet_frobenius_norm", math.nan))
            if not math.isfinite(packet_norm) or packet_norm < 0.0:
                raise ValueError(f"{condition} packet norm is invalid")
            if row.get("packet_kind") != condition:
                raise ValueError(f"{condition} packet kind changed")
            if row.get("packet_layer_indices") != target_layers:
                raise ValueError(f"{condition} packet layers changed")
            if row.get("packet_offsets") != target_offsets:
                raise ValueError(f"{condition} packet offsets changed")
        else:
            residual_norms = []
            if any(
                row.get(field) not in (None, [])
                for field in (
                    "packet_kind",
                    "packet_layer_indices",
                    "packet_offsets",
                    "packet_frobenius_norm",
                    "packet_layer_norms",
                    "packet_residual_layer_norms",
                )
            ):
                raise ValueError(f"{condition} unexpectedly carries packet metadata")

        if condition == "mean_scaffold":
            if row.get("packet_sha256") != scaffold_sha256 or any(residual_norms):
                raise ValueError("mean scaffold row is not the training-only scaffold")
        if condition == "oracle_teacher_matched":
            teacher_hashes.setdefault(task_id, set()).add(str(row["packet_sha256"]))
        if condition == "learned_matched":
            assert training_seed is not None
            learned_hashes.setdefault((task_id, training_seed), set()).add(
                str(row["packet_sha256"])
            )
            previous = learned_norms.setdefault(
                (task_id, training_seed), residual_norms
            )
            if previous != residual_norms:
                raise ValueError("learned residual norms vary by generation seed")

        random_fields = (
            "random_reference_residual_layer_norms",
            "random_norm_match_maximum_absolute_delta",
            "random_norm_match_maximum_relative_delta",
        )
        if condition == "random_residual_norm_matched":
            expected_seed = stable_seed(
                generation_seed,
                task_ids.index(task_id),
                int(training_seed),
                14017,
            )
            if row.get("random_residual_seed") != expected_seed:
                raise ValueError("random residual seed changed")
            reference_norms = _finite_float_sequence(
                row.get(random_fields[0]),
                label="random reference residual layer norms",
                length=layer_count,
            )
            absolute_delta = float(row.get(random_fields[1], math.nan))
            relative_delta = float(row.get(random_fields[2], math.nan))
            observed_absolute = max(
                abs(observed - expected)
                for observed, expected in zip(residual_norms, reference_norms)
            )
            observed_relative = max(
                delta / max(1.0, abs(expected))
                for delta, expected in zip(
                    [
                        abs(observed - expected)
                        for observed, expected in zip(
                            residual_norms, reference_norms
                        )
                    ],
                    reference_norms,
                )
            )
            if (
                not math.isfinite(absolute_delta)
                or not math.isfinite(relative_delta)
                or abs(absolute_delta - observed_absolute) > 1e-9
                or abs(relative_delta - observed_relative) > 1e-9
                or relative_delta > 5e-6
            ):
                raise ValueError("random residual layer norms are not matched")
            learned_reference = learned_norms.get((task_id, int(training_seed)))
            if learned_reference is not None and reference_norms != learned_reference:
                raise ValueError("random residual references a different learned packet")
        elif row.get("random_residual_seed") is not None or any(
            row.get(field) is not None for field in random_fields
        ):
            raise ValueError("non-random condition declares random-control metadata")

    if any(len(values) != 1 for values in teacher_hashes.values()):
        raise ValueError("oracle teacher packets vary by generation seed")
    if any(len(values) != 1 for values in learned_hashes.values()):
        raise ValueError("learned packets vary by generation seed")

    for row in records:
        task_id, condition, generation_seed, training_seed = _record_key(row)
        donor_id = donor_task_ids[task_id]
        if condition in {"neutral_no_lip", "text_only_no_lip", "mean_scaffold"}:
            expected_source = None
        elif condition in {"oracle_teacher_matched", "learned_matched", "random_residual_norm_matched"}:
            expected_source = task_id
        else:
            expected_source = donor_id
        if row.get("source_task_id") != expected_source:
            raise ValueError(f"{condition} source-task identity changed")
        if condition == "oracle_teacher_shuffled" and donor_id in teacher_hashes:
            donor_hash = next(iter(teacher_hashes[donor_id]))
            if row.get("packet_sha256") != donor_hash:
                raise ValueError("shuffled oracle packet is not its registered donor")
        if condition == "learned_shuffled":
            donor_hashes = learned_hashes.get((donor_id, int(training_seed)))
            if donor_hashes is not None and row.get("packet_sha256") != next(
                iter(donor_hashes)
            ):
                raise ValueError("shuffled learned packet is not its registered donor")
        if condition != "text_only_no_lip":
            if (
                row.get("target_input_ids_sha256")
                != metadata.get("neutral_input_ids_sha256")
                or row.get("target_attention_mask_sha256")
                != metadata.get("neutral_attention_mask_sha256")
            ):
                raise ValueError("neutral-carrier input binding changed")


def validate_confirmation_generation_grid(
    records: Sequence[Mapping],
    metadata: Mapping,
    config: Mapping,
    *,
    allow_incomplete: bool,
) -> dict:
    """Validate the asymmetric but task-balanced 1,344-cell grid."""

    validate_packet_confirmation_contract(config)
    design_sha256 = packet_confirmation_design_fingerprint(config)
    metadata_checks = {
        "experiment": metadata.get("experiment_id")
        == PACKET_CONFIRMATION_EXPERIMENT_ID,
        "protocol": metadata.get("protocol_version")
        == PACKET_CONFIRMATION_PROTOCOL_VERSION,
        "design": metadata.get("design_sha256") == design_sha256,
        "scope": metadata.get("run_scope") == "confirmation",
        "task_count": metadata.get("task_count")
        == PACKET_CONFIRMATION_TASK_COUNT,
        "conditions": tuple(metadata.get("conditions", ()))
        == PACKET_CONFIRMATION_CONDITIONS,
        "shared_conditions": tuple(metadata.get("shared_conditions", ()))
        == PACKET_CONFIRMATION_SHARED_CONDITIONS,
        "replica_conditions": tuple(metadata.get("replica_conditions", ()))
        == PACKET_CONFIRMATION_REPLICA_CONDITIONS,
        "generation_seeds": tuple(metadata.get("generation_seeds", ()))
        == PACKET_CONFIRMATION_GENERATION_SEEDS,
        "training_seeds": tuple(metadata.get("training_seeds", ()))
        == PACKET_CONFIRMATION_TRAINING_SEEDS,
        "evaluation_policy": metadata.get("evaluation_policy")
        == PACKET_CONFIRMATION_EVALUATION_POLICY,
    }
    failed_metadata = [
        name for name, passed in metadata_checks.items() if not passed
    ]
    if failed_metadata:
        raise ValueError(
            "confirmation metadata failed its contract: "
            + ", ".join(failed_metadata)
        )
    task_ids = [str(task_id) for task_id in metadata.get("task_ids", [])]
    expected = expected_confirmation_generation_keys(task_ids)
    if metadata.get("expected_records") != len(expected):
        raise ValueError("confirmation metadata expected-record count changed")

    observed = []
    task_specs: dict[str, Mapping] = {}
    for row in records:
        key = _record_key(row)
        observed.append(key)
        task_id, condition, generation_seed, training_seed = key
        if condition in PACKET_CONFIRMATION_SHARED_CONDITIONS:
            if training_seed is not None:
                raise ValueError(f"shared condition has a training seed: {key}")
        elif condition in PACKET_CONFIRMATION_REPLICA_CONDITIONS:
            if training_seed not in PACKET_CONFIRMATION_TRAINING_SEEDS:
                raise ValueError(f"replica condition lacks a registered seed: {key}")
        else:
            raise ValueError(f"unknown confirmation condition: {condition}")
        task_spec = row.get("task_spec")
        if not isinstance(task_spec, Mapping):
            raise ValueError("every confirmation row must contain task_spec")
        if str(task_spec.get("task_id", "")) != task_id:
            raise ValueError(f"task specification identity changed: {task_id}")
        existing = task_specs.setdefault(task_id, task_spec)
        if existing != task_spec:
            raise ValueError(f"task specification changes across rows: {task_id}")
        packet_expected = condition not in {"neutral_no_lip", "text_only_no_lip"}
        source_task_expected = condition in {
            "oracle_teacher_matched",
            "learned_matched",
            "random_residual_norm_matched",
        }
        row_checks = {
            "protocol": row.get("protocol_version")
            == PACKET_CONFIRMATION_PROTOCOL_VERSION,
            "design": row.get("design_sha256") == design_sha256,
            "config": row.get("config_sha256") == metadata.get("config_sha256"),
            "scope": row.get("run_scope") == "confirmation",
            "claim_flag": row.get("claim_eligible") is True,
            "effective_seed": row.get("effective_generation_seed")
            == stable_seed(int(generation_seed), task_ids.index(task_id), 14014),
            "target_prompt": row.get("target_prompt_kind")
            == ("task" if condition == "text_only_no_lip" else "neutral"),
            "packet_presence": row.get("packet_present") is packet_expected,
            "packet_hash": (
                isinstance(row.get("packet_sha256"), str)
                and len(row["packet_sha256"]) == 64
            )
            if packet_expected
            else row.get("packet_sha256") is None,
            "matched_source": row.get("source_task_id") == task_id
            if source_task_expected
            else True,
            "task_manifest": row.get("confirmation_manifest_sha256")
            == metadata.get("task_manifest_sha256"),
            "confirmation_bundle": row.get(
                "confirmation_bundle_manifest_sha256"
            )
            == metadata.get("confirmation_bundle_manifest_sha256"),
            "training_bundle": row.get("training_bundle_manifest_sha256")
            == metadata.get("training_bundle_manifest_sha256"),
            "matrix": row.get("matrix_summary_sha256")
            == metadata.get("matrix_summary_sha256"),
            "target_revision": row.get("target_model_revision")
            == metadata.get("target_model_revision"),
            "output_text": isinstance(row.get("output_text"), str),
        }
        failed_row = [name for name, passed in row_checks.items() if not passed]
        if failed_row:
            raise ValueError(
                f"confirmation row {key} failed: " + ", ".join(failed_row)
            )
    if len(set(observed)) != len(observed):
        raise ValueError("confirmation grid contains duplicate rows")
    unexpected = set(observed).difference(expected)
    missing = expected.difference(observed)
    if unexpected:
        raise ValueError(f"confirmation grid has {len(unexpected)} unexpected rows")
    if missing and not allow_incomplete:
        raise ValueError(f"confirmation grid is missing {len(missing)} rows")

    donor_task_ids = metadata.get("donor_task_ids")
    if (
        not isinstance(donor_task_ids, Mapping)
        or set(donor_task_ids) != set(task_ids)
        or any(
            str(donor) not in task_ids or str(target) == str(donor)
            for target, donor in donor_task_ids.items()
        )
    ):
        raise ValueError("confirmation metadata donor plan is invalid")
    donor_task_ids = {
        str(target): str(donor) for target, donor in donor_task_ids.items()
    }
    if set(task_specs) == set(task_ids):
        ordered_tasks = [task_specs[task_id] for task_id in task_ids]
        donors = stratified_confirmation_donors(
            ordered_tasks,
            seed=int(config["confirmation"]["derangement_seed"]),
        )
        recomputed_donor_task_ids = {
            task_ids[target]: task_ids[source]
            for target, source in donors.items()
        }
        if donor_task_ids != recomputed_donor_task_ids:
            raise ValueError("confirmation donor plan changed")
    elif not allow_incomplete:
        raise ValueError("full confirmation lacks task specifications")
    for target_task_id, donor_task_id in donor_task_ids.items():
        if target_task_id in task_specs and donor_task_id in task_specs:
            target_stratum = validate_terminal_layout(
                task_specs[target_task_id].get("terminal_layout", {})
            )
            donor_stratum = validate_terminal_layout(
                task_specs[donor_task_id].get("terminal_layout", {})
            )
            if target_stratum != donor_stratum:
                raise ValueError("confirmation donor crosses tokenizer strata")
    for row in records:
        condition = str(row["condition"])
        task_id = str(row["task_id"])
        if "shuffled" in condition:
            if (
                row.get("source_task_id") != donor_task_ids[task_id]
                or row.get("donor_task_id") != donor_task_ids[task_id]
                or row.get("source_task_id") == task_id
            ):
                raise ValueError("shuffled confirmation row violates its donor plan")
        elif row.get("donor_task_id") is not None:
            raise ValueError("non-shuffled confirmation row declares a donor")
        if condition == "random_residual_norm_matched":
            if not isinstance(row.get("random_residual_seed"), int):
                raise ValueError("random residual row lacks its deterministic seed")
        elif row.get("random_residual_seed") is not None:
            raise ValueError("non-random row declares a residual seed")

    _validate_packet_artifact_semantics(
        records,
        metadata,
        config,
        task_ids=task_ids,
        donor_task_ids=donor_task_ids,
    )

    complete = not missing and len(task_ids) == PACKET_CONFIRMATION_TASK_COUNT
    if not allow_incomplete and not complete:
        raise ValueError("only the full frozen confirmation grid is claim-eligible")
    if bool(metadata.get("complete")) != complete:
        raise ValueError("generation metadata completeness disagrees with the grid")
    if bool(metadata.get("claim_eligible")) != complete:
        raise ValueError("generation claim flag disagrees with grid completeness")
    if complete and metadata.get("records") != len(records):
        raise ValueError("generation metadata record count disagrees with the grid")
    return {
        "complete": complete,
        "run_scope": "confirmation",
        "task_count": len(task_ids),
        "record_count": len(records),
        "expected_record_count": len(expected),
        "missing_record_count": len(missing),
        "design_sha256": design_sha256,
        "cluster_unit": "task_id",
        "replicates_within_task": "generation and bridge seeds averaged",
    }


def evaluate(
    config: dict[str, Any],
    generations_path: Path,
    output_dir: Path,
    *,
    functional: bool,
    allow_incomplete: bool,
    overwrite: bool,
    candidate_process_policy: CandidateProcessPolicy | None = None,
    security_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata_path = generations_path.with_suffix(".metadata.json")
    metadata = load_json_object(metadata_path)
    records = read_jsonl(generations_path)
    design_validation = validate_confirmation_generation_grid(
        records,
        metadata,
        config,
        allow_incomplete=allow_incomplete,
    )
    prepare_output_dir(output_dir, overwrite=overwrite)
    policy = PACKET_CONFIRMATION_EVALUATION_POLICY
    scored = []
    for row in records:
        scored_row = evaluate_generation(
            row,
            row["task_spec"],
            run_functional=functional,
            timeout_seconds=float(policy["timeout_seconds"]),
            memory_mb=int(policy["memory_mb"]),
            process_policy=candidate_process_policy,
        )
        scored_row["entry_point_declared"] = declares_entry_point(
            scored_row["extracted_code"],
            row["task_spec"].get("entry_point"),
        )
        scored.append(scored_row)
    statistics_kwargs = {
        "bootstrap_iterations": int(policy["bootstrap_iterations"]),
        "confidence": float(policy["confidence"]),
        "seed": int(config["confirmation"]["statistics_seed"]),
    }
    metrics = {
        "syntax_pass": summarize_metric(
            scored,
            "syntax_pass",
            PACKET_CONFIRMATION_CONDITIONS,
            DESCRIPTIVE_COMPARISONS,
            **statistics_kwargs,
        ),
        "entry_point_declared": summarize_metric(
            scored,
            "entry_point_declared",
            PACKET_CONFIRMATION_CONDITIONS,
            DESCRIPTIVE_COMPARISONS,
            **statistics_kwargs,
        ),
    }
    primary_inference = None
    semantic_gate = None
    recovery = None
    if functional:
        metrics["functional_pass"] = summarize_metric(
            scored,
            "functional_pass",
            PACKET_CONFIRMATION_CONDITIONS,
            DESCRIPTIVE_COMPARISONS,
            **statistics_kwargs,
        )
        primary_inference = summarize_gatekept_holm(
            scored,
            "functional_pass",
            PRIMARY_ANCHOR,
            PRIMARY_FAMILY,
            alpha=float(policy["alpha"]),
            alternative=str(policy["alternative"]),
            **statistics_kwargs,
        )
        anchor_passed = bool(primary_inference["anchor"]["rejected"])
        family_passed = bool(
            primary_inference["family"]
            and all(item["rejected"] for item in primary_inference["family"])
        )
        semantic_gate = {
            "oracle_identity_gate_passed": anchor_passed,
            "learned_transport_family_passed": family_passed,
            "passed": bool(anchor_passed and family_passed),
            "criterion": (
                "oracle matched exceeds same-stratum teacher donor, then learned "
                "matched exceeds shuffled, mean scaffold, and norm-matched random "
                "under one Holm family"
            ),
        }
        condition_means = {
            condition: report["mean"]
            for condition, report in metrics["functional_pass"][
                "conditions"
            ].items()
        }
        recovery = normalized_transport_recovery(
            learned_matched=condition_means["learned_matched"],
            learned_shuffled=condition_means["learned_shuffled"],
            oracle_matched=condition_means["oracle_teacher_matched"],
            oracle_shuffled=condition_means["oracle_teacher_shuffled"],
            text=condition_means["text_only_no_lip"],
            neutral=condition_means["neutral_no_lip"],
        )
    sandbox_validated = bool(
        security_context and security_context.get("validated")
    )
    claim_eligible = bool(
        functional
        and design_validation["complete"]
        and design_validation["run_scope"] == "confirmation"
        and sandbox_validated
    )
    summary = {
        "experiment_id": PACKET_CONFIRMATION_EXPERIMENT_ID,
        "protocol_version": PACKET_CONFIRMATION_PROTOCOL_VERSION,
        "generations_jsonl": str(generations_path),
        "generation_metadata": str(metadata_path),
        "scored_jsonl": str(output_dir / "scored_generations.jsonl"),
        "execution_mode": (
            "functional_hardened_namespace"
            if functional and sandbox_validated
            else "functional_subprocess"
            if functional
            else "syntax_only"
        ),
        "subprocess_is_security_sandbox": sandbox_validated if functional else None,
        "claim_eligible": claim_eligible,
        "semantic_gate": semantic_gate,
        "primary_inference": primary_inference,
        "functional_recovery": recovery,
        "semantic_transport_supported": bool(
            claim_eligible and semantic_gate and semantic_gate["passed"]
        ),
        "design_validation": design_validation,
        "metrics": metrics,
        "artifact_provenance": {
            key: metadata.get(key)
            for key in (
                "task_manifest_sha256",
                "selection_report_sha256",
                "confirmation_bundle_manifest_sha256",
                "training_bundle_manifest_sha256",
                "matrix_summary_sha256",
                "primary_variant",
                "primary_replicas",
                "source_model_revision",
                "target_model_revision",
            )
        },
    }
    if security_context is not None:
        summary["sandbox"] = dict(security_context)
    scored_path = output_dir / "scored_generations.jsonl"
    write_jsonl(scored_path, scored)
    summary["scored_jsonl_sha256"] = sha256_path(scored_path)
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.functional and not args.allow_unsafe_execution:
        raise RuntimeError(
            "functional evaluation executes untrusted code; use the hardened "
            "namespace runner for claim-oriented scoring"
        )
    config = load_yaml(args.config)
    summary = evaluate(
        config,
        args.generations,
        args.output_dir,
        functional=args.functional,
        allow_incomplete=args.allow_incomplete,
        overwrite=args.overwrite,
    )
    print("LIP packet bridge confirmation evaluation completed")
    print(f"execution_mode: {summary['execution_mode']}")
    print(f"claim_eligible: {summary['claim_eligible']}")
    print(f"semantic_transport_supported: {summary['semantic_transport_supported']}")
    print(f"summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
