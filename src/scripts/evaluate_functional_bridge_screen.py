"""Validate, harden-score, and test the development-only LIP-EVAL-033."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.evaluation.functional_bridge_screen import (
    FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS,
    FUNCTIONAL_BRIDGE_SCREEN_EXPERIMENT_ID,
    FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS,
    FUNCTIONAL_BRIDGE_SCREEN_PROTOCOL_VERSION,
    FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS,
    expected_functional_bridge_screen_keys,
    functional_bridge_screen_design_fingerprint,
    summarize_functional_bridge_screen,
    validate_functional_bridge_screen_contract,
)
from src.evaluation.oracle_functional import declares_entry_point, stable_seed
from src.evaluation.semantics import CandidateProcessPolicy, evaluate_generation
from src.evaluation.statistics import summarize_metric
from src.pipelines.oracle_experiment import (
    load_json_object,
    load_yaml,
    prepare_output_dir,
    sha256_path,
    write_json,
    write_jsonl,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON constant {value}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line, parse_constant=reject_constant)
            if not isinstance(row, dict):
                raise ValueError(f"row {line_number} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError("EVAL-033 generation file is empty")
    return rows


def _key(row: Mapping) -> tuple[str, str, int, int]:
    return (
        str(row.get("task_id", "")),
        str(row.get("condition", "")),
        int(row.get("generation_seed", -1)),
        int(row.get("training_seed", -1)),
    )


def validate_functional_bridge_screen_grid(
    rows: Sequence[Mapping],
    metadata: Mapping,
    config: Mapping,
    *,
    allow_incomplete: bool,
) -> dict:
    validate_functional_bridge_screen_contract(config)
    design_sha = functional_bridge_screen_design_fingerprint(config)
    metadata_checks = {
        "experiment": metadata.get("experiment_id")
        == FUNCTIONAL_BRIDGE_SCREEN_EXPERIMENT_ID,
        "protocol": metadata.get("protocol_version")
        == FUNCTIONAL_BRIDGE_SCREEN_PROTOCOL_VERSION,
        "design": metadata.get("design_sha256") == design_sha,
        "scope": metadata.get("run_scope")
        == "development_only_reused_open_P014_cohort",
        "claim": metadata.get("claim_eligible") is False,
        "task_count": metadata.get("task_count") == 32,
        "conditions": tuple(metadata.get("conditions", ()))
        == FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS,
        "generation_seeds": tuple(metadata.get("generation_seeds", ()))
        == FUNCTIONAL_BRIDGE_SCREEN_GENERATION_SEEDS,
        "training_seeds": tuple(metadata.get("training_seeds", ()))
        == FUNCTIONAL_BRIDGE_SCREEN_TRAINING_SEEDS,
        "P014_hash": metadata.get("P014_generations_sha256")
        == config["cohort"]["source_artifacts"]["generations_sha256"],
        "bundle_hash": metadata.get("confirmation_bundle_manifest_sha256")
        == config["cohort"]["source_artifacts"][
            "confirmation_bundle_manifest_sha256"
        ],
    }
    failed = [name for name, passed in metadata_checks.items() if not passed]
    if failed:
        raise ValueError("EVAL-033 metadata drifted: " + ", ".join(failed))
    task_ids = [str(value) for value in metadata["task_ids"]]
    expected = expected_functional_bridge_screen_keys(task_ids)
    if metadata.get("expected_records") != len(expected):
        raise ValueError("EVAL-033 expected-record count drifted")
    donors = {
        str(target): str(donor)
        for target, donor in metadata.get("donor_task_ids", {}).items()
    }
    if set(donors) != set(task_ids) or any(
        target == donor or donor not in task_ids for target, donor in donors.items()
    ):
        raise ValueError("EVAL-033 donor map is invalid")
    observed = []
    tasks = {}
    matched_hashes: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        key = _key(row)
        observed.append(key)
        task_id, condition, generation_seed, training_seed = key
        if condition not in FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS:
            raise ValueError(f"unknown EVAL-033 condition: {condition}")
        task = row.get("task_spec")
        if not isinstance(task, Mapping) or str(task.get("task_id", "")) != task_id:
            raise ValueError("EVAL-033 task_spec identity changed")
        previous = tasks.setdefault(task_id, dict(task))
        if previous != dict(task):
            raise ValueError("EVAL-033 task_spec varies across rows")
        packet_norm = float(row.get("packet_frobenius_norm", math.nan))
        packet_hash = row.get("packet_sha256")
        source_expected = task_id if condition == "learned_matched" else donors[task_id]
        row_checks = {
            "experiment": row.get("experiment_id")
            == FUNCTIONAL_BRIDGE_SCREEN_EXPERIMENT_ID,
            "protocol": row.get("protocol_version")
            == FUNCTIONAL_BRIDGE_SCREEN_PROTOCOL_VERSION,
            "design": row.get("design_sha256") == design_sha,
            "config": row.get("config_sha256") == metadata.get("config_sha256"),
            "scope": row.get("run_scope")
            == "development_only_reused_open_P014_cohort",
            "claim": row.get("claim_eligible") is False,
            "effective_seed": row.get("effective_generation_seed")
            == stable_seed(generation_seed, task_ids.index(task_id), 14014),
            "neutral": row.get("target_prompt_kind") == "neutral",
            "input_ids": row.get("target_input_ids_sha256")
            == metadata.get("neutral_input_ids_sha256"),
            "attention": row.get("target_attention_mask_sha256")
            == metadata.get("neutral_attention_mask_sha256"),
            "packet": row.get("packet_present") is True
            and isinstance(packet_hash, str)
            and len(packet_hash) == 64
            and math.isfinite(packet_norm),
            "packet_kind": row.get("packet_kind") == condition,
            "entry_layer": row.get("packet_layer_indices") == [0],
            "offsets": row.get("packet_offsets")
            == config["packets"]["target"]["offsets"],
            "one_layer_norm": len(row.get("packet_layer_norms", ())) == 1
            and len(row.get("packet_residual_layer_norms", ())) == 1,
            "source": row.get("source_task_id") == source_expected,
            "donor": row.get("donor_task_id")
            == (donors[task_id] if condition == "learned_shuffled" else None),
            "P014": row.get("P014_generations_sha256")
            == metadata.get("P014_generations_sha256"),
            "bundle": row.get("confirmation_bundle_manifest_sha256")
            == metadata.get("confirmation_bundle_manifest_sha256"),
            "revision": row.get("target_model_revision")
            == metadata.get("target_model_revision"),
            "output": isinstance(row.get("output_text"), str),
        }
        failed_row = [name for name, passed in row_checks.items() if not passed]
        if failed_row:
            raise ValueError(f"EVAL-033 row {key} drifted: " + ", ".join(failed_row))
        if condition == "learned_matched":
            matched_hashes.setdefault((task_id, training_seed), set()).add(packet_hash)
    if len(set(observed)) != len(observed):
        raise ValueError("EVAL-033 contains duplicate rows")
    unexpected = set(observed).difference(expected)
    missing = expected.difference(observed)
    if unexpected:
        raise ValueError("EVAL-033 contains unexpected rows")
    if missing and not allow_incomplete:
        raise ValueError(f"EVAL-033 is missing {len(missing)} rows")
    if any(len(values) != 1 for values in matched_hashes.values()):
        raise ValueError("matched entry packets vary by generation seed")
    for row in rows:
        task_id, condition, _, training_seed = _key(row)
        if condition == "learned_shuffled":
            donor_hashes = matched_hashes.get((donors[task_id], training_seed))
            if donor_hashes is not None and row.get("packet_sha256") != next(
                iter(donor_hashes)
            ):
                raise ValueError("shuffled entry packet is not its frozen donor")
    complete = not missing
    if bool(metadata.get("complete")) != complete:
        raise ValueError("EVAL-033 metadata completeness disagrees with grid")
    if complete and metadata.get("records") != len(rows):
        raise ValueError("EVAL-033 metadata record count disagrees with grid")
    return {
        "complete": complete,
        "run_scope": "development_only_reused_open_P014_cohort",
        "claim_eligible": False,
        "task_count": len(task_ids),
        "record_count": len(rows),
        "expected_record_count": len(expected),
        "missing_record_count": len(missing),
        "design_sha256": design_sha,
        "cluster_unit": "task_id",
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
    metadata = load_json_object(generations_path.with_suffix(".metadata.json"))
    rows = _read_jsonl(generations_path)
    design_validation = validate_functional_bridge_screen_grid(
        rows, metadata, config, allow_incomplete=allow_incomplete
    )
    prepare_output_dir(output_dir, overwrite=overwrite)
    policy = config["evaluation"]
    scored = []
    for row in rows:
        scored_row = evaluate_generation(
            row,
            row["task_spec"],
            run_functional=functional,
            timeout_seconds=5.0,
            memory_mb=512,
            process_policy=candidate_process_policy,
        )
        scored_row["entry_point_declared"] = declares_entry_point(
            scored_row["extracted_code"], row["task_spec"].get("entry_point")
        )
        scored.append(scored_row)
    statistics_kwargs = {
        "bootstrap_iterations": int(policy["bootstrap_iterations"]),
        "confidence": float(policy["confidence"]),
        "seed": int(policy["statistics_seed"]),
    }
    descriptive = {
        metric: summarize_metric(
            scored,
            metric,
            FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS,
            [("learned_matched", "learned_shuffled")],
            **statistics_kwargs,
        )
        for metric in ("syntax_pass", "entry_point_declared")
    }
    inference = None
    if functional:
        descriptive["functional_pass"] = summarize_metric(
            scored,
            "functional_pass",
            FUNCTIONAL_BRIDGE_SCREEN_CONDITIONS,
            [("learned_matched", "learned_shuffled")],
            **statistics_kwargs,
        )
        if design_validation["complete"]:
            inference = summarize_functional_bridge_screen(scored, config)
    sandbox_validated = bool(security_context and security_context.get("validated"))
    signal = bool(
        functional
        and sandbox_validated
        and design_validation["complete"]
        and inference
        and inference["development_functional_signal_detected"]
    )
    summary = {
        "experiment_id": FUNCTIONAL_BRIDGE_SCREEN_EXPERIMENT_ID,
        "protocol_version": FUNCTIONAL_BRIDGE_SCREEN_PROTOCOL_VERSION,
        "execution_mode": (
            "functional_hardened_namespace"
            if functional and sandbox_validated
            else "functional_subprocess"
            if functional
            else "syntax_only"
        ),
        "claim_eligible": False,
        "development_functional_signal_detected": signal,
        "design_validation": design_validation,
        "inference": inference,
        "metrics": descriptive,
        "artifact_provenance": {
            key: metadata.get(key)
            for key in (
                "P014_generations_sha256",
                "P014_metadata_sha256",
                "P014_functional_summary_sha256",
                "confirmation_bundle_manifest_sha256",
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
    summary["scored_jsonl"] = str(scored_path)
    summary["scored_jsonl_sha256"] = sha256_path(scored_path)
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/LIP-EVAL-033_functional_bridge_screen.yaml"),
    )
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--functional", action="store_true")
    parser.add_argument("--allow-unsafe-execution", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.functional and not args.allow_unsafe_execution:
        raise RuntimeError("use the hardened namespace runner for functional scoring")
    summary = evaluate(
        load_yaml(args.config),
        args.generations,
        args.output_dir,
        functional=args.functional,
        allow_incomplete=args.allow_incomplete,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "execution_mode": summary["execution_mode"],
        "claim_eligible": summary["claim_eligible"],
        "development_functional_signal_detected": summary[
            "development_functional_signal_detected"
        ],
    }))


if __name__ == "__main__":
    main()
