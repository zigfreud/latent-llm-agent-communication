"""Score syntax or opt-in functional behavior for oracle packet outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.oracle_functional import (
    declares_entry_point,
    design_fingerprint,
    packet_contract,
    protocol_version_for_config,
    semantic_gate,
)
from src.evaluation.semantics import CandidateProcessPolicy, evaluate_generation
from src.evaluation.statistics import summarize_metric
from src.pipelines.oracle_experiment import (
    load_json_object,
    load_yaml,
    prepare_output_dir,
    write_json,
    write_jsonl,
)


DEFAULT_CONFIG = Path("config/LIP-PROTO-005_oracle_packet_functional.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--generations", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--functional", action="store_true")
    parser.add_argument("--allow-unsafe-execution", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"generation row {line_number} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError("generation file contains no records")
    return rows


def validate_generation_grid(
    records: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    allow_incomplete: bool,
) -> dict[str, Any]:
    design_sha256 = design_fingerprint(dict(config))
    protocol_version = protocol_version_for_config(config)
    if metadata.get("protocol_version") != protocol_version:
        raise ValueError("generation metadata uses the wrong protocol version")
    if metadata.get("design_sha256") != design_sha256:
        raise ValueError("generation metadata does not match the frozen config")
    task_ids = [str(task_id) for task_id in metadata.get("task_ids", [])]
    conditions = [str(condition) for condition in config["conditions"]]
    generation_seeds = [int(seed) for seed in metadata.get("generation_seeds", [])]
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("metadata task IDs must be a non-empty unique sequence")
    if not generation_seeds or len(set(generation_seeds)) != len(generation_seeds):
        raise ValueError("metadata generation seeds must be a non-empty unique sequence")

    expected = {
        (task_id, condition, seed)
        for task_id in task_ids
        for condition in conditions
        for seed in generation_seeds
    }
    observed = []
    task_specs: dict[str, Mapping[str, Any]] = {}
    for row in records:
        if row.get("protocol_version") != protocol_version:
            raise ValueError("generation record uses the wrong protocol version")
        if row.get("design_sha256") != design_sha256:
            raise ValueError("generation record does not match the frozen config")
        key = (
            str(row.get("task_id")),
            str(row.get("condition")),
            int(row.get("generation_seed")),
        )
        observed.append(key)
        task_spec = row.get("task_spec")
        if not isinstance(task_spec, Mapping):
            raise ValueError("each generation record must contain task_spec")
        existing = task_specs.setdefault(key[0], task_spec)
        if existing != task_spec:
            raise ValueError(f"task specification changes across records: {key[0]}")
    if len(set(observed)) != len(observed):
        raise ValueError("generation grid contains duplicate records")
    unexpected = set(observed).difference(expected)
    missing = expected.difference(observed)
    if unexpected:
        raise ValueError(f"generation grid has {len(unexpected)} unexpected records")
    if missing and not allow_incomplete:
        raise ValueError(f"generation grid is missing {len(missing)} records")
    complete = not missing and len(task_ids) == int(
        config["data"]["functional_task_count"]
    )
    if not allow_incomplete and not complete:
        raise ValueError("only the full frozen task slice is claim-eligible")
    return {
        "complete": complete,
        "run_scope": metadata.get("run_scope"),
        "task_count": len(task_ids),
        "record_count": len(records),
        "expected_record_count": len(expected),
        "missing_record_count": len(missing),
        "design_sha256": design_sha256,
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
    design_validation = validate_generation_grid(
        records,
        metadata,
        config,
        allow_incomplete=allow_incomplete,
    )
    prepare_output_dir(output_dir, overwrite=overwrite)
    evaluation_config = config["evaluation"]
    scored = []
    for row in records:
        scored_row = evaluate_generation(
            row,
            row["task_spec"],
            run_functional=functional,
            timeout_seconds=float(evaluation_config["timeout_seconds"]),
            memory_mb=int(evaluation_config["memory_mb"]),
            process_policy=candidate_process_policy,
        )
        scored_row["entry_point_declared"] = declares_entry_point(
            scored_row["extracted_code"],
            row["task_spec"].get("entry_point"),
        )
        scored.append(scored_row)
    conditions = list(config["conditions"])
    comparisons = list(evaluation_config["comparisons"])
    statistics_kwargs = {
        "bootstrap_iterations": int(evaluation_config["bootstrap_iterations"]),
        "confidence": float(evaluation_config["confidence"]),
        "seed": int(evaluation_config["statistics_seed"]),
    }
    metrics = {
        "syntax_pass": summarize_metric(
            scored,
            "syntax_pass",
            conditions,
            comparisons,
            **statistics_kwargs,
        ),
        "entry_point_declared": summarize_metric(
            scored,
            "entry_point_declared",
            conditions,
            comparisons,
            **statistics_kwargs,
        ),
    }
    if functional:
        metrics["functional_pass"] = summarize_metric(
            scored,
            "functional_pass",
            conditions,
            comparisons,
            **statistics_kwargs,
        )
    gate = None
    if functional:
        packet_sizes, replication_size = packet_contract(config)
        gate = semantic_gate(
            {
                condition: values["mean"]
                for condition, values in metrics["functional_pass"][
                    "conditions"
                ].items()
            },
            packet_sizes=packet_sizes,
            replication_size=replication_size,
        )
    summary = {
        "experiment_id": config["experiment_id"],
        "protocol_version": protocol_version_for_config(config),
        "generations_jsonl": str(generations_path),
        "generation_metadata": str(metadata_path),
        "scored_jsonl": str(output_dir / "scored_generations.jsonl"),
        "execution_mode": (
            "functional_hardened_namespace"
            if functional and security_context
            else "functional_subprocess"
            if functional
            else "syntax_only"
        ),
        "subprocess_is_security_sandbox": (
            bool(security_context and security_context.get("validated"))
            if functional
            else None
        ),
        "claim_eligible": bool(
            functional
            and design_validation["complete"]
            and design_validation["run_scope"] == "full"
        ),
        "semantic_gate": gate,
        "semantic_transport_supported": bool(
            functional
            and design_validation["complete"]
            and design_validation["run_scope"] == "full"
            and gate
            and gate["passed"]
        ),
        "design_validation": design_validation,
        "metrics": metrics,
    }
    if security_context is not None:
        summary["sandbox"] = dict(security_context)
    write_jsonl(output_dir / "scored_generations.jsonl", scored)
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.functional and not args.allow_unsafe_execution:
        raise RuntimeError(
            "functional evaluation executes untrusted code; run only in a disposable, "
            "network-isolated environment and pass --allow-unsafe-execution"
        )
    config = load_yaml(args.config)
    generations_path = args.generations or Path(
        str(config["output"]["generations_jsonl"])
    )
    output_dir = args.output_dir or Path(str(config["output"]["evaluation_dir"]))
    summary = evaluate(
        config,
        generations_path,
        output_dir,
        functional=args.functional,
        allow_incomplete=args.allow_incomplete,
        overwrite=args.overwrite,
    )
    print("Oracle packet semantic evaluation completed")
    print(f"execution_mode: {summary['execution_mode']}")
    print(f"claim_eligible: {summary['claim_eligible']}")
    print(f"summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
