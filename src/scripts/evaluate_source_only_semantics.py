"""Score source-only generations and report task-clustered uncertainty."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.semantics import evaluate_generation
from src.evaluation.source_only import design_fingerprint
from src.evaluation.statistics import summarize_metric


DEFAULT_CONFIG = Path("config/LIP-PROTO-001_source_only_eval.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate source-only Python generations.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--generations", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--functional",
        action="store_true",
        help="Execute task tests in a resource-limited subprocess.",
    )
    parser.add_argument(
        "--allow-unsafe-execution",
        action="store_true",
        help="Acknowledge that the subprocess runner is not a security sandbox.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Score a partial run while still enforcing per-record protocol invariants.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"no generation records found in {path}")
    return rows


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(path: Path, summary: dict):
    lines = [
        f"# {summary['experiment_id']} source-only evaluation",
        "",
        f"- Records: {summary['record_count']}",
        f"- Tasks: {summary['task_count']}",
        f"- Execution mode: {summary['execution_mode']}",
        f"- Complete factorial design: {str(summary['design_validation']['complete']).lower()}",
        "- Confidence intervals resample tasks and average repeated training/generation seeds within task.",
        "",
    ]
    for metric, metric_summary in summary["metrics"].items():
        lines.extend([f"## {metric}", ""])
        for condition, values in metric_summary["conditions"].items():
            lines.append(
                f"- `{condition}`: {values['mean']:.4f} "
                f"({metric_summary['confidence']:.0%} CI "
                f"{values['ci_lower']:.4f} to {values['ci_upper']:.4f}; "
                f"n={values['task_count']} tasks)"
            )
            if values["by_training_seed"]:
                seed_text = ", ".join(
                    f"{seed}={seed_values['mean']:.4f}"
                    for seed, seed_values in values["by_training_seed"].items()
                )
                lines.append(f"  - training-seed means: {seed_text}")
        if metric_summary["comparisons"]:
            lines.extend(["", "Paired task-level comparisons:", ""])
            for comparison in metric_summary["comparisons"]:
                lines.append(
                    f"- `{comparison['treatment']}` minus `{comparison['control']}`: "
                    f"{comparison['mean_difference']:.4f} "
                    f"(CI {comparison['ci_lower']:.4f} to {comparison['ci_upper']:.4f}; "
                    f"two-sided p={comparison['p_value_two_sided']:.4g}, "
                    f"Holm p={comparison['p_value_holm']:.4g}, "
                    f"{comparison['p_value_method']})"
                )
        lines.append("")
    lines.extend(
        [
            "## Interpretation guardrail",
            "",
            "This report is an experimental result, not evidence of semantic transport by itself. "
            "The source-latent condition must outperform neutral, shuffled-vector, and random-vector controls across independent adapter seeds before a stronger claim is considered.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def validate_generation_records(
    records: list[dict],
    config: dict,
    *,
    allow_incomplete: bool = False,
) -> dict:
    conditions = list(config.get("conditions", []))
    training_seeds = [
        int(item["training_seed"])
        for item in config.get("adapter", {}).get("checkpoints", [])
    ]
    generation_seeds = [int(seed) for seed in config.get("generation", {}).get("seeds", [])]
    if not conditions or not training_seeds or not generation_seeds:
        raise ValueError("config must define conditions, adapter checkpoints, and generation seeds")

    keys = set()
    tasks = {}
    design_hashes = set()
    training_bundle_hashes = set()
    checkpoint_hashes = {}
    neutral_hash = hashlib.sha256(
        str(config.get("neutral_target_prompt", "")).strip().encode("utf-8")
    ).hexdigest()
    vector_expectations = {
        "neutral_no_lip": (None, None),
        "text_only_no_lip": (None, None),
        "source_latent": ("translated_source", "matching"),
        "shuffled_source_latent": ("translated_source", "different"),
        "random_norm_matched": ("random_norm_matched", None),
        "oracle_target_latent": ("target_hidden", "matching"),
    }

    for row in records:
        key = (
            str(row.get("task_id", "")),
            str(row.get("condition", "")),
            int(row.get("training_seed", -1)),
            int(row.get("generation_seed", -1)),
        )
        if key in keys:
            raise ValueError(f"duplicate generation record: {key}")
        keys.add(key)
        task_id, condition, training_seed, generation_seed = key
        if not task_id or condition not in conditions:
            raise ValueError(f"unexpected task/condition in generation record: {key}")
        if training_seed not in training_seeds or generation_seed not in generation_seeds:
            raise ValueError(f"unexpected seed in generation record: {key}")
        if row.get("protocol_version") != "lip-source-only-v1":
            raise ValueError("generation record has the wrong protocol_version")
        design_hashes.add(row.get("design_sha256"))
        training_bundle_hashes.add(row.get("training_bundle_manifest_sha256"))
        checkpoint_hashes.setdefault(training_seed, set()).add(
            row.get("adapter_checkpoint_sha256")
        )
        formatted_prompt_hash = row.get("target_formatted_prompt_sha256")
        if not is_sha256(formatted_prompt_hash):
            raise ValueError(f"generation record {key} has no formatted-prompt digest")

        task_spec = row.get("task_spec")
        if not isinstance(task_spec, dict) or str(task_spec.get("task_id")) != task_id:
            raise ValueError(f"generation record {key} has an invalid task_spec")
        previous_task = tasks.setdefault(task_id, task_spec)
        if previous_task != task_spec:
            raise ValueError(f"task {task_id} has inconsistent task specifications")

        expected_kind, vector_relation = vector_expectations[condition]
        if row.get("vector_kind") != expected_kind:
            raise ValueError(f"generation record {key} has the wrong vector_kind")
        vector_task_id = row.get("vector_task_id")
        if vector_relation == "matching" and str(vector_task_id) != task_id:
            raise ValueError(f"generation record {key} must use its matching vector")
        if vector_relation == "different" and (
            vector_task_id is None or str(vector_task_id) == task_id
        ):
            raise ValueError(f"generation record {key} is not actually shuffled")
        if vector_relation is None and vector_task_id is not None:
            raise ValueError(f"generation record {key} has unexpected vector provenance")
        if expected_kind is None and row.get("injected_vector_norm") is not None:
            raise ValueError(f"generation record {key} must not inject a vector")
        if expected_kind is not None:
            norm = row.get("injected_vector_norm")
            if norm is None or float(norm) <= 0:
                raise ValueError(f"generation record {key} has no positive vector norm")

        if condition == "text_only_no_lip":
            if row.get("target_prompt_kind") != "task":
                raise ValueError(f"generation record {key} must expose task text")
            task_prompt_hash = hashlib.sha256(
                str(task_spec.get("prompt", "")).encode("utf-8")
            ).hexdigest()
            if row.get("target_user_prompt_sha256") != task_prompt_hash:
                raise ValueError(f"generation record {key} used the wrong task prompt")
        else:
            if row.get("target_prompt_kind") != "neutral":
                raise ValueError(f"generation record {key} leaks task text to the target")
            if row.get("target_user_prompt_sha256") != neutral_hash:
                raise ValueError(f"generation record {key} used a non-neutral target prompt")

    expected_design_hash = design_fingerprint(config)
    if design_hashes != {expected_design_hash}:
        raise ValueError("generation records do not match the configured design fingerprint")
    if len(training_bundle_hashes) != 1 or not all(
        is_sha256(value) for value in training_bundle_hashes
    ):
        raise ValueError("generation records do not share one training-bundle digest")
    if any(
        len(values) != 1
        or not all(is_sha256(value) for value in values)
        for values in checkpoint_hashes.values()
    ):
        raise ValueError("generation records have inconsistent checkpoint digests")
    expected_task_count = int(config.get("data", {}).get("task_count", len(tasks)))
    if not allow_incomplete and len(tasks) != expected_task_count:
        raise ValueError(
            f"expected {expected_task_count} tasks, found {len(tasks)}; "
            "use --allow-incomplete only for diagnostics"
        )
    expected_keys = {
        (task_id, condition, training_seed, generation_seed)
        for task_id in tasks
        for condition in conditions
        for training_seed in training_seeds
        for generation_seed in generation_seeds
    }
    missing = sorted(expected_keys.difference(keys))
    if missing and not allow_incomplete:
        raise ValueError(
            f"generation design is missing {len(missing)} records; first missing: {missing[0]}"
        )
    return {
        "complete": not missing and len(tasks) == expected_task_count,
        "task_count": len(tasks),
        "expected_task_count": expected_task_count,
        "record_count": len(records),
        "expected_record_count_for_present_tasks": len(expected_keys),
        "missing_record_count": len(missing),
        "design_sha256": next(iter(design_hashes)),
        "training_bundle_manifest_sha256": next(iter(training_bundle_hashes)),
    }


def evaluate(
    config: dict,
    generations_path: Path,
    output_dir: Path,
    functional: bool,
    *,
    allow_incomplete: bool = False,
):
    records = read_jsonl(generations_path)
    design_validation = validate_generation_records(
        records,
        config,
        allow_incomplete=allow_incomplete,
    )
    evaluation_config = config.get("evaluation", {})
    scored = []
    for record in records:
        task = record.get("task_spec")
        if not isinstance(task, dict):
            raise ValueError("each generation record must contain task_spec")
        scored.append(
            evaluate_generation(
                record,
                task,
                run_functional=functional,
                timeout_seconds=float(evaluation_config.get("timeout_seconds", 5.0)),
                memory_mb=int(evaluation_config.get("memory_mb", 512)),
            )
        )

    conditions = list(config.get("conditions", sorted({row["condition"] for row in scored})))
    comparisons = evaluation_config.get(
        "comparisons",
        [
            ["source_latent", "neutral_no_lip"],
            ["source_latent", "shuffled_source_latent"],
            ["source_latent", "random_norm_matched"],
        ],
    )
    stats_kwargs = {
        "bootstrap_iterations": int(evaluation_config.get("bootstrap_iterations", 10_000)),
        "confidence": float(evaluation_config.get("confidence", 0.95)),
        "seed": int(evaluation_config.get("statistics_seed", 1729)),
    }
    metrics = {
        "syntax_pass": summarize_metric(
            scored,
            "syntax_pass",
            conditions,
            comparisons,
            **stats_kwargs,
        )
    }
    if functional:
        metrics["functional_pass"] = summarize_metric(
            scored,
            "functional_pass",
            conditions,
            comparisons,
            **stats_kwargs,
        )

    summary = {
        "experiment_id": str(config.get("experiment_id", "LIP-PROTO-001")),
        "generations_jsonl": str(generations_path),
        "scored_jsonl": str(output_dir / "scored_generations.jsonl"),
        "execution_mode": "functional_subprocess" if functional else "syntax_only",
        "subprocess_is_security_sandbox": False if functional else None,
        "record_count": len(scored),
        "task_count": len({row["task_id"] for row in scored}),
        "design_validation": design_validation,
        "metrics": metrics,
    }
    write_jsonl(output_dir / "scored_generations.jsonl", scored)
    write_json(output_dir / "summary.json", summary)
    write_markdown(output_dir / "summary.md", summary)
    return summary


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must contain a YAML mapping")
    if args.functional and not args.allow_unsafe_execution:
        raise RuntimeError(
            "functional evaluation executes untrusted code; rerun only in a disposable, "
            "network-isolated environment with --allow-unsafe-execution"
        )
    generations_path = args.generations or Path(
        config.get("output", {}).get(
            "generations_jsonl",
            "runs/LIP-PROTO-001/generations.jsonl",
        )
    )
    output_dir = args.output_dir or Path(
        config.get("output", {}).get(
            "evaluation_dir",
            "runs/LIP-PROTO-001/evaluation",
        )
    )
    summary = evaluate(
        config,
        generations_path,
        output_dir,
        args.functional,
        allow_incomplete=args.allow_incomplete,
    )
    print("Source-only semantic evaluation completed")
    print(f"records: {summary['record_count']}")
    print(f"summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
