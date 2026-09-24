"""Strict grid validation and sandbox-only scoring for the joint diagnostic."""
import json
import random
from pathlib import Path

from src.evaluation.alias_normalized_diagnostic import build_single_function_alias
from src.evaluation.oracle_functional import declares_entry_point
from src.evaluation.semantics import evaluate_generation, run_functional_tests
from src.pipelines.oracle_experiment import prepare_output_dir, task_sha256, write_json, write_jsonl


def donor_map(task_ids, seed):
    cycle = list(task_ids)
    if len(cycle) < 2 or len(set(cycle)) != len(cycle):
        raise ValueError("shuffle requires unique tasks")
    random.Random(seed).shuffle(cycle)
    return {task: cycle[(i + 1) % len(cycle)] for i, task in enumerate(cycle)}


def validate_grid(config, rows, metadata):
    expected = {(task, condition) for task in config["task_ids"] for condition in config["conditions"]}
    keys = [(row["task_id"], row["condition"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("functional grid is incomplete, duplicated, or contains extra tasks")
    if metadata["policy"] != config:
        raise ValueError("generation policy changed")
    donors = donor_map(config["task_ids"], config["seed"])
    for row in rows:
        if task_sha256(row["task_spec"]) != metadata["task_hashes"][row["task_id"]]:
            raise ValueError("task tests/specification changed")
        condition = row["condition"]
        expected_donor = (donors[row["task_id"]] if condition.endswith("shuffled") else row["task_id"])
        if condition in ("text", "no_message"):
            expected_donor = None
        if row["donor_task_id"] != expected_donor:
            raise ValueError("incorrect packet donor")


def evaluate(config, generations_path, output_dir, *, functional=False,
             allow_incomplete=False, overwrite=False,
             candidate_process_policy=None, security_context=None):
    if not functional or not security_context or not security_context.get("validated") or candidate_process_policy is None:
        raise ValueError("functional diagnostic requires the validated Linux sandbox")
    rows = [json.loads(line) for line in Path(generations_path).read_text().splitlines() if line.strip()]
    metadata = json.loads(Path(generations_path).with_suffix(".metadata.json").read_text())
    validate_grid(config, rows, metadata)
    prepare_output_dir(Path(output_dir), overwrite=overwrite)
    scored = []
    for row in rows:
        task = row["task_spec"]
        result = evaluate_generation(row, task, run_functional=True, process_policy=candidate_process_policy)
        result["declares_expected_name"] = declares_entry_point(result["extracted_code"], task["entry_point"])
        alias = build_single_function_alias(result["extracted_code"], task["entry_point"])
        result["alias_eligible"] = alias["eligible"]
        result["alias_applied"] = alias["alias_binding_applied"]
        result["alias_functional_pass"] = result["functional_pass"]
        if alias["alias_binding_applied"]:
            result["alias_functional_pass"] = run_functional_tests(alias["normalized_code"], task, process_policy=candidate_process_policy)["functional_pass"]
        scored.append(result)
    totals = {}
    for condition in config["conditions"]:
        group = [row for row in scored if row["condition"] == condition]
        totals[condition] = {"tasks": len(group), **{
            key: sum(bool(row[key]) for row in group)
            for key in ("functional_pass", "alias_functional_pass", "syntax_pass", "declares_expected_name", "hit_token_limit")
        }}
    comparisons = {}
    for arm in ("regional", "joint_only", "oracle"):
        matched = {r["task_id"]: r["functional_pass"] for r in scored if r["condition"] == arm + "_matched"}
        shuffled = {r["task_id"]: r["functional_pass"] for r in scored if r["condition"] == arm + "_shuffled"}
        comparisons[arm] = {"matched_only": sum(matched[t] and not shuffled[t] for t in matched),
                            "shuffled_only": sum(shuffled[t] and not matched[t] for t in matched)}
    summary = {"experiment_id": config["experiment_id"], "execution_mode": "hardened_functional",
               "claim_eligible": False, "diagnostic_route": "exploratory_joint_functional_completed",
               "subprocess_is_security_sandbox": True, "sandbox": security_context,
               "conditions": totals, "paired_comparisons": comparisons,
               "scope": "32 reused development tasks, greedy decoding, frozen checkpoints; not generalization or confirmation"}
    write_jsonl(Path(output_dir) / "scored.jsonl", scored)
    write_json(Path(output_dir) / "summary.json", summary)
    return summary
