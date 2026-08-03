"""Audit whether one exact target hidden state survives cross-prompt transport."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from src.core.prompt_protocol import protocol_metadata
from src.core.utils import set_seed
from src.evaluation.oracle_transport import (
    continuation_token_metrics,
    normalize_layer_indices,
    recovery_fraction,
    summarize_oracle_transport,
)
from src.pipelines.infer import load_target, model_input_device
from src.pipelines.oracle_experiment import (
    bind_tasks_to_manifest,
    load_tasks,
    load_yaml,
    prepare_output_dir,
    prompt_sha256,
    sha256_path,
    write_json,
    write_jsonl,
)
from src.pipelines.oracle_transport import (
    append_reference,
    build_neutral_carrier,
    encode_prompt,
    forward_with_layer_capture,
    forward_with_optional_replacement,
)


DEFAULT_CONFIG = Path("config/LIP-PROTO-002_oracle_transport.yaml")
EXPERIMENT_CARRIER_MODES = {
    "LIP-PROTO-002": "native",
    "LIP-PROTO-003": "left_pad_masked_to_task_length",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.preflight and args.max_tasks is not None:
        parser.error("--preflight and --max-tasks are mutually exclusive")
    return args


def validate_config(config: Mapping[str, Any]) -> None:
    expected_top_level = {
        "experiment_id",
        "source_protocol_experiment",
        "models",
        "prompt_protocol",
        "runtime",
        "data",
        "neutral_target_prompt",
        "carrier",
        "audit",
        "output",
    }
    unknown = sorted(set(config).difference(expected_top_level))
    if unknown:
        raise ValueError(f"unknown config field(s): {', '.join(unknown)}")
    experiment_id = str(config.get("experiment_id", ""))
    if experiment_id not in EXPERIMENT_CARRIER_MODES:
        allowed = ", ".join(EXPERIMENT_CARRIER_MODES)
        raise ValueError(f"experiment_id must be one of: {allowed}")
    if config.get("source_protocol_experiment") != "LIP-PROTO-001":
        raise ValueError("source_protocol_experiment must bind LIP-PROTO-001")

    protocol = protocol_metadata(config.get("prompt_protocol"))
    if protocol["mode"] != "chat_template" or not protocol["add_generation_prompt"]:
        raise ValueError(
            "oracle audit requires the target chat template and generation marker"
        )
    if not str(config.get("neutral_target_prompt", "")).strip():
        raise ValueError("neutral_target_prompt must be non-empty")
    carrier = config.get("carrier", {"mode": "native"})
    if not isinstance(carrier, Mapping):
        raise ValueError("carrier must be a mapping")
    expected_carrier = EXPERIMENT_CARRIER_MODES[experiment_id]
    if carrier.get("mode", "native") != expected_carrier:
        raise ValueError(
            f"{experiment_id} requires carrier.mode={expected_carrier}"
        )

    data = config.get("data", {})
    audit = config.get("audit", {})
    runtime = config.get("runtime", {})
    output = config.get("output", {})
    for name, value in (
        ("data", data),
        ("audit", audit),
        ("runtime", runtime),
        ("output", output),
    ):
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping")

    task_count = int(data.get("task_count", 0))
    selection_count = int(data.get("selection_task_count", 0))
    preflight_count = int(data.get("preflight_task_count", 0))
    if task_count < 2 or not 0 < selection_count < task_count:
        raise ValueError("task_count must exceed a non-empty selection split")
    if not 2 <= preflight_count <= task_count:
        raise ValueError("preflight_task_count must be between 2 and task_count")
    if str(audit.get("injection_mode", "")) != "replace":
        raise ValueError("LIP-PROTO-002 fixes injection_mode=replace")
    if int(audit.get("reference_max_new_tokens", 0)) <= 0:
        raise ValueError("reference_max_new_tokens must be positive")
    if int(audit.get("minimum_reference_tokens", 0)) <= 0:
        raise ValueError("minimum_reference_tokens must be positive")
    if int(audit.get("self_check_tasks", 0)) <= 0:
        raise ValueError("self_check_tasks must be positive")
    if int(audit.get("minimum_informative_tasks_per_split", 0)) <= 0:
        raise ValueError("minimum_informative_tasks_per_split must be positive")
    if float(audit.get("minimum_task_advantage_nll", -1.0)) < 0:
        raise ValueError("minimum_task_advantage_nll must be non-negative")
    if float(audit.get("minimum_confirmation_recovery", -1.0)) < 0:
        raise ValueError("minimum_confirmation_recovery must be non-negative")
    if float(audit.get("maximum_self_nll_delta", -1.0)) < 0:
        raise ValueError("maximum_self_nll_delta must be non-negative")
    if not isinstance(runtime.get("load_4bit"), bool):
        raise ValueError("runtime.load_4bit must be a boolean")
    if not str(output.get("directory", "")).strip():
        raise ValueError("output.directory must be configured")


def run_audit(
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    *,
    preflight: bool,
    max_tasks: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    validate_config(config)
    tasks = load_tasks(Path(str(config["data"]["tasks_jsonl"])))
    tasks, manifest, manifest_path = bind_tasks_to_manifest(config, tasks)
    configured_task_count = int(config["data"]["task_count"])
    if preflight:
        tasks = tasks[: int(config["data"]["preflight_task_count"])]
        run_scope = "preflight"
    elif max_tasks is not None:
        if not 2 <= max_tasks <= len(tasks):
            raise ValueError("--max-tasks must be between 2 and the configured task count")
        tasks = tasks[:max_tasks]
        run_scope = "full" if max_tasks == configured_task_count else "diagnostic_subset"
    else:
        run_scope = "full"

    prepare_output_dir(output_dir, overwrite=overwrite)
    protocol = protocol_metadata(config.get("prompt_protocol"))
    carrier_mode = str(config.get("carrier", {}).get("mode", "native"))
    runtime = config["runtime"]
    audit = config["audit"]
    target_revision = str(manifest["target_model_revision"])
    set_seed(int(audit.get("seed", 1729)))

    print("Loading target model for oracle transport audit...")
    model, tokenizer = load_target(
        str(config["models"]["target_model"]),
        str(runtime.get("device", "auto")),
        bool(runtime["load_4bit"]),
        revision=target_revision,
    )
    device = model_input_device(model)
    layers = normalize_layer_indices(
        audit["layers"], len(model.model.layers)
    )
    neutral_formatted, neutral_inputs = encode_prompt(
        str(config["neutral_target_prompt"]), tokenizer, protocol, device
    )

    references: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    self_check_tasks = int(audit["self_check_tasks"])
    minimum_reference_tokens = int(audit["minimum_reference_tokens"])
    minimum_task_advantage = float(audit["minimum_task_advantage_nll"])

    for task_index, task in enumerate(tasks):
        print(f"Auditing task {task_index + 1}/{len(tasks)}: {task['task_id']}")
        task_formatted, task_inputs = encode_prompt(
            task["prompt"], tokenizer, protocol, device
        )
        task_prompt_length = int(task_inputs["input_ids"].shape[1])
        native_neutral_prompt_length = int(neutral_inputs["input_ids"].shape[1])
        task_neutral_inputs = build_neutral_carrier(
            neutral_inputs,
            task_prompt_length=task_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            mode=carrier_mode,
        )
        neutral_prompt_length = int(task_neutral_inputs["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **task_inputs,
                max_new_tokens=int(audit["reference_max_new_tokens"]),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        reference_ids = generated[0, task_prompt_length:].detach()
        if int(reference_ids.numel()) < minimum_reference_tokens:
            raise RuntimeError(
                f"task {task['task_id']} produced only {reference_ids.numel()} reference tokens"
            )
        reference_text = tokenizer.decode(reference_ids, skip_special_tokens=True)
        references.append(
            {
                "task_id": task["task_id"],
                "task_prompt_sha256": prompt_sha256(task["prompt"]),
                "formatted_task_prompt_sha256": prompt_sha256(task_formatted),
                "reference_token_count": int(reference_ids.numel()),
                "reference_token_ids": reference_ids.detach().cpu().tolist(),
                "reference_text": reference_text,
            }
        )

        task_teacher_inputs = append_reference(task_inputs, reference_ids)
        neutral_teacher_inputs = append_reference(task_neutral_inputs, reference_ids)
        task_outputs, oracle_vectors = forward_with_layer_capture(
            model,
            task_teacher_inputs,
            layers=layers,
            position=task_prompt_length - 1,
        )
        task_metrics = continuation_token_metrics(
            task_outputs.logits, reference_ids, task_prompt_length
        )
        del task_outputs

        neutral_outputs = forward_with_optional_replacement(
            model,
            neutral_teacher_inputs,
        )
        neutral_metrics = continuation_token_metrics(
            neutral_outputs.logits, reference_ids, neutral_prompt_length
        )
        del neutral_outputs

        for layer in layers:
            injected_outputs = forward_with_optional_replacement(
                model,
                neutral_teacher_inputs,
                layer_idx=layer,
                position=neutral_prompt_length - 1,
                vector=oracle_vectors[layer],
            )
            injected_metrics = continuation_token_metrics(
                injected_outputs.logits, reference_ids, neutral_prompt_length
            )
            del injected_outputs
            recovery, advantage, informative = recovery_fraction(
                float(task_metrics["nll"]),
                float(neutral_metrics["nll"]),
                float(injected_metrics["nll"]),
                minimum_task_advantage=minimum_task_advantage,
            )

            self_nll_delta = None
            if task_index < self_check_tasks:
                self_outputs = forward_with_optional_replacement(
                    model,
                    task_teacher_inputs,
                    layer_idx=layer,
                    position=task_prompt_length - 1,
                    vector=oracle_vectors[layer],
                )
                self_metrics = continuation_token_metrics(
                    self_outputs.logits, reference_ids, task_prompt_length
                )
                self_nll_delta = float(self_metrics["nll"]) - float(
                    task_metrics["nll"]
                )
                del self_outputs

            records.append(
                {
                    "experiment_id": config["experiment_id"],
                    "run_scope": run_scope,
                    "task_index": task_index,
                    "task_id": task["task_id"],
                    "task_prompt_sha256": prompt_sha256(task["prompt"]),
                    "neutral_prompt_sha256": prompt_sha256(
                        str(config["neutral_target_prompt"])
                    ),
                    "formatted_neutral_prompt_sha256": prompt_sha256(
                        neutral_formatted
                    ),
                    "layer_idx": layer,
                    "injection_mode": "replace",
                    "carrier_mode": carrier_mode,
                    "injection_position": "last_non_padding_generation_boundary",
                    "task_prompt_token_count": task_prompt_length,
                    "native_neutral_prompt_token_count": native_neutral_prompt_length,
                    "neutral_prompt_token_count": neutral_prompt_length,
                    "reference_token_count": int(reference_ids.numel()),
                    "task_nll": task_metrics["nll"],
                    "task_top1_accuracy": task_metrics["top1_accuracy"],
                    "neutral_nll": neutral_metrics["nll"],
                    "neutral_top1_accuracy": neutral_metrics["top1_accuracy"],
                    "injected_nll": injected_metrics["nll"],
                    "injected_top1_accuracy": injected_metrics["top1_accuracy"],
                    "task_advantage_nll": advantage,
                    "informative": informative,
                    "recovery_fraction": recovery,
                    "self_nll_delta": self_nll_delta,
                }
            )
        del oracle_vectors, task_teacher_inputs, neutral_teacher_inputs, reference_ids

    selection_task_count = int(config["data"]["selection_task_count"])
    if run_scope != "full":
        selection_task_count = max(1, len(tasks) // 2)
    effective_minimum_informative_tasks = (
        int(audit["minimum_informative_tasks_per_split"])
        if run_scope == "full"
        else 1
    )
    summary = summarize_oracle_transport(
        records,
        task_ids=[task["task_id"] for task in tasks],
        layers=layers,
        selection_task_count=selection_task_count,
        minimum_informative_tasks_per_split=effective_minimum_informative_tasks,
        minimum_confirmation_recovery=float(
            audit["minimum_confirmation_recovery"]
        ),
        maximum_self_nll_delta=float(audit["maximum_self_nll_delta"]),
        run_scope=run_scope,
    )
    summary.update(
        {
            "experiment_id": config["experiment_id"],
            "source_protocol_experiment": config["source_protocol_experiment"],
            "config": str(config_path),
            "config_sha256": sha256_path(config_path),
            "heldout_bundle_manifest": str(manifest_path),
            "heldout_bundle_manifest_sha256": sha256_path(manifest_path),
            "target_model": config["models"]["target_model"],
            "target_model_revision": target_revision,
            "prompt_protocol": protocol,
            "injection_mode": "replace",
            "carrier": {
                "mode": carrier_mode,
                "pad_token_id": tokenizer.pad_token_id,
                "padding_attention": (
                    "masked" if carrier_mode != "native" else "not_applicable"
                ),
            },
            "reference_generation": {
                "do_sample": False,
                "max_new_tokens": int(audit["reference_max_new_tokens"]),
            },
            "position_confounded_by_prompt_length": carrier_mode == "native",
        }
    )

    write_jsonl(output_dir / "references.jsonl", references)
    write_jsonl(output_dir / "oracle_transport_records.jsonl", records)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    print("Oracle transport audit completed")
    print(f"run_scope: {run_scope}")
    print(f"tasks: {len(tasks)}")
    print(f"layers: {layers}")
    print(f"selected_layer: {summary['selected_layer']}")
    print(f"gate_passed: {summary['gate']['passed']}")
    print(f"summary: {output_dir / 'summary.json'}")
    return summary


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.preflight:
        output_dir = Path(str(config.get("output", {}).get("directory"))).parent / "preflight"
    else:
        output_dir = Path(str(config.get("output", {}).get("directory")))
    run_audit(
        config,
        args.config,
        output_dir,
        preflight=args.preflight,
        max_tasks=args.max_tasks,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
