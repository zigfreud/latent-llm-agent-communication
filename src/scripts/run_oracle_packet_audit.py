"""Measure how target-oracle transport changes with latent packet capacity."""

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
    continuation_token_profile,
    normalize_layer_indices,
    recovery_fraction,
    summarize_packet_capacity,
    summarize_packet_position_recovery,
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
    forward_with_optional_replacement,
    forward_with_packet_capture,
    forward_with_packet_replacement,
)


DEFAULT_CONFIG = Path("config/LIP-PROTO-004_oracle_packet_capacity.yaml")


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
        "layer_selection_experiment",
        "capacity_source_experiment",
        "functional_source_experiment",
        "models",
        "prompt_protocol",
        "runtime",
        "data",
        "neutral_target_prompt",
        "carrier",
        "audit",
        "position_analysis",
        "output",
    }
    unknown = sorted(set(config).difference(expected_top_level))
    if unknown:
        raise ValueError(f"unknown config field(s): {', '.join(unknown)}")
    experiment_id = str(config.get("experiment_id", ""))
    if experiment_id not in {"LIP-PROTO-004", "LIP-PROTO-006"}:
        raise ValueError("experiment_id must be LIP-PROTO-004 or LIP-PROTO-006")
    if config.get("source_protocol_experiment") != "LIP-PROTO-001":
        raise ValueError("source_protocol_experiment must bind LIP-PROTO-001")
    if config.get("layer_selection_experiment") != "LIP-PROTO-003":
        raise ValueError("layer_selection_experiment must bind LIP-PROTO-003")
    if experiment_id == "LIP-PROTO-006":
        if config.get("capacity_source_experiment") != "LIP-PROTO-004":
            raise ValueError("capacity_source_experiment must bind LIP-PROTO-004")
        if config.get("functional_source_experiment") != "LIP-PROTO-005":
            raise ValueError("functional_source_experiment must bind LIP-PROTO-005")

    protocol = protocol_metadata(config.get("prompt_protocol"))
    if protocol["mode"] != "chat_template" or not protocol["add_generation_prompt"]:
        raise ValueError(
            "oracle packet audit requires the target chat template and generation marker"
        )
    if not str(config.get("neutral_target_prompt", "")).strip():
        raise ValueError("neutral_target_prompt must be non-empty")
    carrier = config.get("carrier", {})
    if not isinstance(carrier, Mapping) or carrier.get("mode") != (
        "left_pad_masked_to_task_length"
    ):
        raise ValueError(
            "oracle packet audits require carrier.mode=left_pad_masked_to_task_length"
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
        raise ValueError("oracle packet audits fix injection_mode=replace")
    if int(audit.get("layer_idx", 0)) >= 0:
        raise ValueError("layer_idx must be a negative transformer-layer index")
    packet_sizes = [int(size) for size in audit.get("packet_sizes", [])]
    if not packet_sizes or packet_sizes != sorted(set(packet_sizes)):
        raise ValueError("packet_sizes must be strictly increasing and unique")
    if any(size <= 0 for size in packet_sizes):
        raise ValueError("packet_sizes must be positive")
    if packet_sizes[0] != 1:
        raise ValueError("packet_sizes must start at 1 to anchor LIP-PROTO-003")
    for field in (
        "reference_max_new_tokens",
        "minimum_reference_tokens",
        "self_check_tasks",
        "minimum_informative_tasks_per_split",
    ):
        if int(audit.get(field, 0)) <= 0:
            raise ValueError(f"{field} must be positive")
    for field in (
        "minimum_task_advantage_nll",
        "minimum_recovery",
        "maximum_self_nll_delta",
    ):
        if float(audit.get(field, -1.0)) < 0:
            raise ValueError(f"{field} must be non-negative")
    if not isinstance(runtime.get("load_4bit"), bool):
        raise ValueError("runtime.load_4bit must be a boolean")
    if not str(output.get("directory", "")).strip():
        raise ValueError("output.directory must be configured")

    position_analysis = config.get("position_analysis")
    if experiment_id == "LIP-PROTO-004":
        if config.get("capacity_source_experiment") is not None or config.get(
            "functional_source_experiment"
        ) is not None:
            raise ValueError("LIP-PROTO-004 must not declare later-protocol lineage")
        if position_analysis is not None:
            raise ValueError("LIP-PROTO-004 must not enable position_analysis")
        return
    if not isinstance(position_analysis, Mapping):
        raise ValueError("LIP-PROTO-006 requires position_analysis")
    allowed_position_fields = {
        "prefix_token_counts",
        "gate_prefix_token_count",
        "minimum_task_support_per_split",
        "estimator",
    }
    unknown_position_fields = sorted(
        set(position_analysis).difference(allowed_position_fields)
    )
    if unknown_position_fields:
        raise ValueError(
            "unknown position_analysis field(s): "
            + ", ".join(unknown_position_fields)
        )
    prefix_counts = [
        int(count) for count in position_analysis.get("prefix_token_counts", [])
    ]
    if not prefix_counts or prefix_counts != sorted(set(prefix_counts)) or any(
        count <= 0 for count in prefix_counts
    ):
        raise ValueError(
            "position_analysis.prefix_token_counts must be positive, increasing, "
            "and unique"
        )
    gate_prefix = int(position_analysis.get("gate_prefix_token_count", 0))
    if gate_prefix not in prefix_counts:
        raise ValueError(
            "position_analysis.gate_prefix_token_count must be a configured prefix"
        )
    if int(audit["minimum_reference_tokens"]) < max(prefix_counts):
        raise ValueError(
            "minimum_reference_tokens must cover the largest position-analysis prefix"
        )
    minimum_position_support = int(
        position_analysis.get("minimum_task_support_per_split", 0)
    )
    if minimum_position_support <= 0:
        raise ValueError("minimum_task_support_per_split must be positive")
    if minimum_position_support > min(selection_count, task_count - selection_count):
        raise ValueError("minimum_task_support_per_split exceeds a frozen split")
    if position_analysis.get("estimator") != "pooled_nll_ratio":
        raise ValueError("position_analysis.estimator must be pooled_nll_ratio")


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
    carrier_mode = str(config["carrier"]["mode"])
    runtime = config["runtime"]
    audit = config["audit"]
    position_analysis = config.get("position_analysis")
    score_continuation = (
        continuation_token_profile
        if isinstance(position_analysis, Mapping)
        else continuation_token_metrics
    )
    target_revision = str(manifest["target_model_revision"])
    set_seed(int(audit.get("seed", 1729)))

    print("Loading target model for oracle packet audit...")
    model, tokenizer = load_target(
        str(config["models"]["target_model"]),
        str(runtime.get("device", "auto")),
        bool(runtime["load_4bit"]),
        revision=target_revision,
    )
    device = model_input_device(model)
    layer_idx = normalize_layer_indices(
        [int(audit["layer_idx"])], len(model.model.layers)
    )[0]
    packet_sizes = [int(size) for size in audit["packet_sizes"]]
    maximum_packet_size = max(packet_sizes)
    neutral_formatted, neutral_inputs = encode_prompt(
        str(config["neutral_target_prompt"]), tokenizer, protocol, device
    )
    native_neutral_prompt_length = int(neutral_inputs["input_ids"].shape[1])
    if maximum_packet_size > native_neutral_prompt_length:
        raise ValueError(
            "maximum packet size exceeds the visible native neutral carrier; "
            "injected positions would remain attention-masked"
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
        if maximum_packet_size > task_prompt_length:
            raise ValueError(
                f"task {task['task_id']} is shorter than the maximum packet size"
            )
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
                f"task {task['task_id']} produced only {reference_ids.numel()} "
                "reference tokens"
            )
        references.append(
            {
                "task_id": task["task_id"],
                "task_prompt_sha256": prompt_sha256(task["prompt"]),
                "formatted_task_prompt_sha256": prompt_sha256(task_formatted),
                "reference_token_count": int(reference_ids.numel()),
                "reference_token_ids": reference_ids.detach().cpu().tolist(),
                "reference_text": tokenizer.decode(
                    reference_ids, skip_special_tokens=True
                ),
            }
        )

        task_teacher_inputs = append_reference(task_inputs, reference_ids)
        neutral_teacher_inputs = append_reference(task_neutral_inputs, reference_ids)
        maximum_positions = torch.arange(
            task_prompt_length - maximum_packet_size,
            task_prompt_length,
            device=device,
        )
        visible_packet_mask = task_neutral_inputs["attention_mask"][
            0, maximum_positions
        ]
        if not bool(torch.all(visible_packet_mask == 1).item()):
            raise RuntimeError(
                "packet overlaps attention-masked carrier positions; capacity audit "
                "would be invalid"
            )
        task_outputs, maximum_packet = forward_with_packet_capture(
            model,
            task_teacher_inputs,
            layer_idx=layer_idx,
            positions=maximum_positions,
        )
        task_metrics = score_continuation(
            task_outputs.logits, reference_ids, task_prompt_length
        )
        del task_outputs

        neutral_outputs = forward_with_optional_replacement(
            model, neutral_teacher_inputs
        )
        neutral_metrics = score_continuation(
            neutral_outputs.logits, reference_ids, neutral_prompt_length
        )
        del neutral_outputs

        for packet_size in packet_sizes:
            positions = maximum_positions[-packet_size:]
            vectors = maximum_packet[-packet_size:, :]
            injected_outputs = forward_with_packet_replacement(
                model,
                neutral_teacher_inputs,
                layer_idx=layer_idx,
                positions=positions,
                vectors=vectors,
            )
            injected_metrics = score_continuation(
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
                self_outputs = forward_with_packet_replacement(
                    model,
                    task_teacher_inputs,
                    layer_idx=layer_idx,
                    positions=positions,
                    vectors=vectors,
                )
                self_metrics = continuation_token_metrics(
                    self_outputs.logits, reference_ids, task_prompt_length
                )
                self_nll_delta = float(self_metrics["nll"]) - float(
                    task_metrics["nll"]
                )
                del self_outputs

            record = {
                "experiment_id": config["experiment_id"],
                "run_scope": run_scope,
                "task_index": task_index,
                "task_id": task["task_id"],
                "task_prompt_sha256": prompt_sha256(task["prompt"]),
                "neutral_prompt_sha256": prompt_sha256(
                    str(config["neutral_target_prompt"])
                ),
                "formatted_neutral_prompt_sha256": prompt_sha256(neutral_formatted),
                "layer_idx": layer_idx,
                "packet_size": packet_size,
                "packet_start_position": int(positions[0].item()),
                "packet_stop_position_exclusive": int(positions[-1].item()) + 1,
                "packet_positions_attention_visible": True,
                "injection_mode": "replace",
                "carrier_mode": carrier_mode,
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
            if isinstance(position_analysis, Mapping):
                record.update(
                    {
                        "task_token_nlls": task_metrics["token_nlls"],
                        "task_token_top1_correct": task_metrics[
                            "token_top1_correct"
                        ],
                        "neutral_token_nlls": neutral_metrics["token_nlls"],
                        "neutral_token_top1_correct": neutral_metrics[
                            "token_top1_correct"
                        ],
                        "injected_token_nlls": injected_metrics["token_nlls"],
                        "injected_token_top1_correct": injected_metrics[
                            "token_top1_correct"
                        ],
                    }
                )
            records.append(record)
        del maximum_packet, task_teacher_inputs, neutral_teacher_inputs, reference_ids

    selection_task_count = int(config["data"]["selection_task_count"])
    if run_scope != "full":
        selection_task_count = max(1, len(tasks) // 2)
    effective_minimum_informative_tasks = (
        int(audit["minimum_informative_tasks_per_split"])
        if run_scope == "full"
        else 1
    )
    task_ids = [task["task_id"] for task in tasks]
    if isinstance(position_analysis, Mapping):
        summary = summarize_packet_position_recovery(
            records,
            task_ids=task_ids,
            packet_sizes=packet_sizes,
            selection_task_count=selection_task_count,
            prefix_token_counts=position_analysis["prefix_token_counts"],
            gate_prefix_token_count=int(
                position_analysis["gate_prefix_token_count"]
            ),
            minimum_task_support_per_split=(
                int(position_analysis["minimum_task_support_per_split"])
                if run_scope == "full"
                else 1
            ),
            minimum_task_advantage=float(audit["minimum_task_advantage_nll"]),
            minimum_recovery=float(audit["minimum_recovery"]),
            maximum_self_nll_delta=float(audit["maximum_self_nll_delta"]),
            run_scope=run_scope,
        )
    else:
        summary = summarize_packet_capacity(
            records,
            task_ids=task_ids,
            packet_sizes=packet_sizes,
            selection_task_count=selection_task_count,
            minimum_informative_tasks_per_split=effective_minimum_informative_tasks,
            minimum_recovery=float(audit["minimum_recovery"]),
            maximum_self_nll_delta=float(audit["maximum_self_nll_delta"]),
            run_scope=run_scope,
        )
    summary.update(
        {
            "experiment_id": config["experiment_id"],
            "source_protocol_experiment": config["source_protocol_experiment"],
            "layer_selection_experiment": config["layer_selection_experiment"],
            "config": str(config_path),
            "config_sha256": sha256_path(config_path),
            "heldout_bundle_manifest": str(manifest_path),
            "heldout_bundle_manifest_sha256": sha256_path(manifest_path),
            "target_model": config["models"]["target_model"],
            "target_model_revision": target_revision,
            "prompt_protocol": protocol,
            "layer_idx": layer_idx,
            "layer_selection_source": "selection split",
            "injection_mode": "replace",
            "carrier": {
                "mode": carrier_mode,
                "pad_token_id": tokenizer.pad_token_id,
                "padding_attention": "masked",
                "packet_positions_attention_visible": True,
            },
            "reference_generation": {
                "do_sample": False,
                "max_new_tokens": int(audit["reference_max_new_tokens"]),
            },
            "capacity_axis_only": position_analysis is None,
        }
    )
    if isinstance(position_analysis, Mapping):
        summary.update(
            {
                "capacity_source_experiment": config["capacity_source_experiment"],
                "functional_source_experiment": config[
                    "functional_source_experiment"
                ],
                "analysis_axis": "target_continuation_token_position",
                "position_analysis": dict(position_analysis),
            }
        )

    write_jsonl(output_dir / "references.jsonl", references)
    write_jsonl(output_dir / "oracle_packet_records.jsonl", records)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    print("Oracle packet audit completed")
    print(f"run_scope: {run_scope}")
    print(f"tasks: {len(tasks)}")
    print(f"layer_idx: {layer_idx}")
    print(f"packet_sizes: {packet_sizes}")
    print(f"selected_packet_size: {summary['selected_packet_size']}")
    print(f"gate_passed: {summary['gate']['passed']}")
    print(f"summary: {output_dir / 'summary.json'}")
    return summary


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.preflight:
        output_dir = Path(str(config.get("output", {}).get("directory"))).parent / (
            "preflight"
        )
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
