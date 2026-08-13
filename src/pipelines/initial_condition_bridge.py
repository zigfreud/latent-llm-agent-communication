"""Train and evaluate the H0-010 receiver initial-condition bridge."""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

import torch

from src.core.packet_bridge import reconstruct_target_packet
from src.core.packet_bundle import (
    PacketRecordDataset,
    compute_target_packet_statistics,
    load_packet_records,
    sha256_file,
    validate_packet_bundle,
)
from src.core.packet_loss import build_terminal_component_masks
from src.core.receiver_initial_condition import evolve_receiver_from_entry_seed
from src.evaluation.packet_bridge import (
    checkpoint_selection_key,
    summarize_packet_latent_metrics,
    summarize_replica_development_gate,
)
from src.pipelines.infer import load_target, model_input_device
from src.pipelines.oracle_experiment import load_json_object, load_yaml
from src.pipelines.packet_bridge import (
    _grad_scaler,
    _json_ready_metrics,
    _make_loader,
    build_packet_bridge,
    build_packet_loss,
    set_packet_training_seed,
)
from src.pipelines.packet_confirmation import _neutral_inputs, _suffix_positions
from src.pipelines.packet_trajectory import _atomic_json
from src.pipelines.receiver_aware_replay import _lf_sha256_file


INITIAL_CONDITION_PROTOCOL_VERSION = "lip-unrolled-initial-condition-bridge-v3"


def _git_head() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("run commit is not a full lowercase Git SHA")
    return value


def _validate_contract(
    experiment: Mapping,
    parent: Mapping,
    *,
    experiment_path: Path,
    parent_path: Path,
    predecessor_registry_path: Path,
) -> None:
    if experiment.get("experiment_id") != "LIP-H0-010":
        raise ValueError("unexpected initial-condition experiment_id")
    if experiment.get("protocol_version") != INITIAL_CONDITION_PROTOCOL_VERSION:
        raise ValueError("unexpected initial-condition protocol_version")
    if experiment.get("claim_status") != "development_only_training_experiment":
        raise ValueError("H0-010 must remain development-only")
    if parent.get("experiment_id") != "LIP-PROTO-014":
        raise ValueError("H0-010 parent must be LIP-PROTO-014")
    if experiment["parent"]["config_sha256"] != _lf_sha256_file(parent_path):
        raise ValueError("parent config hash differs from the frozen contract")
    predecessor = load_json_object(predecessor_registry_path)
    if predecessor.get("experiment_id") != "LIP-H0-009":
        raise ValueError("predecessor registry is not LIP-H0-009")
    expected_artifact = experiment["predecessor"]["trajectory_gate_sha256"]
    observed_artifact = predecessor.get("artifacts", {}).get(
        "trajectory_gate", {}
    ).get("sha256")
    if observed_artifact != expected_artifact:
        raise ValueError("H0-009 artifact hash differs from the frozen contract")

    expected_counts = experiment["data"]["expected_counts"]
    parent_counts = parent["data"]["selection"]
    count_bindings = {
        "train": "train_count",
        "development_selection": "development_selection_count",
        "development_gate": "development_gate_count",
    }
    for split, parent_key in count_bindings.items():
        if int(expected_counts[split]) != int(parent_counts[parent_key]):
            raise ValueError(f"{split} count drifted from PROTO-014")
    if list(experiment["data"]["allowed_splits"]) != [
        "train",
        "development_selection",
        "development_gate",
    ]:
        raise ValueError("H0-010 split allowlist changed")
    if list(experiment["data"]["prohibited_splits"]) != ["confirmation"]:
        raise ValueError("H0-010 must prohibit confirmation")

    receiver = experiment["receiver"]
    parent_target = parent["models"]["target"]
    if receiver["model_id"] != parent_target["model_id"] or receiver[
        "revision"
    ] != parent_target["revision"]:
        raise ValueError("receiver endpoint drifted from PROTO-014")
    layers = [int(value) for value in receiver["evolved_layer_indices"]]
    if layers != [int(value) for value in parent["packets"]["target"]["layer_indices"]]:
        raise ValueError("receiver layer prefix drifted from PROTO-014")
    offsets = [int(value) for value in receiver["packet_offsets"]]
    if offsets != [int(value) for value in parent["packets"]["target"]["offsets"]]:
        raise ValueError("receiver offsets drifted from PROTO-014")
    if int(receiver["entry_layer"]) != 0 or int(receiver["stop_before_layer"]) != 8:
        raise ValueError("H0-010 must seed block 0 and stop before block 8")
    if receiver["freeze_all_parameters"] is not True:
        raise ValueError("receiver parameters must remain frozen")
    if receiver["neutral_prompt"] != parent["packets"]["target"]["neutral_prompt"]:
        raise ValueError("receiver neutral carrier drifted from PROTO-014")

    systems = experiment["variants"]["systems"]
    if set(systems) != {"static_entry_snapshot", "unrolled_initial_condition"}:
        raise ValueError("H0-010 objective systems changed")
    if experiment["variants"]["primary"] != "unrolled_initial_condition":
        raise ValueError("unrolled initial condition must remain primary")
    if float(systems["static_entry_snapshot"]["lambda_induced_trajectory"]) != 0.0:
        raise ValueError("static control cannot use an induced trajectory loss")
    if float(systems["unrolled_initial_condition"]["lambda_induced_trajectory"]) <= 0.0:
        raise ValueError("primary system must optimize the induced trajectory")
    losses = experiment["loss"]
    if set(losses) != {"entry_snapshot", "induced_trajectory"}:
        raise ValueError("H0-010 must define separate entry and trajectory losses")
    if float(losses["entry_snapshot"]["lambda_norm"]) != 0.0:
        raise ValueError("layer-0 entry loss must disable singular relative norms")
    if float(losses["induced_trajectory"]["lambda_norm"]) <= 0.0:
        raise ValueError("induced trajectory loss must retain its norm regularizer")
    pilot = experiment["training"]["pilot"]
    if pilot["variant"] != "unrolled_initial_condition" or int(pilot["seed"]) != 4001:
        raise ValueError("H0-010 pilot cell changed")
    if int(pilot["max_updates"]) != int(experiment["pilot_gate"]["required_updates"]):
        raise ValueError("pilot update count differs from its feasibility gate")
    if experiment["confirmation"]["status"] != "prohibited_in_H0-010":
        raise ValueError("confirmation must remain prohibited")
    if experiment["compute"]["preferred_accelerator"] != "L4":
        raise ValueError("H0-010 is frozen for an L4")
    if experiment["compute"]["allow_silent_fallback"] is not False:
        raise ValueError("silent accelerator fallback is prohibited")
    if not experiment_path.is_file():
        raise ValueError("experiment contract path does not exist")


def _bridge_model_config(experiment: Mapping) -> dict:
    bridge = experiment["bridge"]
    if bridge["model_kind"] != "query_conditioned":
        raise ValueError("H0-010 supports only the frozen query-conditioned bridge")
    return {
        "kind": "query_conditioned",
        "protocol_slots": int(bridge["protocol_slots"]),
        "bridge_width": int(bridge["bridge_width"]),
        "attention_heads": int(bridge["attention_heads"]),
        "feedforward_width": int(bridge["feedforward_width"]),
        "encoder_blocks": int(bridge["encoder_blocks"]),
        "decoder_blocks": int(bridge["decoder_blocks"]),
        "dropout": float(bridge["dropout"]),
    }


def _repeat_receiver_inputs(
    inputs: Mapping[str, torch.Tensor], batch_size: int
) -> dict[str, torch.Tensor]:
    repeated = {}
    for key, value in inputs.items():
        if value.ndim == 0 or value.shape[0] != 1:
            raise ValueError("base receiver inputs must have batch size one")
        repeated[key] = value.expand(batch_size, *value.shape[1:])
    return repeated


def _entry_raw_packet(
    normalized_entry: torch.Tensor,
    scaffold: torch.Tensor,
    site_scale: torch.Tensor,
) -> torch.Tensor:
    if normalized_entry.ndim != 4 or normalized_entry.shape[1] != 1:
        raise ValueError("initial-condition bridge must emit exactly one layer")
    return reconstruct_target_packet(
        normalized_entry,
        scaffold[:1],
        site_scale[:1],
    )[:, 0]


def _normalize_trajectory(
    raw: torch.Tensor,
    scaffold: torch.Tensor,
    site_scale: torch.Tensor,
) -> torch.Tensor:
    if raw.ndim != 4 or tuple(raw.shape[1:]) != tuple(scaffold.shape):
        raise ValueError("induced receiver trajectory shape differs from statistics")
    return (raw.float() - scaffold[None]) / site_scale[None, :, :, None]


def _gpu_telemetry() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("LIP-H0-010 requires the registered L4 accelerator")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    name = torch.cuda.get_device_name(device)
    if "L4" not in name.upper():
        raise RuntimeError(f"registered accelerator is L4, observed {name!r}")
    return {
        "gpu_name": name,
        "total_vram_bytes": int(properties.total_memory),
        "cuda_runtime": str(torch.version.cuda),
    }


def _induced_trajectory(
    receiver,
    receiver_inputs: Mapping[str, torch.Tensor],
    *,
    positions: torch.Tensor,
    normalized_entry: torch.Tensor,
    scaffold: torch.Tensor,
    site_scale: torch.Tensor,
    layers: list[int],
) -> torch.Tensor:
    entry = _entry_raw_packet(normalized_entry, scaffold, site_scale)
    raw = evolve_receiver_from_entry_seed(
        receiver,
        _repeat_receiver_inputs(receiver_inputs, normalized_entry.shape[0]),
        positions=positions,
        entry_packet=entry,
        layer_indices=layers,
    )
    return _normalize_trajectory(raw, scaffold, site_scale)


def evaluate_initial_condition_bridge(
    bridge,
    receiver,
    dataset,
    *,
    receiver_inputs: Mapping[str, torch.Tensor],
    positions: torch.Tensor,
    scaffold: torch.Tensor,
    site_scale: torch.Tensor,
    layers: list[int],
    batch_size: int,
    device: torch.device,
    boundary_positions: int,
) -> dict:
    loader = _make_loader(dataset, batch_size=batch_size, shuffle=False, seed=0)
    entries = []
    induced = []
    targets = []
    counts = []
    task_ids = []
    was_training = bridge.training
    bridge.eval()
    receiver.eval()
    with torch.no_grad():
        for batch in loader:
            source = batch["source_packet"].to(device)
            target = batch["target_residual"].to(device)
            predicted_entry = bridge(source)
            entries.append(predicted_entry.float().cpu())
            induced.append(
                _induced_trajectory(
                    receiver,
                    receiver_inputs,
                    positions=positions,
                    normalized_entry=predicted_entry,
                    scaffold=scaffold,
                    site_scale=site_scale,
                    layers=layers,
                ).cpu()
            )
            targets.append(target.float().cpu())
            counts.append(batch["name_token_count"])
            task_ids.extend(batch["task_ids"])
    if was_training:
        bridge.train()
    entry_prediction = torch.cat(entries)
    trajectory_prediction = torch.cat(induced)
    target = torch.cat(targets)
    name_counts = torch.cat(counts)
    masks = build_terminal_component_masks(
        name_counts,
        target_positions=target.shape[2],
        boundary_positions=boundary_positions,
    )
    return {
        "entry_snapshot": summarize_packet_latent_metrics(
            entry_prediction,
            target[:, :1],
            masks,
            task_ids=task_ids,
        ),
        "induced_trajectory": summarize_packet_latent_metrics(
            trajectory_prediction,
            target,
            masks,
            task_ids=task_ids,
        ),
    }


def _resolved_stage(experiment: Mapping, *, pilot: bool) -> dict:
    training = experiment["training"]
    stage = dict(training["pilot"] if pilot else training["full_matrix"])
    for key in (
        "learning_rate",
        "weight_decay",
        "gradient_clip",
        "fp16_autocast",
        "num_workers",
    ):
        stage[key] = training[key]
    return stage


def run_initial_condition_training(
    *,
    experiment_path: Path,
    parent_path: Path,
    predecessor_registry_path: Path,
    bundle_dir: Path,
    output_dir: Path,
    variant_name: str,
    seed: int,
    pilot: bool,
    target_device: str,
    colab_compute_units_before: float | None,
) -> dict:
    experiment = load_yaml(experiment_path)
    parent = load_yaml(parent_path)
    _validate_contract(
        experiment,
        parent,
        experiment_path=experiment_path,
        parent_path=parent_path,
        predecessor_registry_path=predecessor_registry_path,
    )
    systems = experiment["variants"]["systems"]
    if variant_name not in systems:
        raise ValueError("variant is outside the frozen H0-010 systems")
    configured_seeds = [int(value) for value in experiment["training"]["seeds"]]
    if int(seed) not in configured_seeds:
        raise ValueError("seed is outside the frozen H0-010 seed set")
    stage = _resolved_stage(experiment, pilot=pilot)
    if pilot and (variant_name != stage["variant"] or int(seed) != int(stage["seed"])):
        raise ValueError("pilot must use its single frozen variant and seed")
    if not pilot:
        stage["output_dir"] = str(output_dir)

    telemetry = _gpu_telemetry()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    set_packet_training_seed(int(seed))
    validation = validate_packet_bundle(bundle_dir, require_real=True)
    expected_counts = experiment["data"]["expected_counts"]
    for split, expected in expected_counts.items():
        if int(validation["split_counts"][split]) != int(expected):
            raise ValueError(f"bundle {split} count differs from H0-010")
    if int(validation["split_counts"]["confirmation"]) != 0:
        raise ValueError("H0-010 bundle must contain zero confirmation records")
    if tuple(validation["target_shape"]) != (
        len(experiment["receiver"]["evolved_layer_indices"]),
        len(experiment["receiver"]["packet_offsets"]),
        int(experiment["receiver"]["target_width"]),
    ):
        raise ValueError("bundle target shape differs from the frozen receiver contract")

    records = load_packet_records(bundle_dir)
    by_split = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "development_selection", "development_gate")
    }
    scaffold, site_scale = compute_target_packet_statistics(by_split["train"])
    datasets = {
        split: PacketRecordDataset(
            split_records,
            scaffold=scaffold,
            site_scale=site_scale,
        )
        for split, split_records in by_split.items()
    }
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"initial-condition output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = output_dir / "target_statistics.pt"
    torch.save({"scaffold": scaffold, "site_scale": site_scale}, statistics_path)

    receiver_config = experiment["receiver"]
    receiver, tokenizer = load_target(
        str(receiver_config["model_id"]),
        target_device,
        bool(receiver_config["load_4bit"]),
        revision=str(receiver_config["revision"]),
    )
    receiver.eval()
    for parameter in receiver.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in receiver.parameters()):
        raise RuntimeError("receiver freezing failed")
    device = model_input_device(receiver)
    _, receiver_inputs = _neutral_inputs(parent, tokenizer, device)
    positions = _suffix_positions(receiver_inputs, receiver_config["packet_offsets"])
    layers = [int(value) for value in receiver_config["evolved_layer_indices"]]
    scaffold = scaffold.to(device)
    site_scale = site_scale.to(device)

    source_shape = tuple(validation["source_shape"])
    target_shape = tuple(validation["target_shape"])
    bridge_target_shape = (1, target_shape[1], target_shape[2])
    model_config = _bridge_model_config(experiment)
    bridge = build_packet_bridge(model_config, source_shape, bridge_target_shape).to(device)
    entry_criterion = build_packet_loss(experiment["loss"]["entry_snapshot"])
    trajectory_criterion = build_packet_loss(
        experiment["loss"]["induced_trajectory"]
    )
    optimizer = torch.optim.AdamW(
        bridge.parameters(),
        lr=float(stage["learning_rate"]),
        weight_decay=float(stage["weight_decay"]),
    )
    batch_size = int(stage["batch_size"])
    accumulation_steps = int(stage["gradient_accumulation_steps"])
    if batch_size < 2 or accumulation_steps <= 0:
        raise ValueError("H0-010 requires batch size at least two and positive accumulation")
    max_updates = int(stage["max_updates"])
    validation_interval = int(stage["validation_interval"])
    boundary_positions = int(experiment["data"]["boundary_positions"])
    loader = _make_loader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        seed=int(seed),
        num_workers=int(stage["num_workers"]),
    )
    use_amp = bool(stage["fp16_autocast"])
    scaler = _grad_scaler(use_amp)
    variant = systems[variant_name]
    lambda_entry = float(variant["lambda_entry_snapshot"])
    lambda_trajectory = float(variant["lambda_induced_trajectory"])
    gradient_clip = float(stage["gradient_clip"])

    history = []
    amp_overflow_events = []
    best_key = None
    best_step = None
    best_path = output_dir / "best_checkpoint.pt"
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro_step = 0
    epoch = 0
    while step < max_updates:
        epoch += 1
        for batch in loader:
            if step >= max_updates:
                break
            bridge.train()
            source = batch["source_packet"].to(device)
            target = batch["target_residual"].to(device)
            counts = batch["name_token_count"].to(device)
            masks = build_terminal_component_masks(
                counts,
                target_positions=target.shape[2],
                boundary_positions=boundary_positions,
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                predicted_entry = bridge(source)
                entry_metrics = entry_criterion(predicted_entry, target[:, :1], masks)
                trajectory_metrics = None
                total_loss = lambda_entry * entry_metrics["total_loss"]
                if lambda_trajectory > 0.0:
                    induced = _induced_trajectory(
                        receiver,
                        receiver_inputs,
                        positions=positions,
                        normalized_entry=predicted_entry,
                        scaffold=scaffold,
                        site_scale=site_scale,
                        layers=layers,
                    )
                    trajectory_metrics = trajectory_criterion(
                        induced[:, 1:], target[:, 1:], masks
                    )
                    total_loss = (
                        total_loss
                        + lambda_trajectory * trajectory_metrics["total_loss"]
                    )
                scaled_loss = total_loss / accumulation_steps
            if not bool(torch.isfinite(total_loss.detach()).all().item()):
                raise FloatingPointError("initial-condition training produced non-finite loss")
            scaler.scale(scaled_loss).backward()
            micro_step += 1
            if micro_step % accumulation_steps:
                continue

            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                bridge.parameters(), gradient_clip
            )
            gradient_norm_value = float(gradient_norm)
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            optimizer.zero_grad(set_to_none=True)
            if use_amp and scale_after < scale_before:
                amp_overflow_events.append(
                    {
                        "step_candidate": step + 1,
                        "scale_before": scale_before,
                        "scale_after": scale_after,
                    }
                )
                continue
            if not math.isfinite(gradient_norm_value) or gradient_norm_value <= 0.0:
                raise FloatingPointError(
                    "initial-condition training produced zero or non-finite bridge gradient"
                )
            step += 1
            row = {
                "step": step,
                "epoch": epoch,
                "gradient_norm": gradient_norm_value,
                "total_loss": float(total_loss.detach()),
                "entry": _json_ready_metrics(entry_metrics),
                "trajectory": (
                    _json_ready_metrics(trajectory_metrics)
                    if trajectory_metrics is not None
                    else None
                ),
            }
            history.append(row)

            if step % validation_interval == 0 or step == max_updates:
                selection = evaluate_initial_condition_bridge(
                    bridge,
                    receiver,
                    datasets["development_selection"],
                    receiver_inputs=receiver_inputs,
                    positions=positions,
                    scaffold=scaffold,
                    site_scale=site_scale,
                    layers=layers,
                    batch_size=batch_size,
                    device=device,
                    boundary_positions=boundary_positions,
                )
                row["development_selection"] = selection
                key = checkpoint_selection_key(selection["induced_trajectory"], step=step)
                if best_key is None or key > best_key:
                    best_key = key
                    best_step = step
                    torch.save(
                        {
                            "model_state": bridge.state_dict(),
                            "step": step,
                            "selection_key": list(key),
                            "selection_metrics": selection,
                            "source_shape": list(source_shape),
                            "bridge_target_shape": list(bridge_target_shape),
                            "receiver_target_shape": list(target_shape),
                            "model_config": model_config,
                            "variant": variant_name,
                        },
                        best_path,
                    )
                _atomic_json(output_dir / "train_history.json", history)
                print(
                    json.dumps(
                        {
                            "event": "initial_condition_validation",
                            "variant": variant_name,
                            "seed": int(seed),
                            "step": step,
                            "training_loss": row["total_loss"],
                            "selection_trajectory_rmse": selection[
                                "induced_trajectory"
                            ]["normalized_residual_rmse"],
                            "best_step": best_step,
                        }
                    ),
                    flush=True,
                )
            del source, target, predicted_entry, entry_metrics, total_loss, scaled_loss
            if trajectory_metrics is not None:
                del trajectory_metrics, induced
            gc.collect()

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    bridge.load_state_dict(checkpoint["model_state"])
    gate_metrics = evaluate_initial_condition_bridge(
        bridge,
        receiver,
        datasets["development_gate"],
        receiver_inputs=receiver_inputs,
        positions=positions,
        scaffold=scaffold,
        site_scale=site_scale,
        layers=layers,
        batch_size=batch_size,
        device=device,
        boundary_positions=boundary_positions,
    )
    gate_report = summarize_replica_development_gate(
        gate_metrics["induced_trajectory"],
        alpha=float(experiment["development_gate"]["alpha"]),
        statistics_seed=int(experiment["development_gate"]["statistics_seed"]),
    )
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    total_seconds = float(time.perf_counter() - started)
    pilot_gate = None
    if pilot:
        frozen = experiment["pilot_gate"]
        pilot_gate = {
            "required_updates": int(frozen["required_updates"]),
            "updates_completed": step,
            "finite_loss": all(math.isfinite(row["total_loss"]) for row in history),
            "nonzero_bridge_gradient_each_update": all(
                row["gradient_norm"] > 0.0 for row in history
            ),
            "amp_overflow_events": len(amp_overflow_events),
            "maximum_amp_overflow_events": int(
                frozen["maximum_amp_overflow_events"]
            ),
            "peak_allocated_vram_bytes": peak_allocated,
            "maximum_peak_allocated_vram_bytes": int(
                frozen["maximum_peak_allocated_vram_bytes"]
            ),
        }
        pilot_gate["passed"] = bool(
            step == pilot_gate["required_updates"]
            and pilot_gate["finite_loss"]
            and pilot_gate["nonzero_bridge_gradient_each_update"]
            and pilot_gate["amp_overflow_events"]
            <= pilot_gate["maximum_amp_overflow_events"]
            and pilot_gate["peak_allocated_vram_bytes"]
            <= pilot_gate["maximum_peak_allocated_vram_bytes"]
        )

    result = {
        "experiment_id": "LIP-H0-010",
        "protocol_version": INITIAL_CONDITION_PROTOCOL_VERSION,
        "claim_status": experiment["claim_status"],
        "stage": "pilot" if pilot else "full_training_cell",
        "variant": variant_name,
        "seed": int(seed),
        "run_commit": _git_head(),
        "provenance": {
            "experiment_config_sha256": sha256_file(experiment_path),
            "parent_config_sha256": sha256_file(parent_path),
            "predecessor_registry_sha256": sha256_file(
                predecessor_registry_path
            ),
            "bundle_manifest_sha256": sha256_file(bundle_dir / "manifest.json"),
            "target_statistics_sha256": sha256_file(statistics_path),
        },
        "bundle_validation": validation,
        "training": {
            "updates_completed": step,
            "best_step": best_step,
            "best_selection_key": list(best_key),
            "batch_size": batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "effective_batch_size": batch_size * accumulation_steps,
            "resolved_stage": stage,
            "model_config": model_config,
            "parameter_count": sum(parameter.numel() for parameter in bridge.parameters()),
            "amp_overflow_events": amp_overflow_events,
        },
        "development_selection": checkpoint["selection_metrics"],
        "development_gate_metrics": gate_metrics,
        "development_gate": gate_report,
        "pilot_gate": pilot_gate,
        "telemetry": {
            **telemetry,
            "peak_allocated_vram_bytes": peak_allocated,
            "peak_reserved_vram_bytes": peak_reserved,
            "wall_seconds": total_seconds,
            "updates_per_hour": step * 3600.0 / max(total_seconds, 1e-12),
            "colab_compute_units_before": colab_compute_units_before,
            "colab_compute_units_after": None,
            "colab_compute_units_consumed": None,
        },
        "checkpoint": str(best_path),
        "complete": True,
    }
    _atomic_json(output_dir / "run_summary.json", result)
    _atomic_json(output_dir / "train_history.json", history)
    return result


def finalize_compute_units(summary_path: Path, after: float) -> dict:
    payload = load_json_object(summary_path)
    telemetry = payload.setdefault("telemetry", {})
    before = telemetry.get("colab_compute_units_before")
    if before is None:
        raise ValueError("compute-unit finalization requires a recorded before value")
    telemetry["colab_compute_units_after"] = float(after)
    telemetry["colab_compute_units_consumed"] = max(0.0, float(before) - float(after))
    _atomic_json(summary_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("config/LIP-H0-010_initial_condition_bridge.yaml"),
    )
    parser.add_argument(
        "--parent-config",
        type=Path,
        default=Path("config/LIP-PROTO-014_source_conditioned_residual_packet.yaml"),
    )
    parser.add_argument(
        "--predecessor-registry",
        type=Path,
        default=Path("experiments/registry/LIP-H0-009_entry_seed_free_evolution.json"),
    )
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--variant", default="unrolled_initial_condition")
    parser.add_argument("--seed", type=int, default=4001)
    parser.add_argument("--target-device", default="auto")
    parser.add_argument("--colab-compute-units-before", type=float)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--dry-run-contract", action="store_true")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--finalize-compute-units-after", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment = load_yaml(args.experiment_config)
    parent = load_yaml(args.parent_config)
    _validate_contract(
        experiment,
        parent,
        experiment_path=args.experiment_config,
        parent_path=args.parent_config,
        predecessor_registry_path=args.predecessor_registry,
    )
    if args.dry_run_contract:
        print("LIP-H0-010 contract validated")
        return
    if args.finalize_compute_units_after is not None:
        if args.summary is None:
            raise ValueError("--summary is required to finalize compute units")
        finalize_compute_units(args.summary, args.finalize_compute_units_after)
        print(args.summary)
        return
    if args.bundle_dir is None or args.output_dir is None:
        raise ValueError("--bundle-dir and --output-dir are required")
    result = run_initial_condition_training(
        experiment_path=args.experiment_config,
        parent_path=args.parent_config,
        predecessor_registry_path=args.predecessor_registry,
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
        variant_name=str(args.variant),
        seed=int(args.seed),
        pilot=bool(args.pilot),
        target_device=str(args.target_device),
        colab_compute_units_before=args.colab_compute_units_before,
    )
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "stage": result["stage"],
                "pilot_passed": (
                    result["pilot_gate"]["passed"]
                    if result["pilot_gate"] is not None
                    else None
                ),
                "summary": str(args.output_dir / "run_summary.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
