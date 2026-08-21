"""Train and evaluate the H0-017 closed-loop receiver corrector."""

from __future__ import annotations

import gc
import json
import math
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.core.closed_loop_trajectory import (
    ClosedLoopTrajectoryBridge,
    ReceiverStateCorrector,
)
from src.core.hard_negative_batching import (
    EpochShuffledBatchSampler,
    build_balanced_hard_negative_batches,
    hard_negative_mapping,
)
from src.core.packet_bridge import SourcePacketEncoder
from src.core.packet_bundle import (
    PacketRecordDataset,
    compute_target_packet_statistics,
    load_packet_records,
    sha256_file,
    validate_packet_bundle,
)
from src.core.packet_loss import build_terminal_component_masks
from src.core.receiver_closed_loop import (
    evolve_receiver_with_closed_loop_corrector,
)
from src.evaluation.packet_bridge import (
    checkpoint_selection_key,
    summarize_packet_latent_metrics,
)
from src.pipelines.infer import load_target, model_input_device
from src.pipelines.initial_condition_bridge import (
    _grad_scaler,
    _json_ready_metrics,
    _make_loader,
    _normalize_trajectory,
    _repeat_receiver_inputs,
)
from src.pipelines.oracle_experiment import load_json_object, load_yaml
from src.pipelines.packet_bridge import build_packet_loss, packet_collate
from src.pipelines.packet_confirmation import _neutral_inputs, _suffix_positions
from src.pipelines.packet_trajectory import _atomic_json
from src.pipelines.receiver_aware_replay import _lf_sha256_file


CLOSED_LOOP_PROTOCOL_VERSION = "lip-closed-loop-trajectory-corrector-v1"


def _git_head() -> str:
    value = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("run commit is not a full lowercase Git SHA")
    return value


def validate_closed_loop_contract(
    experiment: Mapping,
    parent: Mapping,
    *,
    experiment_path: Path,
    parent_path: Path,
    learned_registry_path: Path,
    functional_registry_path: Path,
    source_registry_path: Path,
) -> None:
    if experiment.get("experiment_id") != "LIP-H0-017":
        raise ValueError("unexpected closed-loop experiment_id")
    if experiment.get("protocol_version") != CLOSED_LOOP_PROTOCOL_VERSION:
        raise ValueError("unexpected closed-loop protocol_version")
    if experiment.get("claim_status") != "development_only_operator_feasibility":
        raise ValueError("H0-017 must remain development-only")
    if parent.get("experiment_id") != "LIP-PROTO-014":
        raise ValueError("H0-017 parent must be LIP-PROTO-014")
    if experiment["parent"]["config_sha256"] != _lf_sha256_file(parent_path):
        raise ValueError("parent config hash differs from the frozen contract")

    paths = {
        "learned_system": learned_registry_path,
        "functional_mechanism": functional_registry_path,
        "source_encoder_checkpoint": source_registry_path,
    }
    registries = {name: load_json_object(path) for name, path in paths.items()}
    for name, path in paths.items():
        frozen = experiment["predecessors"][name]
        if frozen["registry_sha256"] != _lf_sha256_file(path):
            raise ValueError(f"{name} registry hash differs from the contract")
        if registries[name].get("experiment_id") != frozen["experiment_id"]:
            raise ValueError(f"{name} registry experiment differs from the contract")
    if (
        registries["functional_mechanism"].get("decision", {}).get(
            "diagnostic_route"
        )
        != experiment["predecessors"]["functional_mechanism"]["required_route"]
    ):
        raise ValueError("EVAL-037 did not take the required no-candidate route")
    if (
        registries["learned_system"].get("aggregate_gate", {}).get(
            "strong_replication_passed"
        )
        is not True
    ):
        raise ValueError("H0-016 learned-system gate is not registered as passed")
    checkpoint_registry = registries["source_encoder_checkpoint"]
    expected_checkpoint = experiment["predecessors"]["source_encoder_checkpoint"]
    if (
        checkpoint_registry.get("artifacts", {})
        .get("screen", {})
        .get("best_checkpoint", {})
        .get("sha256")
        != expected_checkpoint["sha256"]
    ):
        raise ValueError("source encoder checkpoint differs from H0-015 registry")

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
    if list(experiment["data"]["prohibited_splits"]) != ["confirmation"]:
        raise ValueError("H0-017 must prohibit confirmation")

    receiver = experiment["receiver"]
    parent_target = parent["models"]["target"]
    if (receiver["model_id"], receiver["revision"]) != (
        parent_target["model_id"],
        parent_target["revision"],
    ):
        raise ValueError("receiver endpoint drifted from PROTO-014")
    layers = [int(value) for value in receiver["corrected_layer_indices"]]
    if layers != list(range(8)):
        raise ValueError("H0-017 must correct the contiguous layer prefix 0..7")
    if layers != [int(value) for value in parent["packets"]["target"]["layer_indices"]]:
        raise ValueError("receiver layers drifted from PROTO-014")
    if [int(value) for value in receiver["packet_offsets"]] != [
        int(value) for value in parent["packets"]["target"]["offsets"]
    ]:
        raise ValueError("receiver packet sites drifted from PROTO-014")
    if receiver["freeze_all_parameters"] is not True:
        raise ValueError("receiver parameters must remain frozen")
    if int(receiver["stop_before_layer"]) != 8:
        raise ValueError("H0-017 must stop before receiver block 8")

    encoder = experiment["source_encoder"]
    if encoder["initialize_from_frozen_checkpoint"] is not True:
        raise ValueError("source encoder must initialize from the frozen checkpoint")
    if encoder["freeze_parameters"] is not True:
        raise ValueError("source encoder must remain frozen")
    if encoder["model_kind"] != "query_conditioned":
        raise ValueError("H0-017 requires the query-conditioned source encoder")
    corrector = experiment["corrector"]
    if corrector["operator"] != "live_residual_plus_site_scaled_normalized_delta":
        raise ValueError("closed-loop update operator drifted")
    if corrector["delta_head_initialization"] != "zeros":
        raise ValueError("closed-loop corrector must initialize as a no-op")
    systems = experiment["variants"]["systems"]
    if set(systems) != {"closed_loop_live", "open_loop_zero_live"}:
        raise ValueError("H0-017 causal variants changed")
    if systems["closed_loop_live"]["condition_on_live_state"] is not True:
        raise ValueError("primary system must observe the live receiver state")
    if systems["open_loop_zero_live"]["condition_on_live_state"] is not False:
        raise ValueError("control system must not observe the live receiver state")
    if list(experiment["loss"]["incoming_trajectory"]["layers"]) != list(range(1, 8)):
        raise ValueError("primary causal loss must use pre-correction layers 1..7")
    if float(experiment["loss"]["incoming_trajectory"]["lambda"]) <= 0.0:
        raise ValueError("incoming trajectory loss must be active")
    if float(experiment["loss"]["corrected_state"]["lambda"]) <= 0.0:
        raise ValueError("corrected-state auxiliary loss must be active")
    if experiment["training"]["pilot"]["variant"] != "closed_loop_live":
        raise ValueError("pilot must test the state-conditioned primary system")
    if int(experiment["training"]["pilot"]["max_updates"]) != int(
        experiment["pilot_gate"]["required_updates"]
    ):
        raise ValueError("pilot updates differ from the feasibility gate")
    if experiment["confirmation"]["status"] != "prohibited_in_H0-017":
        raise ValueError("confirmation must remain prohibited in H0-017")
    if experiment["confirmation"]["eval_038_execution_authorized"] is not False:
        raise ValueError("H0-017 cannot authorize EVAL-038 execution in advance")
    if experiment["compute"]["preferred_accelerator"] != "L4":
        raise ValueError("H0-017 is frozen for an L4")
    if experiment["compute"]["allow_silent_fallback"] is not False:
        raise ValueError("silent accelerator fallback is prohibited")
    if not experiment_path.is_file():
        raise ValueError("experiment contract path does not exist")


def _build_source_encoder(
    experiment: Mapping,
    source_shape: tuple[int, int, int],
) -> SourcePacketEncoder:
    layers, positions, width = source_shape
    config = experiment["source_encoder"]
    return SourcePacketEncoder(
        source_width=width,
        source_layers=layers,
        source_positions=positions,
        protocol_slots=int(config["protocol_slots"]),
        bridge_width=int(config["bridge_width"]),
        attention_heads=int(config["attention_heads"]),
        feedforward_width=int(config["feedforward_width"]),
        decoder_blocks=int(config["encoder_blocks"]),
        dropout=float(config["dropout"]),
    )


def build_closed_loop_bridge(
    experiment: Mapping,
    *,
    source_shape: tuple[int, int, int],
    source_checkpoint_path: Path,
    variant_name: str,
) -> ClosedLoopTrajectoryBridge:
    expected = experiment["predecessors"]["source_encoder_checkpoint"]["sha256"]
    if not source_checkpoint_path.is_file():
        raise FileNotFoundError(source_checkpoint_path)
    if sha256_file(source_checkpoint_path) != expected:
        raise ValueError("source encoder checkpoint hash differs from the contract")
    checkpoint = torch.load(source_checkpoint_path, map_location="cpu", weights_only=True)
    if tuple(checkpoint.get("source_shape", ())) != tuple(source_shape):
        raise ValueError("source encoder checkpoint shape differs from the bundle")
    encoder = _build_source_encoder(experiment, source_shape)
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("source checkpoint does not contain a model state")
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_state, strict=True)

    systems = experiment["variants"]["systems"]
    if variant_name not in systems:
        raise ValueError(f"unknown H0-017 variant: {variant_name}")
    config = experiment["corrector"]
    corrector = ReceiverStateCorrector(
        target_width=int(config["target_width"]),
        target_layers=int(config["target_layers"]),
        target_positions=int(config["target_positions"]),
        bridge_width=int(config["bridge_width"]),
        attention_heads=int(config["attention_heads"]),
        feedforward_width=int(config["feedforward_width"]),
        decoder_blocks=int(config["decoder_blocks"]),
        dropout=float(config["dropout"]),
        condition_on_live_state=bool(
            systems[variant_name]["condition_on_live_state"]
        ),
    )
    bridge = ClosedLoopTrajectoryBridge(encoder, corrector)
    bridge.freeze_encoder()
    return bridge


def _hard_negative_loader(
    experiment: Mapping,
    dataset: PacketRecordDataset,
    *,
    candidate_bank_path: Path,
    batch_size: int,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, dict]:
    candidate = experiment["diagnostic_source"]["candidate_bank"]
    if sha256_file(candidate_bank_path) != candidate["sha256"]:
        raise ValueError("candidate bank hash differs from the H0-017 contract")
    bank = load_json_object(candidate_bank_path)
    mapping = hard_negative_mapping(bank, label=candidate["checkpoint_label"])
    task_ids = [str(record["task_id"]) for record in dataset.records]
    policy = experiment["training"]["batch_policy"]
    batches, metadata = build_balanced_hard_negative_batches(
        task_ids,
        mapping,
        batch_size=batch_size,
        seed=int(policy["partition_seed"]),
        restarts=int(policy["search_restarts"]),
        max_swaps=int(policy["maximum_improving_swaps_per_restart"]),
    )
    if metadata["global_hardest_covered_anchors"] != int(
        policy["expected_global_hardest_covered_anchors"]
    ) or metadata["global_hardest_coverage"] != float(
        policy["expected_global_hardest_coverage"]
    ):
        raise ValueError("hard-negative partition differs from H0-015")
    sampler = EpochShuffledBatchSampler(batches, seed=int(seed))
    return (
        DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=packet_collate,
        ),
        {
            **metadata,
            "candidate_bank_sha256": candidate["sha256"],
            "policy_kind": policy["kind"],
        },
    )


def _evaluate(
    bridge: ClosedLoopTrajectoryBridge,
    receiver,
    dataset: PacketRecordDataset,
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
    incoming_rows = []
    corrected_rows = []
    targets = []
    counts = []
    task_ids = []
    bridge.encoder.eval()
    bridge.corrector.eval()
    receiver.eval()
    with torch.no_grad():
        for batch in loader:
            source = batch["source_packet"].to(device)
            code = bridge.encode(source)
            states = evolve_receiver_with_closed_loop_corrector(
                receiver,
                _repeat_receiver_inputs(receiver_inputs, source.shape[0]),
                positions=positions,
                protocol_code=code,
                corrector=bridge.correction,
                scaffold=scaffold,
                site_scale=site_scale,
                layer_indices=layers,
            )
            incoming_rows.append(
                _normalize_trajectory(
                    states["incoming_before_correction"], scaffold, site_scale
                ).cpu()
            )
            corrected_rows.append(
                _normalize_trajectory(
                    states["residual_input"], scaffold, site_scale
                ).cpu()
            )
            targets.append(batch["target_residual"].float())
            counts.append(batch["name_token_count"])
            task_ids.extend(batch["task_ids"])
    incoming = torch.cat(incoming_rows)
    corrected = torch.cat(corrected_rows)
    target = torch.cat(targets)
    name_counts = torch.cat(counts)
    masks = build_terminal_component_masks(
        name_counts,
        target_positions=target.shape[2],
        boundary_positions=boundary_positions,
    )
    return {
        "incoming_trajectory": summarize_packet_latent_metrics(
            incoming[:, 1:], target[:, 1:], masks, task_ids=task_ids
        ),
        "corrected_state": summarize_packet_latent_metrics(
            corrected, target, masks, task_ids=task_ids
        ),
        "mean_squared_normalized_delta": float(
            (corrected - incoming).square().mean().item()
        ),
    }


def _gpu_telemetry() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("LIP-H0-017 requires the registered L4 accelerator")
    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    if "L4" not in name.upper():
        raise RuntimeError(f"registered accelerator is L4, observed {name!r}")
    return {
        "gpu_name": name,
        "total_vram_bytes": int(torch.cuda.get_device_properties(index).total_memory),
        "cuda_runtime": str(torch.version.cuda),
    }


def run_closed_loop_training(
    *,
    experiment_path: Path,
    parent_path: Path,
    learned_registry_path: Path,
    functional_registry_path: Path,
    source_registry_path: Path,
    bundle_dir: Path,
    source_checkpoint_path: Path,
    output_dir: Path,
    variant_name: str,
    pilot: bool,
    candidate_bank_path: Path | None = None,
    target_device: str = "cuda",
    colab_compute_units_before: float | None = None,
) -> dict:
    experiment = load_yaml(experiment_path)
    parent = load_yaml(parent_path)
    validate_closed_loop_contract(
        experiment,
        parent,
        experiment_path=experiment_path,
        parent_path=parent_path,
        learned_registry_path=learned_registry_path,
        functional_registry_path=functional_registry_path,
        source_registry_path=source_registry_path,
    )
    stage = dict(
        experiment["training"]["pilot" if pilot else "paired_screen"]
    )
    if pilot and variant_name != stage["variant"]:
        raise ValueError("pilot variant differs from the frozen contract")
    if not pilot and variant_name not in stage["variants"]:
        raise ValueError("screen variant differs from the frozen contract")
    if target_device != "cuda":
        raise RuntimeError("H0-017 training requires target_device=cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    telemetry = _gpu_telemetry()
    torch.cuda.reset_peak_memory_stats()

    validation = validate_packet_bundle(bundle_dir, require_real=True)
    if validation["split_counts"] != {
        "train": 256,
        "development_selection": 32,
        "development_gate": 32,
        "confirmation": 0,
    }:
        raise ValueError("training bundle split counts differ from H0-017")
    records = load_packet_records(bundle_dir)
    by_split = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "development_selection", "development_gate")
    }
    scaffold, site_scale = compute_target_packet_statistics(by_split["train"])
    statistics_path = output_dir / "target_statistics.pt"
    torch.save({"scaffold": scaffold, "site_scale": site_scale}, statistics_path)
    datasets = {
        split: PacketRecordDataset(rows, scaffold=scaffold, site_scale=site_scale)
        for split, rows in by_split.items()
    }

    receiver_config = experiment["receiver"]
    receiver, tokenizer = load_target(
        receiver_config["model_id"],
        target_device,
        bool(receiver_config["load_4bit"]),
        revision=receiver_config["revision"],
    )
    receiver.eval()
    receiver.requires_grad_(False)
    device = model_input_device(receiver)
    scaffold = scaffold.to(device)
    site_scale = site_scale.to(device)
    _, receiver_inputs = _neutral_inputs(parent, tokenizer, device)
    positions = _suffix_positions(receiver_inputs, receiver_config["packet_offsets"])
    layers = [int(value) for value in receiver_config["corrected_layer_indices"]]

    bridge = build_closed_loop_bridge(
        experiment,
        source_shape=tuple(validation["source_shape"]),
        source_checkpoint_path=source_checkpoint_path,
        variant_name=variant_name,
    ).to(device)
    bridge.freeze_encoder()
    optimizer = torch.optim.AdamW(
        bridge.corrector.parameters(),
        lr=float(experiment["training"]["learning_rate"]),
        weight_decay=float(experiment["training"]["weight_decay"]),
    )
    incoming_loss_config = experiment["loss"]["incoming_trajectory"]
    corrected_loss_config = experiment["loss"]["corrected_state"]
    incoming_criterion = build_packet_loss(incoming_loss_config)
    corrected_criterion = build_packet_loss(corrected_loss_config)
    lambda_incoming = float(incoming_loss_config["lambda"])
    lambda_corrected = float(corrected_loss_config["lambda"])
    lambda_energy = float(experiment["loss"]["correction_energy"]["lambda"])
    batch_size = int(stage["batch_size"])
    seed = int(experiment["training"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    batch_plan = None
    if pilot:
        loader = _make_loader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            num_workers=int(experiment["training"]["num_workers"]),
        )
    else:
        if candidate_bank_path is None:
            raise ValueError("paired screen requires the frozen candidate bank")
        loader, batch_plan = _hard_negative_loader(
            experiment,
            datasets["train"],
            candidate_bank_path=candidate_bank_path,
            batch_size=batch_size,
            seed=seed,
            num_workers=int(experiment["training"]["num_workers"]),
        )
        _atomic_json(output_dir / "training_batch_plan.json", batch_plan)

    use_amp = bool(experiment["training"]["fp16_autocast"])
    scaler = _grad_scaler(use_amp)
    max_updates = int(stage["max_updates"])
    validation_interval = int(stage["validation_interval"])
    boundary_positions = int(experiment["data"]["boundary_positions"])
    gradient_clip = float(experiment["training"]["gradient_clip"])
    history = []
    amp_overflow_events = []
    best_key = None
    best_step = None
    best_path = output_dir / "best_checkpoint.pt"
    step = 0
    epoch = 0
    while step < max_updates:
        epoch += 1
        for batch in loader:
            if step >= max_updates:
                break
            bridge.encoder.eval()
            bridge.corrector.train()
            source = batch["source_packet"].to(device)
            target = batch["target_residual"].to(device)
            counts = batch["name_token_count"].to(device)
            masks = build_terminal_component_masks(
                counts,
                target_positions=target.shape[2],
                boundary_positions=boundary_positions,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                with torch.no_grad():
                    code = bridge.encode(source)
                states = evolve_receiver_with_closed_loop_corrector(
                    receiver,
                    _repeat_receiver_inputs(receiver_inputs, source.shape[0]),
                    positions=positions,
                    protocol_code=code,
                    corrector=bridge.correction,
                    scaffold=scaffold,
                    site_scale=site_scale,
                    layer_indices=layers,
                )
                incoming = _normalize_trajectory(
                    states["incoming_before_correction"], scaffold, site_scale
                )
                corrected = _normalize_trajectory(
                    states["residual_input"], scaffold, site_scale
                )
                incoming_metrics = incoming_criterion(
                    incoming[:, 1:], target[:, 1:], masks
                )
                corrected_metrics = corrected_criterion(corrected, target, masks)
                correction_energy = (corrected - incoming).square().mean()
                total_loss = (
                    lambda_incoming * incoming_metrics["total_loss"]
                    + lambda_corrected * corrected_metrics["total_loss"]
                    + lambda_energy * correction_energy
                )
            if not bool(torch.isfinite(total_loss.detach()).all().item()):
                raise FloatingPointError("closed-loop training produced non-finite loss")
            scaler.scale(total_loss).backward()
            if any(parameter.grad is not None for parameter in bridge.encoder.parameters()):
                raise RuntimeError("frozen source encoder received gradients")
            if any(parameter.grad is not None for parameter in receiver.parameters()):
                raise RuntimeError("frozen receiver received parameter gradients")
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                bridge.corrector.parameters(), gradient_clip
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
                    "closed-loop training produced zero or non-finite gradient"
                )
            step += 1
            row = {
                "step": step,
                "epoch": epoch,
                "gradient_norm": gradient_norm_value,
                "total_loss": float(total_loss.detach()),
                "correction_energy": float(correction_energy.detach()),
                "incoming": _json_ready_metrics(incoming_metrics),
                "corrected": _json_ready_metrics(corrected_metrics),
            }
            history.append(row)
            if step % validation_interval == 0 or step == max_updates:
                selection = _evaluate(
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
                key = checkpoint_selection_key(
                    selection["incoming_trajectory"], step=step
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_step = step
                    torch.save(
                        {
                            "corrector_state": bridge.corrector.state_dict(),
                            "step": step,
                            "selection_key": list(key),
                            "selection_metrics": selection,
                            "source_shape": validation["source_shape"],
                            "target_shape": validation["target_shape"],
                            "variant": variant_name,
                            "source_encoder_checkpoint_sha256": experiment[
                                "predecessors"
                            ]["source_encoder_checkpoint"]["sha256"],
                        },
                        best_path,
                    )
                _atomic_json(output_dir / "train_history.json", history)
                print(
                    json.dumps(
                        {
                            "event": "closed_loop_validation",
                            "variant": variant_name,
                            "step": step,
                            "incoming_rmse": selection["incoming_trajectory"][
                                "normalized_residual_rmse"
                            ],
                            "best_step": best_step,
                        }
                    ),
                    flush=True,
                )
            del source, target, code, states, incoming, corrected
            del incoming_metrics, corrected_metrics, total_loss, correction_energy
            gc.collect()

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    bridge.corrector.load_state_dict(checkpoint["corrector_state"])
    gate_metrics = _evaluate(
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
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    pilot_gate = None
    if pilot:
        frozen = experiment["pilot_gate"]
        pilot_gate = {
            "required_updates": int(frozen["required_updates"]),
            "updates_completed": step,
            "finite_loss": all(math.isfinite(row["total_loss"]) for row in history),
            "nonzero_corrector_gradient_each_update": all(
                row["gradient_norm"] > 0.0 for row in history
            ),
            "source_encoder_frozen": all(
                not parameter.requires_grad for parameter in bridge.encoder.parameters()
            ),
            "receiver_frozen": all(
                not parameter.requires_grad for parameter in receiver.parameters()
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
            pilot_gate["updates_completed"] == pilot_gate["required_updates"]
            and pilot_gate["finite_loss"]
            and pilot_gate["nonzero_corrector_gradient_each_update"]
            and pilot_gate["source_encoder_frozen"]
            and pilot_gate["receiver_frozen"]
            and pilot_gate["amp_overflow_events"]
            <= pilot_gate["maximum_amp_overflow_events"]
            and pilot_gate["peak_allocated_vram_bytes"]
            <= pilot_gate["maximum_peak_allocated_vram_bytes"]
        )
    result = {
        "experiment_id": "LIP-H0-017",
        "protocol_version": CLOSED_LOOP_PROTOCOL_VERSION,
        "claim_status": experiment["claim_status"],
        "stage": "pilot" if pilot else "paired_screen_cell",
        "variant": variant_name,
        "seed": seed,
        "run_commit": _git_head(),
        "provenance": {
            "experiment_config_sha256": _lf_sha256_file(experiment_path),
            "parent_config_sha256": _lf_sha256_file(parent_path),
            "learned_registry_sha256": _lf_sha256_file(learned_registry_path),
            "functional_registry_sha256": _lf_sha256_file(functional_registry_path),
            "source_registry_sha256": _lf_sha256_file(source_registry_path),
            "bundle_manifest_sha256": sha256_file(bundle_dir / "manifest.json"),
            "source_encoder_checkpoint_sha256": sha256_file(
                source_checkpoint_path
            ),
            "target_statistics_sha256": sha256_file(statistics_path),
            **(
                {"candidate_bank_sha256": sha256_file(candidate_bank_path)}
                if candidate_bank_path is not None
                else {}
            ),
        },
        "bundle_validation": validation,
        "training": {
            "updates_completed": step,
            "best_step": best_step,
            "best_selection_key": list(best_key),
            "batch_size": batch_size,
            "resolved_stage": stage,
            "corrector_parameter_count": sum(
                parameter.numel() for parameter in bridge.corrector.parameters()
            ),
            "source_encoder_parameter_count": sum(
                parameter.numel() for parameter in bridge.encoder.parameters()
            ),
            "source_encoder_trainable_parameters": sum(
                parameter.numel()
                for parameter in bridge.encoder.parameters()
                if parameter.requires_grad
            ),
            "amp_overflow_events": amp_overflow_events,
            **({"batch_policy": batch_plan} if batch_plan is not None else {}),
        },
        "development_selection": checkpoint["selection_metrics"],
        "development_gate_metrics": gate_metrics,
        "pilot_gate": pilot_gate,
        "telemetry": {
            **telemetry,
            "peak_allocated_vram_bytes": peak_allocated,
            "peak_reserved_vram_bytes": peak_reserved,
            "wall_seconds": float(time.perf_counter() - started),
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
