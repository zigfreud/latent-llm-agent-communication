"""Receiver-unrolled gradient-geometry evaluation for LIP-EVAL-032."""

from __future__ import annotations

import gc
import itertools
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

from src.core.packet_bundle import (
    PacketRecordDataset,
    compute_target_packet_statistics,
    load_packet_records,
    sha256_file,
    validate_packet_bundle,
)
from src.core.packet_loss import (
    CONTRASTIVE_REGIONS,
    build_terminal_component_masks,
    packet_similarity,
)
from src.pipelines.initial_condition_bridge import (
    _bridge_model_config,
    _git_head,
    _gpu_telemetry,
    _induced_trajectory,
)
from src.pipelines.infer import load_target, model_input_device
from src.pipelines.oracle_experiment import load_json_object, load_yaml
from src.pipelines.packet_bridge import (
    build_packet_bridge,
    build_packet_loss,
    packet_collate,
)
from src.pipelines.packet_confirmation import _neutral_inputs, _suffix_positions
from src.pipelines.packet_trajectory import _atomic_json


GRADIENT_GEOMETRY_PROTOCOL_VERSION = "lip-receiver-unrolled-gradient-geometry-v1"
CHECKPOINT_LABELS = ("H0_011", "H0_013")
OBJECTIVE_NAMES = (
    "core_margin",
    "symmetric_nce",
    "reconstruction",
    "non_margin",
    "configured_total",
)


def validate_gradient_geometry_contract(config: Mapping) -> None:
    if config.get("experiment_id") != "LIP-EVAL-032":
        raise ValueError("unexpected gradient-geometry experiment_id")
    if config.get("protocol_version") != GRADIENT_GEOMETRY_PROTOCOL_VERSION:
        raise ValueError("unexpected gradient-geometry protocol_version")
    if config.get("claim_status") != "development_only_mechanism_evaluation":
        raise ValueError("LIP-EVAL-032 must remain development-only")

    checkpoints = config["frozen_inputs"]["checkpoints"]
    if tuple(checkpoints) != CHECKPOINT_LABELS:
        raise ValueError("LIP-EVAL-032 must compare only H0-011 and H0-013")
    expected = {
        "H0_011": (4007, 120, "large_negative_batch_unrolled", 0.10),
        "H0_013": (4007, 120, "core_margin_pressure_unrolled", 1.00),
    }
    for label, (seed, step, variant, margin_weight) in expected.items():
        item = checkpoints[label]
        observed = (
            int(item["expected_seed"]),
            int(item["expected_best_step"]),
            item["expected_variant"],
            float(item["expected_lambda_margin"]),
        )
        if observed != (seed, step, variant, margin_weight):
            raise ValueError(f"{label} frozen checkpoint contract drifted")

    data = config["data"]
    if list(data["allowed_splits"]) != ["train"]:
        raise ValueError("LIP-EVAL-032 may use only the train split")
    if set(data["prohibited_splits"]) != {
        "development_selection",
        "development_gate",
        "confirmation",
    }:
        raise ValueError("development and confirmation splits must remain prohibited")
    if (
        int(data["expected_train_tasks"]),
        int(data["batch_size"]),
        int(data["batch_count"]),
        data["ordering"],
    ) != (256, 16, 16, "task_id_ascending"):
        raise ValueError("deterministic train partition drifted")

    receiver = config["receiver"]
    if [int(value) for value in receiver["evolved_layer_indices"]] != list(range(8)):
        raise ValueError("receiver evolution must remain blocks 0 through 7")
    if [int(value) for value in receiver["gradient_objective_layers"]] != list(
        range(1, 8)
    ):
        raise ValueError("gradient objective must remain induced layers 1 through 7")
    if receiver["freeze_all_parameters"] is not True:
        raise ValueError("receiver must remain frozen")

    measurements = config["measurements"]
    if tuple(measurements["objectives"]) != OBJECTIVE_NAMES:
        raise ValueError("gradient objective family drifted")
    if measurements["gradient_accumulation_dtype"] != "float32":
        raise ValueError("diagnostic gradient accumulation must remain float32")
    if int(measurements["candidate_bank_size"]) != 256:
        raise ValueError("candidate bank must contain all 256 train tasks")

    routing = config["routing"]
    scale = routing["scale_limited"]
    conflict = routing["conflict_limited"]
    coverage = routing["coverage_limited"]
    if (
        float(scale["maximum_median_effective_core_to_nonmargin_gradient_ratio"]),
        float(scale["minimum_median_core_nonmargin_cosine"]),
        float(conflict["maximum_core_nonmargin_cosine"]),
        int(conflict["minimum_conflicting_batches"]),
        float(coverage["maximum_global_hardest_negative_coverage"]),
    ) != (0.10, -0.10, -0.10, 12, 0.25):
        raise ValueError("predeclared routing thresholds drifted")
    decision = config["decision_boundary"]
    if any(
        bool(decision[key])
        for key in ("replication_authorized", "functional_confirmation_authorized")
    ) or decision["proto_015_status"] != "premature":
        raise ValueError("LIP-EVAL-032 cannot authorize confirmation or PROTO-015")


def _float(value: torch.Tensor) -> float:
    result = float(value.detach().float().item())
    if not math.isfinite(result):
        raise FloatingPointError("gradient diagnostic produced a non-finite scalar")
    return result


def _gradient_geometry(
    objectives: Mapping[str, torch.Tensor],
    entry: torch.Tensor,
) -> tuple[dict[str, float], dict[str, float]]:
    if tuple(objectives) != OBJECTIVE_NAMES:
        raise ValueError("objectives must use the frozen LIP-EVAL-032 order")
    gradients: dict[str, torch.Tensor] = {}
    names = list(OBJECTIVE_NAMES)
    for index, name in enumerate(names):
        gradient = torch.autograd.grad(
            objectives[name],
            entry,
            retain_graph=index < len(names) - 1,
            create_graph=False,
        )[0]
        gradient = gradient.detach().float()
        if not bool(torch.isfinite(gradient).all().item()):
            raise FloatingPointError(f"{name} gradient is non-finite")
        gradients[name] = gradient

    norms = {
        name: float(torch.linalg.vector_norm(gradient).item())
        for name, gradient in gradients.items()
    }
    cosines = {}
    for left, right in itertools.combinations(names, 2):
        denominator = norms[left] * norms[right]
        value = (
            float(torch.sum(gradients[left] * gradients[right]).item()) / denominator
            if denominator > 0.0
            else 0.0
        )
        cosines[f"{left}__{right}"] = max(-1.0, min(1.0, value))
    return norms, cosines


def _core_hinge_activity(
    prediction: torch.Tensor,
    target: torch.Tensor,
    core_mask: torch.Tensor,
    *,
    margin_target: float,
) -> dict[str, float]:
    similarity = packet_similarity(prediction, target, core_mask)
    batch_size = similarity.shape[0]
    diagonal = torch.diagonal(similarity)
    diagonal_mask = torch.eye(batch_size, dtype=torch.bool, device=similarity.device)
    negatives = similarity.masked_fill(diagonal_mask, float("-inf"))
    row_margin = diagonal - negatives.max(dim=1).values
    column_margin = diagonal - negatives.max(dim=0).values
    row_active = row_margin < float(margin_target)
    column_active = column_margin < float(margin_target)
    return {
        "row_active_fraction": float(row_active.float().mean().item()),
        "column_active_fraction": float(column_active.float().mean().item()),
        "combined_active_fraction": float(
            torch.cat((row_active, column_active)).float().mean().item()
        ),
        "row_margin_mean": float(row_margin.mean().item()),
        "column_margin_mean": float(column_margin.mean().item()),
    }


def _resolved_objectives(metrics: Mapping, criterion) -> dict[str, torch.Tensor]:
    reconstruction = (
        criterion.lambda_huber * metrics["huber_loss"]
        + criterion.lambda_cosine * metrics["cosine_loss"]
        + criterion.lambda_norm * metrics["norm_loss"]
    )
    symmetric_nce = criterion.lambda_symmetric_nce * metrics["symmetric_nce_loss"]
    non_margin = reconstruction + symmetric_nce
    return {
        "core_margin": metrics["core_margin_loss"],
        "symmetric_nce": symmetric_nce,
        "reconstruction": reconstruction,
        "non_margin": non_margin,
        "configured_total": metrics["total_loss"],
    }


def _batch_gradient_row(
    *,
    label: str,
    batch_index: int,
    bridge,
    receiver,
    batch: Mapping,
    receiver_inputs: Mapping[str, torch.Tensor],
    positions: torch.Tensor,
    scaffold: torch.Tensor,
    site_scale: torch.Tensor,
    layers: list[int],
    criterion,
    device: torch.device,
    boundary_positions: int,
    use_amp: bool,
) -> dict:
    source = batch["source_packet"].to(device)
    target = batch["target_residual"].to(device)
    counts = batch["name_token_count"].to(device)
    masks = build_terminal_component_masks(
        counts,
        target_positions=target.shape[2],
        boundary_positions=boundary_positions,
    )
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=use_amp
    ):
        predicted_entry = bridge(source)
    entry = predicted_entry.detach().float().requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        induced = _induced_trajectory(
            receiver,
            receiver_inputs,
            positions=positions,
            normalized_entry=entry,
            scaffold=scaffold,
            site_scale=site_scale,
            layers=layers,
        )
        objective_prediction = induced[:, 1:]
        objective_target = target[:, 1:]
        metrics = criterion(objective_prediction, objective_target, masks)
        objectives = _resolved_objectives(metrics, criterion)

    hinge = _core_hinge_activity(
        objective_prediction,
        objective_target,
        masks["core"],
        margin_target=criterion.margin_target,
    )
    norms, cosines = _gradient_geometry(objectives, entry)
    core_region_coefficient = criterion.lambda_margin / len(CONTRASTIVE_REGIONS)
    nonmargin_norm = norms["non_margin"]
    effective_ratio = (
        core_region_coefficient * norms["core_margin"] / nonmargin_norm
        if nonmargin_norm > 0.0
        else float("inf")
    )
    row = {
        "checkpoint": label,
        "batch_index": int(batch_index),
        "task_ids": list(batch["task_ids"]),
        "losses": {name: _float(value) for name, value in objectives.items()},
        "raw_losses": {
            "huber": _float(metrics["huber_loss"]),
            "cosine": _float(metrics["cosine_loss"]),
            "symmetric_nce": _float(metrics["symmetric_nce_loss"]),
            "aggregate_margin": _float(metrics["margin_loss"]),
            "core_margin": _float(metrics["core_margin_loss"]),
            "norm": _float(metrics["norm_loss"]),
        },
        "gradient_norms": norms,
        "gradient_cosines": cosines,
        "core_hinge": hinge,
        "configured_lambda_margin": float(criterion.lambda_margin),
        "configured_core_region_coefficient": float(core_region_coefficient),
        "effective_core_to_nonmargin_gradient_ratio": float(effective_ratio),
    }
    del source, target, counts, masks, predicted_entry, entry, induced
    del objective_prediction, objective_target, metrics, objectives
    gc.collect()
    torch.cuda.empty_cache()
    return row


def _normalized_core_rows(packet: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = packet * mask[:, None, :, None]
    return F.normalize(masked.flatten(start_dim=1).float(), p=2, dim=1)


def _candidate_diagnostics_from_similarity(
    similarity: torch.Tensor,
    task_ids: Sequence[str],
    *,
    batch_size: int,
) -> dict:
    similarity = similarity.float().cpu()
    count = similarity.shape[0]
    if tuple(similarity.shape) != (count, count) or len(task_ids) != count:
        raise ValueError("candidate similarity must be square and align with task_ids")
    diagonal = torch.diagonal(similarity)
    masked = similarity.masked_fill(torch.eye(count, dtype=torch.bool), float("-inf"))
    global_values, global_indices = masked.max(dim=1)
    rows = []
    for anchor in range(count):
        batch_start = (anchor // batch_size) * batch_size
        batch_stop = min(batch_start + batch_size, count)
        local_indices = [index for index in range(batch_start, batch_stop) if index != anchor]
        local_tensor = similarity[anchor, local_indices]
        local_offset = int(local_tensor.argmax().item())
        local_index = local_indices[local_offset]
        global_index = int(global_indices[anchor].item())
        global_margin = float((diagonal[anchor] - global_values[anchor]).item())
        local_margin = float(
            (diagonal[anchor] - similarity[anchor, local_index]).item()
        )
        rows.append(
            {
                "anchor_task_id": str(task_ids[anchor]),
                "global_hardest_task_id": str(task_ids[global_index]),
                "local_hardest_task_id": str(task_ids[local_index]),
                "global_hardest_in_assigned_batch": bool(
                    batch_start <= global_index < batch_stop
                ),
                "global_margin": global_margin,
                "local_margin": local_margin,
                "local_minus_global_margin": local_margin - global_margin,
            }
        )
    return {
        "task_count": count,
        "global_hardest_coverage": sum(
            row["global_hardest_in_assigned_batch"] for row in rows
        )
        / count,
        "local_minus_global_margin_mean": sum(
            row["local_minus_global_margin"] for row in rows
        )
        / count,
        "rows": rows,
    }


def _candidate_bank(
    *,
    bridge,
    receiver,
    dataset,
    receiver_inputs: Mapping[str, torch.Tensor],
    positions: torch.Tensor,
    scaffold: torch.Tensor,
    site_scale: torch.Tensor,
    layers: list[int],
    batch_size: int,
    boundary_positions: int,
    device: torch.device,
    use_amp: bool,
) -> dict:
    predicted_rows = []
    target_rows = []
    task_ids = []
    for start in range(0, len(dataset), batch_size):
        batch = packet_collate(
            [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        )
        source = batch["source_packet"].to(device)
        target = batch["target_residual"].to(device)
        counts = batch["name_token_count"].to(device)
        masks = build_terminal_component_masks(
            counts,
            target_positions=target.shape[2],
            boundary_positions=boundary_positions,
        )
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_amp
        ):
            entry = bridge(source)
            induced = _induced_trajectory(
                receiver,
                receiver_inputs,
                positions=positions,
                normalized_entry=entry,
                scaffold=scaffold,
                site_scale=site_scale,
                layers=layers,
            )[:, 1:]
        predicted_rows.append(_normalized_core_rows(induced, masks["core"]).cpu())
        target_rows.append(_normalized_core_rows(target[:, 1:], masks["core"]).cpu())
        task_ids.extend(batch["task_ids"])
        del source, target, counts, masks, entry, induced
        gc.collect()
        torch.cuda.empty_cache()

    predicted_matrix = torch.cat(predicted_rows)
    target_matrix = torch.cat(target_rows).to(device)
    similarities = []
    with torch.no_grad():
        for chunk in predicted_matrix.split(batch_size):
            similarities.append((chunk.to(device) @ target_matrix.T).float().cpu())
    similarity = torch.cat(similarities)
    result = _candidate_diagnostics_from_similarity(
        similarity,
        task_ids,
        batch_size=batch_size,
    )
    del predicted_matrix, target_matrix, similarities, similarity
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _bootstrap_interval(
    values: Sequence[float],
    *,
    statistic: str,
    confidence: float,
    resamples: int,
    seed: int,
) -> list[float]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("bootstrap values must be finite and non-empty")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        0,
        tensor.numel(),
        (int(resamples), tensor.numel()),
        generator=generator,
    )
    samples = tensor[indices]
    if statistic == "median":
        estimates = samples.median(dim=1).values
    elif statistic == "mean":
        estimates = samples.mean(dim=1)
    else:
        raise ValueError("bootstrap statistic must be median or mean")
    tail = (1.0 - float(confidence)) / 2.0
    return [
        float(torch.quantile(estimates, tail).item()),
        float(torch.quantile(estimates, 1.0 - tail).item()),
    ]


def _crosses(interval: Sequence[float], threshold: float) -> bool:
    return float(interval[0]) <= float(threshold) <= float(interval[1])


def route_gradient_geometry(
    batch_rows: Sequence[Mapping],
    candidate_bank: Mapping,
    config: Mapping,
) -> dict:
    h013 = [row for row in batch_rows if row["checkpoint"] == "H0_013"]
    if len(h013) != int(config["data"]["batch_count"]):
        raise ValueError("routing requires all 16 H0-013 batches")
    ratios = [float(row["effective_core_to_nonmargin_gradient_ratio"]) for row in h013]
    alignments = [
        float(row["gradient_cosines"]["core_margin__non_margin"]) for row in h013
    ]
    conflict_threshold = float(
        config["routing"]["conflict_limited"]["maximum_core_nonmargin_cosine"]
    )
    conflict_flags = [float(value < conflict_threshold) for value in alignments]
    coverage_flags = [
        float(row["global_hardest_in_assigned_batch"])
        for row in candidate_bank["H0_013"]["rows"]
    ]
    bootstrap = config["bootstrap"]
    common = {
        "confidence": float(bootstrap["confidence"]),
        "resamples": int(bootstrap["resamples"]),
    }
    seed = int(bootstrap["seed"])
    statistics = {
        "median_effective_core_to_nonmargin_gradient_ratio": {
            "estimate": float(torch.tensor(ratios).median().item()),
            "bootstrap_interval": _bootstrap_interval(
                ratios, statistic="median", seed=seed, **common
            ),
        },
        "median_core_nonmargin_cosine": {
            "estimate": float(torch.tensor(alignments).median().item()),
            "bootstrap_interval": _bootstrap_interval(
                alignments, statistic="median", seed=seed + 1, **common
            ),
        },
        "conflicting_batch_fraction": {
            "estimate": sum(conflict_flags) / len(conflict_flags),
            "count": int(sum(conflict_flags)),
            "bootstrap_interval": _bootstrap_interval(
                conflict_flags, statistic="mean", seed=seed + 2, **common
            ),
        },
        "global_hardest_negative_coverage": {
            "estimate": sum(coverage_flags) / len(coverage_flags),
            "bootstrap_interval": _bootstrap_interval(
                coverage_flags, statistic="mean", seed=seed + 3, **common
            ),
        },
    }
    routing = config["routing"]
    ratio_threshold = float(
        routing["scale_limited"][
            "maximum_median_effective_core_to_nonmargin_gradient_ratio"
        ]
    )
    alignment_threshold = float(
        routing["scale_limited"]["minimum_median_core_nonmargin_cosine"]
    )
    conflict_fraction_threshold = int(
        routing["conflict_limited"]["minimum_conflicting_batches"]
    ) / len(h013)
    coverage_threshold = float(
        routing["coverage_limited"]["maximum_global_hardest_negative_coverage"]
    )
    scale_limited = bool(
        statistics["median_effective_core_to_nonmargin_gradient_ratio"]["estimate"]
        < ratio_threshold
        and statistics["median_core_nonmargin_cosine"]["estimate"]
        >= alignment_threshold
    )
    conflict_limited = bool(
        statistics["conflicting_batch_fraction"]["estimate"]
        >= conflict_fraction_threshold
    )
    coverage_limited = bool(
        not scale_limited
        and not conflict_limited
        and statistics["global_hardest_negative_coverage"]["estimate"]
        < coverage_threshold
    )
    crossing = {
        "ratio": _crosses(
            statistics["median_effective_core_to_nonmargin_gradient_ratio"][
                "bootstrap_interval"
            ],
            ratio_threshold,
        ),
        "alignment": _crosses(
            statistics["median_core_nonmargin_cosine"]["bootstrap_interval"],
            alignment_threshold,
        ),
        "conflict_fraction": _crosses(
            statistics["conflicting_batch_fraction"]["bootstrap_interval"],
            conflict_fraction_threshold,
        ),
        "coverage": _crosses(
            statistics["global_hardest_negative_coverage"]["bootstrap_interval"],
            coverage_threshold,
        ),
    }
    conditions = {
        "scale_limited": scale_limited,
        "conflict_limited": conflict_limited,
        "coverage_limited": coverage_limited,
    }
    if any(crossing.values()) or sum(conditions.values()) != 1:
        route = "mixed_or_unresolved"
        intervention = None
    elif scale_limited:
        route = "scale_limited"
        intervention = "explicit_core_only_or_adaptive_gradient_weight"
    elif conflict_limited:
        route = "conflict_limited"
        intervention = "staged_optimization_or_conflict_aware_update"
    else:
        route = "coverage_limited"
        intervention = "cross_batch_memory_or_explicit_hard_negative_mining"
    return {
        "route": route,
        "authorized_H0_014_intervention_family": intervention,
        "statistics": statistics,
        "point_conditions": conditions,
        "bootstrap_threshold_crossings": crossing,
        "replication_authorized": False,
        "functional_confirmation_authorized": False,
        "proto_015_status": "premature",
    }


def _load_checkpoint_binding(
    *,
    label: str,
    binding: Mapping,
    checkpoint_dir: Path,
    config_root: Path,
    device: torch.device,
    source_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> tuple[object, object, dict]:
    registry_path = config_root / binding["registry"]
    experiment_path = config_root / binding["experiment_config"]
    registry = load_json_object(registry_path)
    experiment = load_yaml(experiment_path)
    summary_path = checkpoint_dir / "run_summary.json"
    checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
    statistics_path = checkpoint_dir / "target_statistics.pt"
    if not all(path.is_file() for path in (summary_path, checkpoint_path, statistics_path)):
        raise FileNotFoundError(f"{label} frozen screen artifacts are incomplete")
    expected_summary_sha = registry["artifacts"]["screen"]["sha256"]
    if sha256_file(summary_path) != expected_summary_sha:
        raise ValueError(f"{label} run summary hash differs from its registry")
    summary = load_json_object(summary_path)
    if (
        int(summary["seed"]) != int(binding["expected_seed"])
        or int(summary["training"]["best_step"]) != int(binding["expected_best_step"])
        or summary["variant"] != binding["expected_variant"]
    ):
        raise ValueError(f"{label} checkpoint identity differs from the frozen binding")
    criterion = build_packet_loss(experiment["loss"]["induced_trajectory"])
    if float(criterion.lambda_margin) != float(binding["expected_lambda_margin"]):
        raise ValueError(f"{label} configured lambda_margin drifted")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    bridge_target_shape = (1, target_shape[1], target_shape[2])
    if tuple(checkpoint["source_shape"]) != source_shape or tuple(
        checkpoint["bridge_target_shape"]
    ) != bridge_target_shape:
        raise ValueError(f"{label} bridge checkpoint shape drifted")
    expected_model_config = _bridge_model_config(experiment)
    if checkpoint["model_config"] != expected_model_config:
        raise ValueError(f"{label} bridge model config drifted")
    bridge = build_packet_bridge(
        expected_model_config, source_shape, bridge_target_shape
    ).to(device)
    bridge.load_state_dict(checkpoint["model_state"])
    bridge.eval()
    return bridge, criterion, {
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "experiment_config_path": str(experiment_path),
        "experiment_config_sha256": sha256_file(experiment_path),
        "run_summary_path": str(summary_path),
        "run_summary_sha256": sha256_file(summary_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "target_statistics_path": str(statistics_path),
        "target_statistics_sha256": sha256_file(statistics_path),
        "run_commit": summary["run_commit"],
        "best_step": int(summary["training"]["best_step"]),
    }


def run_gradient_geometry_evaluation(
    *,
    experiment_path: Path,
    bundle_dir: Path,
    h011_checkpoint_dir: Path,
    h013_checkpoint_dir: Path,
    output_dir: Path,
    pilot: bool,
    target_device: str,
    colab_compute_units_before: float | None,
) -> dict:
    config = load_yaml(experiment_path)
    validate_gradient_geometry_contract(config)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"gradient-geometry output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry = _gpu_telemetry()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    validation = validate_packet_bundle(bundle_dir, require_real=True)
    if int(validation["split_counts"]["train"]) != int(
        config["data"]["expected_train_tasks"]
    ):
        raise ValueError("train task count differs from LIP-EVAL-032")
    records = sorted(
        load_packet_records(bundle_dir, split="train"), key=lambda record: record["task_id"]
    )
    if len({record["task_id"] for record in records}) != len(records):
        raise ValueError("train task IDs must be unique")
    scaffold_cpu, site_scale_cpu = compute_target_packet_statistics(records)
    dataset = PacketRecordDataset(
        records, scaffold=scaffold_cpu, site_scale=site_scale_cpu
    )

    parent_path = experiment_path.resolve().parents[1] / config["frozen_inputs"][
        "parent_config"
    ]
    parent = load_yaml(parent_path)
    receiver_config = config["receiver"]
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
    scaffold = scaffold_cpu.to(device)
    site_scale = site_scale_cpu.to(device)
    source_shape = tuple(validation["source_shape"])
    target_shape = tuple(validation["target_shape"])
    checkpoint_dirs = {
        "H0_011": Path(h011_checkpoint_dir),
        "H0_013": Path(h013_checkpoint_dir),
    }
    config_root = experiment_path.resolve().parents[1]
    batch_size = int(config["data"]["batch_size"])
    batch_count = (
        int(config["compute"]["pilot_batches_per_checkpoint"])
        if pilot
        else int(config["data"]["batch_count"])
    )
    boundary_positions = int(config["data"]["boundary_positions"])
    use_amp = bool(config["compute"]["amp_receiver_path"])
    batch_rows = []
    candidate_banks = {}
    provenance = {}
    for label in CHECKPOINT_LABELS:
        binding = config["frozen_inputs"]["checkpoints"][label]
        bridge, criterion, checkpoint_provenance = _load_checkpoint_binding(
            label=label,
            binding=binding,
            checkpoint_dir=checkpoint_dirs[label],
            config_root=config_root,
            device=device,
            source_shape=source_shape,
            target_shape=target_shape,
        )
        stored_statistics = torch.load(
            checkpoint_dirs[label] / "target_statistics.pt",
            map_location="cpu",
            weights_only=True,
        )
        if not torch.equal(stored_statistics["scaffold"].float(), scaffold_cpu) or not torch.equal(
            stored_statistics["site_scale"].float(), site_scale_cpu
        ):
            raise ValueError(f"{label} target statistics differ from the train bundle")
        provenance[label] = checkpoint_provenance
        for batch_index in range(batch_count):
            start = batch_index * batch_size
            batch = packet_collate(
                [dataset[index] for index in range(start, start + batch_size)]
            )
            row = _batch_gradient_row(
                label=label,
                batch_index=batch_index,
                bridge=bridge,
                receiver=receiver,
                batch=batch,
                receiver_inputs=receiver_inputs,
                positions=positions,
                scaffold=scaffold,
                site_scale=site_scale,
                layers=layers,
                criterion=criterion,
                device=device,
                boundary_positions=boundary_positions,
                use_amp=use_amp,
            )
            batch_rows.append(row)
            _atomic_json(output_dir / "batch_rows.json", batch_rows)
            print(
                {
                    "event": "gradient_geometry_batch",
                    "checkpoint": label,
                    "batch": batch_index,
                    "effective_ratio": row[
                        "effective_core_to_nonmargin_gradient_ratio"
                    ],
                    "core_nonmargin_cosine": row["gradient_cosines"][
                        "core_margin__non_margin"
                    ],
                },
                flush=True,
            )
        if not pilot:
            candidate_banks[label] = _candidate_bank(
                bridge=bridge,
                receiver=receiver,
                dataset=dataset,
                receiver_inputs=receiver_inputs,
                positions=positions,
                scaffold=scaffold,
                site_scale=site_scale,
                layers=layers,
                batch_size=batch_size,
                boundary_positions=boundary_positions,
                device=device,
                use_amp=use_amp,
            )
            _atomic_json(output_dir / "candidate_banks.json", candidate_banks)
        del bridge, criterion, stored_statistics
        gc.collect()
        torch.cuda.empty_cache()

    hard_negative_agreement = None
    routing = None
    if not pilot:
        h011_rows = candidate_banks["H0_011"]["rows"]
        h013_rows = candidate_banks["H0_013"]["rows"]
        hard_negative_agreement = sum(
            left["global_hardest_task_id"] == right["global_hardest_task_id"]
            for left, right in zip(h011_rows, h013_rows)
        ) / len(h011_rows)
        routing = route_gradient_geometry(batch_rows, candidate_banks, config)

    result = {
        "experiment_id": "LIP-EVAL-032",
        "protocol_version": GRADIENT_GEOMETRY_PROTOCOL_VERSION,
        "claim_status": config["claim_status"],
        "stage": "pilot" if pilot else "full_evaluation",
        "run_commit": _git_head(),
        "complete": True,
        "provenance": {
            "experiment_config_path": str(experiment_path),
            "experiment_config_sha256": sha256_file(experiment_path),
            "parent_config_path": str(parent_path),
            "parent_config_sha256": sha256_file(parent_path),
            "bundle_manifest_sha256": sha256_file(bundle_dir / "manifest.json"),
            "checkpoints": provenance,
        },
        "bundle_validation": validation,
        "batch_partition": {
            "ordering": "task_id_ascending",
            "batch_size": batch_size,
            "batch_count_per_checkpoint": batch_count,
        },
        "batch_rows_path": str(output_dir / "batch_rows.json"),
        "candidate_banks_path": (
            str(output_dir / "candidate_banks.json") if not pilot else None
        ),
        "candidate_bank_summary": {
            label: {
                key: value
                for key, value in bank.items()
                if key != "rows"
            }
            for label, bank in candidate_banks.items()
        },
        "hard_negative_identity_agreement_H0_011_H0_013": hard_negative_agreement,
        "routing": routing,
        "decision_boundary": config["decision_boundary"],
        "telemetry": {
            **telemetry,
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
            "wall_seconds": float(time.perf_counter() - started),
            "colab_compute_units_before": colab_compute_units_before,
            "colab_compute_units_after": None,
            "colab_compute_units_consumed": None,
        },
    }
    _atomic_json(output_dir / "run_summary.json", result)
    return result
