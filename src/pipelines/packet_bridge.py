"""Bridge-only training over cached, validated LIP packet bundles."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.core.packet_bridge import (
    LIPPacketBridge,
    ReceiverPacketDecoder,
    SourcePacketEncoder,
    StructuredLinearPacketBridge,
)
from src.core.packet_bundle import (
    PacketRecordDataset,
    compute_target_packet_statistics,
    load_packet_records,
    sha256_file,
    validate_packet_bundle,
)
from src.core.packet_loss import (
    ComponentAwarePacketLoss,
    build_terminal_component_masks,
)
from src.evaluation.packet_bridge import (
    checkpoint_selection_key,
    summarize_packet_latent_metrics,
    summarize_replica_development_gate,
)


def _required(mapping, key, label):
    if not isinstance(mapping, dict) or key not in mapping:
        raise ValueError(f"missing required config field: {label}.{key}")
    return mapping[key]


def _positive_int(value, label):
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def set_packet_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_packet_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu, cuda, or auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def packet_collate(rows):
    return {
        "task_ids": [str(row["task_id"]) for row in rows],
        "source_packet": torch.stack([row["source_packet"] for row in rows]),
        "target_residual": torch.stack([row["target_residual"] for row in rows]),
        "name_token_count": torch.tensor(
            [row["name_token_count"] for row in rows], dtype=torch.long
        ),
    }


def build_packet_bridge(model_config, source_shape, target_shape):
    source_layers, source_positions, source_width = source_shape
    target_layers, target_positions, target_width = target_shape
    kind = _required(model_config, "kind", "model")
    if kind == "structured_linear":
        return StructuredLinearPacketBridge(
            source_width=source_width,
            source_layers=source_layers,
            source_positions=source_positions,
            target_width=target_width,
            target_layers=target_layers,
            target_positions=target_positions,
        )
    if kind != "query_conditioned":
        raise ValueError("model.kind must be structured_linear or query_conditioned")

    bridge_width = _positive_int(
        _required(model_config, "bridge_width", "model"), "model.bridge_width"
    )
    attention_heads = _positive_int(
        _required(model_config, "attention_heads", "model"), "model.attention_heads"
    )
    feedforward_width = _positive_int(
        _required(model_config, "feedforward_width", "model"),
        "model.feedforward_width",
    )
    encoder = SourcePacketEncoder(
        source_width=source_width,
        source_layers=source_layers,
        source_positions=source_positions,
        protocol_slots=_positive_int(
            _required(model_config, "protocol_slots", "model"),
            "model.protocol_slots",
        ),
        bridge_width=bridge_width,
        attention_heads=attention_heads,
        feedforward_width=feedforward_width,
        decoder_blocks=_positive_int(
            model_config.get("encoder_blocks", 2), "model.encoder_blocks"
        ),
        dropout=float(model_config.get("dropout", 0.1)),
    )
    decoder = ReceiverPacketDecoder(
        target_width=target_width,
        target_layers=target_layers,
        target_positions=target_positions,
        bridge_width=bridge_width,
        attention_heads=attention_heads,
        feedforward_width=feedforward_width,
        decoder_blocks=_positive_int(
            model_config.get("decoder_blocks", 2), "model.decoder_blocks"
        ),
        dropout=float(model_config.get("dropout", 0.1)),
    )
    return LIPPacketBridge(encoder, decoder)


def build_packet_loss(loss_config) -> ComponentAwarePacketLoss:
    return ComponentAwarePacketLoss(
        temperature=float(loss_config.get("temperature", 0.07)),
        margin_target=float(loss_config.get("margin_target", 0.05)),
        lambda_huber=float(loss_config.get("lambda_huber", 1.0)),
        lambda_cosine=float(loss_config.get("lambda_cosine", 0.25)),
        lambda_symmetric_nce=float(loss_config.get("lambda_symmetric_nce", 1.0)),
        lambda_margin=float(loss_config.get("lambda_margin", 0.1)),
        lambda_norm=float(loss_config.get("lambda_norm", 0.05)),
        component_weights=loss_config.get("component_weights"),
    )


def _make_loader(dataset, *, batch_size, shuffle, seed, num_workers=0):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle,
        pin_memory=False,
        collate_fn=packet_collate,
        generator=generator,
    )


def evaluate_packet_bridge(
    model,
    dataset,
    *,
    batch_size: int,
    device: torch.device,
    boundary_positions: int,
) -> dict:
    loader = _make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )
    predictions = []
    targets = []
    name_counts = []
    task_ids = []
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            source = batch["source_packet"].to(device)
            predictions.append(model(source).float().cpu())
            targets.append(batch["target_residual"].float().cpu())
            name_counts.append(batch["name_token_count"])
            task_ids.extend(batch["task_ids"])
    if was_training:
        model.train()
    prediction = torch.cat(predictions, dim=0)
    target = torch.cat(targets, dim=0)
    counts = torch.cat(name_counts, dim=0)
    masks = build_terminal_component_masks(
        counts,
        target_positions=target.shape[2],
        boundary_positions=boundary_positions,
    )
    return summarize_packet_latent_metrics(
        prediction,
        target,
        masks,
        task_ids=task_ids,
    )


def _json_ready_metrics(metrics):
    ready = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            ready[key] = float(value.detach().item())
        elif value is None or isinstance(value, (str, int, float, bool)):
            ready[key] = value
    return ready


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_packet_bridge(config_path: Path | str) -> dict:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("packet bridge config must contain a mapping")

    data_config = _required(config, "data", "config")
    training_config = _required(config, "training", "config")
    model_config = _required(config, "model", "config")
    loss_config = _required(config, "loss", "config")
    output_dir = Path(_required(config, "output_dir", "config"))
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(_required(config, "seed", "config"))
    set_packet_training_seed(seed)
    device = resolve_packet_device(str(config.get("device", "auto")))

    bundle_dir = Path(_required(data_config, "bundle_dir", "data"))
    validation = validate_packet_bundle(
        bundle_dir,
        require_real=bool(data_config.get("require_real", True)),
    )
    if validation["split_counts"]["confirmation"] != 0:
        raise ValueError("training bundle must not contain confirmation records")
    for split in ("train", "development_selection", "development_gate"):
        if validation["split_counts"][split] < 2:
            raise ValueError(f"packet training requires at least two {split} records")

    records = load_packet_records(bundle_dir)
    by_split = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "development_selection", "development_gate")
    }
    scaffold, site_scale = compute_target_packet_statistics(by_split["train"])
    statistics_path = output_dir / "target_statistics.pt"
    torch.save({"scaffold": scaffold, "site_scale": site_scale}, statistics_path)
    datasets = {
        split: PacketRecordDataset(
            split_records,
            scaffold=scaffold,
            site_scale=site_scale,
        )
        for split, split_records in by_split.items()
    }

    model = build_packet_bridge(
        model_config,
        tuple(validation["source_shape"]),
        tuple(validation["target_shape"]),
    ).to(device)
    criterion = build_packet_loss(loss_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(_required(training_config, "learning_rate", "training")),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
    )
    batch_size = _positive_int(
        _required(training_config, "batch_size", "training"), "training.batch_size"
    )
    if batch_size > len(datasets["train"]):
        raise ValueError("training.batch_size cannot exceed the training task count")
    max_updates = _positive_int(
        _required(training_config, "max_updates", "training"), "training.max_updates"
    )
    validation_interval = _positive_int(
        _required(training_config, "validation_interval", "training"),
        "training.validation_interval",
    )
    gradient_clip = float(training_config.get("gradient_clip", 1.0))
    boundary_positions = _positive_int(
        data_config.get("boundary_positions", 6), "data.boundary_positions"
    )
    use_amp = bool(training_config.get("fp16_autocast", True)) and device.type == "cuda"
    scaler = _grad_scaler(use_amp)
    loader = _make_loader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=int(training_config.get("num_workers", 0)),
    )

    best_key = None
    best_step = None
    history = []
    step = 0
    epoch = 0
    best_path = output_dir / "best_checkpoint.pt"
    while step < max_updates:
        epoch += 1
        for batch in loader:
            if step >= max_updates:
                break
            model.train()
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
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                prediction = model(source)
                loss_metrics = criterion(prediction, target, masks)
                loss = loss_metrics["total_loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            row = {
                "step": step,
                "epoch": epoch,
                "gradient_norm": float(gradient_norm),
                **_json_ready_metrics(loss_metrics),
            }
            history.append(row)

            should_validate = step % validation_interval == 0 or step == max_updates
            if should_validate:
                selection_metrics = evaluate_packet_bridge(
                    model,
                    datasets["development_selection"],
                    batch_size=batch_size,
                    device=device,
                    boundary_positions=boundary_positions,
                )
                key = checkpoint_selection_key(selection_metrics, step=step)
                row["development_selection"] = selection_metrics
                if best_key is None or key > best_key:
                    best_key = key
                    best_step = step
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "step": step,
                            "selection_key": list(key),
                            "selection_metrics": selection_metrics,
                            "source_shape": validation["source_shape"],
                            "target_shape": validation["target_shape"],
                            "model_config": model_config,
                        },
                        best_path,
                    )

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    gate_metrics = evaluate_packet_bridge(
        model,
        datasets["development_gate"],
        batch_size=batch_size,
        device=device,
        boundary_positions=boundary_positions,
    )
    gate_report = summarize_replica_development_gate(
        gate_metrics,
        alpha=float(config.get("development_gate", {}).get("alpha", 0.05)),
        statistics_seed=int(
            config.get("development_gate", {}).get("statistics_seed", 4481)
        ),
    )
    result = {
        "experiment_id": str(_required(config, "experiment_id", "config")),
        "seed": seed,
        "device": str(device),
        "model_kind": model_config["kind"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "updates_completed": step,
        "best_step": best_step,
        "best_selection_key": list(best_key),
        "bundle_validation": validation,
        "bundle_manifest_sha256": sha256_file(bundle_dir / "manifest.json"),
        "target_statistics_sha256": sha256_file(statistics_path),
        "development_selection": checkpoint["selection_metrics"],
        "development_gate_metrics": gate_metrics,
        "development_gate": gate_report,
        "checkpoint": str(best_path),
    }
    (output_dir / "train_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (output_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return result
