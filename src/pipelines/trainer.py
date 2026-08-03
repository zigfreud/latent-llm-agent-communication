import argparse
import csv
import glob
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core.loss import HybridContrastiveLoss
from src.core.models import LIPAdapter
from src.core.utils import get_device, set_seed, setup_logger
from src.scripts.validate_latent_bundle import read_manifest, validate_bundle


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_bundle_provenance(cfg):
    data_config = cfg.get("data", {})
    if not data_config.get("require_bundle_manifest", False):
        return None
    shards_path = os.path.abspath(data_config["dataset_path"])
    bundle_dir = os.path.dirname(shards_path)
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"required training manifest not found: {manifest_path}")
    report = validate_bundle(Path(bundle_dir))
    manifest = read_manifest(Path(bundle_dir))
    expected_trace_id = data_config.get("expected_trace_id")
    if not expected_trace_id:
        raise ValueError("data.expected_trace_id is required with bundle validation")
    if report["trace_id"] != expected_trace_id:
        raise ValueError(
            f"training bundle trace_id={report['trace_id']} != {expected_trace_id}"
        )
    expected_mode = data_config.get("expected_extraction_mode")
    if expected_mode and manifest.get("extraction_mode") != expected_mode:
        raise ValueError(
            f"training bundle extraction_mode={manifest.get('extraction_mode')} "
            f"!= {expected_mode}"
        )
    if report["input_dim"] != int(cfg["model"]["input_dim"]):
        raise ValueError("training bundle input_dim does not match model.input_dim")
    if report["output_dim"] != int(cfg["model"]["output_dim"]):
        raise ValueError("training bundle output_dim does not match model.output_dim")
    return {
        "path": manifest_path,
        "sha256": sha256_file(manifest_path),
        "trace_id": report["trace_id"],
        "records": report["total_records"],
        "extraction_mode": manifest.get("extraction_mode"),
    }


def load_pt_shard(filepath):
    try:
        return torch.load(filepath, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This trainer requires a PyTorch version supporting "
            "torch.load(..., weights_only=True) for shard loading."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to safely load shard {filepath} with weights_only=True. "
            "Shards must be saved in a weights_only-compatible tensor/list/dict format."
        ) from exc


class ShardDataset(Dataset):
    def __init__(self, filepath):
        data = load_pt_shard(filepath)
        if not isinstance(data, list):
            raise TypeError(f"{filepath} must load as a list of records")
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["src_vector"].squeeze().float(), item["tgt_vector"].squeeze().float()


def load_sharded_dataset(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.pt")))

    if not files:
        raise FileNotFoundError(f"No shard (.pt) files found in {directory}")

    print(f"Indexing {len(files)} shards...")
    valid_files = [f for f in files if os.path.getsize(f) > 0]
    if not valid_files:
        raise FileNotFoundError(f"No non-empty shard (.pt) files found in {directory}")
    return ConcatDataset([ShardDataset(f) for f in valid_files])


def apply_overrides(
    cfg,
    experiment_id=None,
    output_dir=None,
    max_steps=None,
    device=None,
    seed=None,
):
    if experiment_id:
        cfg["experiment_id"] = experiment_id
        cfg["experiment_name"] = experiment_id
    else:
        cfg["experiment_id"] = cfg.get("experiment_id") or cfg.get(
            "experiment_name",
            "experiment",
        )

    if output_dir:
        cfg["output_dir"] = output_dir

    if device:
        cfg["device"] = device

    if max_steps is not None:
        cfg.setdefault("training", {})["max_steps"] = max_steps

    if seed is not None:
        cfg["seed"] = int(seed)

    return cfg


LOSS_METRIC_FIELDS = [
    "loss",
    "nce_loss",
    "forward_nce_loss",
    "reverse_nce_loss",
    "mse_loss",
    "margin_loss",
    "norm_loss",
    "cosine_diag_mean",
    "offdiag_cosine_mean",
    "diagonal_margin_mean",
    "hard_negative_cosine_mean",
    "norm_ratio_mean",
    "accuracy",
]


def unpack_loss_result(loss_result):
    if isinstance(loss_result, dict):
        loss = loss_result.get("total_loss", loss_result.get("loss"))
        if loss is None:
            raise ValueError("Loss result dictionary must include total_loss or loss")

        metrics = {
            "loss": loss,
            "nce_loss": loss_result.get("nce_loss", loss_result.get("forward_nce_loss")),
            "forward_nce_loss": loss_result.get("forward_nce_loss", loss_result.get("nce_loss")),
            "reverse_nce_loss": loss_result.get("reverse_nce_loss"),
            "mse_loss": loss_result.get("mse_loss"),
            "margin_loss": loss_result.get("margin_loss"),
            "norm_loss": loss_result.get("norm_loss"),
            "cosine_diag_mean": loss_result.get("cosine_diag_mean"),
            "offdiag_cosine_mean": loss_result.get("offdiag_cosine_mean"),
            "diagonal_margin_mean": loss_result.get("diagonal_margin_mean"),
            "hard_negative_cosine_mean": loss_result.get("hard_negative_cosine_mean"),
            "norm_ratio_mean": loss_result.get("norm_ratio_mean"),
            "accuracy": loss_result.get("accuracy", loss_result.get("acc")),
        }
        return loss, metrics

    if not isinstance(loss_result, tuple):
        raise TypeError("Loss function must return a tuple or dictionary")

    if len(loss_result) == 4:
        loss, nce, mse, acc = loss_result
        return loss, {
            "loss": loss,
            "nce_loss": nce,
            "forward_nce_loss": nce,
            "mse_loss": mse,
            "accuracy": acc,
        }

    if len(loss_result) == 2:
        loss, acc = loss_result
        return loss, {
            "loss": loss,
            "accuracy": acc,
        }

    raise ValueError(
        "Loss function must return a metrics dictionary, (loss, acc), or "
        "(total_loss, loss_nce, loss_mse, acc)"
    )


def tensor_item(value):
    if value is None:
        return None
    return value.item() if hasattr(value, "item") else float(value)


def write_run_artifacts(output_dir, cfg, metrics, rows):
    metrics_path = os.path.join(output_dir, "metrics.json")
    log_path = os.path.join(output_dir, "train_log.csv")
    summary_path = os.path.join(output_dir, "run_summary.md")
    resolved_config_path = os.path.join(output_dir, "resolved_config.yaml")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fieldnames = ["step", "epoch", "batch"] + LOSS_METRIC_FIELDS
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(resolved_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    best_loss = metrics["best_loss"]
    final_loss = metrics["final_loss"]
    final_accuracy = metrics["final_accuracy"]
    lines = [
        f"# {cfg['experiment_id']} Training Run",
        "",
        f"- Experiment: {cfg['experiment_id']}",
        f"- Output directory: {output_dir}",
        f"- Device: {metrics['device']}",
        f"- Dataset path: {metrics['dataset_path']}",
        f"- Samples: {metrics['samples']}",
        f"- Steps completed: {metrics['steps_completed']}",
        f"- Best loss: {best_loss:.6f}" if best_loss is not None else "- Best loss: n/a",
        f"- Final loss: {final_loss:.6f}" if final_loss is not None else "- Final loss: n/a",
        f"- Final accuracy: {final_accuracy:.6f}" if final_accuracy is not None else "- Final accuracy: n/a",
        "",
    ]
    optional_summary_fields = [
        ("Final reverse NCE loss", "final_reverse_nce_loss"),
        ("Final margin loss", "final_margin_loss"),
        ("Final norm loss", "final_norm_loss"),
        ("Final diagonal margin mean", "final_diagonal_margin_mean"),
        ("Final hard negative cosine mean", "final_hard_negative_cosine_mean"),
        ("Final norm ratio mean", "final_norm_ratio_mean"),
    ]
    for label, key in optional_summary_fields:
        value = metrics.get(key)
        if value is not None:
            lines.insert(-1, f"- {label}: {value:.6f}")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_loss(cfg):
    loss_cfg = cfg.get("loss", {})
    kwargs = {"temperature": loss_cfg.get("temperature", 0.07)}
    for key in (
        "lambda_mse",
        "lambda_reverse_nce",
        "reverse_nce_weight",
        "lambda_margin",
        "margin_target",
        "lambda_norm",
    ):
        if key in loss_cfg:
            kwargs[key] = loss_cfg[key]
    return HybridContrastiveLoss(**kwargs)


def train(
    config_path,
    experiment_id=None,
    output_dir=None,
    max_steps=None,
    device=None,
    seed=None,
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, experiment_id, output_dir, max_steps, device, seed)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    logger = setup_logger(cfg["output_dir"])
    device = get_device(cfg.get("device", "auto"))
    set_seed(cfg.get("seed", 42))
    bundle_provenance = validate_training_bundle_provenance(cfg)

    logger.info(f"Starting training: {cfg['experiment_id']}")
    logger.info(f"Device: {device}")

    dataset = load_sharded_dataset(cfg["data"]["dataset_path"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=(getattr(device, "type", str(device)) == "cuda"),
        drop_last=cfg["data"].get("drop_last", True),
    )
    logger.info(f"Total samples: {len(dataset)}")

    model = LIPAdapter(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"].get("hidden_dim", 1024),
        output_dim=cfg["model"]["output_dim"],
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=cfg["training"].get("weight_decay", 0.01),
    )

    criterion = build_loss(cfg)

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    ckpt_path = os.path.join(cfg["output_dir"], "last_checkpoint.pth")
    max_steps = cfg["training"].get("max_steps")
    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

    if cfg["training"].get("resume", True) and os.path.exists(ckpt_path):
        logger.info(f"Resuming from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        best_loss = checkpoint.get("best_loss", float("inf"))

    model.train()
    train_rows = []
    last_epoch_metrics = None

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        epoch_sums = {field: 0.0 for field in LOSS_METRIC_FIELDS}
        epoch_counts = {field: 0 for field in LOSS_METRIC_FIELDS}
        steps = 0

        for batch_idx, (src, tgt) in enumerate(loader):
            if max_steps is not None and global_step >= max_steps:
                break

            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src)
            loss, batch_metrics = unpack_loss_result(criterion(output, tgt))
            loss.backward()
            optimizer.step()

            metric_values = {
                field: tensor_item(batch_metrics.get(field))
                for field in LOSS_METRIC_FIELDS
            }
            metric_values["loss"] = tensor_item(loss)

            for field, value in metric_values.items():
                if value is not None:
                    epoch_sums[field] += value
                    epoch_counts[field] += 1

            steps += 1
            global_step += 1

            row = {
                "step": global_step,
                "epoch": epoch + 1,
                "batch": batch_idx + 1,
            }
            row.update(metric_values)
            train_rows.append(row)

        if steps == 0:
            if max_steps is not None and global_step >= max_steps:
                break
            continue

        avg_metrics = {
            field: (epoch_sums[field] / epoch_counts[field] if epoch_counts[field] else None)
            for field in LOSS_METRIC_FIELDS
        }
        avg_loss = avg_metrics["loss"]
        avg_acc = avg_metrics["accuracy"]
        last_epoch_metrics = {
            "epoch": epoch + 1,
            **avg_metrics,
        }
        acc_pct = (avg_acc * 100.0) if avg_acc is not None else 0.0
        logger.info(f"Ep {epoch+1:03d} | Loss: {avg_loss:.4f} | Acc: {acc_pct:.2f}%")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(cfg["output_dir"], "best_model.pth"))
            logger.info("New best model saved.")

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_loss": best_loss,
                "config": cfg,
                "metrics": last_epoch_metrics,
            },
            ckpt_path,
        )

        if max_steps is not None and global_step >= max_steps:
            break

    if global_step == 0:
        raise RuntimeError("Training completed zero steps; check dataset size, batch_size, and drop_last.")

    final_loss = last_epoch_metrics["loss"] if last_epoch_metrics else None
    final_accuracy = last_epoch_metrics["accuracy"] if last_epoch_metrics else None
    metrics = {
        "experiment_id": cfg["experiment_id"],
        "output_dir": cfg["output_dir"],
        "device": str(device),
        "seed": int(cfg.get("seed", 42)),
        "dataset_path": cfg["data"]["dataset_path"],
        "samples": len(dataset),
        "batch_size": cfg["data"]["batch_size"],
        "epochs_requested": cfg["training"]["epochs"],
        "max_steps": max_steps,
        "steps_completed": global_step,
        "best_loss": best_loss if math.isfinite(best_loss) else None,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "dataset_manifest": bundle_provenance,
    }
    if last_epoch_metrics:
        for field in LOSS_METRIC_FIELDS:
            metrics[f"final_{field}"] = last_epoch_metrics.get(field)

    write_run_artifacts(cfg["output_dir"], cfg, metrics, train_rows)

    logger.info("Process completed.")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    train(
        args.config,
        experiment_id=args.experiment_id,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        device=args.device,
        seed=args.seed,
    )
