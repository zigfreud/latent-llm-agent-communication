import argparse
import csv
import glob
import json
import math
import os
import sys

import torch
import torch.optim as optim
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core.loss import HybridContrastiveLoss
from src.core.models import LIPAdapter
from src.core.utils import get_device, set_seed, setup_logger


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


def apply_overrides(cfg, experiment_id=None, output_dir=None, max_steps=None, device=None):
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

    return cfg


def unpack_loss_result(loss_result):
    if not isinstance(loss_result, tuple):
        raise TypeError("Loss function must return a tuple")

    if len(loss_result) == 4:
        loss, nce, mse, acc = loss_result
        return loss, nce, mse, acc

    if len(loss_result) == 2:
        loss, acc = loss_result
        return loss, None, None, acc

    raise ValueError(
        "Loss function must return either (loss, acc) or "
        "(total_loss, loss_nce, loss_mse, acc)"
    )


def tensor_item(value):
    if value is None:
        return None
    return value.item() if hasattr(value, "item") else float(value)


def add_optional_metric(total, value):
    return total + (tensor_item(value) or 0.0)


def write_run_artifacts(output_dir, cfg, metrics, rows):
    metrics_path = os.path.join(output_dir, "metrics.json")
    log_path = os.path.join(output_dir, "train_log.csv")
    summary_path = os.path.join(output_dir, "run_summary.md")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fieldnames = [
        "step",
        "epoch",
        "batch",
        "loss",
        "nce_loss",
        "mse_loss",
        "accuracy",
    ]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

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
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_loss(cfg):
    loss_cfg = cfg.get("loss", {})
    kwargs = {"temperature": loss_cfg.get("temperature", 0.07)}
    if "lambda_mse" in loss_cfg:
        kwargs["lambda_mse"] = loss_cfg["lambda_mse"]
    return HybridContrastiveLoss(**kwargs)


def train(config_path, experiment_id=None, output_dir=None, max_steps=None, device=None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, experiment_id, output_dir, max_steps, device)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    logger = setup_logger(cfg["output_dir"])
    device = get_device(cfg.get("device", "auto"))
    set_seed(cfg.get("seed", 42))

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
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        best_loss = checkpoint.get("best_loss", float("inf"))

    model.train()
    train_rows = []
    last_epoch_metrics = None

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_nce = 0.0
        epoch_mse = 0.0
        nce_count = 0
        mse_count = 0
        steps = 0

        for batch_idx, (src, tgt) in enumerate(loader):
            if max_steps is not None and global_step >= max_steps:
                break

            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src)
            loss, nce, mse, acc = unpack_loss_result(criterion(output, tgt))
            loss.backward()
            optimizer.step()

            loss_value = tensor_item(loss)
            nce_value = tensor_item(nce)
            mse_value = tensor_item(mse)
            acc_value = tensor_item(acc)

            epoch_loss += loss_value
            epoch_nce = add_optional_metric(epoch_nce, nce)
            epoch_mse = add_optional_metric(epoch_mse, mse)
            nce_count += int(nce is not None)
            mse_count += int(mse is not None)
            epoch_acc += acc_value
            steps += 1
            global_step += 1

            train_rows.append(
                {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                    "loss": loss_value,
                    "nce_loss": nce_value,
                    "mse_loss": mse_value,
                    "accuracy": acc_value,
                }
            )

        if steps == 0:
            if max_steps is not None and global_step >= max_steps:
                break
            continue

        avg_loss = epoch_loss / steps
        avg_nce = epoch_nce / nce_count if nce_count else None
        avg_mse = epoch_mse / mse_count if mse_count else None
        avg_acc = epoch_acc / steps
        last_epoch_metrics = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "nce_loss": avg_nce,
            "mse_loss": avg_mse,
            "accuracy": avg_acc,
        }
        logger.info(f"Ep {epoch+1:03d} | Loss: {avg_loss:.4f} | Acc: {avg_acc*100:.2f}%")

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
        "dataset_path": cfg["data"]["dataset_path"],
        "samples": len(dataset),
        "batch_size": cfg["data"]["batch_size"],
        "epochs_requested": cfg["training"]["epochs"],
        "max_steps": max_steps,
        "steps_completed": global_step,
        "best_loss": best_loss if math.isfinite(best_loss) else None,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
    }
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
    args = parser.parse_args()
    train(
        args.config,
        experiment_id=args.experiment_id,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        device=args.device,
    )
