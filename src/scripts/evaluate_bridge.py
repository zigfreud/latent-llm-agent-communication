import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from src.core.models import LIPAdapter
from src.scripts.validate_latent_bundle import load_shard, validate_bundle


DEFAULT_CONFIG = Path("config/LIP-EVAL-001_bridge_eval.yaml")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LIPAdapter against a validated latent bundle."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("config must contain a YAML mapping")

    return config


def resolve_path(cli_value, config, *keys):
    if cli_value is not None:
        return cli_value

    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"config missing required field: {'.'.join(keys)}")
        value = value[key]

    return Path(value)


def resolve_device(config):
    requested = config.get("device", "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("config requested device=cuda, but CUDA is not available")
        return torch.device("cuda")
    raise ValueError("device must be cpu or cuda")


def positive_int(value, name):
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def model_dims(config, validation_report):
    model_config = config.get("model", {})
    input_dim = positive_int(
        model_config.get("input_dim", validation_report["input_dim"]),
        "model.input_dim",
    )
    hidden_dim = positive_int(model_config.get("hidden_dim", 1024), "model.hidden_dim")
    output_dim = positive_int(
        model_config.get("output_dim", validation_report["output_dim"]),
        "model.output_dim",
    )

    if input_dim != validation_report["input_dim"]:
        raise ValueError(
            f"model.input_dim={input_dim} does not match bundle input_dim="
            f"{validation_report['input_dim']}"
        )
    if output_dim != validation_report["output_dim"]:
        raise ValueError(
            f"model.output_dim={output_dim} does not match bundle output_dim="
            f"{validation_report['output_dim']}"
        )

    return input_dim, hidden_dim, output_dim


def load_checkpoint_safely(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This evaluator requires a PyTorch version supporting "
            "torch.load(..., weights_only=True) for checkpoint loading."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to safely load checkpoint {checkpoint_path} with "
            "weights_only=True."
        ) from exc

    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must load as a state_dict or checkpoint dictionary")

    state_dict = checkpoint.get("model_state", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint['model_state'] must be a state_dict when present")

    return state_dict


def load_bundle_tensors(bundle_dir, validation_report):
    src_vectors = []
    tgt_vectors = []
    for shard in validation_report["shards"]:
        shard_path = bundle_dir.joinpath(*Path(shard["path"]).parts)
        records = load_shard(shard_path)
        for record in records:
            src_vectors.append(record["src_vector"].squeeze().float())
            tgt_vectors.append(record["tgt_vector"].squeeze().float())

    if not src_vectors:
        raise RuntimeError("validated bundle contained zero records")

    return torch.stack(src_vectors), torch.stack(tgt_vectors)


def to_number(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return float(value)


def retrieval_metrics(cosine_matrix, requested_topk):
    sample_count = cosine_matrix.shape[0]
    labels = torch.arange(sample_count, device=cosine_matrix.device)
    nearest = torch.argmax(cosine_matrix, dim=1)
    top1_hits = nearest == labels
    top1 = top1_hits.float().mean().item()

    topk = {}
    for k in requested_topk:
        if not isinstance(k, int) or k <= 0:
            raise ValueError("metrics.retrieval_topk entries must be positive integers")
        if k <= sample_count:
            topk_indices = torch.topk(cosine_matrix, k=k, dim=1).indices
            hits = (topk_indices == labels.unsqueeze(1)).any(dim=1)
            topk[str(k)] = hits.float().mean().item()

    return nearest, top1_hits, top1, topk


def compute_metrics(predictions, targets, requested_topk):
    sample_count = predictions.shape[0]
    mse_per_sample = torch.mean((predictions - targets) ** 2, dim=1)
    rmse_per_sample = torch.sqrt(mse_per_sample)
    cosine_matrix = F.normalize(predictions, p=2, dim=1) @ F.normalize(targets, p=2, dim=1).T
    cosine_diag = torch.diag(cosine_matrix)
    prediction_norm = torch.linalg.vector_norm(predictions, dim=1)
    target_norm = torch.linalg.vector_norm(targets, dim=1)
    norm_ratio = prediction_norm / target_norm.clamp_min(1e-12)
    energy_drift = torch.abs(prediction_norm - target_norm)
    nearest, top1_hits, retrieval_top1, retrieval_topk = retrieval_metrics(
        cosine_matrix,
        requested_topk,
    )

    offdiag_cosine_mean = None
    diagonal_margin = None
    if sample_count > 1:
        offdiag_mask = ~torch.eye(sample_count, dtype=torch.bool, device=cosine_matrix.device)
        offdiag_values = cosine_matrix[offdiag_mask]
        offdiag_cosine_mean = offdiag_values.mean()
        offdiag_by_row = cosine_matrix.masked_fill(~offdiag_mask, float("-inf"))
        diagonal_margin = cosine_diag - torch.max(offdiag_by_row, dim=1).values

    metrics = {
        "sample_count": sample_count,
        "latent_mse_mean": mse_per_sample.mean().item(),
        "latent_rmse_mean": rmse_per_sample.mean().item(),
        "cosine_diag_mean": cosine_diag.mean().item(),
        "cosine_diag_std": cosine_diag.std(unbiased=False).item(),
        "prediction_norm_mean": prediction_norm.mean().item(),
        "target_norm_mean": target_norm.mean().item(),
        "norm_ratio_mean": norm_ratio.mean().item(),
        "energy_drift_mean": energy_drift.mean().item(),
        "retrieval_top1": retrieval_top1,
        "retrieval_topk": retrieval_topk,
        "offdiag_cosine_mean": to_number(offdiag_cosine_mean),
        "diagonal_margin_mean": to_number(diagonal_margin.mean() if diagonal_margin is not None else None),
    }
    pair_rows = []
    for index in range(sample_count):
        margin_value = diagonal_margin[index].item() if diagonal_margin is not None else None
        pair_rows.append(
            {
                "sample_index": index,
                "latent_mse": mse_per_sample[index].item(),
                "latent_rmse": rmse_per_sample[index].item(),
                "cosine_diag": cosine_diag[index].item(),
                "prediction_norm": prediction_norm[index].item(),
                "target_norm": target_norm[index].item(),
                "norm_ratio": norm_ratio[index].item(),
                "energy_drift": energy_drift[index].item(),
                "nearest_target_index": int(nearest[index].item()),
                "retrieval_top1_hit": bool(top1_hits[index].item()),
                "diagonal_margin": margin_value,
            }
        )

    return metrics, pair_rows


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_pairs_csv(path, pair_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "latent_mse",
        "latent_rmse",
        "cosine_diag",
        "prediction_norm",
        "target_norm",
        "norm_ratio",
        "energy_drift",
        "nearest_target_index",
        "retrieval_top1_hit",
        "diagonal_margin",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pair_rows)


def write_summary(path, experiment_id, checkpoint_path, bundle_dir, metrics):
    lines = [
        f"# {experiment_id} Bridge Evaluation",
        "",
        f"- Checkpoint: {checkpoint_path}",
        f"- Bundle directory: {bundle_dir}",
        f"- Samples: {metrics['sample_count']}",
        f"- Input dim: {metrics['input_dim']}",
        f"- Output dim: {metrics['output_dim']}",
        f"- Latent MSE mean: {metrics['latent_mse_mean']:.6f}",
        f"- Latent RMSE mean: {metrics['latent_rmse_mean']:.6f}",
        f"- Cosine diagonal mean: {metrics['cosine_diag_mean']:.6f}",
        f"- Retrieval top-1: {metrics['retrieval_top1']:.6f}",
        "",
        "This evaluation reports latent-space bridge metrics only. It does not claim",
        "semantic transfer, text-level fidelity, model-to-model alignment, or production readiness.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def evaluate(config_path, checkpoint_override=None, bundle_dir_override=None, output_dir_override=None):
    config = load_config(config_path)
    bundle_dir = resolve_path(bundle_dir_override, config, "data", "bundle_dir")
    checkpoint_path = resolve_path(checkpoint_override, config, "checkpoint", "path")
    output_dir = resolve_path(output_dir_override, config, "output_dir")
    experiment_id = config.get("experiment_id", "LIP-EVAL-001")
    device = resolve_device(config)

    validation_report = validate_bundle(bundle_dir)
    input_dim, hidden_dim, output_dim = model_dims(config, validation_report)
    sources, targets = load_bundle_tensors(bundle_dir, validation_report)

    model = LIPAdapter(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )
    model.load_state_dict(load_checkpoint_safely(checkpoint_path))
    model.to(device)
    model.eval()

    with torch.no_grad():
        predictions = model(sources.to(device)).cpu()

    requested_topk = config.get("metrics", {}).get("retrieval_topk", [1])
    metrics, pair_rows = compute_metrics(predictions, targets, requested_topk)
    metrics.update(
        {
            "experiment_id": experiment_id,
            "checkpoint": str(checkpoint_path),
            "bundle_dir": str(bundle_dir),
            "device": str(device),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "validation_status": validation_report["validation_status"],
            "bundle_validation": validation_report,
        }
    )

    write_json(output_dir / "eval_metrics.json", metrics)
    write_pairs_csv(output_dir / "eval_pairs.csv", pair_rows)
    write_summary(
        output_dir / "eval_summary.md",
        experiment_id,
        checkpoint_path,
        bundle_dir,
        metrics,
    )

    return metrics


def main():
    args = parse_args()
    metrics = evaluate(
        args.config,
        checkpoint_override=args.checkpoint,
        bundle_dir_override=args.bundle_dir,
        output_dir_override=args.output_dir,
    )
    print("Bridge evaluation completed")
    print(f"sample_count: {metrics['sample_count']}")
    print(f"latent_mse_mean: {metrics['latent_mse_mean']:.6f}")
    print(f"cosine_diag_mean: {metrics['cosine_diag_mean']:.6f}")
    print(f"retrieval_top1: {metrics['retrieval_top1']:.6f}")


if __name__ == "__main__":
    main()
