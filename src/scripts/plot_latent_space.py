import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

from src.core.models import LIPAdapter
from src.core.utils import get_device, set_seed
from src.pipelines.trainer import load_sharded_dataset
from torch.utils.data import DataLoader, Subset


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_output_dir(cfg: dict) -> Path:
    output_dir = cfg.get("output_dir", "checkpoints")
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT_DIR / output_dir
    return output_dir


def _load_model(cfg: dict, device: torch.device, output_dir: Path) -> LIPAdapter:
    model_cfg = cfg.get("model", {})
    model = LIPAdapter(
        input_dim=model_cfg.get("input_dim", 2048),
        hidden_dim=model_cfg.get("hidden_dim", 1024),
        output_dim=model_cfg.get("output_dim", 4096),
    )
    weights_path = output_dir / "adapter.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"adapter.pth not found at {weights_path}")
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _sample_dataset(dataset, num_samples: int, seed: int):
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check dataset_path.")
    if len(dataset) <= num_samples:
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=num_samples, replace=False)
    return Subset(dataset, indices.tolist())


def _collect_vectors(model, loader, device: torch.device):
    src_batches = []
    tgt_batches = []
    mapped_batches = []

    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            mapped = model(src)

            src_batches.append(src.detach().cpu().numpy())
            tgt_batches.append(tgt.detach().cpu().numpy())
            mapped_batches.append(mapped.detach().cpu().numpy())

    src_mat = np.concatenate(src_batches, axis=0)
    tgt_mat = np.concatenate(tgt_batches, axis=0)
    mapped_mat = np.concatenate(mapped_batches, axis=0)
    return src_mat, tgt_mat, mapped_mat


def _pad_to_dim(matrix: np.ndarray, target_dim: int) -> np.ndarray:
    if matrix.shape[1] == target_dim:
        return matrix
    if matrix.shape[1] > target_dim:
        return matrix[:, :target_dim]
    pad_width = target_dim - matrix.shape[1]
    return np.pad(matrix, ((0, 0), (0, pad_width)), mode="constant")


def _reduce_and_project(src_mat, tgt_mat, mapped_mat, seed: int, perplexity: float = 30.0):
    target_dim = max(src_mat.shape[1], tgt_mat.shape[1], mapped_mat.shape[1])
    src_mat = _pad_to_dim(src_mat, target_dim)
    tgt_mat = _pad_to_dim(tgt_mat, target_dim)
    mapped_mat = _pad_to_dim(mapped_mat, target_dim)

    combined = np.concatenate([src_mat, tgt_mat, mapped_mat], axis=0)

    pca = PCA(n_components=50, svd_solver="randomized", random_state=seed)
    combined_pca = pca.fit_transform(combined)

    total_points = combined_pca.shape[0]
    if total_points <= perplexity:
        perplexity = max(5, (total_points - 1) / 3)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    combined_tsne = tsne.fit_transform(combined_pca)

    n = src_mat.shape[0]
    src_tsne = combined_tsne[:n]
    tgt_tsne = combined_tsne[n:2 * n]
    mapped_tsne = combined_tsne[2 * n:3 * n]
    return src_tsne, tgt_tsne, mapped_tsne


def _plot_latent_space(src_tsne, tgt_tsne, mapped_tsne, output_path: Path):
    plt.style.use("seaborn-v0_8-paper")
    sns.set_context("paper")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    ax.scatter(
        src_tsne[:, 0],
        src_tsne[:, 1],
        c="#1f77b4",
        alpha=0.3,
        s=22,
        label="Source Original (DeepSeek)",
    )
    ax.scatter(
        tgt_tsne[:, 0],
        tgt_tsne[:, 1],
        c="#d62728",
        alpha=0.3,
        s=22,
        label="Target Real (Llama-3)",
    )
    ax.scatter(
        mapped_tsne[:, 0],
        mapped_tsne[:, 1],
        c="#2ca02c",
        alpha=0.9,
        s=36,
        marker="x",
        label="Mapped Source (LIP Injection)",
    )

    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot latent space alignment with PCA + t-SNE.")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--num_samples", type=int, default=800)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path
    cfg = _load_config(config_path)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))

    output_dir = _resolve_output_dir(cfg)
    model = _load_model(cfg, device, output_dir)

    dataset_path = Path(cfg["data"]["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = ROOT_DIR / dataset_path
    dataset = load_sharded_dataset(dataset_path)
    subset = _sample_dataset(dataset, args.num_samples, seed)

    batch_size = int(cfg.get("data", {}).get("batch_size", 16))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    src_mat, tgt_mat, mapped_mat = _collect_vectors(model, loader, device)
    src_tsne, tgt_tsne, mapped_tsne = _reduce_and_project(src_mat, tgt_mat, mapped_mat, seed)

    output_path = output_dir / "latent_space_tsne.png"
    _plot_latent_space(src_tsne, tgt_tsne, mapped_tsne, output_path)

    print(f"Saved latent space figure to {output_path}")


if __name__ == "__main__":
    main()
