import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(directory: Path):
    metrics_file = directory / "training_metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"training_metrics.json not found in {directory}")
    with metrics_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_metrics(metrics, directory: Path):
    epochs = [m["epoch"] for m in metrics]
    losses = [m["loss"] for m in metrics]
    accs = [m["acc"] for m in metrics]

    plt.style.use("default")
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for ax in (ax_loss, ax_acc):
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_facecolor("white")

    ax_loss.plot(epochs, losses, label="Hybrid Contrastive Loss", color="#1f77b4", linewidth=2)
    ax_loss.set_ylabel("Loss", fontsize=11)
    ax_loss.legend(loc="upper right", frameon=False)

    ax_acc.plot(epochs, accs, label="Top-1 Contrastive Acc", color="#d62728", linewidth=2)
    ax_acc.set_ylabel("Accuracy", fontsize=11)
    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.legend(loc="lower right", frameon=False)

    plt.tight_layout()
    output_path = directory / "training_convergence.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics.")
    parser.add_argument("--dir", required=True, help="Checkpoint directory containing training_metrics.json")
    args = parser.parse_args()

    directory = Path(args.dir)
    metrics = load_metrics(directory)
    output_path = plot_metrics(metrics, directory)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
