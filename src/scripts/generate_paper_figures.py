import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 300,
    }
)


def plot_rag_latency_comparison(csv_path, output_dir):
    """Generate Figure 3: end-to-end latency comparison for heavy-context RAG."""
    df = pd.read_csv(csv_path)

    df["total_latency_ms"] = df["total_latency"] / 1e6
    df["scenario"] = df["scenario"].replace(
        {
            "JsonBaselineScenario": "JSON (Baseline)",
            "LipProtocolScenario": "LIP (Ours)",
        }
    )

    rag_data = df[df["prompt_len"] > 1000].copy()
    if rag_data.empty:
        print("Warning: no RAG data found to plot.")
        return

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=rag_data,
        x="scenario",
        y="total_latency_ms",
        hue="scenario",
        palette=["#e74c3c", "#2ecc71"],
        edgecolor=".2",
    )

    means = rag_data.groupby("scenario")["total_latency_ms"].mean()
    json_mean = means["JSON (Baseline)"]
    lip_mean = means["LIP (Ours)"]
    speedup = json_mean / lip_mean

    plt.title(f"End-to-End Latency: Heavy Context RAG (Speedup: {speedup:.1f}x)")
    plt.ylabel("Latency (ms)")
    plt.xlabel("")

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "fig_rag_latency.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")


def plot_component_breakdown(csv_path, output_dir):
    """Generate Figure 4: latency breakdown by component."""
    df = pd.read_csv(csv_path)
    df = df[df["prompt_len"] > 1000].copy()

    metrics = ["source_processing_time", "transport_overhead", "target_ingestion_time"]
    df_melt = df.groupby("scenario")[metrics].mean().reset_index()

    for metric in metrics:
        df_melt[metric] /= 1e6

    df_melt["scenario"] = df_melt["scenario"].replace(
        {
            "JsonBaselineScenario": "JSON",
            "LipProtocolScenario": "LIP",
        }
    )

    df_melt.set_index("scenario").plot(
        kind="bar",
        stacked=True,
        color=["#3498db", "#95a5a6", "#f1c40f"],
        figsize=(8, 5),
    )

    plt.title("Latency Breakdown by Component")
    plt.ylabel("Time (ms)")
    plt.xticks(rotation=0)
    plt.legend(["Source Compute", "Transport/Network", "Target Ingestion"])

    save_path = os.path.join(output_dir, "fig_latency_breakdown.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot: {save_path}")


if __name__ == "__main__":
    csv_path = "experiments/logs/benchmark_results_gen6.csv"
    logs_dir = "experiments/logs"
    output_dir = "paper/figures"

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print("Place the CSV file in experiments/logs/ before running.")
    else:
        plot_rag_latency_comparison(csv_path, output_dir)
        plot_component_breakdown(csv_path, output_dir)
