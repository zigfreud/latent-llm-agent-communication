"""Render the predictive-to-functional capacity gap for LIP-PROTO-006/007."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_POSITION_SUMMARY = Path(
    "runs/LIP-PROTO-006/oracle-token-position/summary.json"
)
DEFAULT_FUNCTIONAL_SUMMARY = Path(
    "runs/LIP-PROTO-007/functional-evaluation/summary.json"
)
DEFAULT_OUTPUT_STEM = Path(
    "paper/figures/LIP-PROTO-007_predictive_functional_gap"
)
FUNCTIONAL_PACKET_SIZES = (8, 16, 32)


def load_summaries(
    position_path: Path, functional_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    with position_path.open("r", encoding="utf-8") as handle:
        position = json.load(handle)
    with functional_path.open("r", encoding="utf-8") as handle:
        functional = json.load(handle)
    if position.get("experiment_id") != "LIP-PROTO-006":
        raise ValueError("position summary must come from LIP-PROTO-006")
    if position.get("gate_window") != "first_8_tokens":
        raise ValueError("position summary must use the frozen first-eight-token gate")
    if functional.get("experiment_id") != "LIP-PROTO-007":
        raise ValueError("functional summary must come from LIP-PROTO-007")
    if functional.get("execution_mode") != "functional_hardened_namespace":
        raise ValueError("functional summary must come from the hardened evaluator")
    if not functional.get("claim_eligible"):
        raise ValueError("functional summary is not claim-eligible")
    return position, functional


def prefix_recovery(
    summary: Mapping[str, Any], *, split: str
) -> tuple[list[int], list[float]]:
    values = []
    for packet_size in FUNCTIONAL_PACKET_SIZES:
        value = summary["by_packet_size"][str(packet_size)][split]["windows"][
            "first_8_tokens"
        ]["recovery_fraction"]
        if value is None:
            raise ValueError(
                f"K={packet_size} {split} first-eight-token window is not informative"
            )
        values.append(float(value))
    return list(FUNCTIONAL_PACKET_SIZES), values


def functional_rates(
    summary: Mapping[str, Any], *, condition_prefix: str
) -> tuple[list[int], list[float], list[float], list[float]]:
    conditions = summary["metrics"]["functional_pass"]["conditions"]
    means = []
    lower_errors = []
    upper_errors = []
    for packet_size in FUNCTIONAL_PACKET_SIZES:
        condition = conditions[f"{condition_prefix}_k{packet_size}"]
        mean = float(condition["mean"])
        means.append(mean)
        lower_errors.append(mean - float(condition["ci_lower"]))
        upper_errors.append(float(condition["ci_upper"]) - mean)
    return list(FUNCTIONAL_PACKET_SIZES), means, lower_errors, upper_errors


def control_interval(
    summary: Mapping[str, Any], condition_name: str
) -> tuple[float, float, float]:
    condition = summary["metrics"]["functional_pass"]["conditions"][condition_name]
    return (
        float(condition["mean"]),
        float(condition["ci_lower"]),
        float(condition["ci_upper"]),
    )


def plot_summaries(
    position: Mapping[str, Any],
    functional: Mapping[str, Any],
    output_stem: Path,
    *,
    formats: Sequence[str] = ("svg", "pdf", "png"),
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("plotting requires matplotlib") from exc

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, (predictive_ax, functional_ax) = plt.subplots(
        1, 2, figsize=(7.15, 3.05), constrained_layout=True
    )

    split_styles = {
        "selection": {"color": "#0072B2", "marker": "o"},
        "confirmation": {"color": "#D55E00", "marker": "s"},
    }
    for split, style in split_styles.items():
        packet_sizes, recovery = prefix_recovery(position, split=split)
        predictive_ax.plot(
            packet_sizes,
            recovery,
            linewidth=2.0,
            markersize=5.0,
            label=split.capitalize(),
            **style,
        )
        for packet_size, value in zip(packet_sizes, recovery):
            predictive_ax.annotate(
                f"{value:.0%}",
                (packet_size, value),
                xytext=(0, 7 if split == "selection" else -12),
                textcoords="offset points",
                ha="center",
                color=style["color"],
                fontsize=7.0,
            )
    predictive_ax.axhline(0.0, color="#777777", linewidth=0.7)
    predictive_ax.axhline(1.0, color="#aaaaaa", linewidth=0.7, linestyle=":")
    predictive_ax.set_xscale("log", base=2)
    predictive_ax.set_xticks(FUNCTIONAL_PACKET_SIZES, labels=["8", "16", "32"])
    predictive_ax.set_ylim(-0.05, 1.05)
    predictive_ax.set_xlabel("Latent packet size K")
    predictive_ax.set_ylabel("Recovered text advantage")
    predictive_ax.set_title(
        "A  First-eight-token prediction (tasks 0:16)", loc="left"
    )
    predictive_ax.grid(axis="y", color="#e1e1e1", linewidth=0.6)
    predictive_ax.legend(frameon=False, loc="lower right")

    matched = functional_rates(functional, condition_prefix="oracle_packet")
    shuffled = functional_rates(
        functional, condition_prefix="shuffled_oracle_packet"
    )
    for values, label, color, marker, offset in (
        (matched, "Matched oracle", "#0072B2", "o", -0.35),
        (shuffled, "Task-shuffled", "#999999", "x", 0.35),
    ):
        packet_sizes, means, lower, upper = values
        functional_ax.errorbar(
            [size + offset for size in packet_sizes],
            means,
            yerr=[lower, upper],
            color=color,
            marker=marker,
            linewidth=1.6,
            markersize=5.0,
            capsize=2.5,
            label=label,
        )
    neutral_mean, _, _ = control_interval(functional, "neutral_no_lip")
    text_mean, text_low, text_high = control_interval(functional, "text_only_no_lip")
    functional_ax.axhline(
        neutral_mean,
        color="#555555",
        linewidth=0.9,
        linestyle=":",
        label="Neutral control",
    )
    functional_ax.axhspan(
        text_low, text_high, color="#009E73", alpha=0.12, linewidth=0
    )
    functional_ax.axhline(
        text_mean,
        color="#009E73",
        linewidth=1.3,
        linestyle="--",
        label=f"Text control ({text_mean:.0%})",
    )
    functional_ax.annotate(
        "0/48 at every K",
        (16, 0.0),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        color="#0072B2",
        fontsize=7.5,
    )
    functional_ax.set_xscale("log", base=2)
    functional_ax.set_xticks(FUNCTIONAL_PACKET_SIZES, labels=["8", "16", "32"])
    functional_ax.set_ylim(-0.05, 1.05)
    functional_ax.set_xlabel("Latent packet size K")
    functional_ax.set_ylabel("Task-clustered functional pass rate")
    functional_ax.set_title("B  Free execution (untouched tasks 16:32)", loc="left")
    functional_ax.grid(axis="y", color="#e1e1e1", linewidth=0.6)
    functional_ax.legend(frameon=False, loc="upper right")

    written = []
    for extension in formats:
        normalized = extension.lower().lstrip(".")
        if normalized not in {"svg", "pdf", "png"}:
            raise ValueError(f"unsupported figure format: {extension}")
        output_path = output_stem.with_suffix(f".{normalized}")
        save_options: dict[str, Any] = {"bbox_inches": "tight"}
        if normalized == "png":
            save_options["dpi"] = 300
        figure.savefig(output_path, **save_options)
        written.append(output_path)
    plt.close(figure)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--position-summary", type=Path, default=DEFAULT_POSITION_SUMMARY
    )
    parser.add_argument(
        "--functional-summary", type=Path, default=DEFAULT_FUNCTIONAL_SUMMARY
    )
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument(
        "--formats", nargs="+", default=["svg", "pdf", "png"], metavar="FORMAT"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    position, functional = load_summaries(
        args.position_summary, args.functional_summary
    )
    for output_path in plot_summaries(
        position, functional, args.output_stem, formats=args.formats
    ):
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
