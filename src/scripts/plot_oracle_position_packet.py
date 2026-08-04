"""Render functional and positional capacity for LIP-PROTO-011."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.oracle_position_packet import (
    ORACLE_POSITION_EXPERIMENT_ID,
    ORACLE_POSITION_PATTERN_CONTRACT,
    ORACLE_POSITION_PATTERN_ORDER,
    ORACLE_POSITION_SCOPE_NAME,
)


DEFAULT_SUMMARY = Path("runs/LIP-PROTO-011/functional-evaluation/summary.json")
DEFAULT_OUTPUT_STEM = Path("paper/figures/LIP-PROTO-011_position_sparse_packet")
PATTERN_LABELS = {
    "full_k32": "Full K=32",
    "diagnostic_top_k8": "Diagnostic top K=8",
    "peak_window_k8": "Peak window K=8",
    "suffix_k8": "Suffix K=8",
}


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("experiment_id") != ORACLE_POSITION_EXPERIMENT_ID:
        raise ValueError("summary must come from LIP-PROTO-011")
    if summary.get("execution_mode") != "functional_hardened_namespace":
        raise ValueError("summary must come from the hardened evaluator")
    if not summary.get("claim_eligible"):
        raise ValueError("position-sparse figure requires a claim-eligible full run")
    return summary


def pattern_rates(
    summary: Mapping[str, Any],
    *,
    shuffled: bool = False,
) -> tuple[list[float], list[float], list[float]]:
    conditions = summary["metrics"]["functional_pass"]["conditions"]
    means = []
    lower_errors = []
    upper_errors = []
    prefix = "shuffled_oracle" if shuffled else "oracle"
    for pattern_name in ORACLE_POSITION_PATTERN_ORDER:
        condition = conditions[
            f"{prefix}_{ORACLE_POSITION_SCOPE_NAME}_{pattern_name}"
        ]
        mean = float(condition["mean"])
        means.append(mean)
        lower_errors.append(mean - float(condition["ci_lower"]))
        upper_errors.append(float(condition["ci_upper"]) - mean)
    return means, lower_errors, upper_errors


def primary_annotations(summary: Mapping[str, Any]) -> list[str]:
    by_treatment = {
        str(item["treatment"]): item
        for item in summary["primary_inference"]["hypotheses"]
    }
    annotations = []
    for pattern_name in ORACLE_POSITION_PATTERN_ORDER:
        treatment = f"oracle_{ORACLE_POSITION_SCOPE_NAME}_{pattern_name}"
        item = by_treatment[treatment]
        annotations.append(
            f"p={float(item['p_value']):.3g}"
            if item["tested"]
            else "gate stopped"
        )
    return annotations


def position_matrix() -> list[list[int]]:
    offsets = list(range(-32, 0))
    patterns = {
        pattern["name"]: set(pattern["packet_offsets"])
        for pattern in ORACLE_POSITION_PATTERN_CONTRACT
    }
    return [
        [int(offset in patterns[pattern_name]) for offset in offsets]
        for pattern_name in ORACLE_POSITION_PATTERN_ORDER
    ]


def plot_summary(
    summary: Mapping[str, Any],
    output_stem: Path,
    *,
    formats: Sequence[str] = ("svg", "pdf", "png"),
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
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
    figure, (rate_axis, position_axis) = plt.subplots(
        2,
        1,
        figsize=(5.5, 4.8),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )
    x_values = list(range(len(ORACLE_POSITION_PATTERN_ORDER)))
    for values, label, color, marker, offset in (
        (pattern_rates(summary), "Matched replay", "#0072B2", "o", -0.08),
        (
            pattern_rates(summary, shuffled=True),
            "Task-shuffled replay",
            "#888888",
            "x",
            0.08,
        ),
    ):
        means, lower, upper = values
        rate_axis.errorbar(
            [x + offset for x in x_values],
            means,
            yerr=[lower, upper],
            color=color,
            marker=marker,
            linewidth=1.7,
            markersize=5.2,
            capsize=2.5,
            label=label,
        )
    conditions = summary["metrics"]["functional_pass"]["conditions"]
    neutral = float(conditions["neutral_no_lip"]["mean"])
    text = conditions["text_only_no_lip"]
    text_mean = float(text["mean"])
    rate_axis.axhline(
        neutral,
        color="#444444",
        linewidth=0.9,
        linestyle=":",
        label="Neutral control",
    )
    rate_axis.axhspan(
        float(text["ci_lower"]),
        float(text["ci_upper"]),
        color="#009E73",
        alpha=0.12,
        linewidth=0,
    )
    rate_axis.axhline(
        text_mean,
        color="#009E73",
        linewidth=1.3,
        linestyle="--",
        label=f"Task text ({text_mean:.0%})",
    )
    matched_means, _, _ = pattern_rates(summary)
    for x_value, mean, annotation in zip(
        x_values,
        matched_means,
        primary_annotations(summary),
    ):
        rate_axis.annotate(
            annotation,
            (x_value, mean),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=6.8,
            color="#005A8C",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
        )
    rate_axis.set_xticks(
        x_values,
        [PATTERN_LABELS[name] for name in ORACLE_POSITION_PATTERN_ORDER],
        rotation=12,
        ha="right",
    )
    rate_axis.set_xlim(-0.45, len(x_values) - 0.55)
    rate_axis.set_ylim(-0.05, 1.05)
    rate_axis.set_ylabel("Task-clustered functional pass rate")
    rate_axis.set_title(
        "Functional capacity of position-sparse latent packets",
        loc="left",
    )
    rate_axis.grid(axis="y", color="#e1e1e1", linewidth=0.6)
    rate_axis.legend(
        frameon=False,
        loc="center",
        bbox_to_anchor=(0.5, 0.40),
        ncol=2,
        columnspacing=1.0,
        handlelength=2.0,
    )

    position_axis.imshow(
        position_matrix(),
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["#f0f0f0", "#0072B2"]),
        vmin=0,
        vmax=1,
    )
    position_axis.set_yticks(
        x_values,
        [PATTERN_LABELS[name] for name in ORACLE_POSITION_PATTERN_ORDER],
    )
    tick_offsets = (-32, -24, -16, -8, -1)
    position_axis.set_xticks(
        [offset + 32 for offset in tick_offsets],
        [str(offset) for offset in tick_offsets],
    )
    position_axis.set_xlabel("Prompt suffix position")
    position_axis.set_title("Replayed positions (blue)", loc="left")
    for spine in position_axis.spines.values():
        spine.set_visible(False)

    written = []
    for extension in formats:
        normalized = extension.lower().lstrip(".")
        if normalized not in {"svg", "pdf", "png"}:
            raise ValueError(f"unsupported figure format: {extension}")
        output_path = output_stem.with_suffix(f".{normalized}")
        options: dict[str, Any] = {"bbox_inches": "tight"}
        if normalized == "png":
            options["dpi"] = 300
        figure.savefig(output_path, **options)
        written.append(output_path)
    plt.close(figure)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument(
        "--formats", nargs="+", default=["svg", "pdf", "png"], metavar="FORMAT"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for output_path in plot_summary(
        load_summary(args.summary),
        args.output_stem,
        formats=args.formats,
    ):
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
