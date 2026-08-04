"""Render claim-eligible functional capacity across oracle replay depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.oracle_layer_depth import ORACLE_LAYER_DEPTH_SCOPE_ORDER


DEFAULT_SUMMARY = Path(
    "runs/LIP-PROTO-009/functional-evaluation/summary.json"
)
DEFAULT_OUTPUT_STEM = Path(
    "paper/figures/LIP-PROTO-009_functional_layer_depth"
)
SCOPE_DEPTHS = dict(zip(ORACLE_LAYER_DEPTH_SCOPE_ORDER, (8, 16, 24, 32)))


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("experiment_id") != "LIP-PROTO-009":
        raise ValueError("summary must come from LIP-PROTO-009")
    if summary.get("execution_mode") != "functional_hardened_namespace":
        raise ValueError("summary must come from the hardened evaluator")
    if not summary.get("claim_eligible"):
        raise ValueError("layer-depth figure requires a claim-eligible full run")
    return summary


def depth_rates(
    summary: Mapping[str, Any],
    *,
    shuffled: bool = False,
) -> tuple[list[int], list[float], list[float], list[float]]:
    conditions = summary["metrics"]["functional_pass"]["conditions"]
    depths = []
    means = []
    lower_errors = []
    upper_errors = []
    prefix = "shuffled_oracle" if shuffled else "oracle"
    for scope in ORACLE_LAYER_DEPTH_SCOPE_ORDER:
        condition = conditions[f"{prefix}_{scope}_k32"]
        mean = float(condition["mean"])
        depths.append(SCOPE_DEPTHS[scope])
        means.append(mean)
        lower_errors.append(mean - float(condition["ci_lower"]))
        upper_errors.append(float(condition["ci_upper"]) - mean)
    return depths, means, lower_errors, upper_errors


def control_interval(
    summary: Mapping[str, Any], condition_name: str
) -> tuple[float, float, float]:
    condition = summary["metrics"]["functional_pass"]["conditions"][condition_name]
    return (
        float(condition["mean"]),
        float(condition["ci_lower"]),
        float(condition["ci_upper"]),
    )


def primary_annotations(summary: Mapping[str, Any]) -> dict[int, str]:
    annotations = {}
    for hypothesis in summary["primary_inference"]["hypotheses"]:
        treatment = str(hypothesis["treatment"])
        scope = treatment.removeprefix("oracle_").removesuffix("_k32")
        depth = SCOPE_DEPTHS[scope]
        annotations[depth] = (
            f"p={float(hypothesis['p_value']):.3g}"
            if hypothesis["tested"]
            else "gate stopped"
        )
    return annotations


def plot_summary(
    summary: Mapping[str, Any],
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
    figure, axis = plt.subplots(figsize=(4.6, 3.25), constrained_layout=True)
    for values, label, color, marker, offset in (
        (depth_rates(summary), "Matched replay", "#0072B2", "o", -0.18),
        (
            depth_rates(summary, shuffled=True),
            "Task-shuffled replay",
            "#888888",
            "x",
            0.18,
        ),
    ):
        depths, means, lower, upper = values
        axis.errorbar(
            [depth + offset for depth in depths],
            means,
            yerr=[lower, upper],
            color=color,
            marker=marker,
            linewidth=1.7,
            markersize=5.2,
            capsize=2.5,
            label=label,
        )

    neutral_mean, _, _ = control_interval(summary, "neutral_no_lip")
    text_mean, text_low, text_high = control_interval(summary, "text_only_no_lip")
    axis.axhline(
        neutral_mean,
        color="#444444",
        linewidth=0.9,
        linestyle=":",
        label="Neutral control",
    )
    axis.axhspan(text_low, text_high, color="#009E73", alpha=0.12, linewidth=0)
    axis.axhline(
        text_mean,
        color="#009E73",
        linewidth=1.3,
        linestyle="--",
        label=f"Task text ({text_mean:.0%})",
    )

    matched_depths, matched_means, _, _ = depth_rates(summary)
    for depth, mean in zip(matched_depths, matched_means):
        axis.annotate(
            primary_annotations(summary)[depth],
            (depth, mean),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=6.8,
            color="#005A8C",
        )

    axis.set_xticks(list(SCOPE_DEPTHS.values()))
    axis.set_xlim(6, 34)
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel("Decoder blocks receiving the K=32 latent packet")
    axis.set_ylabel("Task-clustered functional pass rate")
    axis.set_title("Functional capacity by replay depth", loc="left")
    axis.grid(axis="y", color="#e1e1e1", linewidth=0.6)
    axis.legend(frameon=False, loc="upper left")

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
    summary = load_summary(args.summary)
    for output_path in plot_summary(
        summary,
        args.output_stem,
        formats=args.formats,
    ):
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
