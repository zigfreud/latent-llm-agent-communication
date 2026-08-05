"""Render the constant-capacity terminal-source factorial for LIP-PROTO-013."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.oracle_terminal_factorial import (
    ORACLE_TERMINAL_ASSIGNMENTS,
    ORACLE_TERMINAL_EXPERIMENT_ID,
    ORACLE_TERMINAL_HYPOTHESIS_LABELS,
    ORACLE_TERMINAL_SCOPE_NAME,
)


DEFAULT_SUMMARY = Path("runs/LIP-PROTO-013/evaluation/summary.json")
DEFAULT_OUTPUT_STEM = Path("paper/figures/LIP-PROTO-013_terminal_source_factorial")
CLAIM_LABELS = {
    "core_contribution": "Core contribution",
    "name_contribution": "Function-name contribution",
    "boundary_contribution": "Boundary contribution",
    "core_only_sufficiency": "Core-only rescue",
    "name_only_sufficiency": "Name-only rescue",
    "boundary_only_sufficiency": "Boundary-only rescue",
    "tail_only_sufficiency": "Name + boundary rescue",
}


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("experiment_id") != ORACLE_TERMINAL_EXPERIMENT_ID:
        raise ValueError("summary must come from LIP-PROTO-013")
    if summary.get("execution_mode") != "functional_hardened_namespace":
        raise ValueError("summary must come from the hardened evaluator")
    if not summary.get("claim_eligible"):
        raise ValueError("terminal-factorial figure requires a claim-eligible full run")
    return summary


def factorial_rates(
    summary: Mapping[str, Any],
) -> tuple[list[float], list[float], list[float]]:
    conditions = summary["metrics"]["functional_pass"]["conditions"]
    means = []
    lower_errors = []
    upper_errors = []
    for assignment in ORACLE_TERMINAL_ASSIGNMENTS:
        condition = conditions[
            f"oracle_{ORACLE_TERMINAL_SCOPE_NAME}_terminal_k24_{assignment}"
        ]
        value = float(condition["mean"])
        means.append(value)
        lower_errors.append(value - float(condition["ci_lower"]))
        upper_errors.append(float(condition["ci_upper"]) - value)
    return means, lower_errors, upper_errors


def primary_contrasts(
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    family = summary["primary_inference"]["family"]
    if len(family) != len(ORACLE_TERMINAL_HYPOTHESIS_LABELS):
        raise ValueError("primary inference does not contain seven component contrasts")
    return [
        {
            "label": label,
            "mean_difference": float(item["mean_difference"]),
            "ci_lower": float(item["ci_lower"]),
            "ci_upper": float(item["ci_upper"]),
            "tested": bool(item["tested"]),
            "rejected": bool(item["rejected"]),
            "p_value_holm": float(item["p_value_holm"]),
        }
        for label, item in zip(ORACLE_TERMINAL_HYPOTHESIS_LABELS, family)
    ]


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
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, (rate_axis, contrast_axis) = plt.subplots(
        2,
        1,
        figsize=(6.5, 5.5),
        gridspec_kw={"height_ratios": [1.45, 1.55]},
        constrained_layout=True,
    )
    means, lower, upper = factorial_rates(summary)
    colors = [
        "#0072B2" if assignment == "mmm" else
        "#888888" if assignment == "sss" else
        "#56B4E9"
        for assignment in ORACLE_TERMINAL_ASSIGNMENTS
    ]
    x_values = list(range(len(means)))
    rate_axis.bar(x_values, means, color=colors, width=0.72)
    rate_axis.errorbar(
        x_values,
        means,
        yerr=[lower, upper],
        fmt="none",
        ecolor="#222222",
        capsize=2.5,
        linewidth=0.9,
    )
    rate_axis.set_xticks(
        x_values,
        [assignment.upper() for assignment in ORACLE_TERMINAL_ASSIGNMENTS],
    )
    rate_axis.set_ylim(-0.05, 1.05)
    rate_axis.set_ylabel("Task-clustered functional pass rate")
    rate_axis.set_title(
        "K=24 source-identity factorial (M=matched, S=same-stratum donor)",
        loc="left",
    )
    rate_axis.grid(axis="y", color="#e1e1e1", linewidth=0.6)
    rate_axis.text(
        0.99,
        0.96,
        "letters: core / function name / boundary",
        transform=rate_axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color="#444444",
    )

    contrasts = primary_contrasts(summary)
    y_values = list(reversed(range(len(contrasts))))
    for y_value, item in zip(y_values, contrasts):
        color = "#0072B2" if item["rejected"] else "#777777"
        contrast_axis.plot(
            [item["ci_lower"], item["ci_upper"]],
            [y_value, y_value],
            color=color,
            linewidth=1.8,
        )
        contrast_axis.scatter(
            [item["mean_difference"]],
            [y_value],
            color=color,
            s=25,
            zorder=3,
        )
        annotation = (
            f"Holm p={item['p_value_holm']:.3g}"
            if item["tested"]
            else "replication gate closed"
        )
        contrast_axis.annotate(
            annotation,
            (item["ci_upper"], y_value),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=6.8,
            color=color,
        )
    contrast_axis.axvline(0.0, color="#222222", linewidth=0.9, linestyle=":")
    contrast_axis.set_yticks(
        y_values,
        [CLAIM_LABELS[item["label"]] for item in contrasts],
    )
    contrast_axis.set_xlabel("Paired task-level pass-rate difference")
    contrast_axis.set_title(
        "Confirmatory component contrasts (one seven-test Holm family)",
        loc="left",
    )
    contrast_axis.grid(axis="x", color="#e1e1e1", linewidth=0.6)

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
