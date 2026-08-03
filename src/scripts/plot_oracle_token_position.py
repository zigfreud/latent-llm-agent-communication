"""Render paper-ready LIP-PROTO-006 token-position recovery figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_SUMMARY = Path("runs/LIP-PROTO-006/oracle-token-position/summary.json")
DEFAULT_OUTPUT_STEM = Path("paper/figures/LIP-PROTO-006_token_position_recovery")
PROFILE_PACKET_SIZES = (8, 16, 32)


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("experiment_id") != "LIP-PROTO-006":
        raise ValueError("summary must come from LIP-PROTO-006")
    if summary.get("analysis_axis") != "target_continuation_token_position":
        raise ValueError("summary is missing the token-position analysis axis")
    if summary.get("gate_window") != "first_8_tokens":
        raise ValueError("figure contract requires the frozen first-eight-token gate")
    return summary


def recovery_by_packet_size(
    summary: Mapping[str, Any], *, split: str, window: str
) -> tuple[list[int], list[float]]:
    packet_sizes = [int(size) for size in summary["packet_sizes"]]
    values = []
    for size in packet_sizes:
        value = summary["by_packet_size"][str(size)][split]["windows"][window][
            "recovery_fraction"
        ]
        values.append(math.nan if value is None else float(value))
    return packet_sizes, values


def recovery_by_token_position(
    summary: Mapping[str, Any], *, packet_size: int, split: str
) -> tuple[list[int], list[float]]:
    by_position = summary["by_packet_size"][str(packet_size)][split][
        "by_token_position"
    ]
    positions = sorted(int(position) for position in by_position)
    values = []
    for position in positions:
        metric = by_position[str(position)]
        value = metric["recovery_fraction"] if metric["informative"] else None
        values.append(math.nan if value is None else float(value))
    return positions, values


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
    fig = plt.figure(figsize=(7.15, 5.15), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
    capacity_ax = fig.add_subplot(grid[0, :])
    selection_ax = fig.add_subplot(grid[1, 0])
    confirmation_ax = fig.add_subplot(grid[1, 1], sharey=selection_ax)

    split_styles = {
        "selection": {"color": "#1565c0", "marker": "o"},
        "confirmation": {"color": "#d1495b", "marker": "s"},
    }
    for split, style in split_styles.items():
        sizes, early = recovery_by_packet_size(
            summary, split=split, window="first_8_tokens"
        )
        _, full = recovery_by_packet_size(
            summary, split=split, window="full_sequence"
        )
        capacity_ax.plot(
            sizes,
            early,
            linewidth=2.0,
            markersize=4.5,
            label=f"{split.capitalize()} · first 8",
            **style,
        )
        capacity_ax.plot(
            sizes,
            full,
            color=style["color"],
            linewidth=1.1,
            linestyle="--",
            alpha=0.6,
            label=f"{split.capitalize()} · full",
        )
    capacity_ax.axhline(
        float(summary["gate"]["minimum_recovery"]),
        color="#666666",
        linewidth=1.0,
        linestyle=":",
        label="10% gate",
    )
    selected = summary.get("selected_packet_size")
    if selected is not None:
        capacity_ax.axvline(
            int(selected), color="#333333", linewidth=0.8, linestyle=":"
        )
        capacity_ax.annotate(
            f"selected K={selected}",
            xy=(int(selected), 0.02),
            xytext=(7, -5),
            textcoords="offset points",
            fontsize=7.5,
        )
    capacity_ax.set_xscale("log", base=2)
    capacity_ax.set_xticks(sizes, labels=[str(size) for size in sizes])
    capacity_ax.set_ylim(-0.08, 1.05)
    capacity_ax.set_xlabel("Latent packet size K (prompt states)")
    capacity_ax.set_ylabel("Recovered text advantage")
    capacity_ax.set_title("A  Early-prefix capacity crosses before full-sequence saturation", loc="left")
    capacity_ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    capacity_ax.legend(ncol=3, frameon=False, loc="upper left")

    profile_colors = {8: "#1565c0", 16: "#e07a1f", 32: "#2e7d32"}
    for axis, split, panel in (
        (selection_ax, "selection", "B"),
        (confirmation_ax, "confirmation", "C"),
    ):
        for packet_size in PROFILE_PACKET_SIZES:
            positions, values = recovery_by_token_position(
                summary, packet_size=packet_size, split=split
            )
            axis.plot(
                positions,
                values,
                color=profile_colors[packet_size],
                linewidth=1.15,
                marker="o",
                markersize=2.0,
                alpha=0.9,
                label=f"K={packet_size}",
            )
        axis.axhline(0.0, color="#888888", linewidth=0.7)
        axis.axhline(1.0, color="#aaaaaa", linewidth=0.7, linestyle=":")
        axis.axvspan(1, 8, color="#f1c40f", alpha=0.12, linewidth=0)
        axis.set_xlim(1, 64)
        axis.set_ylim(-0.25, 1.25)
        axis.set_xlabel("Continuation token position")
        axis.set_title(f"{panel}  {split.capitalize()} split", loc="left")
        axis.grid(axis="y", color="#e5e5e5", linewidth=0.5)
    selection_ax.set_ylabel("Per-position recovery")
    confirmation_ax.tick_params(labelleft=False)
    confirmation_ax.legend(frameon=False, loc="upper right")

    written = []
    for extension in formats:
        normalized = extension.lower().lstrip(".")
        if normalized not in {"svg", "pdf", "png"}:
            raise ValueError(f"unsupported figure format: {extension}")
        output_path = output_stem.with_suffix(f".{normalized}")
        save_options = {"bbox_inches": "tight"}
        if normalized == "png":
            save_options["dpi"] = 300
        fig.savefig(output_path, **save_options)
        written.append(output_path)
    plt.close(fig)
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
    for output_path in plot_summary(summary, args.output_stem, formats=args.formats):
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
