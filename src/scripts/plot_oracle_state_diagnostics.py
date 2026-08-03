"""Render layer-by-position target-oracle state diagnostics for LIP-PROTO-008."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.oracle_state_diagnostics import (
    ORACLE_STATE_DIAGNOSTICS_VERSION,
    ORACLE_STATE_TYPES,
)


DEFAULT_DIAGNOSTICS = Path("runs/LIP-PROTO-008/state-diagnostics.json")
DEFAULT_OUTPUT_STEM = Path("paper/figures/LIP-PROTO-008_state_diagnostics")


def load_diagnostics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        diagnostics = json.load(handle)
    if diagnostics.get("protocol_version") != ORACLE_STATE_DIAGNOSTICS_VERSION:
        raise ValueError("unsupported oracle state diagnostics protocol")
    if diagnostics.get("state_types") != list(ORACLE_STATE_TYPES):
        raise ValueError("diagnostics state types do not match the frozen contract")
    return diagnostics


def metric_grid(
    diagnostics: Mapping[str, Any],
    *,
    state_type: str,
    metric: str,
) -> list[list[float]]:
    layers = [int(layer) for layer in diagnostics["layer_indices"]]
    offsets = [int(offset) for offset in diagnostics["packet_offsets"]]
    cells = {
        (str(cell["state_type"]), int(cell["layer_index"]), int(cell["packet_offset"])): cell
        for cell in diagnostics["cells"]
    }
    expected = len(ORACLE_STATE_TYPES) * len(layers) * len(offsets)
    if len(cells) != expected:
        raise ValueError("diagnostics do not form one complete state/layer/position grid")
    grid = []
    for layer in layers:
        row = []
        for offset in offsets:
            key = (state_type, layer, offset)
            if key not in cells or metric not in cells[key]:
                raise ValueError(f"diagnostics are missing {metric} at {key}")
            value = float(cells[key][metric])
            if metric == "mean_pairwise_cosine":
                value = 1.0 - value
            row.append(value)
        grid.append(row)
    return grid


def _metric_limits(metric: str) -> tuple[float, float]:
    if metric in {"task_signal_fraction", "task_effective_rank_fraction"}:
        return 0.0, 1.0
    if metric == "mean_pairwise_cosine":
        return 0.0, 2.0
    raise ValueError(f"no registered plotting limits for metric: {metric}")


def plot_diagnostics(
    diagnostics: Mapping[str, Any],
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
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    metric_specs = (
        ("task_signal_fraction", "Task-centered energy", "viridis"),
        ("mean_pairwise_cosine", "Angular separation (1 − cosine)", "magma"),
        ("task_effective_rank_fraction", "Normalized effective rank", "cividis"),
    )
    state_labels = {
        "residual_input": "Residual input",
        "key_pre_rope": "Key projection",
        "value_pre_cache": "Value projection",
    }
    grids_by_metric = {
        metric: [
            metric_grid(diagnostics, state_type=state_type, metric=metric)
            for state_type in ORACLE_STATE_TYPES
        ]
        for metric, _, _ in metric_specs
    }
    limits = {metric: _metric_limits(metric) for metric in grids_by_metric}
    layers = [int(layer) for layer in diagnostics["layer_indices"]]
    offsets = [int(offset) for offset in diagnostics["packet_offsets"]]
    figure, axes = plt.subplots(
        len(ORACLE_STATE_TYPES),
        len(metric_specs),
        figsize=(7.15, 6.6),
        constrained_layout=True,
        squeeze=False,
    )
    for column, (metric, title, color_map) in enumerate(metric_specs):
        vmin, vmax = limits[metric]
        images = []
        for row, state_type in enumerate(ORACLE_STATE_TYPES):
            axis = axes[row][column]
            image = axis.imshow(
                grids_by_metric[metric][row],
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap=color_map,
                vmin=vmin,
                vmax=vmax,
            )
            images.append(image)
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(
                    f"{state_labels[state_type]}\nDecoder layer",
                    labelpad=5,
                )
            if row == len(ORACLE_STATE_TYPES) - 1:
                axis.set_xlabel("Prompt suffix position")
            x_indices = sorted(set((0, len(offsets) // 2, len(offsets) - 1)))
            y_indices = sorted(set((0, len(layers) // 2, len(layers) - 1)))
            axis.set_xticks(x_indices, [str(offsets[index]) for index in x_indices])
            axis.set_yticks(y_indices, [str(layers[index]) for index in y_indices])
        figure.colorbar(
            images[0],
            ax=[axes[row][column] for row in range(len(ORACLE_STATE_TYPES))],
            shrink=0.78,
            pad=0.02,
        )

    written = []
    for extension in formats:
        normalized = extension.lower().lstrip(".")
        if normalized not in {"svg", "pdf", "png"}:
            raise ValueError(f"unsupported figure format: {extension}")
        output_path = output_stem.with_suffix(f".{normalized}")
        options = {"bbox_inches": "tight"}
        if normalized == "png":
            options["dpi"] = 300
        figure.savefig(output_path, **options)
        written.append(output_path)
    plt.close(figure)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument(
        "--formats", nargs="+", default=["svg", "pdf", "png"], metavar="FORMAT"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostics = load_diagnostics(args.diagnostics)
    for output_path in plot_diagnostics(
        diagnostics, args.output_stem, formats=args.formats
    ):
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
