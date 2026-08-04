import pytest

from src.evaluation.oracle_state_diagnostics import (
    ORACLE_STATE_DIAGNOSTICS_VERSION,
    ORACLE_STATE_TYPES,
)
from src.scripts.plot_oracle_state_diagnostics import _metric_limits, metric_grid


def synthetic_diagnostics():
    layers = [0, 1]
    offsets = [-2, -1]
    cells = []
    for state_index, state_type in enumerate(ORACLE_STATE_TYPES):
        for layer in layers:
            for offset in offsets:
                cells.append(
                    {
                        "state_type": state_type,
                        "layer_index": layer,
                        "packet_offset": offset,
                        "task_signal_fraction": state_index + layer + abs(offset),
                        "mean_pairwise_cosine": 0.9,
                        "task_effective_rank_fraction": 0.5,
                    }
                )
    return {
        "protocol_version": ORACLE_STATE_DIAGNOSTICS_VERSION,
        "state_types": list(ORACLE_STATE_TYPES),
        "layer_indices": layers,
        "packet_offsets": offsets,
        "cells": cells,
    }


def test_metric_grid_preserves_layer_and_suffix_position_order():
    grid = metric_grid(
        synthetic_diagnostics(),
        state_type="residual_input",
        metric="task_signal_fraction",
    )
    assert grid == [[2.0, 1.0], [3.0, 2.0]]


def test_metric_grid_converts_cosine_to_angular_separation():
    grid = metric_grid(
        synthetic_diagnostics(),
        state_type="key_pre_rope",
        metric="mean_pairwise_cosine",
    )
    assert grid[0][0] == pytest.approx(0.1)


def test_metric_grid_rejects_incomplete_diagnostics():
    diagnostics = synthetic_diagnostics()
    diagnostics["cells"].pop()
    with pytest.raises(ValueError, match="complete"):
        metric_grid(
            diagnostics,
            state_type="value_pre_cache",
            metric="task_signal_fraction",
        )


def test_plotting_limits_preserve_registered_absolute_scales():
    assert _metric_limits("task_signal_fraction") == (0.0, 1.0)
    assert _metric_limits("task_effective_rank_fraction") == (0.0, 1.0)
    assert _metric_limits("mean_pairwise_cosine") == (0.0, 2.0)
