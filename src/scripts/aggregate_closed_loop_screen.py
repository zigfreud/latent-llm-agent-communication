"""Aggregate the frozen H0-017 paired development screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipelines.closed_loop_screen import aggregate_closed_loop_screen
from src.pipelines.oracle_experiment import load_json_object, load_yaml
from src.pipelines.packet_trajectory import _atomic_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregation-config",
        type=Path,
        default=Path("config/LIP-H0-017_paired_screen_aggregation.yaml"),
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("config/LIP-H0-017_closed_loop_trajectory_corrector.yaml"),
    )
    parser.add_argument("--pilot-summary", type=Path, required=True)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--treatment-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = aggregate_closed_loop_screen(
        load_yaml(args.aggregation_config),
        load_yaml(args.experiment_config),
        load_json_object(args.pilot_summary),
        load_json_object(args.control_summary),
        load_json_object(args.treatment_summary),
        aggregation_path=args.aggregation_config,
        experiment_path=args.experiment_config,
    )
    _atomic_json(args.output, payload)
    print(json.dumps(payload["aggregate_gate"]))


if __name__ == "__main__":
    main()
