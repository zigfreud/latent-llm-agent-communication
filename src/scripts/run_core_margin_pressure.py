"""Run the frozen H0-013 core-margin-pressure pilot or screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipelines.core_margin_pressure import (
    run_core_margin_pressure_training,
    validate_core_margin_pressure_contract,
)
from src.pipelines.initial_condition_bridge import finalize_compute_units
from src.pipelines.oracle_experiment import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("config/LIP-H0-013_core_margin_pressure.yaml"),
    )
    parser.add_argument(
        "--parent-config",
        type=Path,
        default=Path("config/LIP-PROTO-014_source_conditioned_residual_packet.yaml"),
    )
    parser.add_argument(
        "--predecessor-registry",
        type=Path,
        default=Path("experiments/registry/LIP-H0-012_core_negative_replication.json"),
    )
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-device", default="auto")
    parser.add_argument("--colab-compute-units-before", type=float)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--dry-run-contract", action="store_true")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--finalize-compute-units-after", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment = load_yaml(args.experiment_config)
    parent = load_yaml(args.parent_config)
    validate_core_margin_pressure_contract(
        experiment,
        parent,
        experiment_path=args.experiment_config,
        parent_path=args.parent_config,
        predecessor_registry_path=args.predecessor_registry,
    )
    if args.dry_run_contract:
        print("LIP-H0-013 contract validated")
        return
    if args.finalize_compute_units_after is not None:
        if args.summary is None:
            raise ValueError("--summary is required to finalize compute units")
        finalize_compute_units(args.summary, args.finalize_compute_units_after)
        print(args.summary)
        return
    if args.bundle_dir is None or args.output_dir is None:
        raise ValueError("--bundle-dir and --output-dir are required")
    result = run_core_margin_pressure_training(
        experiment_path=args.experiment_config,
        parent_path=args.parent_config,
        predecessor_registry_path=args.predecessor_registry,
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
        variant_name="core_margin_pressure_unrolled",
        seed=4007,
        pilot=bool(args.pilot),
        target_device=str(args.target_device),
        colab_compute_units_before=args.colab_compute_units_before,
    )
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "stage": result["stage"],
                "pilot_passed": (
                    result["pilot_gate"]["passed"]
                    if result["pilot_gate"] is not None
                    else None
                ),
                "summary": str(args.output_dir / "run_summary.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
