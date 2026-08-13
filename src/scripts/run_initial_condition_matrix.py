"""CLI for the frozen H0-010 development matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipelines.initial_condition_matrix import (
    finalize_matrix_compute_units,
    run_initial_condition_matrix,
    validate_pilot_authorization,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("config/LIP-H0-010_initial_condition_bridge.yaml"),
    )
    parser.add_argument(
        "--parent-config",
        type=Path,
        default=Path("config/LIP-PROTO-014_source_conditioned_residual_packet.yaml"),
    )
    parser.add_argument(
        "--predecessor-registry",
        type=Path,
        default=Path("experiments/registry/LIP-H0-009_entry_seed_free_evolution.json"),
    )
    parser.add_argument(
        "--pilot-registry",
        type=Path,
        default=Path(
            "experiments/registry/LIP-H0-010_initial_condition_bridge_pilot_v3.json"
        ),
    )
    parser.add_argument("--pilot-summary", type=Path)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--target-device", default="auto")
    parser.add_argument("--colab-compute-units-before", type=float)
    parser.add_argument("--dry-run-authorization", action="store_true")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--finalize-compute-units-after", type=float)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.finalize_compute_units_after is not None:
        if args.summary is None:
            raise ValueError("--summary is required to finalize compute units")
        finalize_matrix_compute_units(args.summary, args.finalize_compute_units_after)
        print(args.summary)
        return
    if args.pilot_summary is None:
        raise ValueError("--pilot-summary is required")
    if args.dry_run_authorization:
        validate_pilot_authorization(args.pilot_registry, args.pilot_summary)
        print("LIP-H0-010 matrix authorization validated")
        return
    if args.bundle_dir is None or args.output_root is None:
        raise ValueError("--bundle-dir and --output-root are required")
    result = run_initial_condition_matrix(
        experiment_path=args.experiment_config,
        parent_path=args.parent_config,
        predecessor_registry_path=args.predecessor_registry,
        pilot_registry_path=args.pilot_registry,
        pilot_summary_path=args.pilot_summary,
        bundle_dir=args.bundle_dir,
        output_root=args.output_root,
        target_device=str(args.target_device),
        colab_compute_units_before=args.colab_compute_units_before,
    )
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "development_gate_passed": result["development_gate"]["passed"],
                "summary": str(args.output_root / "matrix_summary.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
