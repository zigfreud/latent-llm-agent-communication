"""Validate or run the development-only LIP-H0-017 trajectory corrector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipelines.closed_loop_trajectory import (
    run_closed_loop_training,
    validate_closed_loop_contract,
)
from src.pipelines.oracle_experiment import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("config/LIP-H0-017_closed_loop_trajectory_corrector.yaml"),
    )
    parser.add_argument(
        "--parent-config",
        type=Path,
        default=Path("config/LIP-PROTO-014_source_conditioned_residual_packet.yaml"),
    )
    parser.add_argument(
        "--learned-registry",
        type=Path,
        default=Path(
            "experiments/registry/LIP-H0-016_hard_negative_replication.json"
        ),
    )
    parser.add_argument(
        "--functional-registry",
        type=Path,
        default=Path(
            "experiments/registry/LIP-EVAL-037_oracle_native_packet_blend_screen.json"
        ),
    )
    parser.add_argument(
        "--source-registry",
        type=Path,
        default=Path("experiments/registry/LIP-H0-015_hard_negative_batches.json"),
    )
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--candidate-bank", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--variant", default="closed_loop_live")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--target-device", default="cuda")
    parser.add_argument("--colab-compute-units-before", type=float)
    parser.add_argument("--dry-run-contract", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment = load_yaml(args.experiment_config)
    parent = load_yaml(args.parent_config)
    validate_closed_loop_contract(
        experiment,
        parent,
        experiment_path=args.experiment_config,
        parent_path=args.parent_config,
        learned_registry_path=args.learned_registry,
        functional_registry_path=args.functional_registry,
        source_registry_path=args.source_registry,
    )
    if args.dry_run_contract:
        print("LIP-H0-017 contract validated")
        return
    required = {
        "bundle_dir": args.bundle_dir,
        "source_checkpoint": args.source_checkpoint,
        "output_dir": args.output_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("missing runtime path(s): " + ", ".join(missing))
    result = run_closed_loop_training(
        experiment_path=args.experiment_config,
        parent_path=args.parent_config,
        learned_registry_path=args.learned_registry,
        functional_registry_path=args.functional_registry,
        source_registry_path=args.source_registry,
        bundle_dir=args.bundle_dir,
        source_checkpoint_path=args.source_checkpoint,
        candidate_bank_path=args.candidate_bank,
        output_dir=args.output_dir,
        variant_name=str(args.variant),
        pilot=bool(args.pilot),
        target_device=str(args.target_device),
        colab_compute_units_before=args.colab_compute_units_before,
    )
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "stage": result["stage"],
                "variant": result["variant"],
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
