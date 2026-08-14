"""Run the frozen LIP-EVAL-032 receiver-unrolled gradient diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipelines.gradient_geometry import (
    run_gradient_geometry_evaluation,
    validate_gradient_geometry_contract,
)
from src.pipelines.initial_condition_bridge import finalize_compute_units
from src.pipelines.oracle_experiment import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("config/LIP-EVAL-032_gradient_geometry.yaml"),
    )
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--h011-checkpoint-dir", type=Path)
    parser.add_argument("--h013-checkpoint-dir", type=Path)
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
    config = load_yaml(args.experiment_config)
    validate_gradient_geometry_contract(config)
    if args.dry_run_contract:
        print("LIP-EVAL-032 contract validated")
        return
    if args.finalize_compute_units_after is not None:
        if args.summary is None:
            raise ValueError("--summary is required to finalize compute units")
        finalize_compute_units(args.summary, args.finalize_compute_units_after)
        print(args.summary)
        return
    required = {
        "--bundle-dir": args.bundle_dir,
        "--h011-checkpoint-dir": args.h011_checkpoint_dir,
        "--h013-checkpoint-dir": args.h013_checkpoint_dir,
        "--output-dir": args.output_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"required arguments missing: {', '.join(missing)}")
    result = run_gradient_geometry_evaluation(
        experiment_path=args.experiment_config,
        bundle_dir=args.bundle_dir,
        h011_checkpoint_dir=args.h011_checkpoint_dir,
        h013_checkpoint_dir=args.h013_checkpoint_dir,
        output_dir=args.output_dir,
        pilot=bool(args.pilot),
        target_device=str(args.target_device),
        colab_compute_units_before=args.colab_compute_units_before,
    )
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "stage": result["stage"],
                "route": (
                    result["routing"]["route"]
                    if result["routing"] is not None
                    else None
                ),
                "summary": str(args.output_dir / "run_summary.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
