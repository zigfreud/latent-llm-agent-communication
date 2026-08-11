"""Run the registered LIP-PROTO-014 bridge objective/replica matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipelines.packet_matrix import run_packet_bridge_matrix


DEFAULT_CONFIG = Path("config/LIP-PROTO-014_source_conditioned_residual_packet.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--allow-nonclaim-bundle",
        action="store_true",
        help="Permit a dry-run or preflight bundle; never marks the matrix complete.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-updates", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_packet_bridge_matrix(
        args.config,
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
        variants=args.variants,
        seeds=args.seeds,
        device=args.device,
        allow_nonclaim_bundle=args.allow_nonclaim_bundle,
        resume=args.resume,
        overwrite=args.overwrite,
        max_updates=args.max_updates,
    )
    print("LIP packet bridge matrix completed")
    print(f"full_registered_matrix: {result['full_registered_matrix']}")
    print(f"ready_for_confirmation: {result['ready_for_confirmation']}")


if __name__ == "__main__":
    main()
