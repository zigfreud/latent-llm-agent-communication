"""Extract the content-addressed LIP-PROTO-014 source/teacher packet bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipelines.packet_extraction import materialize_packet_bundle


DEFAULT_CONFIG = Path("config/LIP-PROTO-014_source_conditioned_residual_packet.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--preflight-tasks-per-split", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize_packet_bundle(
        args.config,
        bundle_dir=args.bundle_dir,
        dry_run=args.dry_run,
        resume=args.resume,
        overwrite=args.overwrite,
        keep_staging=args.keep_staging,
        preflight_tasks_per_split=args.preflight_tasks_per_split,
    )
    print("LIP packet bundle extraction passed")
    print(f"mode: {result['extraction_mode']}")
    print(f"scope: {result['extraction_scope']}")
    print(f"records: {result['records']}")
    print(f"manifest: {result['manifest']}")


if __name__ == "__main__":
    main()
