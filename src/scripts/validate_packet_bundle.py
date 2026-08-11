"""CLI validation for learned LIP packet bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.packet_bundle import (
    PacketBundleValidationError,
    validate_packet_bundle,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a LIP packet bundle")
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--require-real", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        report = validate_packet_bundle(args.bundle_dir, require_real=args.require_real)
    except PacketBundleValidationError as exc:
        if args.report_json:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(
                json.dumps(
                    {
                        "bundle_dir": str(args.bundle_dir),
                        "validation_status": "failed",
                        "error": str(exc),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        raise SystemExit(f"Invalid packet bundle: {exc}") from exc
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("LIP packet bundle validation passed")
    print(f"trace_id: {report['trace_id']}")
    print(f"records: {report['records']}")
    print(f"source_shape: {report['source_shape']}")
    print(f"target_shape: {report['target_shape']}")


if __name__ == "__main__":
    main()
