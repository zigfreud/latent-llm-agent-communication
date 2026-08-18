"""Generate or resume the frozen LIP-EVAL-035 development grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipelines.constant_entry_point_screen import (
    run_constant_entry_point_screen,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/LIP-EVAL-035_constant_opaque_entry_point_receiver_screen.yaml"
        ),
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--prediction-batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-new-records", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = run_constant_entry_point_screen(
        args.config,
        artifact_root=args.artifact_root,
        output_path=args.output,
        device=args.device,
        prediction_batch_size=args.prediction_batch_size,
        resume=args.resume,
        overwrite=args.overwrite,
        max_new_records=args.max_new_records,
    )
    print(
        json.dumps(
            {
                "records": metadata["records"],
                "expected_records": metadata["expected_records"],
                "complete": metadata["complete"],
                "claim_eligible": metadata["claim_eligible"],
                "metadata": str(args.output.with_suffix(".metadata.json")),
            }
        )
    )


if __name__ == "__main__":
    main()
