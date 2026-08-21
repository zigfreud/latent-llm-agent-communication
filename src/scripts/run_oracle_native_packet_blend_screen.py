"""Generate or resume one phase of the frozen LIP-EVAL-037 screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipelines.oracle_native_packet_blend_screen import (
    run_oracle_native_packet_blend_screen,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/LIP-EVAL-037_oracle_native_packet_blend_screen.yaml"),
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("screen", "confirm"), required=True)
    parser.add_argument("--screen-lock", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-new-records", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = run_oracle_native_packet_blend_screen(
        args.config,
        artifact_root=args.artifact_root,
        output_path=args.output,
        phase=args.phase,
        screen_lock_path=args.screen_lock,
        device=args.device,
        resume=args.resume,
        overwrite=args.overwrite,
        max_new_records=args.max_new_records,
    )
    print(
        json.dumps(
            {
                "records": metadata["records"],
                "expected_records": metadata["expected_records"],
                "active_phase": metadata["active_phase"],
                "screen_phase_complete": metadata["screen_phase_complete"],
                "confirmation_phase_complete": metadata[
                    "confirmation_phase_complete"
                ],
                "complete": metadata["complete"],
                "claim_eligible": metadata["claim_eligible"],
                "metadata": str(args.output.with_suffix(".metadata.json")),
            }
        )
    )


if __name__ == "__main__":
    main()
