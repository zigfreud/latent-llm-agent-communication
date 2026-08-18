"""Generate or resume one phase of the frozen LIP-EVAL-036 screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipelines.constrained_prefix_receiver_screen import (
    run_constrained_prefix_receiver_screen,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/LIP-EVAL-036_constrained_prefix_receiver_screen.yaml"
        ),
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("controls", "learned"), required=True)
    parser.add_argument("--control-lock", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--prediction-batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-new-records", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = run_constrained_prefix_receiver_screen(
        args.config,
        artifact_root=args.artifact_root,
        output_path=args.output,
        phase=args.phase,
        control_lock_path=args.control_lock,
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
                "active_phase": metadata["active_phase"],
                "control_phase_complete": metadata["control_phase_complete"],
                "learned_phase_complete": metadata["learned_phase_complete"],
                "complete": metadata["complete"],
                "claim_eligible": metadata["claim_eligible"],
                "metadata": str(args.output.with_suffix(".metadata.json")),
            }
        )
    )


if __name__ == "__main__":
    main()
